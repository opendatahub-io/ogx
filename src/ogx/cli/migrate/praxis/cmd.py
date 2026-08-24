# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""``ogx migrate praxis`` — argparse surface + orchestration.

This is generic OGX plumbing: it bootstraps the source deployment's config/store
stack exactly as the server does (so it connects to the source the same way),
reads the source tables raw (ACL/tenant-bypassing), and drives the pure
transforms + ``PraxisWriter`` from :mod:`.target`. All Praxis-specific
schema coupling lives in that module; this file has no knowledge of the target
schema beyond the logical table names.

Read-only to the source (see ``reader._SourceReader`` for the
create-no-tables safety argument), write-only to the target.

The per-phase read+transform+write logic lives in sibling modules, one per
source table: :mod:`.responses`, :mod:`.conversations`,
:mod:`.items`. :mod:`.reader` holds the infrastructure shared by
all three (the raw reader, run stats/options, progress bar).
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

import yaml

from ogx.cli.subcommand import Subcommand
from ogx.core.configure import parse_and_maybe_upgrade_config
from ogx.core.datatypes import StackConfig, TenancyMode
from ogx.core.stack import _initialize_storage
from ogx.core.storage.datatypes import ResponsesStoreReference, SqlStoreReference
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx.core.storage.sqlstore.sqlstore import get_system_sqlstore, shutdown_sqlstore_backends
from ogx.core.utils.config_resolution import resolve_config_or_distro
from ogx.log import get_logger

from .conversations import _run_conversations_phase
from .items import _run_items_phase
from .reader import _build_progress, _ReaderFor, _RunOptions, _SourceReader, _Stats
from .responses import _run_responses_phase
from .target import PraxisWriter, TenantDeriver

logger = get_logger(name=__name__, category="cli")

_VALID_TABLES = ("responses", "conversations", "items")


def _parse_tables(raw: str) -> set[str]:
    names = {t.strip() for t in raw.split(",") if t.strip()}
    if not names:
        raise ValueError("Failed to parse --tables: at least one table is required")
    invalid = names - set(_VALID_TABLES)
    if invalid:
        raise ValueError(
            f"Failed to parse --tables: unknown table(s) {sorted(invalid)}; valid values are {list(_VALID_TABLES)}"
        )
    return names


def _load_owner_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    map_path = Path(path)
    if not map_path.is_file():
        raise ValueError(f"Failed to load --owner-tenant-map: {path!r} is not a file")
    data = yaml.safe_load(map_path.read_text())
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(
            f"Failed to load --owner-tenant-map {path!r}: expected a mapping of owner_principal (str) -> tenant_id (str)"
        )
    return data


def _log_summary(stats: _Stats, dry_run: bool) -> None:
    logger.info(
        "Praxis migration complete",
        dry_run=dry_run,
        responses_read=stats.read["responses"],
        responses_transformed=stats.transformed["responses"],
        responses_submitted=stats.submitted["responses"],
        conversations_read=stats.read["conversations"],
        conversations_transformed=stats.transformed["conversations"],
        conversations_orphans_read=stats.read["conversations_orphans"],
        conversations_submitted=stats.submitted["conversations"] + stats.submitted["conversations_orphans"],
        items_read=stats.read["items"],
        items_transformed=stats.transformed["items"],
        items_submitted=stats.submitted["items"],
        legacy_items_transformed=stats.transformed["items_legacy"],
        legacy_items_submitted=stats.submitted["items_legacy"],
        skipped=len(stats.skipped),
    )
    if stats.skipped:
        logger.warning(
            "Skipped-row manifest (transform/validation errors tolerated via --skip-errors)", count=len(stats.skipped)
        )
        for kind, row_id, error in stats.skipped:
            logger.warning("Skipped row", kind=kind, row_id=row_id, error=error)


def _resolve_responses_store(run_config: StackConfig) -> ResponsesStoreReference | None:
    """Resolve the responses store reference from the responses provider config.

    OGX persists responses (and the conversation_messages table, which the
    ResponsesStore creates on the responses backend) via the responses provider's
    own ``config.persistence.responses`` — not ``storage.stores.responses``, which
    stock and operator-generated configs leave unset. Reading from the provider
    config is therefore the only reliable way to find where the data actually lives.
    """
    for provider in run_config.providers.get("responses", []):
        persistence = provider.config.get("persistence") or {}
        responses_cfg = persistence.get("responses")
        if responses_cfg:
            return ResponsesStoreReference(**responses_cfg)
    return None


async def _run(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("Failed to start migration: --batch-size must be a positive integer")

    tables_scope = _parse_tables(args.tables)
    tenant = TenantDeriver(sentinel=args.tenant_sentinel, explicit_map=_load_owner_map(args.owner_tenant_map))
    orphan_created_at = args.orphan_created_at if args.orphan_created_at is not None else int(time.time())

    # Bootstrap the source stack exactly as the server does (env-resolved config,
    # backend + tenancy registration) without booting providers or the HTTP server.
    config_file = resolve_config_or_distro(args.config)
    run_config = parse_and_maybe_upgrade_config(yaml.safe_load(config_file.read_text()))
    _initialize_storage(run_config)

    tenant_enabled = run_config.server.tenancy.mode != TenancyMode.DISABLED
    responses_ref = _resolve_responses_store(run_config)
    conversations_ref = run_config.storage.stores.conversations
    src_conversations_table = conversations_ref.table_name if conversations_ref else "openai_conversations"

    missing_store_reasons = {
        "responses": (responses_ref, "responses provider has no persistence.responses configured"),
        "conversations": (conversations_ref, "storage.stores.conversations is not configured"),
        "items": (conversations_ref, "storage.stores.conversations is not configured (conversation_items lives there)"),
    }
    for table, (ref, reason) in missing_store_reasons.items():
        if table in tables_scope and ref is None:
            logger.warning(
                "Requested table's source store is not configured; skipping phase", table=table, reason=reason
            )
            tables_scope.discard(table)

    if not tables_scope:
        raise ValueError(
            "Failed to start migration: none of the requested --tables have a source store configured; "
            "nothing to migrate"
        )

    if "items" in tables_scope and "conversations" not in tables_scope:
        logger.warning(
            "Legacy inline items are backfilled only during the conversations phase; with 'items' in scope but "
            "'conversations' excluded, deprecated openai_conversations.items rows will NOT be migrated "
            "(the conversation_items straight-copy still runs)"
        )

    writer: PraxisWriter | None = None
    if not args.dry_run:
        dsn = args.praxis_dsn or os.environ.get("PRAXIS_DATABASE_URL")
        if not dsn:
            raise ValueError(
                "Failed to start migration: a Praxis target DSN is required for a live run; pass --praxis-dsn or set "
                "PRAXIS_DATABASE_URL (or use --dry-run to read+transform without writing)"
            )
        target_tables: dict[str, str] = {}
        if "responses" in tables_scope:
            target_tables["responses"] = args.praxis_responses_table
        if "conversations" in tables_scope:
            target_tables["conversations"] = args.praxis_conversations_table
        if "items" in tables_scope:
            target_tables["items"] = args.praxis_items_table
        writer = PraxisWriter(dsn, target_tables)
        await writer.connect()

    logger.info(
        "Starting Praxis migration",
        source_config=str(config_file),
        tables=sorted(tables_scope),
        tenancy_mode=run_config.server.tenancy.mode.value,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    readers: dict[str, _SourceReader] = {}

    async def _reader_for(backend: str, table_hint: str) -> _SourceReader:
        if backend not in readers:
            impl = await get_system_sqlstore(SqlStoreReference(backend=backend, table_name=table_hint))
            assert isinstance(impl, SqlAlchemySqlStoreImpl)
            readers[backend] = _SourceReader(impl, tenant_enabled)
        return readers[backend]

    reader_for: _ReaderFor = _reader_for

    opts = _RunOptions(batch_size=args.batch_size, skip_errors=args.skip_errors)
    stats = _Stats()
    try:
        with _build_progress() as progress:
            if "responses" in tables_scope and responses_ref is not None:
                await _run_responses_phase(responses_ref, reader_for, writer, tenant, stats, progress, opts)

            if "conversations" in tables_scope and conversations_ref is not None:
                await _run_conversations_phase(
                    conversations_ref,
                    responses_ref,
                    src_conversations_table,
                    reader_for,
                    writer,
                    tenant,
                    "items" in tables_scope,
                    orphan_created_at,
                    stats,
                    progress,
                    opts,
                )

            if "items" in tables_scope and conversations_ref is not None:
                await _run_items_phase(conversations_ref, reader_for, writer, tenant, stats, progress, opts)
    finally:
        if writer is not None:
            await writer.close()
        await shutdown_sqlstore_backends()

    _log_summary(stats, args.dry_run)


class PraxisMigrate(Subcommand):
    """``ogx migrate praxis`` — migrate OGX Responses/Conversations data into Praxis."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "praxis",
            prog="ogx migrate praxis",
            description=(
                "Migrate OGX 3.5 Responses/Conversations data (PostgreSQL) into a Praxis target. Read-only to the "
                "source, write-only (INSERT ... ON CONFLICT DO NOTHING) to the target and therefore safe to re-run. "
                "Intended to run as a one-shot Kubernetes Job on the existing OGX image."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_cmd)

    def _add_arguments(self) -> None:
        p = self.parser
        p.add_argument(
            "config",
            type=str,
            metavar="config | distro",
            help="OGX run config path or distro name for the SOURCE deployment; resolved via resolve_config_or_distro.",
        )
        p.add_argument(
            "--praxis-dsn",
            type=str,
            default=None,
            help=(
                "asyncpg DSN for the Praxis target. Local/test use only — a DSN passed here is visible in the "
                "process listing and shell history. For production runs, set the PRAXIS_DATABASE_URL env var "
                "instead. Not required with --dry-run."
            ),
        )
        p.add_argument(
            "--tables",
            type=str,
            default="responses,conversations,items",
            help="Comma-separated target tables to load (subset of responses,conversations,items).",
        )
        p.add_argument(
            "--praxis-responses-table",
            type=str,
            default="openai_responses",
            help="Target responses table name (must match the Praxis deployment).",
        )
        p.add_argument(
            "--praxis-conversations-table",
            type=str,
            default="openai_conversations",
            help="Target conversations table name (must match the Praxis deployment).",
        )
        p.add_argument(
            "--praxis-items-table",
            type=str,
            default="openai_conversation_items",
            help="Target items table name (must match the Praxis deployment).",
        )
        p.add_argument("--batch-size", type=int, default=500, help="Source page size = write batch size.")
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Read + transform + validate only; no target connection and no writes.",
        )
        p.add_argument(
            "--tenant-sentinel",
            type=str,
            default="default",
            help="tenant_id for rows with an empty owner_principal; validated against the OGX tenant regex.",
        )
        p.add_argument(
            "--owner-tenant-map",
            type=str,
            default=None,
            help="Path to a JSON/YAML mapping {owner_principal: tenant_id} that overrides derived tenant_ids.",
        )
        p.add_argument(
            "--orphan-created-at",
            type=int,
            default=None,
            help="created_at (unix epoch) for message-only conversation orphans. Defaults to the migration start time.",
        )
        p.add_argument(
            "--skip-errors",
            action="store_true",
            help="Log and continue past per-row transform/validation errors (default: fail fast); emits a skipped-row manifest.",
        )

    def _run_cmd(self, args: argparse.Namespace) -> None:
        try:
            asyncio.run(_run(args))
        except Exception as exc:
            logger.error("Failed to run Praxis migration", error=str(exc), error_type=type(exc).__name__)
            raise SystemExit(1) from exc
