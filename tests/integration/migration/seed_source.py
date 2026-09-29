# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Deterministic source seeding for the OGX->Praxis migration e2e.

Seeds an OGX source database with a fixed dataset that exercises every migration
path — the responses blob decomposition, the two-pass conversation join, the
message-only orphan pass, the legacy inline-item backfill, and each
``TenantDeriver`` branch. It is driven from the CI workflow (``python
tests/integration/migration/seed_source.py <config>``) and reused in-process by
the golden-guard unit test.

Why store classes, not HTTP: the migration reads the *database*, never the HTTP
layer, and OGX's response write path is ``ResponsesStore.store_response_object``.
Seeding through that same store produces byte-identical on-disk rows to a live
server while staying deterministic (fixed ids/timestamps, no inference, no
recordings) so the migrated rows can be pinned as golden fixtures. Happy-path
rows (responses, conversation continuity) go through ``ResponsesStore`` so the
physical schema and stored blob shape are exactly what OGX writes; edge-case rows
(``openai_conversations`` + ``conversation_items``) are inserted raw so the test
controls ``owner_principal``, ``tenant_id``, and the deprecated inline-items
column directly.

The dataset is the single source of truth: the same ``seed_all`` feeds the
Postgres CI run and the SQLite golden generator, so the committed golden fixtures
and the live Postgres+Praxis round-trip cannot drift apart.
"""

import argparse
import asyncio
import sys

import yaml

from ogx.cli.migrate.praxis.cmd import _resolve_responses_store
from ogx.core.configure import parse_and_maybe_upgrade_config
from ogx.core.datatypes import TenancyConfig, TenancyMode, User
from ogx.core.request_headers import RequestProviderDataContext
from ogx.core.stack import _initialize_storage
from ogx.core.storage.datatypes import ResponsesStoreReference, SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import set_default_tenancy_config
from ogx.core.storage.sqlstore.sqlstore import get_system_sqlstore, shutdown_sqlstore_backends
from ogx.core.utils.config_resolution import resolve_config_or_distro
from ogx.log import get_logger
from ogx.providers.utils.responses.responses_store import ResponsesStore
from ogx_api import (
    OpenAIResponseInputMessageContentText,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIUserMessageParam,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

logger = get_logger(name=__name__, category="cli")

# Physical table names that are not config-driven (see cmd.py for rationale).
_CONVERSATION_ITEMS_TABLE = "conversation_items"

# --- Deterministic dataset constants (mirrored in the golden fixtures) --------
RESPONSE_ID = "resp_1"
RESPONSE_CREATED_AT = 111
RESPONSE_MODEL = "gpt-4o"

CONV_JOINED = "conv_joined"  # openai_conversations row + continuity messages -> Pass A join
CONV_LEGACY = "conv_legacy"  # openai_conversations row with deprecated inline items -> backfill
CONV_EMPTY_OWNER = "conv_empty_owner"  # empty owner + (single) empty tenant_id -> sentinel
CONV_ORPHAN = "conv_orphan"  # continuity messages only, no conversations row -> Pass B orphan

# Per-row tenant_id values used only in SINGLE mode. Chosen distinct from the
# deployment default_tenant_id so the migration's verbatim tenant_id precedence
# (source tenant_id column over owner_principal) is actually exercised.
TENANT_JOINED = "team-a"
TENANT_LEGACY = "team-b"

# created_at synthesized for message-only orphans. The CLI defaults this to the
# migration start time; the e2e pins it via --orphan-created-at so it is golden.
ORPHAN_CREATED_AT = 999


def _tenancy_enabled(mode: TenancyMode) -> bool:
    return mode != TenancyMode.DISABLED


async def _seed_responses_and_messages(responses_ref: ResponsesStoreReference) -> None:
    """Seed openai_responses + conversation_messages through the real ResponsesStore.

    Owner/tenant columns are stamped by the store from the process-wide tenancy
    config exactly as the server does for a request with no authenticated user
    (empty owner_principal; tenant_id defaulted in SINGLE mode).
    """
    store = ResponsesStore(responses_ref, policy=[])
    await store.initialize()

    response = OpenAIResponseObject(
        id=RESPONSE_ID,
        created_at=RESPONSE_CREATED_AT,
        model=RESPONSE_MODEL,
        object="response",
        output=[
            OpenAIResponseMessage(
                id="msg_out",
                role="assistant",
                status="completed",
                content=[OpenAIResponseOutputMessageContentOutputText(text="hello")],
            )
        ],
        status="completed",
        store=True,
    )
    input_items = [
        OpenAIResponseMessage(id="msg_in", role="user", content=[OpenAIResponseInputMessageContentText(text="hi")])
    ]
    messages = [OpenAIUserMessageParam(content="hi"), OpenAIUserMessageParam(content="again")]
    await store.store_response_object(response, input=input_items, messages=messages)

    # Continuity messages for a conversation that also has an openai_conversations
    # row (joins in Pass A) ...
    # Match the conversation's owner and tenant so the migration can join the rows.
    with RequestProviderDataContext(user=User("acme", None, tenant_id=TENANT_JOINED)):
        await store.store_conversation_messages(CONV_JOINED, [OpenAIUserMessageParam(content="joined")])
    # ... and a message-only orphan with no openai_conversations row (Pass B).
    await store.store_conversation_messages(CONV_ORPHAN, [OpenAIUserMessageParam(content="orphan")])


def _conv_schema(tenant_enabled: bool) -> dict[str, ColumnType | ColumnDefinition]:
    schema: dict[str, ColumnType | ColumnDefinition] = {
        "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
        "created_at": ColumnType.INTEGER,
        "items": ColumnType.JSON,
        "metadata": ColumnType.JSON,
        "owner_principal": ColumnType.STRING,
        "access_attributes": ColumnType.JSON,
    }
    if tenant_enabled:
        schema["tenant_id"] = ColumnType.STRING
    return schema


def _item_schema(tenant_enabled: bool) -> dict[str, ColumnType | ColumnDefinition]:
    schema: dict[str, ColumnType | ColumnDefinition] = {
        "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
        "conversation_id": ColumnType.STRING,
        "created_at": ColumnType.INTEGER,
        "sort_order": ColumnType.INTEGER,
        "item_data": ColumnType.JSON,
        "owner_principal": ColumnType.STRING,
        "access_attributes": ColumnType.JSON,
    }
    if tenant_enabled:
        schema["tenant_id"] = ColumnType.STRING
    return schema


def _with_tenant(row: dict, tenant_enabled: bool, tenant_id: str) -> dict:
    """Attach a tenant_id column iff tenancy is enabled (DISABLED has no column)."""
    if tenant_enabled:
        return {**row, "tenant_id": tenant_id}
    return row


async def _seed_conversations_and_items(conversations_ref: SqlStoreReference, tenant_enabled: bool) -> None:
    """Seed openai_conversations + conversation_items raw so the test controls
    owner_principal, tenant_id, and the deprecated inline items column."""
    impl = await get_system_sqlstore(conversations_ref)

    await impl.create_table(conversations_ref.table_name, _conv_schema(tenant_enabled))
    await impl.insert(
        conversations_ref.table_name,
        [
            _with_tenant(
                {
                    "id": CONV_JOINED,
                    "created_at": 100,
                    "items": None,
                    "metadata": {"k": "v"},
                    "owner_principal": "acme",
                    "access_attributes": None,
                },
                tenant_enabled,
                TENANT_JOINED,  # verbatim tenant_id takes precedence over owner_principal
            ),
            _with_tenant(
                {
                    "id": CONV_LEGACY,
                    "created_at": 200,
                    "items": [{"id": "item_legacy", "type": "message", "role": "user"}],
                    "metadata": None,
                    "owner_principal": "",
                    "access_attributes": None,
                },
                tenant_enabled,
                TENANT_LEGACY,
            ),
            _with_tenant(
                {
                    "id": CONV_EMPTY_OWNER,
                    "created_at": 300,
                    "items": None,
                    "metadata": {},
                    "owner_principal": "",
                    "access_attributes": None,
                },
                tenant_enabled,
                "",  # empty tenant_id + empty owner -> sentinel in both modes
            ),
        ],
    )

    await impl.create_table(_CONVERSATION_ITEMS_TABLE, _item_schema(tenant_enabled))
    await impl.insert(
        _CONVERSATION_ITEMS_TABLE,
        [
            _with_tenant(
                {
                    "id": "item_1",
                    "conversation_id": CONV_JOINED,
                    "created_at": 100,
                    "sort_order": 0,
                    "item_data": {"type": "message", "id": "item_1"},
                    "owner_principal": "acme",
                    "access_attributes": None,
                },
                tenant_enabled,
                TENANT_JOINED,
            ),
            _with_tenant(
                {
                    "id": "item_null_sort",
                    "conversation_id": CONV_JOINED,
                    "created_at": 105,
                    "sort_order": None,  # legacy NULL sort_order -> position 0
                    "item_data": {"type": "message", "id": "item_null_sort"},
                    "owner_principal": "acme",
                    "access_attributes": None,
                },
                tenant_enabled,
                TENANT_JOINED,
            ),
        ],
    )


async def seed_all(
    responses_ref: ResponsesStoreReference,
    conversations_ref: SqlStoreReference,
    tenancy_mode: TenancyMode,
    default_tenant_id: str | None,
) -> None:
    """Seed the full deterministic dataset. Backend-agnostic: the caller registers
    the storage backends (via ``_initialize_storage`` for Postgres, or
    ``register_sqlstore_backends`` for the in-process SQLite golden generator)."""
    set_default_tenancy_config(TenancyConfig(mode=tenancy_mode, default_tenant_id=default_tenant_id))
    await _seed_responses_and_messages(responses_ref)
    await _seed_conversations_and_items(conversations_ref, _tenancy_enabled(tenancy_mode))


async def _main(config: str) -> None:
    config_file = resolve_config_or_distro(config)
    run_config = parse_and_maybe_upgrade_config(yaml.safe_load(config_file.read_text()))
    _initialize_storage(run_config)

    responses_ref = _resolve_responses_store(run_config)
    if responses_ref is None:
        raise ValueError("Failed to seed source: the responses provider has no persistence.responses configuration")
    conversations_ref = run_config.storage.stores.conversations
    if conversations_ref is None:
        raise ValueError("Failed to seed source: storage.stores.conversations is not configured")

    tenancy = run_config.server.tenancy
    logger.info(
        "Seeding OGX source for Praxis migration e2e",
        source_config=str(config_file),
        responses_table=responses_ref.table_name,
        conversations_table=conversations_ref.table_name,
        tenancy_mode=tenancy.mode.value,
    )
    try:
        await seed_all(responses_ref, conversations_ref, tenancy.mode, tenancy.default_tenant_id)
    finally:
        await shutdown_sqlstore_backends()
    logger.info("Seeding complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed an OGX source database for the Praxis migration e2e.")
    parser.add_argument(
        "config",
        type=str,
        metavar="config | distro",
        help="OGX run config path or distro name for the SOURCE deployment (e.g. ci-tests::run-with-postgres-store.yaml).",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_main(args.config))
    except Exception as exc:
        logger.error("Failed to seed source", error=str(exc), error_type=type(exc).__name__)
        sys.exit(1)


if __name__ == "__main__":
    main()
