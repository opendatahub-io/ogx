# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""In-process golden guard for the Praxis migration e2e.

Runs the real read + transform pipeline over the deterministic ``seed_source``
dataset against SQLite, normalizes the migrated rows exactly as the Postgres
assertion test does, and compares them to the committed golden fixtures. This:

* unit-tests the shared normalizer (:mod:`tests.integration.migration._compare`);
* pins the golden fixtures the Postgres e2e asserts against, without needing
  Postgres/Praxis — the target rows are backend-independent (every JSON column is
  re-serialized by the pure transforms and stored verbatim as TEXT), so a golden
  generated here is faithful to a real Postgres+Praxis round-trip;
* keeps the seed dataset and the golden in lockstep — regenerate both with
  ``OGX_MIGRATION_UPDATE_GOLDEN=1 uv run pytest tests/unit/cli/migrate/test_golden.py``.

It reuses the exact read layer (``get_system_sqlstore`` -> ``_SourceReader``) and
migration phase functions the CLI runs, so a transform regression fails here.
"""

import os
from collections.abc import Sequence
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import pytest

from ogx.cli.migrate.praxis.conversations import (
    _CONVERSATION_MESSAGES_TABLE,
    _CONVERSATIONS_COLUMNS,
    _MESSAGES_COLUMNS,
    _migrate_conversations,
)
from ogx.cli.migrate.praxis.items import _CONVERSATION_ITEMS_TABLE, _ITEMS_COLUMNS, _migrate_items
from ogx.cli.migrate.praxis.reader import _build_progress, _RunOptions, _SourceReader, _Stats
from ogx.cli.migrate.praxis.responses import _RESPONSES_COLUMNS, _migrate_responses
from ogx.cli.migrate.praxis.target import ItemPositionAllocator, PraxisWriter, TenantDeriver
from ogx.core.datatypes import TenancyConfig, TenancyMode
from ogx.core.storage.datatypes import ResponsesStoreReference, SqliteSqlStoreConfig, SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import set_default_tenancy_config
from ogx.core.storage.sqlstore.sqlstore import (
    get_system_sqlstore,
    register_sqlstore_backends,
    shutdown_sqlstore_backends,
)
from tests.integration.migration import _compare, seed_source

_RESPONSES_TABLE = "responses"
_CONVERSATIONS_TABLE = "openai_conversations"

# default_tenant_id used for the SINGLE-mode leg; must match the workflow's
# OGX_DEFAULT_TENANT_ID and is what the message-only orphan derives (it has no
# owner_principal, so it falls back to the deployment default tenant).
_DEFAULT_TENANT_ID = "acme-corp"

_MODES = {
    TenancyMode.DISABLED: None,
    TenancyMode.SINGLE: _DEFAULT_TENANT_ID,
    TenancyMode.MULTI: None,
}


class _CapturingWriter(PraxisWriter):
    """A PraxisWriter that records batches while enforcing Praxis uniqueness."""

    def __init__(self) -> None:
        super().__init__(
            dsn="postgresql://unused",
            tables={
                "responses": "openai_responses",
                "conversations": "openai_conversations",
                "items": "openai_conversation_items",
            },
        )
        self.batches: dict[str, list[tuple]] = {}
        self._item_primary_keys: set[tuple] = set()
        self._item_positions: set[tuple] = set()

    async def connect(self) -> None:  # pragma: no cover - not used
        return None

    async def close(self) -> None:  # pragma: no cover - not used
        return None

    async def write_batch(self, kind: str, rows: Sequence[tuple[Any, ...]]) -> int:
        if kind == "items":
            pending_rows = []
            pending_primary_keys: set[tuple] = set()
            pending_positions: set[tuple] = set()
            for row in rows:
                primary_key = (row[0], row[1], row[2])
                if primary_key in self._item_primary_keys or primary_key in pending_primary_keys:
                    continue  # Matches ON CONFLICT (item_id, tenant_id, conversation_id) DO NOTHING.
                position_key = (row[1], row[2], row[5])
                if position_key in self._item_positions or position_key in pending_positions:
                    raise ValueError(f"duplicate Praxis item position: {position_key!r}")
                pending_rows.append(row)
                pending_primary_keys.add(primary_key)
                pending_positions.add(position_key)
            self._item_primary_keys.update(pending_primary_keys)
            self._item_positions.update(pending_positions)
            self.batches.setdefault(kind, []).extend(pending_rows)
        else:
            self.batches.setdefault(kind, []).extend(rows)
        return len(rows)


def _disabled_progress():
    progress = _build_progress()
    progress.disable = True
    return progress


async def _migrate_in_process(mode: TenancyMode, default_tenant_id: str | None) -> dict[str, dict]:
    """Seed SQLite with the shared dataset, run the real read+transform pipeline
    with a capturing writer, and return normalized target rows keyed by PK."""
    tenant_enabled = mode != TenancyMode.DISABLED
    with TemporaryDirectory() as tmp:
        backend = f"sql_golden_{uuid4().hex}"
        register_sqlstore_backends({backend: SqliteSqlStoreConfig(db_path=f"{tmp}/source.db")})
        try:
            responses_ref = ResponsesStoreReference(backend=backend, table_name=_RESPONSES_TABLE)
            conversations_ref = SqlStoreReference(backend=backend, table_name=_CONVERSATIONS_TABLE)
            await seed_source.seed_all(responses_ref, conversations_ref, mode, default_tenant_id)
            # Drop seed-time engines so the reader re-inspects the file and registers
            # its own read schema on a fresh engine, exactly like the CLI's _run.
            await shutdown_sqlstore_backends()

            impl = await get_system_sqlstore(SqlStoreReference(backend=backend, table_name=_RESPONSES_TABLE))
            reader = _SourceReader(impl, tenant_enabled)
            assert await reader.prepare(_RESPONSES_TABLE, _RESPONSES_COLUMNS, "id")
            assert await reader.prepare(_CONVERSATIONS_TABLE, _CONVERSATIONS_COLUMNS, "id")
            assert await reader.prepare(_CONVERSATION_MESSAGES_TABLE, _MESSAGES_COLUMNS, "conversation_id")
            assert await reader.prepare(_CONVERSATION_ITEMS_TABLE, _ITEMS_COLUMNS, "id")

            writer = _CapturingWriter()
            position_allocator = ItemPositionAllocator()
            tenant = TenantDeriver()  # sentinel "default", matching the CLI default
            stats = _Stats()
            # A single-row batch exercises position planning across page boundaries.
            opts = _RunOptions(batch_size=1, skip_errors=False)
            with _disabled_progress() as progress:
                await _migrate_responses(reader, writer, tenant, _RESPONSES_TABLE, stats, progress, opts)
                await _migrate_conversations(
                    reader,
                    reader,
                    writer,
                    tenant,
                    _CONVERSATIONS_TABLE,
                    True,
                    True,
                    True,
                    seed_source.ORPHAN_CREATED_AT,
                    stats,
                    progress,
                    opts,
                    position_allocator,
                )
                await _migrate_items(reader, writer, tenant, stats, progress, opts, position_allocator)
        finally:
            await shutdown_sqlstore_backends()
            set_default_tenancy_config(TenancyConfig())

    return _compare.normalize_all(writer.batches)


@pytest.mark.parametrize("mode", list(_MODES), ids=lambda m: m.value)
async def test_golden_matches_pipeline(mode: TenancyMode):
    normalized = await _migrate_in_process(mode, _MODES[mode])

    if os.environ.get("OGX_MIGRATION_UPDATE_GOLDEN") == "1":
        _compare.write_golden(mode.value, normalized)
        pytest.skip(f"Wrote golden fixture expected_{mode.value}.json")

    expected = _compare.load_golden(mode.value)
    problems = _compare.diff_normalized(expected, normalized)
    assert not problems, "Migrated rows do not match golden:\n" + "\n".join(problems)


def _sample_batches():
    return {
        "responses": [("resp_1", "default", 111, "gpt-4o", '{"id":"resp_1"}', "[]", '[{"role":"user"}]')],
        "conversations": [("conv_1", "acme", 100, '{"k":"v"}', "[]")],
        "items": [("item_1", "acme", "conv_1", '{"type":"message"}', 100, 0)],
    }


def test_normalizer_parses_json_columns_and_keys_by_pk():
    normalized = _compare.normalize_all(_sample_batches())
    assert set(normalized) == {"responses", "conversations", "items"}
    # responses keyed by (tenant_id, id); JSON-as-TEXT parsed to structures.
    resp = normalized["responses"]["default\x00resp_1"]
    assert resp["response_object"] == {"id": "resp_1"}
    assert resp["input"] == []
    assert resp["messages"] == [{"role": "user"}]
    assert resp["created_at"] == 111
    assert normalized["conversations"]["conv_1\x00acme"]["metadata"] == {"k": "v"}
    assert normalized["items"]["item_1\x00acme\x00conv_1"]["item_data"] == {"type": "message"}


def test_normalizer_is_order_independent():
    batches = _sample_batches()
    batches["items"] = [
        ("item_b", "t", "c", "{}", 1, 1),
        ("item_a", "t", "c", "{}", 0, 0),
    ]
    reversed_batches = {**batches, "items": list(reversed(batches["items"]))}
    assert _compare.normalize_all(batches) == _compare.normalize_all(reversed_batches)


def test_normalizer_rejects_duplicate_primary_key():
    dup = {"items": [("i", "t", "c", "{}", 0, 0), ("i", "t", "c", "{}", 9, 9)]}
    with pytest.raises(ValueError, match="duplicate primary key"):
        _compare.normalize_all(dup)


def test_normalizer_rejects_wrong_column_count():
    with pytest.raises(ValueError, match="expected 6 columns"):
        _compare.normalize_rows("items", [("i", "t", "c")])


async def test_capturing_writer_rejects_duplicate_item_position():
    writer = _CapturingWriter()
    first = ("item_1", "tenant_a", "conv_1", "{}", 100, 0)
    duplicate_position = ("item_2", "tenant_a", "conv_1", "{}", 101, 0)

    await writer.write_batch("items", [first])
    with pytest.raises(ValueError, match="duplicate Praxis item position"):
        await writer.write_batch("items", [duplicate_position])
