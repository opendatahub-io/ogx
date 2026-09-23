# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""End-to-end assertion for the OGX->Praxis migration against real Postgres.

This is the assertion and regression half of the e2e (the workflow does the
setup — see ``conftest.py``). It reads the three Praxis target tables the
migration wrote to and asserts, row-for-row, that they equal the committed
golden fixture for the tenancy leg under test. It also verifies the live
Postgres conflict behavior for globally unique Praxis IDs.

Why this catches what the in-process SQLite guard (``tests/unit/cli/migrate/
test_golden.py``) cannot, even though both compare against the *same*
golden:

* the asyncpg ``INSERT ... ON CONFLICT DO NOTHING`` write leg actually executes;
* Postgres type coercion round-trips (BIGINT ``created_at``/``position`` and
  response payloads stored as ``BYTEA``) instead of SQLite's looser typing;
* the target tables exist with the DDL/PK contract **Praxis** stamped (not a
  schema the test authored), so table-name or column drift between OGX's writer
  and Praxis's schema surfaces here.

The comparison is delegated to the shared, backend-agnostic normalizer in
:mod:`_compare`, so the golden generated in-process against SQLite is authoritative
for this real round-trip (see that module's docstring for why that holds).
"""

import uuid

import asyncpg

from ogx.cli.migrate.praxis.target import PraxisWriter

from . import _compare

# Physical Praxis target table names. These MUST match both the Praxis DDL and
# the ``ogx migrate praxis`` invocation in the workflow: the CLI defaults cover
# responses/conversations, and it is invoked with
# ``--praxis-items-table openai_conversation_items`` (also the CLI default).
_PRAXIS_TABLES: dict[str, str] = {
    "responses": "openai_responses",
    "conversations": "openai_conversations",
    "items": "openai_conversation_items",
}


def _select(kind: str) -> str:
    """Project the target columns in :data:`_compare.TARGET_COLUMNS` order so the
    fetched tuples normalize identically to the in-process ``as_row()`` tuples.
    Identifiers are quoted because ``position`` collides with a SQL keyword."""
    columns = ", ".join(f'"{col}"' for col in _compare.TARGET_COLUMNS[kind])
    return f'SELECT {columns} FROM "{_PRAXIS_TABLES[kind]}"'


async def _fetch_target(dsn: str) -> dict[str, list[tuple]]:
    conn = await asyncpg.connect(dsn)
    try:
        return {kind: [tuple(record) for record in await conn.fetch(_select(kind))] for kind in _compare.KINDS}
    finally:
        await conn.close()


async def test_praxis_target_matches_golden(praxis_dsn: str, tenancy_mode: str):
    batches = await _fetch_target(praxis_dsn)
    actual = _compare.normalize_all(batches)
    expected = _compare.load_golden(tenancy_mode)

    problems = _compare.diff_normalized(expected, actual)
    assert not problems, f"Praxis target tables do not match golden expected_{tenancy_mode}.json:\n" + "\n".join(
        problems
    )


async def test_praxis_global_primary_keys_skip_cross_tenant_duplicate_ids(praxis_dsn: str):
    """Praxis IDs are global, and items cannot cross a conversation's tenant boundary."""
    writer = PraxisWriter(dsn=praxis_dsn, tables=_PRAXIS_TABLES)
    await writer.connect()
    assert writer._conn is not None
    conn = writer._conn
    suffix = uuid.uuid4().hex
    response_id = f"resp-global-{suffix}"
    conversation_id = f"conv-global-{suffix}"
    item_a_id = f"item-global-a-{suffix}"
    item_b_id = f"item-global-b-{suffix}"
    try:
        await writer.write_batch(
            "responses",
            [
                (response_id, "tenant-a", "owner-a", "urn:rhoai:ogx:production", 1, "gpt-4o", b"{}", b"[]", b"[]"),
                (response_id, "tenant-b", "owner-b", "urn:rhoai:ogx:production", 2, "gpt-4o", b"{}", b"[]", b"[]"),
            ],
        )
        await writer.write_batch(
            "conversations",
            [
                (conversation_id, "tenant-a", "owner-a", "urn:rhoai:ogx:production", 1, "{}", "[]"),
                (conversation_id, "tenant-b", "owner-b", "urn:rhoai:ogx:production", 2, "{}", "[]"),
            ],
        )
        await writer.write_batch(
            "items",
            [
                (item_a_id, "tenant-a", "owner-a", "urn:rhoai:ogx:production", conversation_id, "{}", 1, 0),
                (item_b_id, "tenant-b", "owner-b", "urn:rhoai:ogx:production", conversation_id, "{}", 2, 0),
            ],
        )

        assert await conn.fetchval("SELECT count(*) FROM openai_responses WHERE id=$1", response_id) == 1
        assert await conn.fetchval("SELECT tenant_id FROM openai_responses WHERE id=$1", response_id) == "tenant-a"
        assert (
            await conn.fetchval("SELECT count(*) FROM openai_conversations WHERE conversation_id=$1", conversation_id)
            == 1
        )
        assert (
            await conn.fetchval("SELECT tenant_id FROM openai_conversations WHERE conversation_id=$1", conversation_id)
            == "tenant-a"
        )
        assert await conn.fetchval("SELECT count(*) FROM openai_conversation_items WHERE item_id=$1", item_a_id) == 1
        assert (
            await conn.fetchval("SELECT tenant_id FROM openai_conversation_items WHERE item_id=$1", item_a_id)
            == "tenant-a"
        )
        assert await conn.fetchval("SELECT count(*) FROM openai_conversation_items WHERE item_id=$1", item_b_id) == 0
    finally:
        await conn.execute(
            "DELETE FROM openai_conversation_items WHERE item_id = ANY($1::text[])", [item_a_id, item_b_id]
        )
        await conn.execute("DELETE FROM openai_conversations WHERE conversation_id=$1", conversation_id)
        await conn.execute("DELETE FROM openai_responses WHERE id=$1", response_id)
        await writer.close()
