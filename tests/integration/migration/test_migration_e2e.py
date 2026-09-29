# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""End-to-end assertion for the OGX->Praxis migration against real Postgres.

This is the thin assertion half of the e2e (the workflow does the setup — see
``conftest.py``). It reads the three Praxis target tables the migration wrote to
and asserts, row-for-row, that they equal the committed golden fixture for the
tenancy leg under test.

Why this catches what the in-process SQLite guard (``tests/unit/cli/migrate/
test_golden.py``) cannot, even though both compare against the *same*
golden:

* the asyncpg ``INSERT ... ON CONFLICT DO NOTHING`` write leg actually executes;
* Postgres type coercion round-trips (BIGINT ``created_at``/``position``, JSON
  stored as ``TEXT``) instead of SQLite's looser typing;
* the target tables exist with the DDL/PK contract **Praxis** stamped (not a
  schema the test authored), so table-name or column drift between OGX's writer
  and Praxis's schema surfaces here.

The comparison is delegated to the shared, backend-agnostic normalizer in
:mod:`_compare`, so the golden generated in-process against SQLite is authoritative
for this real round-trip (see that module's docstring for why that holds).
"""

import asyncpg

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
