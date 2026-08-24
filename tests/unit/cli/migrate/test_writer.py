# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for PraxisWriter using a fake asyncpg connection.

Asserts the exact SQL (physical table name + ON CONFLICT target), positional
tuple passthrough, and transaction wrapping without touching a real database.
An optional idempotency integration test runs against PRAXIS_TEST_DSN when set.
"""

import os

import pytest

from ogx.cli.migrate.praxis.target import PraxisWriter

_DEFAULT_TABLES = {
    "responses": "openai_responses",
    "conversations": "openai_conversations",
    "items": "conversation_items",
}


class _FakeTransaction:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn

    async def __aenter__(self) -> "_FakeTransaction":
        self._conn.transaction_entered += 1
        return self

    async def __aexit__(self, *exc: object) -> bool:
        self._conn.transaction_exited += 1
        return False


class _FakeConn:
    def __init__(self) -> None:
        self.executemany_calls: list[tuple[str, list]] = []
        self.transaction_entered = 0
        self.transaction_exited = 0
        self.closed = False

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    async def executemany(self, sql: str, rows: list) -> None:
        self.executemany_calls.append((sql, list(rows)))

    async def close(self) -> None:
        self.closed = True


def _writer_with_fake_conn(tables: dict[str, str] | None = None) -> tuple[PraxisWriter, _FakeConn]:
    writer = PraxisWriter(dsn="postgresql://unused", tables=tables or _DEFAULT_TABLES)
    fake = _FakeConn()
    writer._conn = fake
    return writer, fake


class TestSql:
    def test_responses_sql(self):
        writer, _ = _writer_with_fake_conn()
        sql = writer.sql_for("responses")
        assert "INSERT INTO openai_responses" in sql
        assert "ON CONFLICT (tenant_id, id) DO NOTHING" in sql
        assert "(id, tenant_id, created_at, model, response_object, input, messages)" in sql

    def test_conversations_sql(self):
        writer, _ = _writer_with_fake_conn()
        sql = writer.sql_for("conversations")
        assert "INSERT INTO openai_conversations" in sql
        assert "ON CONFLICT (conversation_id, tenant_id) DO NOTHING" in sql

    def test_items_sql(self):
        writer, _ = _writer_with_fake_conn()
        sql = writer.sql_for("items")
        assert "INSERT INTO conversation_items" in sql
        assert "ON CONFLICT (item_id, tenant_id, conversation_id) DO NOTHING" in sql

    def test_custom_table_names_are_bound(self):
        writer, _ = _writer_with_fake_conn({"responses": "praxis_resp", "conversations": "c", "items": "i"})
        assert "INSERT INTO praxis_resp" in writer.sql_for("responses")

    def test_unknown_kind_rejected(self):
        writer, _ = _writer_with_fake_conn()
        with pytest.raises(ValueError, match="unknown target table"):
            writer.sql_for("bogus")

    def test_invalid_table_name_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Praxis table name"):
            PraxisWriter(dsn="postgresql://unused", tables={"responses": "bad; DROP TABLE"})


class TestWriteBatch:
    async def test_write_batch_passes_rows_and_wraps_transaction(self):
        writer, fake = _writer_with_fake_conn()
        rows = [
            ("resp_1", "t1", 1, "m", "{}", "[]", "[]"),
            ("resp_2", "t1", 2, "m", "{}", "[]", "[]"),
        ]

        submitted = await writer.write_batch("responses", rows)

        assert submitted == 2
        assert len(fake.executemany_calls) == 1
        sql, sent_rows = fake.executemany_calls[0]
        assert "openai_responses" in sql
        assert sent_rows == rows  # positional tuples passed through unchanged, in order
        assert fake.transaction_entered == 1
        assert fake.transaction_exited == 1

    async def test_empty_batch_is_noop(self):
        writer, fake = _writer_with_fake_conn()
        submitted = await writer.write_batch("responses", [])
        assert submitted == 0
        assert fake.executemany_calls == []
        assert fake.transaction_entered == 0

    async def test_write_batch_without_connection_raises(self):
        writer = PraxisWriter(dsn="postgresql://unused", tables=_DEFAULT_TABLES)
        with pytest.raises(RuntimeError, match="not connected"):
            await writer.write_batch("responses", [("resp_1", "t1", 1, "m", "{}", "[]", "[]")])


@pytest.mark.skipif(not os.environ.get("PRAXIS_TEST_DSN"), reason="PRAXIS_TEST_DSN not set")
class TestIdempotencyIntegration:
    """Runs against a throwaway Postgres with Praxis's DDL already applied."""

    async def test_double_write_is_idempotent(self):
        dsn = os.environ["PRAXIS_TEST_DSN"]
        writer = PraxisWriter(dsn=dsn, tables=_DEFAULT_TABLES)
        await writer.connect()
        try:
            row = ("resp_idem", "t1", 1, "gpt-4o", "{}", "[]", "[]")
            await writer.write_batch("responses", [row])
            await writer.write_batch("responses", [row])  # ON CONFLICT DO NOTHING
            count = await writer._conn.fetchval(
                "SELECT count(*) FROM openai_responses WHERE tenant_id=$1 AND id=$2", "t1", "resp_idem"
            )
            assert count == 1
        finally:
            await writer._conn.execute("DELETE FROM openai_responses WHERE tenant_id='t1' AND id='resp_idem'")
            await writer.close()
