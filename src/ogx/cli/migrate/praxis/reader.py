# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared, ACL-bypassing source-reading infrastructure for ``ogx migrate praxis``.

Generic across all migration phases (responses, conversations, items): the raw
per-backend reader, per-run stats/options, and progress-bar helpers. Phase
modules (:mod:`.responses`, :mod:`.conversations`,
:mod:`.items`) build on top of this.
"""

from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sqlalchemy import func, inspect, select

from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx.log import get_logger
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

logger = get_logger(name=__name__, category="cli")


def _read_schema(
    data_columns: Mapping[str, ColumnType], key_col: str, tenant_enabled: bool
) -> dict[str, ColumnType | ColumnDefinition]:
    """Build the physical read schema: declared data columns plus the
    access-control columns OGX's AuthorizedSqlStore always adds. ``tenant_id``
    exists only when tenancy is enabled (it is never created in DISABLED mode)."""
    schema: dict[str, ColumnType | ColumnDefinition] = {}
    for name, col_type in data_columns.items():
        schema[name] = ColumnDefinition(type=col_type, primary_key=True) if name == key_col else col_type
    schema["owner_principal"] = ColumnType.STRING
    schema["access_attributes"] = ColumnType.JSON
    if tenant_enabled:
        schema["tenant_id"] = ColumnType.STRING
    return schema


class _SourceReader:
    """Raw, ACL-bypassing reader over one OGX SQL backend.

    Wraps the plain ``SqlAlchemySqlStoreImpl`` returned by ``get_system_sqlstore``
    (no ``AuthorizedSqlStore``, so no per-request owner filtering and no
    ``ALTER TABLE ADD COLUMN`` on read).

    **Read-only safety.** Registering a table then reading it lazily triggers
    ``metadata.create_all(checkfirst=True)``, which is a no-op for existing
    tables but *creates* an absent one — a source write. To avoid that this
    reader inspects ``information_schema`` (via ``_existing_tables``) before
    registering, and only ever registers/reads tables that already exist. The
    inspection primes the engine while ``metadata`` is still empty, so no
    ``create_all`` runs for a table that is not physically present.
    """

    def __init__(self, impl: SqlAlchemySqlStoreImpl, tenant_enabled: bool) -> None:
        self._impl = impl
        self._tenant_enabled = tenant_enabled
        self._existing: set[str] | None = None
        self._registered: set[str] = set()

    async def _existing_tables(self) -> set[str]:
        if self._existing is None:
            await self._impl._ensure_engine()
            assert self._impl._engine is not None
            async with self._impl._engine.connect() as conn:
                names: set[str] = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            self._existing = names
        return self._existing

    async def prepare(self, table: str, data_columns: Mapping[str, ColumnType], key_col: str) -> bool:
        """Register the read schema for ``table`` iff it physically exists.

        :returns: True if the table is available to read, False if it is absent
            (in which case nothing is registered and no write can occur).
        """
        if table not in await self._existing_tables():
            return False
        if table not in self._registered:
            await self._impl.create_table(table, _read_schema(data_columns, key_col, self._tenant_enabled))
            self._registered.add(table)
        return True

    async def count(self, table: str) -> int:
        await self._impl._ensure_engine()
        assert self._impl.async_session is not None
        table_obj = self._impl.metadata.tables[table]
        async with self._impl.async_session() as session:
            result = await session.execute(select(func.count()).select_from(table_obj))
            return int(result.scalar_one())

    async def page(self, table: str, key_col: str, batch_size: int) -> AsyncIterator[list[dict[str, Any]]]:
        """Yield successive batches ordered by ``key_col`` using keyset pagination."""
        after: str | None = None
        while True:
            result = await self._impl.fetch_all(
                table=table,
                order_by=[(key_col, "asc")],
                cursor=(key_col, after) if after is not None else None,
                limit=batch_size,
            )
            rows = result.data
            if not rows:
                return
            yield rows
            after = rows[-1][key_col]
            if not result.has_more:
                return

    async def fetch_messages(self, table: str, conversation_id: str) -> dict[str, Any] | None:
        return await self._impl.fetch_one(table, where={"conversation_id": conversation_id})


@dataclass
class _Stats:
    """Per-phase counters + a manifest of rows skipped under --skip-errors."""

    read: Counter[str] = field(default_factory=Counter)
    transformed: Counter[str] = field(default_factory=Counter)
    submitted: Counter[str] = field(default_factory=Counter)
    skipped: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, id, error)


@dataclass(frozen=True)
class _RunOptions:
    """CLI knobs threaded unchanged through every migration phase."""

    batch_size: int
    skip_errors: bool


# Looks up (or lazily creates) the raw reader for a backend; captures the
# in-progress `readers` cache local to a single `_run` invocation.
_ReaderFor = Callable[[str, str], Awaitable["_SourceReader"]]


def _handle_row_error(kind: str, row_id: str, exc: Exception, stats: _Stats, skip_errors: bool) -> None:
    """Fail fast by default; under --skip-errors, record and continue."""
    if not skip_errors:
        raise RuntimeError(f"Failed to migrate {kind} row {row_id!r}: {exc}") from exc
    # Store only the exception type, never str(exc): a Pydantic ValidationError
    # embeds the offending input value, which would leak source row data into
    # logs and the skipped manifest (CWE-532).
    error_code = type(exc).__name__
    stats.skipped.append((kind, row_id, error_code))
    logger.warning("Skipping row after transform error", kind=kind, row_id=row_id, error=error_code)


def _build_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description:<24}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("submitted={task.fields[submitted]}"),
        TextColumn("skipped={task.fields[skipped]}"),
        TimeElapsedColumn(),
    )


def _fields(stats: _Stats, label: str) -> dict[str, Any]:
    # dict[str, Any] (not int): these are spread into rich's add_task/update as
    # **fields, and Any keeps mypy from matching them against those methods'
    # typed keyword params (start: bool, description: str, ...).
    return {"submitted": stats.submitted[label], "skipped": len(stats.skipped)}
