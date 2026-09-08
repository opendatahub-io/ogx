# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""``ogx migrate praxis`` — items phase: conversation_items -> Praxis.

Copies the ``conversation_items`` table (which lives on the conversations
backend) while normalizing positions that would violate Praxis's per-
conversation uniqueness constraint. Legacy inline items stored directly on an
``openai_conversations`` row are handled separately, as a backfill during the
conversations phase — see :mod:`.conversations`.
"""

from typing import Any

from rich.progress import Progress

from ogx.core.storage.datatypes import SqlStoreReference
from ogx.log import get_logger
from ogx_api.internal.sqlstore import ColumnType

from .reader import _fields, _handle_row_error, _ReaderFor, _RunOptions, _SourceReader, _Stats
from .target import ItemPositionAllocator, PraxisWriter, TenantDeriver, transform_item

# Not config-driven (unlike the conversations table): conversation_items always
# lives on the conversations backend under this physical name.
_CONVERSATION_ITEMS_TABLE = "conversation_items"

_ITEMS_COLUMNS: dict[str, ColumnType] = {
    "id": ColumnType.STRING,
    "conversation_id": ColumnType.STRING,
    "created_at": ColumnType.INTEGER,
    "sort_order": ColumnType.INTEGER,
    "item_data": ColumnType.JSON,
}

logger = get_logger(name=__name__, category="cli")


async def _write_rows(
    writer: PraxisWriter | None,
    stats: _Stats,
    opts: _RunOptions,
    source: str,
    rows: list[tuple[Any, ...]],
) -> None:
    if writer is None:
        return
    for start in range(0, len(rows), opts.batch_size):
        stats.submitted[source] += await writer.write_batch("items", rows[start : start + opts.batch_size])


async def _write_retained_items(
    writer: PraxisWriter | None,
    stats: _Stats,
    opts: _RunOptions,
    position_allocator: ItemPositionAllocator,
) -> None:
    position_allocator.finalize()
    if writer is None:
        return
    rows = [item.as_row() for item in position_allocator.allocate_retained("items_legacy")]
    await _write_rows(writer, stats, opts, "items_legacy", rows)


async def _migrate_items(
    reader: _SourceReader,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
    position_allocator: ItemPositionAllocator,
) -> None:
    """Plan positions in one pass, then re-read and write bounded batches.

    Only compact position metadata is retained for table-backed items. Legacy
    inline items cannot be cheaply re-read here, so the conversations phase
    retains those comparatively rare rows in the shared allocator.
    """
    task = progress.add_task("items", total=await reader.count(_CONVERSATION_ITEMS_TABLE), **_fields(stats, "items"))
    skipped_item_ids: set[str] = set()
    async for batch in reader.page(_CONVERSATION_ITEMS_TABLE, "id", opts.batch_size):
        for row in batch:
            stats.read["items"] += 1
            try:
                praxis_item = transform_item(row, tenant)
            except Exception as exc:
                item_id = str(row.get("id"))
                _handle_row_error("items", item_id, exc, stats, opts.skip_errors)
                skipped_item_ids.add(item_id)
                continue
            stats.transformed["items"] += 1
            position_allocator.observe("items", praxis_item)
        progress.update(task, advance=len(batch), **_fields(stats, "items"))

    await _write_retained_items(writer, stats, opts, position_allocator)
    if writer is not None:
        async for batch in reader.page(_CONVERSATION_ITEMS_TABLE, "id", opts.batch_size):
            out_rows = [
                position_allocator.allocate("items", transform_item(row, tenant)).as_row()
                for row in batch
                if str(row.get("id")) not in skipped_item_ids
            ]
            await _write_rows(writer, stats, opts, "items", out_rows)
    progress.update(task, **_fields(stats, "items"))


async def _run_items_phase(
    conversations_ref: SqlStoreReference,
    reader_for: _ReaderFor,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
    position_allocator: ItemPositionAllocator,
) -> None:
    reader = await reader_for(conversations_ref.backend, _CONVERSATION_ITEMS_TABLE)
    if not await reader.prepare(_CONVERSATION_ITEMS_TABLE, _ITEMS_COLUMNS, "id"):
        logger.warning("Source conversation_items table absent; skipping items phase")
        await _write_retained_items(writer, stats, opts, position_allocator)
        return
    await _migrate_items(reader, writer, tenant, stats, progress, opts, position_allocator)
