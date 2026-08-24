# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""``ogx migrate praxis`` — items phase: conversation_items -> Praxis.

Straight copy of the ``conversation_items`` table (lives on the conversations
backend). Legacy inline items stored directly on an ``openai_conversations``
row are handled separately, as a backfill during the conversations phase —
see :mod:`.conversations`.
"""

from rich.progress import Progress

from ogx.core.storage.datatypes import SqlStoreReference
from ogx.log import get_logger
from ogx_api.internal.sqlstore import ColumnType

from .reader import _fields, _handle_row_error, _ReaderFor, _RunOptions, _SourceReader, _Stats
from .target import PraxisWriter, TenantDeriver, transform_item

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


async def _migrate_items(
    reader: _SourceReader,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
) -> None:
    task = progress.add_task("items", total=await reader.count(_CONVERSATION_ITEMS_TABLE), **_fields(stats, "items"))
    async for batch in reader.page(_CONVERSATION_ITEMS_TABLE, "id", opts.batch_size):
        out_rows = []
        for row in batch:
            stats.read["items"] += 1
            try:
                praxis_item = transform_item(row, tenant)
            except Exception as exc:
                _handle_row_error("items", str(row.get("id")), exc, stats, opts.skip_errors)
                continue
            stats.transformed["items"] += 1
            out_rows.append(praxis_item.as_row())
        if writer is not None and out_rows:
            stats.submitted["items"] += await writer.write_batch("items", out_rows)
        progress.update(task, advance=len(batch), **_fields(stats, "items"))


async def _run_items_phase(
    conversations_ref: SqlStoreReference,
    reader_for: _ReaderFor,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
) -> None:
    reader = await reader_for(conversations_ref.backend, _CONVERSATION_ITEMS_TABLE)
    if not await reader.prepare(_CONVERSATION_ITEMS_TABLE, _ITEMS_COLUMNS, "id"):
        logger.warning("Source conversation_items table absent; skipping items phase")
        return
    await _migrate_items(reader, writer, tenant, stats, progress, opts)
