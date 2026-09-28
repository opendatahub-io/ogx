# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""``ogx migrate praxis`` — responses phase: openai_responses -> Praxis."""

from rich.progress import Progress

from ogx.core.storage.datatypes import ResponsesStoreReference
from ogx.log import get_logger
from ogx_api.internal.sqlstore import ColumnType

from .reader import _fields, _handle_row_error, _ReaderFor, _RunOptions, _SourceReader, _Stats
from .target import PraxisWriter, TenantDeriver, transform_response

logger = get_logger(name=__name__, category="cli")

# Physical (data) columns to SELECT from the source responses table. Access-control
# columns (owner_principal, access_attributes, and tenant_id when tenancy is
# enabled) are appended by _read_schema. fetch_all issues ``SELECT`` over exactly
# the registered columns, so this set must be a subset of the columns that
# physically exist on the source table (verified against responses_store.py).
# created_at is INTEGER (unix epoch) in every source table.
_RESPONSES_COLUMNS: dict[str, ColumnType] = {
    "id": ColumnType.STRING,
    "created_at": ColumnType.INTEGER,
    "model": ColumnType.STRING,
    "response_object": ColumnType.JSON,
}


async def _migrate_responses(
    reader: _SourceReader,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    table: str,
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
) -> None:
    task = progress.add_task("responses", total=await reader.count(table), **_fields(stats, "responses"))
    async for batch in reader.page(table, "id", opts.batch_size):
        out_rows = []
        for row in batch:
            stats.read["responses"] += 1
            try:
                praxis_row = transform_response(row, tenant)
            except Exception as exc:
                _handle_row_error("responses", str(row.get("id")), exc, stats, opts.skip_errors)
                continue
            stats.transformed["responses"] += 1
            out_rows.append(praxis_row.as_row())
        if writer is not None and out_rows:
            stats.submitted["responses"] += await writer.write_batch("responses", out_rows)
        progress.update(task, advance=len(batch), **_fields(stats, "responses"))


async def _run_responses_phase(
    responses_ref: ResponsesStoreReference,
    reader_for: _ReaderFor,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
) -> None:
    src_table = responses_ref.table_name
    reader = await reader_for(responses_ref.backend, src_table)
    if not await reader.prepare(src_table, _RESPONSES_COLUMNS, "id"):
        logger.warning("Source responses table absent; skipping responses phase", table=src_table)
        return
    await _migrate_responses(reader, writer, tenant, src_table, stats, progress, opts)
