# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""``ogx migrate praxis`` — conversations phase: openai_conversations (+ the
conversation_messages join and legacy inline-item backfill) -> Praxis.

Two passes:

* Pass A (:func:`_migrate_known_conversations`) — every ``openai_conversations``
  row, joined against ``conversation_messages`` where present, plus a backfill of
  any deprecated inline ``items`` still stored on the conversation row.
* Pass B (:func:`_migrate_orphan_conversations`) — ``conversation_messages`` rows
  with no matching ``openai_conversations`` record (message-only orphans).
"""

from typing import Any

from rich.progress import Progress

from ogx.core.storage.datatypes import ResponsesStoreReference, SqlStoreReference
from ogx.log import get_logger
from ogx_api.internal.sqlstore import ColumnType

from .reader import _fields, _handle_row_error, _ReaderFor, _RunOptions, _SourceReader, _Stats
from .target import (
    PraxisWriter,
    TenantDeriver,
    transform_conversation,
    transform_legacy_inline_item,
    transform_message_only_conversation,
)

# conversation_messages lives on the *responses* backend (created by the
# ResponsesStore), not the conversations backend, and is not config-driven.
_CONVERSATION_MESSAGES_TABLE = "conversation_messages"

_CONVERSATIONS_COLUMNS: dict[str, ColumnType] = {
    "id": ColumnType.STRING,
    "created_at": ColumnType.INTEGER,
    "metadata": ColumnType.JSON,
    "items": ColumnType.JSON,  # deprecated inline items, backfilled in Pass A
}
_MESSAGES_COLUMNS: dict[str, ColumnType] = {
    "conversation_id": ColumnType.STRING,
    "messages": ColumnType.JSON,
}

logger = get_logger(name=__name__, category="cli")


async def _transform_conversation_row(
    conv_row: dict[str, Any],
    msg_reader: _SourceReader | None,
    msg_available: bool,
    tenant: TenantDeriver,
    items_in_scope: bool,
    seen: set[str],
    stats: _Stats,
    opts: _RunOptions,
) -> tuple[tuple[Any, ...] | None, list[tuple[Any, ...]]]:
    """Transform one conversations-table row plus its legacy inline items.

    :returns: ``(praxis_conversation_row, praxis_item_rows)``. The conversation
        row is ``None`` if the transform failed and the row was skipped.
    """
    conv_id = str(conv_row.get("id"))
    try:
        messages_row = (
            await msg_reader.fetch_messages(_CONVERSATION_MESSAGES_TABLE, conv_row["id"])
            if msg_available and msg_reader
            else None
        )
        praxis_conv = transform_conversation(conv_row, messages_row, tenant)
    except Exception as exc:
        _handle_row_error("conversations", conv_id, exc, stats, opts.skip_errors)
        return None, []
    seen.add(conv_row["id"])
    stats.transformed["conversations"] += 1

    item_rows: list[tuple[Any, ...]] = []
    inline = conv_row.get("items") if items_in_scope else None
    if isinstance(inline, list):
        for position, element in enumerate(inline):
            try:
                praxis_item = transform_legacy_inline_item(element, position, conv_row, tenant)
            except Exception as exc:
                _handle_row_error("items_legacy", f"{conv_id}[{position}]", exc, stats, opts.skip_errors)
                continue
            stats.transformed["items_legacy"] += 1
            item_rows.append(praxis_item.as_row())
    return praxis_conv.as_row(), item_rows


async def _migrate_known_conversations(
    conv_reader: _SourceReader,
    msg_reader: _SourceReader | None,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    conv_table: str,
    msg_available: bool,
    items_in_scope: bool,
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
) -> set[str]:
    """Pass A — openai_conversations ⋈ conversation_messages, plus legacy inline-item backfill.

    :returns: the set of source conversation ids seen, so Pass B can skip them.
    """
    seen: set[str] = set()
    task = progress.add_task(
        "conversations", total=await conv_reader.count(conv_table), **_fields(stats, "conversations")
    )
    async for batch in conv_reader.page(conv_table, "id", opts.batch_size):
        conv_out = []
        item_out = []
        for conv_row in batch:
            stats.read["conversations"] += 1
            conv_dict, item_rows = await _transform_conversation_row(
                conv_row, msg_reader, msg_available, tenant, items_in_scope, seen, stats, opts
            )
            if conv_dict is not None:
                conv_out.append(conv_dict)
            item_out.extend(item_rows)
        if writer is not None:
            if conv_out:
                stats.submitted["conversations"] += await writer.write_batch("conversations", conv_out)
            if item_out:
                stats.submitted["items_legacy"] += await writer.write_batch("items", item_out)
        progress.update(task, advance=len(batch), **_fields(stats, "conversations"))
    return seen


async def _migrate_orphan_conversations(
    msg_reader: _SourceReader,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    orphan_created_at: int,
    seen: set[str],
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
) -> None:
    """Pass B — message-only orphans: conversation_messages rows with no openai_conversations record."""
    orphan_task = progress.add_task("conversations (orphans)", total=None, **_fields(stats, "conversations_orphans"))
    async for batch in msg_reader.page(_CONVERSATION_MESSAGES_TABLE, "conversation_id", opts.batch_size):
        orphan_out = []
        for msg_row in batch:
            conv_id = str(msg_row.get("conversation_id"))
            if conv_id in seen:
                continue
            stats.read["conversations_orphans"] += 1
            try:
                praxis_conv = transform_message_only_conversation(msg_row, tenant, orphan_created_at)
            except Exception as exc:
                _handle_row_error("conversations_orphans", str(conv_id), exc, stats, opts.skip_errors)
                continue
            seen.add(conv_id)
            stats.transformed["conversations_orphans"] += 1
            orphan_out.append(praxis_conv.as_row())
        if writer is not None and orphan_out:
            stats.submitted["conversations_orphans"] += await writer.write_batch("conversations", orphan_out)
        progress.update(orphan_task, advance=len(batch), **_fields(stats, "conversations_orphans"))


async def _migrate_conversations(
    conv_reader: _SourceReader,
    msg_reader: _SourceReader | None,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    conv_table: str,
    conv_available: bool,
    msg_available: bool,
    items_in_scope: bool,
    orphan_created_at: int,
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
) -> None:
    seen: set[str] = set()
    if conv_available:
        seen = await _migrate_known_conversations(
            conv_reader, msg_reader, writer, tenant, conv_table, msg_available, items_in_scope, stats, progress, opts
        )

    if not (msg_available and msg_reader):
        return
    await _migrate_orphan_conversations(msg_reader, writer, tenant, orphan_created_at, seen, stats, progress, opts)


async def _run_conversations_phase(
    conversations_ref: SqlStoreReference,
    responses_ref: ResponsesStoreReference | None,
    conv_table: str,
    reader_for: _ReaderFor,
    writer: PraxisWriter | None,
    tenant: TenantDeriver,
    items_in_scope: bool,
    orphan_created_at: int,
    stats: _Stats,
    progress: Progress,
    opts: _RunOptions,
) -> None:
    conv_reader = await reader_for(conversations_ref.backend, conv_table)
    conv_available = await conv_reader.prepare(conv_table, _CONVERSATIONS_COLUMNS, "id")
    if not conv_available:
        logger.warning(
            "Source conversations table absent; only message-only orphans (if any) will migrate",
            table=conv_table,
        )

    # conversation_messages lives on the responses backend.
    msg_reader: _SourceReader | None = None
    msg_available = False
    if responses_ref is not None:
        msg_reader = await reader_for(responses_ref.backend, _CONVERSATION_MESSAGES_TABLE)
        msg_available = await msg_reader.prepare(_CONVERSATION_MESSAGES_TABLE, _MESSAGES_COLUMNS, "conversation_id")

    if not (conv_available or msg_available):
        return
    await _migrate_conversations(
        conv_reader,
        msg_reader,
        writer,
        tenant,
        conv_table,
        conv_available,
        msg_available,
        items_in_scope,
        orphan_created_at,
        stats,
        progress,
        opts,
    )
