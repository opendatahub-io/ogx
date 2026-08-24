# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Self-contained, excisable Praxis migration module.

Everything Praxis-specific — the OGX->Praxis schema mapping and the asyncpg
write contract — lives here so the coupling stays thin and disposable: once
migrations complete, deleting ``src/ogx/cli/migrate/praxis/`` and removing one
registration line in ``migrate.py`` fully excises the feature.

To keep this module independently unit-testable and merge-conflict-free against
upstream, it imports **only** ``ogx_api`` Pydantic models (plus stdlib and
asyncpg). It must not import from ``ogx`` core. All logging and OGX
config/store bootstrap live in ``cmd.py``.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import asyncpg  # type: ignore[import-untyped]

from ogx_api import (
    OpenAIMessageParam,
    OpenAIResponseObject,
    OpenAIResponseObjectWithInput,
)

# OGX tenant_id regex (mirrors ogx.core.datatypes._TENANT_ID_RE). Used only to
# validate the caller-supplied sentinel; derived verbatim values are not
# re-validated because Praxis imposes no constraint on tenant_id and verbatim
# preservation keeps isolation exactly as it was in the source.
_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,127}$")

# PostgreSQL identifier rules for target table names. Names are interpolated
# unquoted into INSERT statements, so restrict them to a safe character set.
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_POSTGRES_MAX_IDENTIFIER_LEN = 63

# Blob keys that are internal to OGX storage and must be stripped before the
# public OpenAI response object is re-serialized into Praxis's response_object.
_INTERNAL_RESPONSE_FIELDS = ("input", "messages", "input_storage_mode")


class _StoredResponseBlob(OpenAIResponseObjectWithInput):
    """Local mirror of OGX's internal ``_OpenAIResponseObjectWithInputAndMessages``.

    Defined here rather than imported from ``ogx.providers.utils.responses`` so
    this module stays free of OGX-core imports. It extends the public
    ``OpenAIResponseObjectWithInput`` with the two fields OGX injects into the
    stored ``response_object`` blob: ``messages`` (hidden chat-completion
    messages) and ``input_storage_mode``. Both are optional for backward
    compatibility with responses stored before those features existed.

    Parsing a stored blob through this model is the dry-run validation gate: a
    blob that fails to parse is surfaced as a transform error.
    """

    messages: list[OpenAIMessageParam] | None = None
    input_storage_mode: str | None = None


def _validate_sentinel(sentinel: str) -> str:
    normalized = sentinel.strip().lower()
    if not _TENANT_ID_RE.match(normalized):
        raise ValueError(f"Failed to validate --tenant-sentinel {sentinel!r}: must match [a-z0-9][a-z0-9-_]{{0,127}}")
    return normalized


def validate_table_name(name: str) -> str:
    """Validate a target (Praxis) table name interpolated into SQL.

    :raises ValueError: If the name is not a safe PostgreSQL identifier.
    """
    if not _TABLE_NAME_RE.match(name) or len(name) > _POSTGRES_MAX_IDENTIFIER_LEN:
        raise ValueError(
            f"Failed to validate Praxis table name {name!r}: must start with a letter or underscore, "
            "contain only letters/numbers/underscores, and be at most 63 characters"
        )
    return name


class TenantDeriver:
    """Derive Praxis ``tenant_id`` from a source row, verbatim.

    Precedence: an explicit ``owner_principal -> tenant_id`` override map first,
    then the source ``tenant_id`` column (present only when tenancy is enabled),
    then ``owner_principal`` (the DISABLED-mode default), then the sentinel for
    empty values. Verbatim copying preserves source isolation exactly with zero
    collision risk.
    """

    def __init__(self, sentinel: str = "default", explicit_map: Mapping[str, str] | None = None) -> None:
        self.sentinel = _validate_sentinel(sentinel)
        self.explicit_map: dict[str, str] = dict(explicit_map or {})

    def derive(self, owner_principal: str | None, tenant_id_col: str | None) -> str:
        if owner_principal is not None and owner_principal in self.explicit_map:
            return self.explicit_map[owner_principal]
        raw = tenant_id_col if tenant_id_col not in (None, "") else owner_principal
        return (raw or "").strip() or self.sentinel


@dataclass(frozen=True)
class PraxisResponseRow:
    """A target row for the Praxis ``responses`` table (PK ``(tenant_id, id)``)."""

    tenant_id: str
    id: str
    created_at: int
    model: str
    response_object: str
    input: str
    messages: str

    def as_row(self) -> tuple[Any, ...]:
        """Positional tuple matching the responses INSERT column order."""
        return (self.id, self.tenant_id, self.created_at, self.model, self.response_object, self.input, self.messages)


@dataclass(frozen=True)
class PraxisConversationRow:
    """A target row for the Praxis ``conversations`` table (PK ``(conversation_id, tenant_id)``)."""

    conversation_id: str
    tenant_id: str
    created_at: int
    metadata: str
    messages: str

    def as_row(self) -> tuple[Any, ...]:
        """Positional tuple matching the conversations INSERT column order."""
        return (self.conversation_id, self.tenant_id, self.created_at, self.metadata, self.messages)


@dataclass(frozen=True)
class PraxisItemRow:
    """A target row for the Praxis ``items`` table (PK ``(item_id, tenant_id, conversation_id)``)."""

    item_id: str
    tenant_id: str
    conversation_id: str
    item_data: str
    created_at: int
    position: int

    def as_row(self) -> tuple[Any, ...]:
        """Positional tuple matching the items INSERT column order."""
        return (self.item_id, self.tenant_id, self.conversation_id, self.item_data, self.created_at, self.position)


def transform_response(row: Mapping[str, Any], tenant: TenantDeriver) -> PraxisResponseRow:
    """Decompose an OGX ``openai_responses`` row into a Praxis responses row.

    The stored ``response_object`` blob is ``OpenAIResponseObject.model_dump()``
    plus injected ``input``/``messages``/``input_storage_mode``. The public
    ``response_object`` is that object with the internal fields stripped and
    re-serialized OpenAI-compliant; ``input`` and ``messages`` are the *raw*
    stored blob fields, copied verbatim (no ancestry reconstruction).
    """
    blob = dict(row["response_object"])
    # Parse through the storage model as a validation gate.
    parsed = _StoredResponseBlob(**blob)
    public = OpenAIResponseObject(
        **{k: v for k, v in parsed.model_dump().items() if k not in _INTERNAL_RESPONSE_FIELDS}
    )
    return PraxisResponseRow(
        tenant_id=tenant.derive(row.get("owner_principal"), row.get("tenant_id")),
        id=row["id"],
        created_at=int(row["created_at"]),
        model=row["model"],
        response_object=public.model_dump_json(),
        input=json.dumps(blob.get("input", [])),
        messages=json.dumps(blob.get("messages") or []),
    )


def transform_conversation(
    conv_row: Mapping[str, Any],
    messages_row: Mapping[str, Any] | None,
    tenant: TenantDeriver,
) -> PraxisConversationRow:
    """Build a Praxis conversations row by joining an ``openai_conversations``
    row with its ``conversation_messages`` row (may be absent).

    :raises ValueError: If both rows are present and derive to different
        tenants — the two source tables disagree about who owns this
        conversation, which the join cannot silently resolve.
    """
    tenant_id = tenant.derive(conv_row.get("owner_principal"), conv_row.get("tenant_id"))
    messages = None
    if messages_row is not None:
        messages = messages_row["messages"]
        messages_tenant_id = tenant.derive(messages_row.get("owner_principal"), messages_row.get("tenant_id"))
        if messages_tenant_id != tenant_id:
            raise ValueError(
                f"Failed to join conversation {conv_row.get('id')!r}: openai_conversations derives tenant_id "
                f"{tenant_id!r} but conversation_messages derives {messages_tenant_id!r}"
            )
    return PraxisConversationRow(
        conversation_id=conv_row["id"],
        tenant_id=tenant_id,
        created_at=int(conv_row["created_at"]),
        metadata=json.dumps(conv_row.get("metadata") or {}),
        messages=json.dumps(messages or []),
    )


def transform_message_only_conversation(
    messages_row: Mapping[str, Any],
    tenant: TenantDeriver,
    orphan_created_at: int,
) -> PraxisConversationRow:
    """Synthesize a Praxis conversations row for a ``conversation_messages``
    orphan — a conversation with continuity messages but no
    ``openai_conversations`` record (no source ``created_at`` or ``metadata``)."""
    return PraxisConversationRow(
        conversation_id=messages_row["conversation_id"],
        tenant_id=tenant.derive(messages_row.get("owner_principal"), messages_row.get("tenant_id")),
        created_at=int(orphan_created_at),
        metadata="{}",
        messages=json.dumps(messages_row.get("messages") or []),
    )


def transform_item(row: Mapping[str, Any], tenant: TenantDeriver) -> PraxisItemRow:
    """Copy an OGX ``conversation_items`` row into a Praxis items row.

    ``id -> item_id``, ``sort_order -> position`` (legacy ``NULL`` becomes 0),
    ``item_data`` re-serialized, ``created_at`` widened to int.
    """
    sort_order = row.get("sort_order")
    return PraxisItemRow(
        item_id=row["id"],
        tenant_id=tenant.derive(row.get("owner_principal"), row.get("tenant_id")),
        conversation_id=row["conversation_id"],
        item_data=json.dumps(row["item_data"]),
        created_at=int(row["created_at"]),
        position=int(sort_order) if sort_order is not None else 0,
    )


def transform_legacy_inline_item(
    element: Mapping[str, Any],
    position: int,
    conv_row: Mapping[str, Any],
    tenant: TenantDeriver,
) -> PraxisItemRow:
    """Backfill an item from the deprecated inline ``openai_conversations.items``
    list into the Praxis items stream.

    The element is a stored ``ConversationItem`` dict; it must carry an ``id``
    (its target primary key). Tenant and ``created_at`` are inherited from the
    parent conversation row, and ``position`` is the element's list index.
    """
    item_id = element.get("id")
    if not item_id:
        raise ValueError(
            f"Failed to backfill legacy inline item at position {position} for conversation "
            f"{conv_row.get('id')!r}: element has no 'id'"
        )
    return PraxisItemRow(
        item_id=item_id,
        tenant_id=tenant.derive(conv_row.get("owner_principal"), conv_row.get("tenant_id")),
        conversation_id=conv_row["id"],
        item_data=json.dumps(dict(element)),
        created_at=int(conv_row["created_at"]),
        position=position,
    )


# INSERT ... ON CONFLICT DO NOTHING statements keyed by logical table. Conflict
# targets equal Praxis's primary keys (verified against schemas.rs). Praxis
# stores JSON columns as TEXT holding serde_json strings, so binding Python str
# is the exact contract. {t} is the validated physical table name.
_INSERT_SQL: dict[str, str] = {
    "responses": (
        "INSERT INTO {t} (id, tenant_id, created_at, model, response_object, input, messages) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (tenant_id, id) DO NOTHING"
    ),
    "conversations": (
        "INSERT INTO {t} (conversation_id, tenant_id, created_at, metadata, messages) "
        "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (conversation_id, tenant_id) DO NOTHING"
    ),
    "items": (
        "INSERT INTO {t} (item_id, tenant_id, conversation_id, item_data, created_at, position) "
        "VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (item_id, tenant_id, conversation_id) DO NOTHING"
    ),
}


class PraxisWriter:
    """asyncpg writer for the Praxis target. Carries no DDL — Praxis stamps its
    own schema on boot, so this only ``INSERT``s.

    Idempotency/resumability rests solely on ``ON CONFLICT DO NOTHING`` against
    the natural primary key: a restarted Job re-reads the source and skips
    already-written rows, crash-safe at any point and safe to re-run, with zero
    footprint in the target.
    """

    def __init__(self, dsn: str, tables: Mapping[str, str]) -> None:
        """:param dsn: asyncpg DSN for the target.
        :param tables: logical (``responses``/``conversations``/``items``) ->
            physical table name in the Praxis deployment.
        """
        self.dsn = dsn
        self.tables = {kind: validate_table_name(name) for kind, name in tables.items()}
        self._conn: asyncpg.Connection | None = None

    async def connect(self) -> None:
        self._conn = await asyncpg.connect(self.dsn)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def sql_for(self, kind: str) -> str:
        """Return the INSERT statement for a logical table with its name bound."""
        if kind not in _INSERT_SQL:
            raise ValueError(f"Failed to build SQL for unknown target table {kind!r}")
        if kind not in self.tables:
            raise ValueError(f"Failed to build SQL: no physical table name configured for {kind!r}")
        return _INSERT_SQL[kind].format(t=self.tables[kind])

    async def write_batch(self, kind: str, rows: Sequence[tuple[Any, ...]]) -> int:
        """Write a batch of positional tuples in a single transaction.

        Uses ``executemany`` (not ``COPY``, which cannot express ``ON CONFLICT``).
        Returns the number of rows submitted (attempted), not net inserted —
        conflicts are silently skipped.
        """
        if not rows:
            return 0
        if self._conn is None:
            raise RuntimeError("Failed to write batch: PraxisWriter is not connected")
        sql = self.sql_for(kind)
        async with self._conn.transaction():
            await self._conn.executemany(sql, rows)
        return len(rows)
