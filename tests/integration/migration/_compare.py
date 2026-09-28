# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Backend-agnostic normalization + golden comparison for the Praxis e2e.

This module has **no OGX imports** (stdlib only) so it is trivially importable
from both the in-process golden-guard unit test and the Postgres assertion test.

The migration's target rows are deterministic regardless of source backend: every
JSON column is re-serialized by the pure transforms in ``target.py``
(``model_dump_json`` / ``json.dumps``), and Praxis stores them verbatim as
``TEXT``. So a golden fixture generated in-process against SQLite is byte-faithful
to the rows a real Postgres+Praxis round-trip produces — the normalizer parses the
JSON-as-TEXT columns back into structures and compares them order-independently,
so even JSON key-ordering differences between backends are irrelevant.

Two producers feed the same normalizer:
  * the in-process pipeline yields ``PraxisRow.as_row()`` positional tuples;
  * the e2e ``SELECT`` reads the Praxis tables with columns projected in the
    same order (:data:`TARGET_COLUMNS`).
Both therefore normalize identically against one shared golden.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_EXPECTED_DIR = Path(__file__).parent / "expected"

# Target column order per logical table. MUST match the positional order of the
# ``as_row()`` tuples in ``target.py`` and the INSERT column lists, so
# the in-process (tuple) and Postgres (SELECT) producers align on one normalizer.
TARGET_COLUMNS: dict[str, tuple[str, ...]] = {
    "responses": ("id", "tenant_id", "created_at", "model", "response_object", "input", "messages"),
    "conversations": ("conversation_id", "tenant_id", "created_at", "metadata", "messages"),
    "items": ("item_id", "tenant_id", "conversation_id", "item_data", "created_at", "position"),
}

# Columns stored as JSON-in-TEXT by Praxis; parsed back to structures before compare.
JSON_COLUMNS: dict[str, tuple[str, ...]] = {
    "responses": ("response_object", "input", "messages"),
    "conversations": ("metadata", "messages"),
    "items": ("item_data",),
}

# Natural primary key per table (matches the Praxis PKs in schemas.rs). Used to
# index rows so comparison is order-independent and mismatches point at a row.
PK_COLUMNS: dict[str, tuple[str, ...]] = {
    "responses": ("tenant_id", "id"),
    "conversations": ("conversation_id", "tenant_id"),
    "items": ("item_id", "tenant_id", "conversation_id"),
}

KINDS: tuple[str, ...] = ("responses", "conversations", "items")


def _row_to_mapping(kind: str, row: Sequence[Any] | Mapping[str, Any]) -> dict[str, Any]:
    columns = TARGET_COLUMNS[kind]
    if isinstance(row, Mapping):
        return {col: row[col] for col in columns}
    if len(row) != len(columns):
        raise ValueError(
            f"Failed to normalize {kind} row: expected {len(columns)} columns {columns}, got {len(row)} values"
        )
    return dict(zip(columns, row, strict=True))


def _parse_json_columns(kind: str, record: dict[str, Any]) -> dict[str, Any]:
    for col in JSON_COLUMNS[kind]:
        value = record[col]
        if isinstance(value, str):
            record[col] = json.loads(value)
    return record


def _pk_key(kind: str, record: Mapping[str, Any]) -> str:
    # NUL-joined so it can never collide with a value that contains the separator.
    return "\x00".join(str(record[col]) for col in PK_COLUMNS[kind])


def normalize_rows(kind: str, rows: Sequence[Sequence[Any] | Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Normalize target rows into ``{pk -> record}`` with JSON columns parsed.

    :raises ValueError: on a duplicate primary key (the target PK must be unique).
    """
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        record = _parse_json_columns(kind, _row_to_mapping(kind, row))
        key = _pk_key(kind, record)
        if key in out:
            raise ValueError(f"Failed to normalize {kind} rows: duplicate primary key {key!r}")
        out[key] = record
    return out


def normalize_all(batches: Mapping[str, Sequence[Sequence[Any] | Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Normalize every logical table; absent kinds normalize to an empty mapping."""
    return {kind: normalize_rows(kind, batches.get(kind, [])) for kind in KINDS}


def diff_normalized(
    expected: Mapping[str, Mapping[str, Any]],
    actual: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return human-readable mismatch lines comparing two ``normalize_all`` maps.

    Empty list means the two are structurally equal.
    """
    problems: list[str] = []
    for kind in KINDS:
        exp = expected.get(kind, {})
        act = actual.get(kind, {})
        missing = sorted(set(exp) - set(act))
        extra = sorted(set(act) - set(exp))
        if missing:
            problems.append(f"{kind}: missing rows {missing}")
        if extra:
            problems.append(f"{kind}: unexpected rows {extra}")
        for key in sorted(set(exp) & set(act)):
            if exp[key] != act[key]:
                problems.append(
                    f"{kind}[{key}] differs:\n  expected={json.dumps(exp[key], sort_keys=True)}\n"
                    f"  actual  ={json.dumps(act[key], sort_keys=True)}"
                )
    return problems


def dumps_golden(normalized: Mapping[str, Mapping[str, Any]]) -> str:
    """Serialize a golden fixture: sorted keys + trailing newline for stable diffs."""
    return json.dumps(normalized, sort_keys=True, indent=2) + "\n"


def golden_path(mode: str) -> Path:
    """Filesystem path to the golden fixture for a tenancy mode."""
    return _EXPECTED_DIR / f"expected_{mode}.json"


def load_golden(mode: str) -> dict[str, dict[str, Any]]:
    return json.loads(golden_path(mode).read_text())


def write_golden(mode: str, normalized: Mapping[str, Mapping[str, Any]]) -> None:
    _EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    golden_path(mode).write_text(dumps_golden(normalized))
