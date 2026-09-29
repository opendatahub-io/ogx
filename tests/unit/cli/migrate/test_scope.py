# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for ``ogx migrate praxis --tables`` scope validation."""

import pytest

from ogx.cli.migrate.praxis.cmd import _parse_tables


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("responses", {"responses"}),
        ("conversations", {"conversations"}),
        ("conversations,items", {"conversations", "items"}),
        (
            "responses,conversations,items",
            {"responses", "conversations", "items"},
        ),
    ],
)
def test_parse_tables_accepts_valid_scopes(raw: str, expected: set[str]) -> None:
    assert _parse_tables(raw) == expected


@pytest.mark.parametrize("raw", ["items", "responses,items"])
def test_parse_tables_rejects_items_without_conversations(raw: str) -> None:
    with pytest.raises(ValueError, match="'items' requires 'conversations'"):
        _parse_tables(raw)
