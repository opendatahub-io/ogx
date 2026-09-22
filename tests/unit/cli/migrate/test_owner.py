# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for Praxis owner_subject derivation."""

import pytest

from ogx.cli.migrate.praxis.target import OwnerSubjectDerivationError, OwnerSubjectDeriver


def test_source_owner_principal_takes_precedence() -> None:
    deriver = OwnerSubjectDeriver("fallback-owner")

    assert deriver.derive("source-owner") == "source-owner"


@pytest.mark.parametrize("owner_principal", [None, "", "   "])
def test_empty_source_owner_uses_fallback(owner_principal: str | None) -> None:
    deriver = OwnerSubjectDeriver("fallback-owner")

    assert deriver.derive(owner_principal) == "fallback-owner"


def test_owner_principal_whitespace_is_trimmed() -> None:
    deriver = OwnerSubjectDeriver("fallback-owner")

    assert deriver.derive("  source-owner  ") == "source-owner"


@pytest.mark.parametrize("fallback", ["", "   "])
def test_missing_or_blank_fallback_is_rejected(fallback: str | None) -> None:
    with pytest.raises(ValueError, match="fallback-owner-subject.*non-empty"):
        OwnerSubjectDeriver(fallback)


def test_none_fallback_is_allowed_when_source_owner_exists() -> None:
    assert OwnerSubjectDeriver().derive("source-owner") == "source-owner"


def test_missing_source_owner_without_fallback_fails() -> None:
    with pytest.raises(OwnerSubjectDerivationError, match="owner_subject.*fallback-owner-subject"):
        OwnerSubjectDeriver().derive(None)


def test_fallback_is_trimmed() -> None:
    assert OwnerSubjectDeriver("  fallback-owner  ").derive(None) == "fallback-owner"
