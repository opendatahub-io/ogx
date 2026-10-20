# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Praxis owner issuer validation."""

import pytest

from ogx.cli.migrate.praxis.target import DEFAULT_OWNER_ISSUER, validate_owner_issuer


def test_default_owner_issuer_is_a_valid_urn() -> None:
    assert validate_owner_issuer(DEFAULT_OWNER_ISSUER) == DEFAULT_OWNER_ISSUER


@pytest.mark.parametrize(
    "issuer",
    [
        "urn:example:deployment:production",
        "urn:1abc:foo",
        "urn:uuid:6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "urn:example:animal:ferret:nose",
        "URN:example:deployment:production",
    ],
)
def test_valid_urns_are_accepted(issuer: str) -> None:
    assert validate_owner_issuer(issuer) == issuer


@pytest.mark.parametrize(
    "issuer",
    ["", "https://example.com", "not-a-urn", "urn:x", "urn:example:", "urn:ab-:foo"],
)
def test_invalid_urns_are_rejected(issuer: str) -> None:
    with pytest.raises(ValueError, match="--owner-issuer.*valid URN"):
        validate_owner_issuer(issuer)


def test_owner_issuer_whitespace_is_trimmed() -> None:
    assert validate_owner_issuer("  urn:example:deployment  ") == "urn:example:deployment"
