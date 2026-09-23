# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for Praxis migration command-level validation."""

import argparse

import pytest

from ogx.cli.migrate.praxis.cmd import PraxisMigrate, _run
from ogx.cli.migrate.praxis.target import DEFAULT_OWNER_ISSUER


async def test_continue_on_error_requires_dry_run() -> None:
    args = argparse.Namespace(batch_size=1, continue_on_error=True, dry_run=False)

    with pytest.raises(ValueError, match="continue-on-error requires --dry-run"):
        await _run(args)


def test_owner_issuer_argument_defaults_to_production_urn() -> None:
    subparsers = argparse.ArgumentParser().add_subparsers()
    PraxisMigrate(subparsers)

    args = subparsers.choices["praxis"].parse_args(["starter"])

    assert args.owner_issuer == DEFAULT_OWNER_ISSUER


def test_owner_issuer_argument_rejects_invalid_urn() -> None:
    subparsers = argparse.ArgumentParser().add_subparsers()
    PraxisMigrate(subparsers)

    with pytest.raises(SystemExit):
        subparsers.choices["praxis"].parse_args(["starter", "--owner-issuer", "https://example.com"])
