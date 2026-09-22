# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for Praxis migration command-level validation."""

import argparse

import pytest

from ogx.cli.migrate.praxis.cmd import _run


async def test_continue_on_error_requires_dry_run() -> None:
    args = argparse.Namespace(batch_size=1, continue_on_error=True, dry_run=False)

    with pytest.raises(ValueError, match="continue-on-error requires --dry-run"):
        await _run(args)
