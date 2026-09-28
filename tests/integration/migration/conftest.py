# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Fixtures for the OGX->Praxis migration e2e.

This suite is assertion-only: the GitHub Actions workflow
(``.github/workflows/migration-e2e-praxis.yml``) does the heavy lifting — it
stands up Postgres + Praxis, seeds the OGX source, and runs the real ``ogx
migrate praxis`` CLI — then hands this test two things via the environment:

* ``PRAXIS_TEST_DSN`` — an asyncpg DSN for the *target* Postgres the migration
  wrote to (read-only access is sufficient); and
* ``MIGRATION_TENANCY_MODE`` — which tenancy leg ran (``disabled``, ``single``,
  or ``multi``),
  selecting the golden fixture to compare against.

Both are required; the test skips (rather than silently comparing against the
wrong golden) when either is absent, so a plain ``uv run pytest
tests/integration/migration`` is a no-op without the live target.
"""

import os

import pytest

_VALID_MODES = ("disabled", "single", "multi")


@pytest.fixture(scope="session")
def praxis_dsn() -> str:
    dsn = os.environ.get("PRAXIS_TEST_DSN")
    if not dsn:
        pytest.skip("PRAXIS_TEST_DSN is not set; the Praxis migration e2e requires a live target Postgres DSN")
    return dsn


@pytest.fixture(scope="session")
def tenancy_mode() -> str:
    mode = os.environ.get("MIGRATION_TENANCY_MODE")
    if not mode:
        pytest.skip(
            "MIGRATION_TENANCY_MODE is not set; expected 'disabled', 'single', or 'multi' to select the golden fixture"
        )
    if mode not in _VALID_MODES:
        pytest.fail(f"MIGRATION_TENANCY_MODE={mode!r} is invalid; expected one of {list(_VALID_MODES)}")
    return mode
