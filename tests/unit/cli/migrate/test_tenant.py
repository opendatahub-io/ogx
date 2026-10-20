# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for TenantDeriver — source tenant_id plus CLI fallback."""

import pytest

from ogx.cli.migrate.praxis.reader import _handle_row_error, _Stats
from ogx.cli.migrate.praxis.target import TenantDerivationError, TenantDeriver


def test_tenant_id_column_takes_precedence_when_present():
    deriver = TenantDeriver(fallback_tenant="fallback-tenant")
    # Tenancy enabled: tenant_id column wins over owner_principal.
    assert deriver.derive("owner@example.com", "tenant-a") == "tenant-a"


def test_empty_tenant_id_column_falls_back_to_cli_tenant():
    deriver = TenantDeriver(fallback_tenant="fallback-tenant")
    assert deriver.derive("owner-x", "") == "fallback-tenant"


def test_whitespace_only_tenant_id_column_falls_back_to_cli_tenant():
    deriver = TenantDeriver(fallback_tenant="fallback-tenant")
    assert deriver.derive("owner-x", "   ") == "fallback-tenant"


def test_disabled_mode_none_column_uses_cli_fallback():
    deriver = TenantDeriver(fallback_tenant="fallback-tenant")
    # DISABLED mode surfaces tenant_id_col=None (physical column absent).
    assert deriver.derive("owner-y", None) == "fallback-tenant"


def test_missing_tenant_without_cli_fallback_fails():
    deriver = TenantDeriver()
    with pytest.raises(TenantDerivationError, match="source row has no tenant_id.*fallback-tenant") as error:
        deriver.derive("owner-y", None)
    stats = _Stats()
    _handle_row_error("responses", "resp-1", error.value, stats, continue_on_error=True)
    assert stats.skipped == [("responses", "resp-1", "TenantDerivationError")]


def test_custom_fallback_tenant():
    deriver = TenantDeriver(fallback_tenant="fallback-tenant")
    assert deriver.derive(None, None) == "fallback-tenant"


def test_whitespace_is_stripped_from_verbatim_value():
    deriver = TenantDeriver()
    assert deriver.derive("ignored-owner", "  spaced-tenant  ") == "spaced-tenant"


def test_invalid_fallback_tenant_rejected():
    with pytest.raises(ValueError, match="fallback-tenant"):
        TenantDeriver(fallback_tenant="Not Valid!")


def test_verbatim_values_are_not_revalidated():
    # Praxis imposes no constraint on tenant_id; verbatim values with uppercase
    # or characters the fallback-tenant regex would reject are preserved as-is.
    deriver = TenantDeriver()
    assert deriver.derive("ignored-owner", "UPPER Case!") == "UPPER Case!"
