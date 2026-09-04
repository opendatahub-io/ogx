# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for TenantDeriver — verbatim tenant_id derivation.

Precedence covered: explicit-map override, source tenant_id column (tenancy
enabled), owner_principal fallback (DISABLED mode, no tenant_id column), and
empty -> sentinel.
"""

import pytest

from ogx.cli.migrate.praxis.target import TenantDeriver


def test_owner_principal_verbatim():
    deriver = TenantDeriver()
    # DISABLED mode: no tenant_id column, owner_principal copied verbatim (not lowercased/hashed).
    assert deriver.derive("Acme-Corp_123", None) == "Acme-Corp_123"


def test_tenant_id_column_takes_precedence_when_present():
    deriver = TenantDeriver()
    # Tenancy enabled: tenant_id column wins over owner_principal.
    assert deriver.derive("owner@example.com", "tenant-a") == "tenant-a"


def test_empty_tenant_id_column_falls_back_to_owner_principal():
    deriver = TenantDeriver()
    assert deriver.derive("owner-x", "") == "owner-x"


def test_whitespace_only_tenant_id_column_falls_back_to_owner_principal():
    deriver = TenantDeriver()
    assert deriver.derive("owner-x", "   ") == "owner-x"


def test_disabled_mode_none_column_uses_owner_principal():
    deriver = TenantDeriver()
    # DISABLED mode surfaces tenant_id_col=None (physical column absent).
    assert deriver.derive("owner-y", None) == "owner-y"


def test_empty_owner_and_column_uses_sentinel():
    deriver = TenantDeriver(sentinel="default")
    assert deriver.derive("", None) == "default"
    assert deriver.derive(None, None) == "default"
    assert deriver.derive("   ", None) == "default"


def test_custom_sentinel():
    deriver = TenantDeriver(sentinel="fallback-tenant")
    assert deriver.derive(None, None) == "fallback-tenant"


def test_explicit_map_override_wins():
    deriver = TenantDeriver(explicit_map={"owner@example.com": "mapped-tenant"})
    # Override wins even when a tenant_id column is present.
    assert deriver.derive("owner@example.com", "tenant-a") == "mapped-tenant"


def test_explicit_map_only_applies_to_matching_owner():
    deriver = TenantDeriver(explicit_map={"someone-else": "mapped"})
    assert deriver.derive("owner@example.com", None) == "owner@example.com"


def test_whitespace_is_stripped_from_verbatim_value():
    deriver = TenantDeriver()
    assert deriver.derive("  spaced-owner  ", None) == "spaced-owner"


def test_invalid_sentinel_rejected():
    with pytest.raises(ValueError, match="tenant-sentinel"):
        TenantDeriver(sentinel="Not Valid!")


def test_verbatim_values_are_not_revalidated():
    # Praxis imposes no constraint on tenant_id; verbatim values with uppercase
    # or characters the OGX sentinel regex would reject are preserved as-is.
    deriver = TenantDeriver()
    assert deriver.derive("UPPER Case!", None) == "UPPER Case!"
