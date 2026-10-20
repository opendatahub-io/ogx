# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for source-config resolution in ``ogx migrate praxis``.

Focus: ``_resolve_responses_store`` must read the responses store from the
responses provider's ``config.persistence.responses`` — the location OGX actually
writes to — rather than ``storage.stores.responses``, which stock and
operator-generated configs leave unset. This is the read leg for both the
``openai_responses`` table and the ``conversation_messages`` table (the latter
lives on the responses backend).
"""

import yaml

from ogx.cli.migrate.praxis.cmd import _resolve_responses_store
from ogx.core.configure import parse_and_maybe_upgrade_config
from ogx.core.utils.config_resolution import resolve_config_or_distro


def _load_postgres_run_config():
    config_file = resolve_config_or_distro("ci-tests::run-with-postgres-store.yaml")
    return parse_and_maybe_upgrade_config(yaml.safe_load(config_file.read_text()))


def test_resolves_responses_store_from_provider_persistence():
    """Regression: the stock Postgres config leaves storage.stores.responses unset,
    yet OGX persists responses via the provider's persistence.responses. The
    resolver must find the store there so a real migration does not fail."""
    run_config = _load_postgres_run_config()

    # Guard the premise: if this ever starts being populated, the resolver
    # rationale (and this test) should be revisited.
    assert run_config.storage.stores.responses is None

    ref = _resolve_responses_store(run_config)
    assert ref is not None
    assert ref.table_name == "responses"
    assert ref.backend == "sql_default"


def test_returns_none_when_no_responses_provider():
    run_config = _load_postgres_run_config()
    run_config.providers["responses"] = []
    assert _resolve_responses_store(run_config) is None


def test_returns_none_when_provider_has_no_persistence():
    run_config = _load_postgres_run_config()
    for provider in run_config.providers["responses"]:
        provider.config.pop("persistence", None)
    assert _resolve_responses_store(run_config) is None
