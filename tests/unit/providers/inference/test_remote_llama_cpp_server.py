# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.remote.inference.llama_cpp_server.config import LlamaCppServerConfig
from ogx.providers.remote.inference.llama_cpp_server.llama_cpp_server import (
    LlamaCppServerInferenceAdapter,
)
from ogx_api import ModelType


class TestConstructModelFromIdentifier:
    """construct_model_from_identifier() delegates to the shared classify_model();
    see test_models_dev_registry.py for classification coverage (models.dev lookup,
    metadata enrichment, name-heuristic fallback, rerank, HuggingFace precedence)."""

    def _make_adapter(self) -> LlamaCppServerInferenceAdapter:
        config = LlamaCppServerConfig(base_url="http://mocked.localhost:8080/v1")
        adapter = LlamaCppServerInferenceAdapter(config=config)
        adapter.__provider_id__ = "llama-cpp-server"
        return adapter

    def test_classified_model_uses_adapters_provider_id(self):
        adapter = self._make_adapter()
        model = adapter.construct_model_from_identifier("nomic-embed-text-v1.5")

        assert model.model_type == ModelType.embedding
        assert model.provider_id == "llama-cpp-server"

    def test_unclassified_identifier_falls_through_to_default(self):
        adapter = self._make_adapter()
        model = adapter.construct_model_from_identifier("qwen3-0.6b")

        assert model.model_type == ModelType.llm
