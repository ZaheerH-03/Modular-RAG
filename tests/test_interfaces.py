"""
tests/test_interfaces.py
========================
Structural tests that verify every concrete class properly implements its
corresponding Abstract Base Class (ABC).

These tests do NOT load any models or access the GPU — they only inspect the
class hierarchy, so they run instantly in any environment.

Run from the repo root with:

    .venv\\Scripts\\activate
    python -m pytest tests/test_interfaces.py -v
"""

import pytest

# ---------------------------------------------------------------------------
# ABCs under test
# ---------------------------------------------------------------------------
from interfaces import (
    BaseDocumentLoader,
    BaseEmbedder,
    BaseIndexBuilder,
    BaseLLMLoader,
    BasePromptProvider,
    BaseRetrieverBuilder,
)

# ---------------------------------------------------------------------------
# Concrete implementations under test
# ---------------------------------------------------------------------------
from ingestion.data_loader import IncrementalDocumentLoader
from ingestion.build_vector_index import HuggingFaceEmbedder, VectorIndexBuilder
from retrieval.hybrid_retriever import HybridRRFRetrieverBuilder
from llm_loaders.local_llm_loader import QuantizedLocalLLM
from prompts.base import CustomPromptProvider


# ---------------------------------------------------------------------------
# Interface compliance tests
# ---------------------------------------------------------------------------


class TestBaseDocumentLoader:
    def test_incremental_loader_is_subclass(self):
        assert issubclass(IncrementalDocumentLoader, BaseDocumentLoader)

    def test_incremental_loader_is_instantiable(self):
        loader = IncrementalDocumentLoader()
        assert isinstance(loader, BaseDocumentLoader)

    def test_incremental_loader_has_required_method(self):
        assert callable(getattr(IncrementalDocumentLoader, "get_document_updates", None))


class TestBaseEmbedder:
    def test_hf_embedder_is_subclass(self):
        assert issubclass(HuggingFaceEmbedder, BaseEmbedder)

    def test_hf_embedder_is_instantiable(self):
        embedder = HuggingFaceEmbedder.__new__(HuggingFaceEmbedder)
        assert isinstance(embedder, BaseEmbedder)

    def test_hf_embedder_has_required_method(self):
        assert callable(getattr(HuggingFaceEmbedder, "get_model", None))


class TestBaseIndexBuilder:
    def test_vector_index_builder_is_subclass(self):
        assert issubclass(VectorIndexBuilder, BaseIndexBuilder)

    def test_vector_index_builder_has_required_method(self):
        assert callable(getattr(VectorIndexBuilder, "build", None))


class TestBaseRetrieverBuilder:
    def test_hybrid_retriever_builder_is_subclass(self):
        assert issubclass(HybridRRFRetrieverBuilder, BaseRetrieverBuilder)

    def test_hybrid_retriever_builder_is_instantiable(self):
        builder = HybridRRFRetrieverBuilder()
        assert isinstance(builder, BaseRetrieverBuilder)

    def test_hybrid_retriever_builder_has_required_method(self):
        assert callable(getattr(HybridRRFRetrieverBuilder, "build", None))

    def test_hybrid_retriever_builder_default_params(self):
        builder = HybridRRFRetrieverBuilder()
        assert builder._vector_top_k == 5
        assert builder._bm25_top_k == 5
        assert builder._fusion_top_k == 2
        assert builder._num_queries == 1

    def test_hybrid_retriever_builder_custom_params(self):
        builder = HybridRRFRetrieverBuilder(vector_top_k=10, fusion_top_k=5)
        assert builder._vector_top_k == 10
        assert builder._fusion_top_k == 5


class TestBaseLLMLoader:
    def test_quantized_llm_is_subclass(self):
        assert issubclass(QuantizedLocalLLM, BaseLLMLoader)

    def test_quantized_llm_has_required_method(self):
        assert callable(getattr(QuantizedLocalLLM, "get_llm", None))


class TestBasePromptProvider:
    def test_custom_prompt_provider_is_subclass(self):
        assert issubclass(CustomPromptProvider, BasePromptProvider)

    def test_custom_prompt_provider_is_instantiable(self):
        provider = CustomPromptProvider()
        assert isinstance(provider, BasePromptProvider)

    def test_custom_prompt_provider_has_required_method(self):
        assert callable(getattr(CustomPromptProvider, "get_prompt", None))

    def test_custom_prompt_provider_returns_prompt_template(self):
        from llama_index.core import PromptTemplate
        provider = CustomPromptProvider()
        prompt = provider.get_prompt()
        assert isinstance(prompt, PromptTemplate)

    def test_prompt_contains_required_placeholders(self):
        provider = CustomPromptProvider()
        prompt = provider.get_prompt()
        assert "{context_str}" in prompt.template
        assert "{query_str}" in prompt.template
