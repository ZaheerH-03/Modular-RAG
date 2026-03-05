"""
query_engine.py
===============
Top-level entry point for the ModularRAG pipeline.

Orchestrates the full setup sequence:

1. Load and register the quantized local LLM (must happen first so LlamaIndex
   never falls back to OpenAI when building retrievers or synthesizers).
2. Build / sync the ChromaDB-backed vector index.
3. Construct the hybrid RRF retriever.
4. Assemble the response synthesizer with the custom strict prompt.
5. Wire everything into a ``RetrieverQueryEngine``.
"""

import os

from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from ingestion.build_vector_index import build_vector_index
from interfaces import BaseIndexBuilder, BaseLLMLoader, BasePromptProvider, BaseRetrieverBuilder
from llm_loaders.local_llm_loader import QuantizedLocalLLM
from prompts.base import get_custom_prompt
from retrieval.hybrid_retriever import get_hybrid_rrf_retriever

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH: str = "meta-llama/Llama-3.2-3B-Instruct"
_DEFAULT_MAX_NEW_TOKENS: int = 256
_DEFAULT_CONTEXT_WINDOW: int = 2048
_DEFAULT_TEMPERATURE: float = 0.1


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------


def setup_query_engine(
    year: int,
    branch: str,
    model_path: str = _DEFAULT_MODEL_PATH,
) -> RetrieverQueryEngine:
    """
    Initialise and wire together the full RAG query engine.

    Args:
        year:        Target curriculum year (cumulative — includes all prior years).
        branch:      Branch identifier (e.g. ``"ds"``, ``"cs"``).
        model_path:  HuggingFace Hub ID or local directory path for the LLM.

    Returns:
        A fully assembled ``RetrieverQueryEngine`` ready to answer queries.
    """
    print(f"Initializing Query Engine for Year {year} — {branch.upper()}...")

    # Step 1: Load the LLM and register it globally FIRST.
    #   This must happen before any retriever or synthesizer is constructed.
    #   LlamaIndex's QueryFusionRetriever.__init__ calls resolve_llm("default"),
    #   which will crash trying to reach OpenAI if Settings.llm is not set yet.
    print("Loading Local LLM...")
    llm = QuantizedLocalLLM(
        model_name=model_path,
        max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
        context_window=_DEFAULT_CONTEXT_WINDOW,
        temperature=_DEFAULT_TEMPERATURE,
    )
    Settings.llm = llm

    # Step 2: Build / sync the ChromaDB-backed vector index.
    index = build_vector_index(year, branch)

    # Step 3: Construct the hybrid RRF retriever (vector + BM25).
    hybrid_retriever = get_hybrid_rrf_retriever(index)

    # Step 4: Load the custom strict QA prompt.
    qa_prompt = get_custom_prompt()

    # Step 5: Assemble the response synthesizer.
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=qa_prompt,
    )

    # Step 6: Wire retriever + synthesizer into the final query engine.
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = setup_query_engine(year=4, branch="ds", model_path=_DEFAULT_MODEL_PATH)

    test_query = "Different types of organizational behaviour models?"
    print(f"\nUser Query: {test_query}\n")

    response = engine.query(test_query)

    print("\n" + "=" * 50)
    print("Answer:")
    print("=" * 50)
    print(response.response)
    print("\n" + "=" * 50)

    print("SOURCES USED:")
    for node in response.source_nodes:
        doc_name = os.path.basename(node.metadata.get("file_path", "Unknown"))
        page = node.metadata.get("page_label", "?")
        print(f"  - {doc_name}, p.{page}  (Score: {node.score:.4f})")