"""
query_engine.py
===============
Top-level entry point for the ModularRAG pipeline.

Orchestrates the full setup sequence:

1. Load and register the LLM (must happen first so LlamaIndex never falls
   back to OpenAI when building retrievers or synthesizers).
2. Build / sync the ChromaDB-backed vector index.
3. Construct the hybrid RRF retriever.
4. Assemble the response synthesizer with the custom strict prompt.
5. Wire everything into a ``RetrieverQueryEngine``.

All tunable settings (paths, model names, top-k values, etc.) are read from
``config.yaml`` via ``config.py`` — edit that file instead of this one.
"""

import os

from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from config import cfg
from ingestion.build_vector_index import DB_DIR, build_vector_index
from interfaces import BaseLLMLoader
from llm_loaders.local_llm_loader import QuantizedLocalLLM
from llm_loaders.ollama_loader import OllamaLoader
from prompts.base import get_custom_prompt
from retrieval.hybrid_retriever import get_hybrid_rrf_retriever


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm(backend: str) -> BaseLLMLoader:
    """
    Instantiate the correct LLM loader based on the ``backend`` string.

    Args:
        backend: ``"local"`` for QuantizedLocalLLM or ``"ollama"`` for OllamaLoader.

    Returns:
        A :class:`~interfaces.BaseLLMLoader` whose ``.get_llm()`` returns the LLM.

    Raises:
        ValueError: If ``backend`` is not ``"local"`` or ``"ollama"``.
    """
    if backend == "local":
        c = cfg.local_llm
        return QuantizedLocalLLM(
            model_name=c.model,
            max_new_tokens=c.max_new_tokens,
            context_window=c.context_window,
            temperature=c.temperature,
        )
    if backend == "ollama":
        c = cfg.ollama_llm
        return OllamaLoader(
            model_name=c.model,
            request_timeout=c.request_timeout,
            context_window=c.context_window,
            temperature=c.temperature,
        )
    raise ValueError(
        f"Unknown llm_backend '{backend}'. "
        "Valid options: 'local', 'ollama'. Update config.yaml."
    )


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------


def setup_query_engine(
    year: int | None = None,
    branch: str | None = None,
    backend: str | None = None,
    streaming: bool | None = None,
) -> RetrieverQueryEngine:
    """
    Initialise and wire together the full RAG query engine.

    All arguments fall back to values in ``config.yaml`` when not supplied.

    Args:
        year:      Target curriculum year (cumulative — includes all prior years).
        branch:    Branch identifier (e.g. ``"ds"``, ``"cs"``).
        backend:   LLM backend — ``"local"`` or ``"ollama"``. Overrides config.
        streaming: Enable token-by-token streaming output. Overrides config.
                   Only works with backends that implement ``stream_complete()``
                   (i.e. ``"ollama"``).

    Returns:
        A fully assembled ``RetrieverQueryEngine`` ready to answer queries.
    """
    # Fall back to config values for anything not explicitly passed
    year     = year     if year     is not None else cfg.pipeline.year
    branch   = branch   if branch   is not None else cfg.pipeline.branch
    backend  = backend  if backend  is not None else cfg.pipeline.llm_backend
    streaming = streaming if streaming is not None else cfg.pipeline.streaming

    print(f"Initializing Query Engine for Year {year} — {branch.upper()}...")
    print(f"LLM backend: {backend} | streaming: {streaming}")

    # Step 1: Load the LLM and register it globally FIRST.
    #   This must happen before any retriever or synthesizer is constructed.
    #   LlamaIndex's QueryFusionRetriever.__init__ calls resolve_llm("default"),
    #   which will crash trying to reach OpenAI if Settings.llm is not set yet.
    print("Loading LLM...")
    llm_loader = _build_llm(backend)
    llm = llm_loader.get_llm()
    Settings.llm = llm

    # Step 2: Build / sync the ChromaDB-backed vector index.
    index = build_vector_index(year, branch)

    # Step 3: Construct the hybrid RRF retriever (vector + BM25).
    #   Pass DB_DIR as cache_dir so BM25 nodes are cached to disk after the
    #   first reconstruction and loaded instantly on all subsequent runs.
    retrieval = cfg.retrieval
    from retrieval.hybrid_retriever import HybridRRFRetrieverBuilder
    hybrid_retriever = HybridRRFRetrieverBuilder(
        vector_top_k=retrieval.vector_top_k,
        bm25_top_k=retrieval.bm25_top_k,
        fusion_top_k=retrieval.fusion_top_k,
        num_queries=retrieval.num_queries,
        cache_dir=DB_DIR,
    ).build(index)

    # Step 4: Load the custom strict QA prompt.
    qa_prompt = get_custom_prompt()

    # Step 5: Assemble the response synthesizer.
    #   streaming=True makes LlamaIndex call llm.stream_complete() and return
    #   a StreamingResponse whose .response_gen is a token iterator.
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=qa_prompt,
        streaming=streaming,
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
    # All settings below come from config.yaml — no need to edit this block.
    engine = setup_query_engine()   # uses config.yaml defaults for everything

    test_query = cfg.pipeline.test_query
    print(f"\nUser Query: {test_query}\n")

    response = engine.query(test_query)

    print("\n" + "=" * 50)
    print("Answer:")
    print("=" * 50)

    if cfg.pipeline.streaming:
        # Streaming: print each token as it arrives
        for token in response.response_gen:
            print(token, end="", flush=True)
        print()
    else:
        print(response.response)

    print("\n" + "=" * 50)

    print("SOURCES USED:")
    for node in response.source_nodes:
        doc_name = os.path.basename(node.metadata.get("file_path", "Unknown"))
        page = node.metadata.get("page_label", "?")
        print(f"  - {doc_name}, p.{page}  (Score: {node.score:.4f})")