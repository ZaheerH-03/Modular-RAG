"""
retrieval/hybrid_retriever.py
==============================
Builds a hybrid Reciprocal Rank Fusion (RRF) retriever that combines dense
vector search with sparse BM25 keyword matching.

BM25 nodes are cached to disk after the first reconstruction so that
subsequent runs do not need to re-fetch every node from ChromaDB.
The cache is automatically invalidated by the index builder whenever
the underlying documents change.
"""

import os
import pickle

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from interfaces import BaseRetrieverBuilder


class HybridRRFRetrieverBuilder(BaseRetrieverBuilder):
    """
    Constructs a hybrid Reciprocal Rank Fusion (RRF) retriever by combining
    a dense ``VectorIndexRetriever`` and a sparse ``BM25Retriever`` via
    LlamaIndex's ``QueryFusionRetriever``.

    When the docstore is empty (common when loading an index directly from
    ChromaDB without a persisted docstore), the BM25 retriever is seeded by
    reconstructing ``TextNode`` objects directly from the ChromaDB collection.
    Reconstructed nodes are cached to disk so that subsequent runs load
    instantly instead of re-fetching from ChromaDB every time.

    Args:
        vector_top_k:    Number of top results for the vector retriever.
        bm25_top_k:      Number of top results for the BM25 retriever.
        fusion_top_k:    Number of final results after RRF re-ranking.
        num_queries:     Number of query variants to generate (1 = no expansion).
        cache_dir:       Directory to persist BM25 node cache files.
                         If ``None``, caching is disabled.
    """

    def __init__(
        self,
        vector_top_k: int = 5,
        bm25_top_k: int = 5,
        fusion_top_k: int = 2,
        num_queries: int = 1,
        cache_dir: str | None = None,
    ) -> None:
        self._vector_top_k = vector_top_k
        self._bm25_top_k = bm25_top_k
        self._fusion_top_k = fusion_top_k
        self._num_queries = num_queries
        self._cache_dir = cache_dir

    def _cache_path(self, collection_name: str) -> str | None:
        """Return the pickle cache file path for a given collection, or None if caching is disabled."""
        if self._cache_dir is None:
            return None
        return os.path.join(self._cache_dir, f"{collection_name}_bm25_cache.pkl")

    def _get_bm25_nodes(self, index: VectorStoreIndex) -> list[TextNode]:
        """
        Retrieve text nodes for BM25 seeding.

        Resolution order:
        1. In-memory docstore (populated after a fresh ingestion run).
        2. Disk cache (pickle) — avoids re-fetching ChromaDB between runs.
        3. Live ChromaDB reconstruction — only when no cache exists yet.

        Reconstructed nodes are written to the disk cache so that the next
        call skips straight to step 2.

        Args:
            index: The source ``VectorStoreIndex``.

        Returns:
            A list of ``TextNode`` objects.

        Raises:
            ValueError: If no nodes can be found in the docstore, cache, or ChromaDB.
        """
        # 1. In-memory docstore (cheapest — populated after a fresh ingest)
        nodes: list[TextNode] = list(index.docstore.docs.values())
        if nodes:
            return nodes

        collection = index.storage_context.vector_store._collection
        cache_path = self._cache_path(collection.name)

        # 2. Disk cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading BM25 nodes from cache ({collection.name})...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # 3. Live reconstruction from ChromaDB
        print(f"Reconstructing BM25 nodes from ChromaDB ({collection.name}) — this runs once per index change...")
        chroma_data = collection.get(include=["documents", "metadatas"])

        nodes = [
            TextNode(id_=doc_id, text=text, metadata=metadata or {})
            for doc_id, text, metadata in zip(
                chroma_data["ids"],
                chroma_data["documents"],
                chroma_data["metadatas"],
            )
        ]

        if not nodes:
            raise ValueError(
                "ChromaDB is completely empty. "
                "Ensure your ingestion pipeline has processed documents."
            )

        # Save to disk cache for next run
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(nodes, f)
            print(f"BM25 nodes cached to disk — future runs will load instantly.")

        return nodes

    def build(self, index: VectorStoreIndex) -> BaseRetriever:
        """
        Construct and return the hybrid RRF retriever.

        Args:
            index: A fully populated ``VectorStoreIndex`` to retrieve from.

        Returns:
            A ``QueryFusionRetriever`` combining vector and BM25 search via
            Reciprocal Rank Fusion.
        """
        # 1. Dense vector retriever
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self._vector_top_k,
        )

        # 2. Sparse BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self._get_bm25_nodes(index),
            similarity_top_k=self._bm25_top_k,
        )

        # 3. RRF fusion retriever
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=self._fusion_top_k,
            num_queries=self._num_queries,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
        )

        return retriever


# ---------------------------------------------------------------------------
# Module-level convenience function (keeps existing call sites unbroken)
# ---------------------------------------------------------------------------


def get_hybrid_rrf_retriever(
    index: VectorStoreIndex,
    cache_dir: str | None = None,
) -> BaseRetriever:
    """
    Convenience wrapper — build a hybrid RRF retriever via
    :class:`HybridRRFRetrieverBuilder` with default parameters.

    Args:
        index:     A fully populated ``VectorStoreIndex``.
        cache_dir: Directory to persist BM25 node cache files.
                   Pass the same ``DB_DIR`` used by the index builder.

    Returns:
        A ``QueryFusionRetriever`` combining vector and BM25 search.
    """
    return HybridRRFRetrieverBuilder(cache_dir=cache_dir).build(index)