"""
retrieval/hybrid_retriever.py
==============================
Builds a hybrid Reciprocal Rank Fusion (RRF) retriever that combines dense
vector search with sparse BM25 keyword matching.
"""

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

    Args:
        vector_top_k:    Number of top results for the vector retriever.
        bm25_top_k:      Number of top results for the BM25 retriever.
        fusion_top_k:    Number of final results after RRF re-ranking.
        num_queries:     Number of query variants to generate (1 = no expansion).
    """

    def __init__(
        self,
        vector_top_k: int = 5,
        bm25_top_k: int = 5,
        fusion_top_k: int = 2,
        num_queries: int = 1,
    ) -> None:
        self._vector_top_k = vector_top_k
        self._bm25_top_k = bm25_top_k
        self._fusion_top_k = fusion_top_k
        self._num_queries = num_queries

    def _get_bm25_nodes(self, index: VectorStoreIndex) -> list[TextNode]:
        """
        Retrieve text nodes for BM25 seeding.

        Tries the in-memory docstore first; if empty, reconstructs nodes
        directly from the underlying ChromaDB collection.

        Args:
            index: The source ``VectorStoreIndex``.

        Returns:
            A list of ``TextNode`` objects.

        Raises:
            ValueError: If no nodes can be found in either the docstore or
                        ChromaDB.
        """
        nodes: list[TextNode] = list(index.docstore.docs.values())

        if not nodes:
            print("Docstore is empty — reconstructing text nodes from ChromaDB for BM25...")
            collection = index.storage_context.vector_store._collection
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


def get_hybrid_rrf_retriever(index: VectorStoreIndex) -> BaseRetriever:
    """
    Convenience wrapper — build a hybrid RRF retriever via
    :class:`HybridRRFRetrieverBuilder` with default parameters.

    Args:
        index: A fully populated ``VectorStoreIndex``.

    Returns:
        A ``QueryFusionRetriever`` combining vector and BM25 search.
    """
    return HybridRRFRetrieverBuilder().build(index)