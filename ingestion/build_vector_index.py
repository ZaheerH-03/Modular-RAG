"""
ingestion/build_vector_index.py
================================
Provides embedding-model loading and vector-index building with incremental
upsert support backed by ChromaDB.
"""

import json
import os

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ingestion.data_loader import get_document_updates
from interfaces import BaseEmbedder, BaseIndexBuilder

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BASE_DIR: str = "E:/Documents/data_Science/ModularRAG"
DB_DIR: str = os.path.join(BASE_DIR, "chromadb")
EMBEDDING_MODEL_PATH: str = os.path.join(BASE_DIR, "models/bge-m3")


# ---------------------------------------------------------------------------
# Concrete embedder
# ---------------------------------------------------------------------------


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Loads a HuggingFace sentence-transformer model for use as a LlamaIndex
    embedding model.

    The model is intentionally pinned to CPU so that GPU VRAM is left
    entirely for the quantized LLM.  Running both on GPU simultaneously
    causes CUDA OOM during the LLM's RMSNorm forward pass.

    Args:
        model_path: Local directory path or HuggingFace Hub model ID.
    """

    def __init__(self, model_path: str = EMBEDDING_MODEL_PATH) -> None:
        self._model_path = model_path

    def get_model(self) -> HuggingFaceEmbedding:
        """
        Load and return the HuggingFace embedding model on CPU.

        Returns:
            A ``HuggingFaceEmbedding`` instance ready for node parsing and
            vector retrieval.
        """
        return HuggingFaceEmbedding(model_name=self._model_path, device="cpu")


# ---------------------------------------------------------------------------
# Concrete index builder
# ---------------------------------------------------------------------------


class VectorIndexBuilder(BaseIndexBuilder):
    """
    Builds and incrementally syncs a ``VectorStoreIndex`` backed by ChromaDB.

    On each call to :meth:`build`, the builder:

    1. Checks for file additions, modifications, and deletions via hashing.
    2. Loads only the changed documents and splits them into semantic nodes.
    3. Upserts new/modified nodes and deletes removed ones from the index.
    4. Persists the updated hash snapshot so subsequent runs are efficient.

    Args:
        embedder:   An :class:`~interfaces.BaseEmbedder` instance. Defaults to
                    :class:`HuggingFaceEmbedder`.
        db_dir:     Path to the ChromaDB persistence directory.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        db_dir: str = DB_DIR,
    ) -> None:
        self._embedder = embedder or HuggingFaceEmbedder()
        self._db_dir = db_dir

    def _get_storage_context(self, year: int, branch: str) -> StorageContext:
        """
        Retrieve or create a ChromaDB collection and wrap it in a
        ``StorageContext``.

        Args:
            year:   Target year.
            branch: Branch identifier.

        Returns:
            A ``StorageContext`` backed by the correct ChromaDB collection.
        """
        collection_name = f"{year}_yr_{branch}"
        db = chromadb.PersistentClient(path=self._db_dir)
        existing_collections = [c.name for c in db.list_collections()]

        if collection_name in existing_collections:
            collection = db.get_collection(collection_name)
            if collection.count() > 0:
                vector_store = ChromaVectorStore(chroma_collection=collection)
                return StorageContext.from_defaults(vector_store=vector_store)

        chromadb_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chromadb_collection)
        return StorageContext.from_defaults(vector_store=vector_store)

    def build(self, year: int, branch: str) -> VectorStoreIndex:
        """
        Build (or sync) the vector index for the given year and branch.

        Args:
            year:   Target year (cumulative — includes years 1 through ``year``).
            branch: Branch identifier (e.g. ``"ds"``, ``"cs"``).

        Returns:
            A ``VectorStoreIndex`` backed by ChromaDB and ready to query.
        """
        needs_update, changed_docs, deleted_files, current_hashes, state_log = \
            get_document_updates(year, branch, self._db_dir)

        embedding_model = self._embedder.get_model()
        storage_context = self._get_storage_context(year, branch)

        index = VectorStoreIndex.from_vector_store(
            vector_store=storage_context.vector_store,
            embed_model=embedding_model,
        )

        if needs_update:
            splitter = SemanticSplitterNodeParser(
                embed_model=embedding_model,
                buffer_size=3,
                breakpoint_percentile_threshold=95,
            )

            if changed_docs:
                nodes = splitter.get_nodes_from_documents(changed_docs)
                index.insert_nodes(nodes)

            for fp in deleted_files:
                index.delete_ref_doc(fp, delete_from_docstore=True)

            with open(state_log, "w") as f:
                json.dump(current_hashes, f)

            print(
                f"Updated index: {len(changed_docs)} doc(s) added/modified, "
                f"{len(deleted_files)} file(s) removed."
            )
        else:
            print("Index is up-to-date. No changes detected.")

        return index


# ---------------------------------------------------------------------------
# Module-level convenience functions (keeps existing call sites unbroken)
# ---------------------------------------------------------------------------


def load_embedding_model(model_name: str) -> HuggingFaceEmbedding:
    """
    Convenience wrapper — load a HuggingFace embedding model via
    :class:`HuggingFaceEmbedder`.

    Args:
        model_name: Local directory path or HuggingFace Hub model ID.

    Returns:
        A ``HuggingFaceEmbedding`` instance pinned to CPU.
    """
    return HuggingFaceEmbedder(model_path=model_name).get_model()


def build_vector_index(year: int, branch: str) -> VectorStoreIndex:
    """
    Convenience wrapper — build/sync the vector index via
    :class:`VectorIndexBuilder`.

    Args:
        year:   Target year (cumulative — includes years 1 through ``year``).
        branch: Branch identifier (e.g. ``"ds"``, ``"cs"``).

    Returns:
        A ready-to-query ``VectorStoreIndex``.
    """
    return VectorIndexBuilder().build(year, branch)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vector_index = build_vector_index(4, "ds")
    vectordb = chromadb.PersistentClient(path=DB_DIR)
    coll = vectordb.get_collection("4_yr_ds")
    print(f"Nodes in 4th_yr_ds: {coll.count()}")