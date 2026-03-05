"""
interfaces/base_index.py
========================
Abstract contract for vector-index builders.
"""

from abc import ABC, abstractmethod
from llama_index.core import VectorStoreIndex


class BaseIndexBuilder(ABC):
    """
    Abstract base class for vector-index builders.

    Concrete implementations are responsible for the full ingestion lifecycle:
    loading documents, detecting changes, splitting nodes, and upserting into
    a persistent vector store.  They must return a ready-to-query
    ``VectorStoreIndex``.
    """

    @abstractmethod
    def build(self, year: int, branch: str) -> VectorStoreIndex:
        """
        Build (or sync) the vector index for the given year and branch.

        Args:
            year:   Target year (cumulative — includes years 1 through ``year``).
            branch: Branch identifier (e.g. ``"ds"``, ``"cs"``).

        Returns:
            A ``VectorStoreIndex`` backed by a persistent vector store and
            ready to be queried.
        """
        ...
