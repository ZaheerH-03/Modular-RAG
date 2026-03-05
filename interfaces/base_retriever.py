"""
interfaces/base_retriever.py
============================
Abstract contract for retriever builders.
"""

from abc import ABC, abstractmethod
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever


class BaseRetrieverBuilder(ABC):
    """
    Abstract base class for retriever builders.

    Concrete implementations receive a populated ``VectorStoreIndex`` and
    are responsible for constructing and returning any retriever strategy
    (vector-only, BM25, hybrid RRF, etc.) as a LlamaIndex ``BaseRetriever``.
    """

    @abstractmethod
    def build(self, index: VectorStoreIndex) -> BaseRetriever:
        """
        Construct and return the retriever.

        Args:
            index: A fully populated ``VectorStoreIndex`` to retrieve from.

        Returns:
            A LlamaIndex ``BaseRetriever`` instance ready to be plugged into
            a ``RetrieverQueryEngine``.
        """
        ...
