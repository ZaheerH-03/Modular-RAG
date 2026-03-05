"""
interfaces/base_embedder.py
===========================
Abstract contract for embedding-model providers.
"""

from abc import ABC, abstractmethod
from llama_index.core.embeddings import BaseEmbedding


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding-model providers.

    Concrete implementations are responsible for loading a specific embedding
    model (e.g. HuggingFace, OpenAI, local ONNX) and returning it as a
    LlamaIndex-compatible ``BaseEmbedding`` instance.
    """

    @abstractmethod
    def get_model(self) -> BaseEmbedding:
        """
        Load and return the embedding model.

        Returns:
            A LlamaIndex ``BaseEmbedding`` instance ready for use in indexing
            and retrieval.
        """
        ...
