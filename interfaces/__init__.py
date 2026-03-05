"""
interfaces package
==================
Exports all Abstract Base Classes (ABCs) that define the contracts for
every component in the ModularRAG pipeline.
"""

from .base_loader import BaseDocumentLoader
from .base_embedder import BaseEmbedder
from .base_llm import BaseLLMLoader
from .base_retriever import BaseRetrieverBuilder
from .base_index import BaseIndexBuilder
from .base_prompt import BasePromptProvider

__all__ = [
    "BaseDocumentLoader",
    "BaseEmbedder",
    "BaseLLMLoader",
    "BaseRetrieverBuilder",
    "BaseIndexBuilder",
    "BasePromptProvider",
]
