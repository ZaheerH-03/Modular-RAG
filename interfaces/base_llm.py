"""
interfaces/base_llm.py
======================
Abstract contract for LLM loader / wrapper providers.
"""

from abc import ABC, abstractmethod
from llama_index.core.llms import LLM


class BaseLLMLoader(ABC):
    """
    Abstract base class for LLM loaders.

    Concrete implementations are responsible for loading a specific language
    model (quantized local, cloud API, etc.) and returning it as a
    LlamaIndex-compatible ``LLM`` instance.
    """

    @abstractmethod
    def get_llm(self) -> LLM:
        """
        Load and return the language model.

        Returns:
            A LlamaIndex ``LLM`` instance ready to be assigned to
            ``Settings.llm`` or passed to a response synthesizer.
        """
        ...
