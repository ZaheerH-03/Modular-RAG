"""
interfaces/base_prompt.py
=========================
Abstract contract for prompt-template providers.
"""

from abc import ABC, abstractmethod
from llama_index.core import PromptTemplate


class BasePromptProvider(ABC):
    """
    Abstract base class for prompt-template providers.

    Concrete implementations are responsible for constructing and returning
    a LlamaIndex ``PromptTemplate`` that will be injected into the
    response synthesizer.
    """

    @abstractmethod
    def get_prompt(self) -> PromptTemplate:
        """
        Build and return the prompt template.

        Returns:
            A LlamaIndex ``PromptTemplate`` ready to be passed as
            ``text_qa_template`` to ``get_response_synthesizer``.
        """
        ...
