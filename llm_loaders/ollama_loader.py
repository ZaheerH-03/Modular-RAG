"""
llm_loaders/ollama_loader.py
============================
Provides an Ollama-backed LLM loader that satisfies the BaseLLMLoader
interface. Ollama runs models locally via its own server (no HuggingFace /
bitsandbytes required), making it a lighter alternative to QuantizedLocalLLM.

Usage:
    loader = OllamaLoader("llama3.2")
    llm = loader.get_llm()
    Settings.llm = llm
"""

from typing import Any

from llama_index.llms.ollama import Ollama

from interfaces import BaseLLMLoader


class OllamaLoader(BaseLLMLoader):
    """
    Loads a model served by a local Ollama instance and exposes it as a
    LlamaIndex-compatible LLM.

    Args:
        model_name:      Name of the Ollama model (e.g. ``"llama3.2"``).
        request_timeout: Seconds to wait for each Ollama response (default 120).
        context_window:  Token budget for prompt + generation (default 4096).
        temperature:     Sampling temperature (default 0.6).
    """

    def __init__(
        self,
        model_name: str,
        request_timeout: float = 120.0,
        context_window: int = 4096,
        temperature: float = 0.6,
        **kwargs: Any,
    ) -> None:
        # Do NOT pass kwargs to super().__init__() — BaseLLMLoader is a plain
        # ABC whose __init__ is object.__init__ and accepts no arguments.
        self._model_name = model_name
        self._llm = Ollama(
            model=self._model_name,
            request_timeout=request_timeout,
            context_window=context_window,
            temperature=temperature,
            additional_kwargs={
                "num_gpu": 1,
            },
        )

    def get_llm(self) -> Ollama:
        """
        Return the configured Ollama LLM instance.

        Returns:
            An ``Ollama`` LLM instance ready to be assigned to
            ``Settings.llm`` or passed to a response synthesizer.
        """
        return self._llm
