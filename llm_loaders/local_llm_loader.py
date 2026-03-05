"""
llm_loaders/local_llm_loader.py
================================
Provides a 4-bit quantized local LLM wrapper that is both a LlamaIndex
``CustomLLM`` (for seamless integration with the query engine) and
implements the project-wide ``BaseLLMLoader`` interface.
"""

import torch
from typing import Any

from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

from interfaces import BaseLLMLoader


class QuantizedLocalLLM(CustomLLM, BaseLLMLoader):
    """
    A 4-bit quantized local language model wrapper.

    Inherits from both LlamaIndex's ``CustomLLM`` (so it slots directly into
    any LlamaIndex query engine or response synthesizer) and the project-wide
    ``BaseLLMLoader`` interface (so it can be swapped out with other loaders).

    The model is loaded with ``BitsAndBytesConfig`` NF4 quantisation to keep
    GPU VRAM usage low, and is pinned to ``device_map="auto"`` so the GPU is
    utilised when available.

    Args:
        model_name:      HuggingFace Hub ID or local directory path.
        max_new_tokens:  Maximum number of tokens to generate per call.
        context_window:  Token budget for the combined prompt + generation.
        temperature:     Sampling temperature (lower = more deterministic).
        top_p:           Nucleus sampling probability threshold.
    """

    # Pydantic model fields (validated and serialisable)
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.9
    context_window: int = 4096

    # Private attributes — excluded from Pydantic validation (hold PyTorch objects)
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)

        print(f"Loading Quantized Local LLM: {self.model_name}")

        # 4-bit NF4 quantisation config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Tokenizer — ensure a pad token exists (required for batched generation)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Model — load with quantisation; device_map="auto" handles GPU placement
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )
        self._model.eval()

    # ------------------------------------------------------------------
    # BaseLLMLoader interface
    # ------------------------------------------------------------------

    def get_llm(self) -> "QuantizedLocalLLM":
        """
        Return ``self`` as the LLM instance.

        Because ``QuantizedLocalLLM`` *is* the LLM, the loader and the model
        are the same object.

        Returns:
            This ``QuantizedLocalLLM`` instance.
        """
        return self

    # ------------------------------------------------------------------
    # CustomLLM interface (required by LlamaIndex)
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> LLMMetadata:
        """
        Report model capabilities to LlamaIndex.

        Returns:
            An ``LLMMetadata`` instance describing context window size, maximum
            output tokens, and model name.
        """
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Generate a completion for the given prompt.

        The prompt is truncated to ``context_window - max_new_tokens`` tokens
        to ensure the full generation budget is always available.  Only the
        newly generated tokens are decoded (the prompt tokens are sliced off).

        Args:
            prompt: The input text prompt.
            **kwargs: Additional keyword arguments (unused; accepted for
                      compatibility with the LlamaIndex callback decorator).

        Returns:
            A ``CompletionResponse`` containing the generated text.
        """
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window - self.max_new_tokens,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=1.3,       # Penalise repeated tokens to stop looping
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Slice off the prompt tokens so only the generated part is decoded
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        decoded = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return CompletionResponse(text=decoded)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """
        Streaming completion — not yet implemented.

        Raises:
            NotImplementedError: Always. Token-by-token streaming is not
                                 supported by this wrapper.
        """
        raise NotImplementedError("Streaming is not yet implemented for this custom wrapper.")