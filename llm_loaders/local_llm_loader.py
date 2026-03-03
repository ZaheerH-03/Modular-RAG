import torch
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pydantic import PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


class QuantizedLocalLLM(CustomLLM):
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # Default, can be overridden
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.9
    context_window: int = 4096

    # Private attributes for PyTorch objects to bypass Pydantic validation
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)

        print(f"Loading Quantized Local LLM: {self.model_name}")

        # 1. Your 4-bit Quantization Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # 2. Tokenizer Setup
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 3. Model Loading
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )
        self._model.eval()

    @property
    def metadata(self) -> LLMMetadata:
        """Tells LlamaIndex the capabilities and limits of this model."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """The core generation method called by LlamaIndex query engines."""

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
                repetition_penalty=1.3,  # Penalise repeated tokens to stop looping
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # 4. Your Critical Fix: Slicing off the prompt tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        decoded = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return CompletionResponse(text=decoded)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Streaming is not yet implemented for this custom wrapper.")