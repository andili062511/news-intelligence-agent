"""Transformers backend for Qwen Instruct models."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .base import BaseLLM


class QwenLLM(BaseLLM):
    """One-time-loaded deterministic Qwen text generator."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        model: Any = None,
        tokenizer: Any = None,
        device: str | torch.device | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be injected together")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_new_tokens = max_new_tokens

        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model = model
        self.tokenizer = tokenizer
        self.model.to(self.device)
        self.model.eval()

    def generate(self, messages: Sequence[dict[str, Any]]) -> str:
        """Render Qwen's chat template and decode only newly generated tokens."""

        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, Mapping):
            model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
            input_length = model_inputs["input_ids"].shape[-1]
        else:
            input_ids = encoded.to(self.device)
            model_inputs = {"input_ids": input_ids}
            input_length = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        generated_ids = output_ids[0][input_length:]
        return str(
            self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        ).strip()
