"""Shared helpers for deterministic causal-language-model generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _move_tensor_values(
    inputs: Mapping[str, Any], device: Any, *, torch_module: Any
) -> dict[str, Any]:
    """Move model input tensors without treating encoding metadata as tensors."""

    return {
        key: value.to(device) if torch_module.is_tensor(value) else value
        for key, value in inputs.items()
    }


def generate_chat_completion(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_new_tokens: int,
) -> str:
    """Generate and decode only the completion following a chat prompt."""

    import torch

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError(
            "apply_chat_template must return a mapping containing input_ids and "
            "attention_mask"
        )

    inputs = _move_tensor_values(encoded, model.device, torch_module=torch)
    if "input_ids" not in inputs:
        raise ValueError("tokenized model inputs do not contain input_ids")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    ).strip()
