"""Offline tests for adapter inference input handling."""

from __future__ import annotations

from typing import Any

import torch

from finetune.inference_adapter import PlannerInference


VALID_RESPONSE = (
    '{"intent":"news_search","search_queries":["test query"],'
    '"time_range":null,"tools":["news_search"]}'
)


class FakeBatchEncoding(dict[str, torch.Tensor]):
    """Minimal mapping with the shape of a Transformers BatchEncoding."""


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def __init__(self) -> None:
        self.decoded_tokens: torch.Tensor | None = None

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> FakeBatchEncoding:
        assert kwargs == {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        return FakeBatchEncoding(
            input_ids=torch.tensor([[10, 11, 12]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

    def decode(self, token_ids: torch.Tensor, **kwargs: Any) -> str:
        self.decoded_tokens = token_ids
        assert kwargs == {"skip_special_tokens": True}
        return VALID_RESPONSE


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.positional_inputs: tuple[Any, ...] | None = None
        self.generation_kwargs: dict[str, Any] | None = None

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.positional_inputs = args
        self.generation_kwargs = kwargs
        return torch.tensor([[10, 11, 12, 20, 21]])


def test_generate_unpacks_inputs_and_decodes_only_new_tokens() -> None:
    planner = PlannerInference.__new__(PlannerInference)
    planner.model = FakeModel()
    planner.tokenizer = FakeTokenizer()

    result = planner.generate("What happened?", max_new_tokens=17)

    assert result.schema_valid is True
    assert planner.model.positional_inputs == ()
    assert planner.model.generation_kwargs is not None
    assert torch.equal(
        planner.model.generation_kwargs["input_ids"], torch.tensor([[10, 11, 12]])
    )
    assert torch.equal(
        planner.model.generation_kwargs["attention_mask"], torch.tensor([[1, 1, 1]])
    )
    assert planner.model.generation_kwargs["do_sample"] is False
    assert planner.model.generation_kwargs["pad_token_id"] == 2
    assert planner.model.generation_kwargs["max_new_tokens"] == 17
    assert planner.tokenizer.decoded_tokens is not None
    assert planner.tokenizer.decoded_tokens.tolist() == [20, 21]
