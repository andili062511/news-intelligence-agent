from typing import Any

import torch

from agent.generator import generate_grounded_answer
from llm.base import BaseLLM
from llm.prompts import SYSTEM_PROMPT, build_citation_repair_messages, build_grounded_messages
from llm.qwen import QwenLLM


class FakeLLM(BaseLLM):
    def __init__(self, answer: str = '{"claims":[{"text":"Supported fact.","citations":["E1"]}]}') -> None:
        self.answer = answer
        self.calls: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]]) -> str:
        self.calls.append(messages)
        return self.answer


def pack(ready: bool) -> dict[str, Any]:
    return {
        "ready": ready,
        "evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": "chunk-1",
                "article_id": "article-1",
                "title": "A title",
                "source": "News",
                "url": "https://example.com/1",
                "published_at": "2026-09-14",
                "text": "Validated text",
            }
        ],
    }


def test_ready_grounding_calls_llm_with_evidence() -> None:
    llm = FakeLLM()

    result = generate_grounded_answer("What happened?", pack(True), llm)

    assert result["generated"] is True
    assert '"claims"' in result["answer"]
    assert result["used_evidence_ids"] == ["E1"]
    assert len(llm.calls) == 1
    assert "[E1]" in llm.calls[0][1]["content"]


def test_unready_grounding_never_calls_llm() -> None:
    llm = FakeLLM()

    result = generate_grounded_answer("What happened?", pack(False), llm)

    assert result == {
        "generated": False,
        "answer": None,
        "reason": "insufficient_grounding",
    }
    assert llm.calls == []


def test_invalid_citation_metadata_never_calls_llm() -> None:
    llm = FakeLLM()
    grounding_pack = pack(True)
    grounding_pack["evidence"][0]["url"] = ""

    result = generate_grounded_answer("What happened?", grounding_pack, llm)

    assert result["reason"] == "insufficient_grounding"
    assert llm.calls == []


def test_prompt_contains_only_projected_evidence_fields() -> None:
    grounding_pack = pack(True)
    grounding_pack["evidence"][0]["private_raw_field"] = "must not leak"

    messages = build_grounded_messages("Question", grounding_pack)

    assert "Validated text" in messages[1]["content"]
    assert "must not leak" not in messages[1]["content"]


def test_prompt_requires_structured_claim_json() -> None:
    assert "valid JSON only" in SYSTEM_PROMPT
    assert '"claims"' in SYSTEM_PROMPT
    assert '"citations": ["E1"]' in SYSTEM_PROMPT
    assert "Every claim must contain at least one citation" in SYSTEM_PROMPT
    assert "Do not include markdown" in SYSTEM_PROMPT
    assert "Do not include explanatory text outside the JSON" in SYSTEM_PROMPT


def test_repair_prompt_contains_required_context() -> None:
    messages = build_citation_repair_messages(
        "What happened?",
        pack(True),
        "An uncited answer.",
        "factual_answer_without_citation",
    )
    content = messages[1]["content"]

    assert "Original question:\nWhat happened?" in content
    assert "Validated text" in content
    assert "First raw model output:\nAn uncited answer." in content
    assert "factual_answer_without_citation" in content
    assert "Generate the complete JSON object again" in content


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any):
        self.messages = messages
        assert kwargs["add_generation_prompt"] is True
        return torch.tensor([[10, 11]])

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        assert token_ids.tolist() == [20, 21]
        assert kwargs["skip_special_tokens"] is True
        return " Model answer [E1]. "


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.device = None
        self.generation_kwargs: dict[str, Any] = {}

    def to(self, device: Any):
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs: Any):
        self.generation_kwargs = kwargs
        return torch.tensor([[10, 11, 20, 21]])


def test_qwen_uses_injected_objects_and_deterministic_generation() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    llm = QwenLLM(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_new_tokens=42,
    )

    answer = llm.generate([{"role": "user", "content": "Question"}])

    assert answer == "Model answer [E1]."
    assert model.eval_called is True
    assert str(model.device) == "cpu"
    assert model.generation_kwargs["do_sample"] is False
    assert model.generation_kwargs["max_new_tokens"] == 42
