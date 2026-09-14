from typing import Any

import pytest

from agent.graph import create_agent_graph
from llm.base import BaseLLM


class FakeLLM(BaseLLM):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.messages: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]]) -> str:
        self.messages.append(messages)
        output = self.outputs[self.calls]
        self.calls += 1
        return output


def search(*args: Any) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": f"c-{number}",
            "article_id": f"a-{number}",
            "title": f"Title {number}",
            "source": "News",
            "url": f"https://example.com/{number}",
            "published_at": "2026-09-14",
            "text": f"Evidence {number}",
            "rerank_score": 1 / number,
        }
        for number in (1, 2)
    ]


GOOD = '{"claims":[{"text":"Supported claim.","citations":["E1"]}]}'
NO_CITATION = '{"claims":[{"text":"Unsupported claim.","citations":[]}]}'
BAD_ID = '{"claims":[{"text":"Wrong source.","citations":["E99"]}]}'


@pytest.mark.parametrize("first", [NO_CITATION, BAD_ID, "{malformed"])
def test_invalid_first_output_repairs_to_completed(first: str) -> None:
    llm = FakeLLM([first, GOOD])
    result = create_agent_graph(search, llm=llm).invoke(
        {"question": "What happened?", "max_retries": 0}
    )
    assert result["status"] == "completed"
    assert result["generation_attempts"] == 2
    assert result["final_answer"] == "Supported claim [E1]."
    assert first in llm.messages[1][1]["content"]
    assert result["generation_errors"]


@pytest.mark.parametrize("outputs", [["{bad", "still bad"], [NO_CITATION, NO_CITATION]])
def test_two_invalid_outputs_fail_with_no_final_answer(outputs: list[str]) -> None:
    llm = FakeLLM(outputs)
    result = create_agent_graph(search, llm=llm).invoke(
        {"question": "What happened?", "max_retries": 0}
    )
    assert result["status"] == "citation_failed"
    assert result["generation_attempts"] == 2
    assert result["final_answer"] is None
    assert result["raw_model_output"] == outputs[-1]
    assert llm.calls == 2


def test_valid_first_output_completes_in_one_attempt() -> None:
    llm = FakeLLM([GOOD])
    result = create_agent_graph(search, llm=llm).invoke(
        {"question": "What happened?", "max_retries": 0}
    )
    assert result["status"] == "completed"
    assert result["generation_attempts"] == 1
    assert result["final_answer"] == "Supported claim [E1]."
    assert llm.calls == 1
