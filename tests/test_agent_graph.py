from typing import Any

from agent.graph import create_agent_graph
from llm.base import BaseLLM


def item(article_id: str, score: float) -> dict[str, Any]:
    return {
        "chunk_id": f"chunk-{article_id}",
        "article_id": article_id,
        "title": f"Title {article_id}",
        "source": "example.com",
        "url": f"https://example.com/{article_id}",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": f"Evidence from {article_id}",
        "rrf_score": 0.02,
        "rerank_score": score,
    }


class FakeSearch:
    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, query: str, top_k: int, candidate_k: int) -> list[dict[str, Any]]:
        self.calls.append((query, top_k, candidate_k))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


class FakeLLM(BaseLLM):
    def __init__(
        self,
        answer: str = '{"claims":[{"text":"Evidence is reported.","citations":["E1","E2"]}]}',
        answers: list[str] | None = None,
    ) -> None:
        self.answers = answers or [answer]
        self.calls = 0
        self.messages: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]]) -> str:
        self.messages.append(messages)
        index = min(self.calls, len(self.answers) - 1)
        self.calls += 1
        return self.answers[index]


def test_successful_search_routes_to_finalize() -> None:
    search = FakeSearch([[item("article-1", 0.8), item("article-2", 0.7)]])
    graph = create_agent_graph(search, llm=FakeLLM(), top_k=4, candidate_k=12)

    result = graph.invoke({"question": "AI safety risks", "max_retries": 2})

    assert result["status"] == "completed"
    assert result["final_answer"] == "Evidence is reported [E1][E2]."
    assert result["citation_status"] == "VALID"
    assert result["retry_count"] == 0
    assert len(result["evidence"]) == 2
    assert search.calls == [("AI safety risks", 4, 12)]


def test_insufficient_search_rewrites_then_retries() -> None:
    search = FakeSearch(
        [
            [item("article-1", 0.8)],
            [item("article-2", 0.7)],
        ]
    )
    graph = create_agent_graph(search, llm=FakeLLM())

    result = graph.invoke({"question": "machines becoming dangerously capable"})

    assert result["status"] == "completed"
    assert result["retry_count"] == 1
    assert len(search.calls) == 2
    assert search.calls[0][0] != search.calls[1][0]
    assert len({entry["article_id"] for entry in result["evidence"]}) == 2


def test_exceeding_max_retries_returns_insufficient_evidence() -> None:
    search = FakeSearch([[]])
    graph = create_agent_graph(search)

    result = graph.invoke({"question": "obscure news", "max_retries": 2})

    assert result["status"] == "insufficient_evidence"
    assert result["retry_count"] == 2
    assert len(search.calls) == 3
    assert result["evidence"] == []
    assert result["reason"]


def test_zero_retries_terminates_after_one_search_without_looping() -> None:
    search = FakeSearch([[]])
    graph = create_agent_graph(search)

    result = graph.invoke({"question": "unavailable story", "max_retries": 0})

    assert result["status"] == "insufficient_evidence"
    assert result["retry_count"] == 0
    assert len(search.calls) == 1


def test_invalid_generated_citation_is_repaired_once() -> None:
    search = FakeSearch([[item("article-1", 0.8), item("article-2", 0.7)]])
    llm = FakeLLM(answers=[
        '{"claims":[{"text":"Unsupported claim.","citations":["E99"]}]}',
        '{"claims":[{"text":"Supported claim.","citations":["E1"]}]}',
    ])

    result = create_agent_graph(search, llm=llm).invoke(
        {"question": "AI safety risks", "max_retries": 0}
    )

    assert result["status"] == "completed"
    assert result["answer_validation"]["valid"] is True
    assert "E99" in result["generation_validation_errors"][0]["errors"][0]
    assert result["final_answer"] == "Supported claim [E1]."
    assert result["generation_attempts"] == 2
    assert llm.calls == 2
    assert len(search.calls) == 1
    assert '"E99"' in llm.messages[1][1]["content"]
    assert "[E1]" in llm.messages[1][1]["content"]


def test_uncited_answer_is_repaired_once_and_completed() -> None:
    search = FakeSearch([[item("article-1", 0.8), item("article-2", 0.7)]])
    llm = FakeLLM(answers=[
        '{"claims":[{"text":"An uncited factual answer.","citations":[]}]}',
        '{"claims":[{"text":"Supported claim.","citations":["E1"]}]}',
    ])

    result = create_agent_graph(search, llm=llm).invoke(
        {"question": "AI safety risks", "max_retries": 0}
    )

    assert result["status"] == "completed"
    assert result["final_answer"] == "Supported claim [E1]."
    assert result["generation_attempts"] == 2
    assert result["generation_validation_errors"][0]["errors"][0].startswith(
        "missing_citation"
    )
    assert llm.calls == 2


def test_two_uncited_answers_fail_without_more_retries() -> None:
    search = FakeSearch([[item("article-1", 0.8), item("article-2", 0.7)]])
    llm = FakeLLM(answers=[
        '{"claims":[{"text":"First uncited answer.","citations":[]}]}',
        '{"claims":[{"text":"Second uncited answer.","citations":[]}]}',
    ])

    result = create_agent_graph(search, llm=llm).invoke(
        {"question": "AI safety risks", "max_retries": 0}
    )

    assert result["status"] == "citation_failed"
    assert result["final_answer"] is None
    assert "Second uncited answer" in result["raw_model_output"]
    assert result["generation_attempts"] == 2
    assert len(result["generation_validation_errors"]) == 2
    assert llm.calls == 2
    assert len(search.calls) == 1


def test_insufficient_graph_never_calls_llm_and_terminates() -> None:
    search = FakeSearch([[]])
    llm = FakeLLM()

    result = create_agent_graph(search, llm=llm).invoke(
        {"question": "Unavailable story", "max_retries": 1}
    )

    assert result["status"] == "insufficient_evidence"
    assert result["final_answer"] is None
    assert llm.calls == 0
    assert len(search.calls) == 2
