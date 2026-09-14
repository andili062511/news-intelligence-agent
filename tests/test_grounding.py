from copy import deepcopy
import sys
from typing import Any

from agent import cli
from agent.graph import create_agent_graph
from agent.grounding import assess_grounding_readiness, build_grounding_pack
from llm.base import BaseLLM


def evidence(number: int, **overrides: Any) -> dict[str, Any]:
    item = {
        "chunk_id": f"chunk-{number}",
        "article_id": f"article-{number}",
        "title": f"Title {number}",
        "source": "Example News",
        "url": f"https://example.com/{number}",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": f"Evidence {number}",
        "rerank_score": 1.0 / number,
    }
    item.update(overrides)
    return item


def test_grounding_pack_is_bounded_and_does_not_modify_input() -> None:
    items = [evidence(1), evidence(2), evidence(3)]
    original = deepcopy(items)

    pack = build_grounding_pack("What happened?", items, max_items=2)

    assert pack["evidence_count"] == 2
    assert [item["evidence_id"] for item in pack["evidence"]] == ["E1", "E2"]
    assert items == original


def test_grounding_pack_keeps_only_highest_ranked_chunk_per_article() -> None:
    items = [
        evidence(1, article_id="shared", rerank_score=0.9),
        evidence(2, article_id="shared", rerank_score=0.8),
        evidence(3, article_id="shared", rerank_score=0.7),
        evidence(4, article_id="other", rerank_score=0.1),
    ]
    original = deepcopy(items)

    pack = build_grounding_pack("Question", items)

    assert pack["evidence_count"] == 1
    assert pack["evidence"][0]["chunk_id"] == "chunk-1"
    assert pack["evidence"][0]["evidence_id"] == "E1"
    assert items == original


def test_grounding_pack_can_disable_article_chunk_limit() -> None:
    items = [
        evidence(1, article_id="shared"),
        evidence(2, article_id="shared"),
    ]

    pack = build_grounding_pack(
        "Question", items, max_chunks_per_article=None
    )

    assert [item["chunk_id"] for item in pack["evidence"]] == [
        "chunk-1",
        "chunk-2",
    ]


def test_empty_and_zero_sized_grounding_packs() -> None:
    assert build_grounding_pack("Question", [])["evidence"] == []
    assert build_grounding_pack("Question", [evidence(1)], max_items=0)[
        "evidence_count"
    ] == 0


def test_invalid_or_empty_text_evidence_is_excluded() -> None:
    pack = build_grounding_pack(
        "Question",
        [evidence(1, text=" "), evidence(2, url=""), evidence(3)],
    )

    assert pack["evidence_count"] == 1
    assert pack["evidence"][0]["evidence_id"] == "E3"
    assert len(pack["citation_errors"]) == 2


def test_metadata_outside_ranked_window_cannot_fail_pack_validation() -> None:
    pack = build_grounding_pack(
        "Question",
        [evidence(1), evidence(2), evidence(3), evidence(4, url="")],
    )

    assert pack["evidence_count"] == 3
    assert pack["citation_errors"] == []


def test_fewer_than_minimum_evidence_is_not_ready() -> None:
    pack = build_grounding_pack("Question", [evidence(1)])

    result = assess_grounding_readiness(pack, min_evidence=2)

    assert result["ready"] is False
    assert result["evidence_count"] == 1


def test_one_article_is_not_ready_for_multi_source_answer() -> None:
    pack = build_grounding_pack(
        "Question",
        [evidence(1), evidence(2, article_id="article-1")],
    )

    result = assess_grounding_readiness(pack)

    assert result["ready"] is False
    assert result["unique_articles"] == 1


def test_multiple_articles_are_ready_without_score_threshold() -> None:
    pack = build_grounding_pack(
        "Question",
        [evidence(1, rerank_score=-100), evidence(2, rerank_score=-200)],
    )

    result = assess_grounding_readiness(pack)

    assert result["ready"] is True
    assert result["unique_articles"] == 2


class FakeSearch:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls = 0

    def __call__(
        self, query: str, top_k: int, candidate_k: int
    ) -> list[dict[str, Any]]:
        self.calls += 1
        return self.results


class FakeLLM(BaseLLM):
    def __init__(self, answer: str = '{"claims":[{"text":"The reports agree.","citations":["E1","E2"]}]}') -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, messages: list[dict[str, Any]]) -> str:
        self.calls += 1
        return self.answer


def test_graph_sufficient_evidence_runs_citation_validation() -> None:
    search = FakeSearch([evidence(1), evidence(2)])

    result = create_agent_graph(search, llm=FakeLLM()).invoke(
        {"question": "What happened?", "max_retries": 0}
    )

    assert result["status"] == "completed"
    assert result["citation_status"] == "VALID"
    assert result["grounding_pack"]["evidence_count"] == 2
    assert result["final_answer"] == "The reports agree [E1][E2]."


def test_citation_failure_never_reaches_grounded() -> None:
    search = FakeSearch([evidence(1), evidence(2, url="")])

    result = create_agent_graph(search).invoke(
        {"question": "What happened?", "max_retries": 0}
    )

    assert result["status"] == "citation_failed"
    assert result["citation_status"] == "FAILED"
    assert result["citation_errors"]
    assert result["final_answer"] is None


def test_insufficient_evidence_does_not_prepare_citations_or_loop() -> None:
    search = FakeSearch([])

    result = create_agent_graph(search).invoke(
        {"question": "Unknown story", "max_retries": 2}
    )

    assert result["status"] == "insufficient_evidence"
    assert result["citation_status"] is None
    assert result["grounding_pack"] is None
    assert result["final_answer"] is None
    assert search.calls == 3


def test_cli_displays_citation_ids_and_grounding_status(monkeypatch, capsys) -> None:
    pack = build_grounding_pack("What happened?", [evidence(1), evidence(2)])
    pack["ready"] = True
    monkeypatch.setattr(cli, "QwenLLM", lambda **kwargs: FakeLLM())
    monkeypatch.setattr(
        cli,
        "run_agent",
        lambda *args, **kwargs: {
            "question": "What happened?",
            "intent": "fact_lookup",
            "search_queries": ["What happened?"],
            "retry_count": 0,
            "status": "completed",
            "citation_status": "VALID",
            "grounding_pack": pack,
            "final_answer": "The reports agree [E1] [E2].",
            "generation_attempts": 1,
        },
    )
    monkeypatch.setattr(sys, "argv", ["agent.cli", "--question", "What happened?"])

    cli.main()
    output = capsys.readouterr().out

    assert "Evidence Status:\nSUFFICIENT" in output
    assert "Citation Status:\nVALID" in output
    assert "Grounding Ready:\nYES" in output
    assert "[E1]" in output
    assert "Final Answer:" in output
    assert "[E1] Title 1" in output


def test_cli_displays_raw_answer_and_no_final_answer_on_failure(
    monkeypatch, capsys
) -> None:
    pack = build_grounding_pack("What happened?", [evidence(1), evidence(2)])
    pack["ready"] = True
    monkeypatch.setattr(cli, "QwenLLM", lambda **kwargs: FakeLLM())
    monkeypatch.setattr(
        cli,
        "run_agent",
        lambda *args, **kwargs: {
            "question": "What happened?",
            "intent": "fact_lookup",
            "search_queries": ["What happened?"],
            "retry_count": 0,
            "status": "citation_failed",
            "citation_status": "VALID",
            "grounding_pack": pack,
            "raw_generated_answer": "An uncited factual answer.",
            "raw_model_output": "{bad json",
            "generation_attempts": 2,
            "answer_validation": {"valid": False},
            "reason": "factual_answer_without_citation",
            "final_answer": None,
        },
    )
    monkeypatch.setattr(sys, "argv", ["agent.cli", "--question", "What happened?"])

    cli.main()
    output = capsys.readouterr().out

    assert "Generation Attempts: 2" in output
    assert "Raw Model Output:\n{bad json" in output
    assert "Validation Error:\nfactual_answer_without_citation" in output
    assert "Final Answer:\nNONE" in output
