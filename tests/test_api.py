from typing import Any

import pytest

import app.main as api


def _state(status: str, *, answer: str | None = None) -> dict[str, Any]:
    evidence = [
        {
            "evidence_id": "E1",
            "title": "AI safety report",
            "source": "Example News",
            "url": "https://example.com/ai-safety",
            "published_at": "2026-09-15",
        },
        {
            "evidence_id": "E2",
            "title": "AI policy report",
            "source": "Example News",
            "url": "https://example.com/ai-policy",
            "published_at": "2026-09-14",
        },
    ]
    return {
        "status": status,
        "final_answer": answer,
        "intent": "news_search",
        "search_queries": ["AI safety risks"],
        "evidence": evidence,
        "grounding_pack": {"evidence": evidence},
        "answer_citations": ["E1"],
    }


@pytest.mark.parametrize("status", ["insufficient_evidence", "citation_failed"])
def test_query_does_not_expose_answer_or_citations_on_failure(
    status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(status, answer="unvalidated answer")
    monkeypatch.setattr(api, "get_llm", lambda: object())
    monkeypatch.setattr(api, "run_agent", lambda question, llm: state)

    response = api.query_news(api.QueryRequest(question="  AI safety?  "))

    assert response.status == status
    assert response.answer is None
    assert response.citations == []
    assert response.question == "AI safety?"
    assert response.evidence_count == 2


def test_query_returns_only_citations_used_by_validated_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state("completed", answer="Supported answer [E1].")
    monkeypatch.setattr(api, "get_llm", lambda: object())
    monkeypatch.setattr(api, "run_agent", lambda question, llm: state)

    response = api.query_news(api.QueryRequest(question="AI safety?"))

    assert response.status == "completed"
    assert response.answer == "Supported answer [E1]."
    assert [citation.evidence_id for citation in response.citations] == ["E1"]
    assert response.intent == "news_search"
    assert response.search_queries == ["AI safety risks"]
