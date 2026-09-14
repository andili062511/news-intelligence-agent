from agent.answer_validation import extract_citation_ids, validate_answer_citations


PACK = {"evidence": [{"evidence_id": "E1"}, {"evidence_id": "E2"}]}


def test_extracts_multiple_citations_and_deduplicates() -> None:
    assert extract_citation_ids("One [E1], two [E2], one again [E1].") == [
        "E1",
        "E2",
    ]


def test_unknown_citation_is_rejected() -> None:
    result = validate_answer_citations("Claim [E99].", PACK)

    assert result["valid"] is False
    assert result["invalid_citations"] == ["E99"]
    assert result["reason"] == "Unknown citation [E99]"


def test_factual_answer_without_citation_is_rejected() -> None:
    result = validate_answer_citations("A factual claim.", PACK)

    assert result["valid"] is False
    assert result["reason"] == "factual_answer_without_citation"


def test_empty_answer_is_rejected() -> None:
    result = validate_answer_citations("  ", PACK)

    assert result["valid"] is False
    assert result["reason"] == "empty_answer"


def test_known_citations_are_valid() -> None:
    result = validate_answer_citations("Claim [E2] and context [E1].", PACK)

    assert result == {
        "valid": True,
        "used_citations": ["E2", "E1"],
        "invalid_citations": [],
        "reason": "valid",
    }
