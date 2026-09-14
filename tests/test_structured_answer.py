from typing import Any

from agent.structured_answer import parse_claim_response, render_claims, validate_claims


def pack() -> dict[str, Any]:
    return {"evidence": [{"evidence_id": "E1"}, {"evidence_id": "E2"}]}


def test_parse_valid_response() -> None:
    result = parse_claim_response(
        '{"claims":[{"text":"Supported.","citations":["E1"]}]}'
    )
    assert result["valid"] is True


def test_parse_malformed_json_returns_error_without_crashing() -> None:
    result = parse_claim_response("```json")
    assert result["valid"] is False
    assert result["errors"][0].startswith("invalid_json")


def test_validate_rejects_empty_claims_and_unknown_citations() -> None:
    assert validate_claims([], pack())["valid"] is False
    result = validate_claims(
        [{"text": "Unsupported.", "citations": ["E99"]}], pack()
    )
    assert result["valid"] is False
    assert "E99" in result["errors"][0]


def test_validate_deduplicates_citations() -> None:
    result = validate_claims(
        [{"text": "Supported.", "citations": ["E1", "E1", "E2"]}], pack()
    )
    assert result["claims"][0]["citations"] == ["E1", "E2"]


def test_renderer_outputs_only_claim_citations() -> None:
    answer = render_claims(
        [
            {"text": "First claim.", "citations": ["E1", "E2"]},
            {"text": "Second claim.", "citations": ["E2"]},
        ]
    )
    assert answer == "First claim [E1][E2].\n\nSecond claim [E2]."
    assert answer.count("[E1]") == 1


def test_renderer_does_not_guess_a_citation() -> None:
    answer = render_claims([{"text": "Only E2 was selected.", "citations": ["E2"]}])
    assert answer == "Only E2 was selected [E2]."
    assert "[E1]" not in answer
