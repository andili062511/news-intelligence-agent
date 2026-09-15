"""Offline unit tests for planner generation scoring."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from finetune.evaluate_planner import (
    build_parser,
    main,
    score_predictions,
    select_evaluation_samples,
)
from finetune.inference_adapter import parse_planner_response


GOLD = {
    "intent": "time_sensitive_news",
    "search_queries": ["Anthropic latest news", "Anthropic recent developments"],
    "time_range": "1week",
    "tools": ["news_search", "time_filter"],
}


def test_valid_json_and_schema_scoring() -> None:
    metrics = score_predictions([json.dumps(GOLD)], [GOLD])

    assert metrics.json_valid_rate == 1.0
    assert metrics.schema_valid_rate == 1.0
    assert metrics.intent_accuracy == 1.0
    assert metrics.tool_exact_match == 1.0
    assert metrics.time_range_accuracy == 1.0
    assert metrics.search_query_nonempty_rate == 1.0
    assert metrics.full_exact_match == 1.0


def test_invalid_json_scores_zero() -> None:
    metrics = score_predictions(["not JSON"], [GOLD])

    assert metrics.json_valid_rate == 0.0
    assert metrics.schema_valid_rate == 0.0
    assert metrics.intent_accuracy == 0.0
    assert metrics.tool_exact_match == 0.0
    assert metrics.full_exact_match == 0.0


def test_intent_and_tool_exact_match_are_independent() -> None:
    wrong = dict(GOLD)
    wrong["intent"] = "news_search"
    wrong["tools"] = ["time_filter", "news_search"]
    metrics = score_predictions([json.dumps(wrong)], [GOLD])

    assert metrics.intent_accuracy == 0.0
    assert metrics.tool_exact_match == 0.0
    assert metrics.time_range_accuracy == 1.0
    assert metrics.schema_valid_rate == 1.0


def test_json_can_be_valid_while_schema_is_invalid() -> None:
    raw = json.dumps({"intent": "time_sensitive_news"})
    parsed = parse_planner_response(raw)
    metrics = score_predictions([raw], [GOLD])

    assert parsed.parsed_response == {"intent": "time_sensitive_news"}
    assert parsed.schema_valid is False
    assert metrics.json_valid_rate == 1.0
    assert metrics.schema_valid_rate == 0.0
    assert metrics.intent_accuracy == 1.0


def test_scoring_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        score_predictions([], [GOLD])


def _evaluation_rows(count: int = 100) -> tuple[list[str], list[dict[str, object]]]:
    questions = [f"question-{index}" for index in range(count)]
    outputs = [dict(GOLD, search_queries=[f"query-{index}"]) for index in range(count)]
    return questions, outputs


def test_max_samples_none_uses_all_samples() -> None:
    questions, outputs = _evaluation_rows()

    selected_questions, selected_outputs = select_evaluation_samples(
        questions, outputs, None
    )

    assert selected_questions == questions
    assert selected_outputs == outputs


def test_max_samples_truncates_to_deterministic_prefix() -> None:
    questions, outputs = _evaluation_rows()

    selected_questions, selected_outputs = select_evaluation_samples(
        questions, outputs, 20
    )

    assert selected_questions == questions[:20]
    assert selected_outputs == outputs[:20]


def test_max_samples_larger_than_dataset_uses_all_samples() -> None:
    questions, outputs = _evaluation_rows(10)

    selected_questions, selected_outputs = select_evaluation_samples(
        questions, outputs, 20
    )

    assert selected_questions == questions
    assert selected_outputs == outputs


@pytest.mark.parametrize("max_samples", [0, -1])
def test_max_samples_rejects_non_positive_values(
    max_samples: int, capsys: pytest.CaptureFixture[str]
) -> None:
    questions, outputs = _evaluation_rows()
    with pytest.raises(ValueError, match="positive integer"):
        select_evaluation_samples(questions, outputs, max_samples)

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--max-samples", str(max_samples)])
    assert "must be a positive integer" in capsys.readouterr().err


def test_base_and_adapter_evaluate_identical_sample_prefix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    questions, outputs = _evaluation_rows(30)
    evaluated: dict[str, list[str]] = {"base": [], "adapter": []}

    class FakePlannerInference:
        def __init__(self, model_name, adapter_path, *, load_in_4bit):
            self.target = "adapter" if adapter_path else "base"

        def generate(self, question, *, max_new_tokens):
            evaluated[self.target].append(question)
            return SimpleNamespace(raw_response=json.dumps(GOLD))

    monkeypatch.setattr(
        "finetune.evaluate_planner.load_evaluation_samples",
        lambda path: (questions, outputs),
    )
    monkeypatch.setattr(
        "finetune.evaluate_planner.PlannerInference", FakePlannerInference
    )

    monkeypatch.setattr(sys, "argv", ["evaluate_planner", "--max-samples", "20"])
    main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_planner", "--adapter", "adapter-path", "--max-samples", "20"],
    )
    main()

    assert evaluated["base"] == questions[:20]
    assert evaluated["adapter"] == evaluated["base"]
    assert capsys.readouterr().out.count("Evaluated samples: 20") == 2
