"""Offline unit tests for planner generation scoring."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from finetune.evaluate_planner import (
    INTENTS,
    analyze_predictions,
    build_parser,
    main,
    score_predictions,
    select_evaluation_samples,
    write_error_report,
    write_summary_report,
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
    assert metrics.structural_exact_match == 1.0
    assert metrics.tool_precision == 1.0
    assert metrics.tool_recall == 1.0
    assert metrics.tool_f1 == 1.0
    assert metrics.query_count_accuracy == 1.0
    assert metrics.entity_coverage == 1.0
    assert metrics.query_token_recall == 1.0


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


def test_structural_exact_match_ignores_query_wording() -> None:
    prediction = dict(GOLD, search_queries=["Anthropic coverage", "Anthropic updates"])
    metrics = score_predictions([json.dumps(prediction)], [GOLD])

    assert metrics.structural_exact_match == 1.0
    assert metrics.full_exact_match == 0.0


def test_tool_metrics_use_micro_set_membership() -> None:
    expected = dict(GOLD, tools=["news_search", "time_filter", "source_compare"])
    prediction = dict(GOLD, tools=["news_search", "source_compare"])
    metrics = score_predictions([json.dumps(prediction)], [expected])

    assert metrics.tool_exact_match == 0.0
    assert metrics.tool_precision == 1.0
    assert metrics.tool_recall == pytest.approx(2 / 3)
    assert metrics.tool_f1 == pytest.approx(0.8)


def test_intent_confusion_matrix_and_per_intent_metrics() -> None:
    expected = [GOLD, dict(GOLD, intent="news_search", time_range=None, tools=["news_search"])]
    predictions = [
        json.dumps(dict(GOLD, intent="news_search")),
        json.dumps(expected[1]),
    ]
    result = analyze_predictions(predictions, expected)

    assert tuple(result.intent_confusion_matrix) == INTENTS
    assert result.intent_confusion_matrix["time_sensitive_news"]["news_search"] == 1
    assert result.intent_confusion_matrix["news_search"]["news_search"] == 1
    assert result.per_intent_metrics["news_search"]["precision"] == 0.5
    assert result.per_intent_metrics["news_search"]["recall"] == 1.0
    assert result.per_intent_metrics["news_search"]["f1"] == pytest.approx(2 / 3)
    assert result.per_intent_metrics["time_sensitive_news"]["recall"] == 0.0


def test_query_quality_metrics_are_deterministic() -> None:
    expected = dict(
        GOLD,
        search_queries=["OpenAI latest news", "Anthropic recent developments"],
    )
    prediction = dict(
        GOLD,
        search_queries=["OpenAI breaking coverage", "developments today"],
    )
    metrics = score_predictions([json.dumps(prediction)], [expected])

    # Expected normalized tokens: openai, news, anthropic, developments.
    assert metrics.query_count_accuracy == 1.0
    assert metrics.entity_coverage == 0.5
    assert metrics.query_token_recall == 0.5


def test_time_range_errors_are_counted_and_sorted() -> None:
    one_month = dict(GOLD, time_range="1month")
    predictions = [
        json.dumps(dict(GOLD, time_range="recent")),
        json.dumps(dict(GOLD, time_range="recent")),
        json.dumps(dict(one_month, time_range=None)),
    ]
    result = analyze_predictions(predictions, [GOLD, GOLD, one_month])

    assert result.common_time_range_errors[0] == {
        "expected": "1week",
        "predicted": "recent",
        "count": 2,
    }
    assert result.common_time_range_errors[1]["expected"] == "1month"
    assert result.common_time_range_errors[1]["predicted"] is None


def test_error_and_summary_exports(tmp_path) -> None:
    wrong = dict(GOLD, intent="news_search", tools=["news_search"])
    raw = json.dumps(wrong)
    result = analyze_predictions([raw], [GOLD])
    error_path = tmp_path / "reports" / "errors.jsonl"
    summary_path = tmp_path / "reports" / "summary.json"

    write_error_report(error_path, ["What happened?"], result)
    write_summary_report(summary_path, result)

    error = json.loads(error_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert error["question"] == "What happened?"
    assert error["expected"] == GOLD
    assert error["predicted"] == wrong
    assert error["raw_model_response"] == raw
    assert "intent_mismatch" in error["errors"]
    assert "tool_mismatch" in error["errors"]
    assert summary["sample_count"] == 1
    assert "structural_exact_match" in summary
    assert "intent_confusion_matrix" in summary
    assert "per_intent_metrics" in summary
    assert "common_time_range_errors" in summary


def test_parser_accepts_report_outputs() -> None:
    args = build_parser().parse_args(
        [
            "--error-output",
            "reports/errors.jsonl",
            "--summary-output",
            "reports/summary.json",
        ]
    )

    assert args.error_output.as_posix() == "reports/errors.jsonl"
    assert args.summary_output.as_posix() == "reports/summary.json"


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


def test_parser_accepts_hard_eval_file_without_special_mode() -> None:
    args = build_parser().parse_args(
        ["--val-file", "finetune/data/planner_hard_eval.jsonl"]
    )

    assert args.val_file.as_posix() == "finetune/data/planner_hard_eval.jsonl"


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
