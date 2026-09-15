"""Offline tests for the planner fine-tuning dataset."""

import json
from pathlib import Path

import pytest

from finetune.dataset_builder import generate_samples, split_samples, write_jsonl
from finetune.schemas import (
    ALLOWED_INTENTS,
    ALLOWED_TOOLS,
    DatasetValidationError,
    validate_sample,
)
from finetune.validate_dataset import validate_datasets, validate_jsonl


def _valid_sample(output: dict[str, object], question: str = "What is new with OpenAI?") -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "You are a news planner."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": json.dumps(output)},
        ]
    }


def test_dataset_generation_is_deterministic_and_small() -> None:
    first = generate_samples(total_samples=40, seed=123)
    second = generate_samples(total_samples=40, seed=123)

    assert first == second
    assert len(first) == 40


def test_generated_schema_and_assistant_json_are_valid() -> None:
    for sample in generate_samples(total_samples=40, seed=7):
        instruction, output = validate_sample(sample)
        decoded = json.loads(sample["messages"][2]["content"])

        assert instruction
        assert decoded == output.to_dict()
        assert output.intent in ALLOWED_INTENTS
        assert output.search_queries
        assert set(output.tools) <= ALLOWED_TOOLS


def test_train_validation_have_no_overlap(tmp_path: Path) -> None:
    train, validation = split_samples(generate_samples(total_samples=40, seed=9))
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    train_result, validation_result, overlap = validate_datasets(train_path, validation_path)

    assert train_result.sample_count == 36
    assert validation_result.sample_count == 4
    assert overlap == 0


def test_duplicate_instruction_detection(tmp_path: Path) -> None:
    sample = generate_samples(total_samples=4, seed=2)[0]
    path = tmp_path / "duplicates.jsonl"
    write_jsonl(path, [sample, sample])

    result = validate_jsonl(path)

    assert result.duplicate_count == 1


def test_time_sensitive_intent_requires_time_range() -> None:
    sample = _valid_sample(
        {
            "intent": "time_sensitive_news",
            "search_queries": ["OpenAI latest news"],
            "time_range": None,
            "tools": ["news_search", "time_filter"],
        }
    )

    with pytest.raises(DatasetValidationError, match="requires time_range"):
        validate_sample(sample)


def test_compare_intent_requires_two_queries() -> None:
    sample = _valid_sample(
        {
            "intent": "compare_news",
            "search_queries": ["OpenAI AI agents"],
            "time_range": None,
            "tools": ["news_search", "source_compare"],
        }
    )

    with pytest.raises(DatasetValidationError, match="at least two"):
        validate_sample(sample)


@pytest.mark.parametrize(
    ("field", "bad_value", "error"),
    [
        ("intent", "answer_question", "invalid intent"),
        ("tools", ["web_search"], "allowed tools"),
    ],
)
def test_rejects_invalid_intent_and_tools(field: str, bad_value: object, error: str) -> None:
    output: dict[str, object] = {
        "intent": "news_search",
        "search_queries": ["OpenAI AI agents developments"],
        "time_range": None,
        "tools": ["news_search"],
    }
    output[field] = bad_value

    with pytest.raises(DatasetValidationError, match=error):
        validate_sample(_valid_sample(output))
