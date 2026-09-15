"""Dataset contract and leakage checks for the planner hard evaluation set."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from finetune.schemas import validate_sample


DATA_DIR = Path("finetune/data")
HARD_EVAL_FILE = DATA_DIR / "planner_hard_eval.jsonl"


def _load_questions(path: Path) -> list[str]:
    questions: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            sample = json.loads(line)
            question, _ = validate_sample(sample)
            questions.append(question)
    return questions


def _normalize(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(re.findall(r"[\w]+", normalized))


def test_hard_eval_size_balance_schema_and_query_requirements() -> None:
    intents: Counter[str] = Counter()
    questions: list[str] = []
    with HARD_EVAL_FILE.open(encoding="utf-8") as source:
        for line in source:
            sample = json.loads(line)
            question, output = validate_sample(sample)
            questions.append(question)
            intents[output.intent] += 1
            assert output.search_queries
            if output.intent == "compare_news":
                assert len(output.search_queries) >= 2
                assert "news_search" in output.tools
                assert "source_compare" in output.tools
                if output.time_range is not None:
                    assert "time_filter" in output.tools

    assert len(questions) == 100
    assert len(questions) == len(set(questions))
    assert intents == {
        "news_search": 25,
        "compare_news": 25,
        "time_sensitive_news": 25,
        "fact_check": 25,
    }


def test_hard_eval_has_no_exact_or_normalized_train_val_overlap() -> None:
    hard_questions = _load_questions(HARD_EVAL_FILE)
    hard_exact = set(hard_questions)
    hard_normalized = {_normalize(question) for question in hard_questions}

    for reference_name in ("planner_train.jsonl", "planner_val.jsonl"):
        reference_questions = _load_questions(DATA_DIR / reference_name)
        assert hard_exact.isdisjoint(reference_questions)
        assert hard_normalized.isdisjoint(
            {_normalize(question) for question in reference_questions}
        )
