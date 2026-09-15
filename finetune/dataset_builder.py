"""Build a reproducible, synthetic planner instruction dataset without an LLM."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Iterable

from finetune.schemas import InstructionSample, Message, PlannerOutput


RANDOM_SEED = 20260915
DEFAULT_INTENT_COUNTS = {
    "news_search": 300,
    "compare_news": 250,
    "time_sensitive_news": 300,
    "fact_check": 150,
}
ENTITIES = (
    "OpenAI",
    "Anthropic",
    "Google",
    "Microsoft",
    "NVIDIA",
    "Meta",
    "Tesla",
    "Apple",
    "Amazon",
)
TOPICS = (
    "AI agents",
    "AI safety",
    "large language models",
    "chips",
    "cloud computing",
    "robotics",
    "earnings",
    "regulation",
)
SYSTEM_PROMPT = (
    "You are the planner for a news intelligence agent. Classify the user's intent, "
    "rewrite it into concise search queries, identify any time range, and select only "
    "the available tools. Return only valid JSON with the fields intent, search_queries, "
    "time_range, and tools. Plan retrieval; do not answer the news question or invent citations."
)


def _sample(user_text: str, output: PlannerOutput) -> InstructionSample:
    return {
        "messages": [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_text),
            Message(role="assistant", content=output.to_json()),
        ]
    }


def _news_candidates() -> Iterable[InstructionSample]:
    entity_topic_templates: tuple[tuple[str, Callable[[str, str], list[str]]], ...] = (
        ("What is happening with {entity} and {topic}?", lambda e, t: [f"{e} {t} developments"]),
        ("Find news about {entity}'s work on {topic}.", lambda e, t: [f"{e} {t} news"]),
        ("Summarize the main developments involving {entity} in {topic}.", lambda e, t: [f"{e} {t} key developments"]),
        ("Which stories connect {entity} with {topic}?", lambda e, t: [f"{e} {t} reports"]),
        ("Give me an overview of {entity} activity in {topic}.", lambda e, t: [f"{e} {t} overview"]),
    )
    for (template, query_fn), entity, topic in product(entity_topic_templates, ENTITIES, TOPICS):
        tools = ["news_search", "structured_query"] if topic in {"earnings", "regulation"} else ["news_search"]
        yield _sample(
            template.format(entity=entity, topic=topic),
            PlannerOutput("news_search", query_fn(entity, topic), None, tools),
        )

    topic_templates = (
        "What are the main developments in {topic}?",
        "Show me important reporting about {topic}.",
        "What stories should I know about in {topic}?",
        "Give me a news overview of {topic}.",
    )
    for template, topic in product(topic_templates, TOPICS):
        yield _sample(
            template.format(topic=topic),
            PlannerOutput("news_search", [f"{topic} major developments"], None, ["news_search"]),
        )


def _compare_candidates() -> Iterable[InstructionSample]:
    templates = (
        "Compare {entity1} and {entity2} on {topic}.",
        "What is the difference between recent {entity1} and {entity2} developments in {topic}?",
        "How do {entity1} and {entity2} differ in their {topic} developments?",
        "Contrast reporting on {entity1} versus {entity2} in {topic}.",
    )
    for template, (entity1, entity2), topic in product(templates, combinations(ENTITIES, 2), TOPICS):
        yield _sample(
            template.format(entity1=entity1, entity2=entity2, topic=topic),
            PlannerOutput(
                "compare_news",
                [f"{entity1} {topic} developments", f"{entity2} {topic} developments"],
                None,
                ["news_search", "source_compare"],
            ),
        )


def _time_candidates() -> Iterable[InstructionSample]:
    entity_templates = (
        ("What is the latest news about {entity}?", "latest", "1week"),
        ("What happened with {entity} this week?", "this week", "1week"),
        ("Catch me up on recent {entity} developments.", "recent", "1week"),
        ("What has {entity} announced today?", "today", "24hours"),
        ("Which new {entity} stories emerged this month?", "this month", "1month"),
    )
    for (template, qualifier, time_range), entity in product(entity_templates, ENTITIES):
        queries = [f"{entity} latest news", f"{entity} recent developments"]
        if qualifier == "today":
            queries = [f"{entity} announcements today", f"{entity} news today"]
        elif qualifier == "this month":
            queries = [f"{entity} news this month", f"{entity} recent developments"]
        yield _sample(
            template.format(entity=entity),
            PlannerOutput("time_sensitive_news", queries, time_range, ["news_search", "time_filter"]),
        )

    topic_templates = (
        ("What are the recent developments in {topic}?", "1week"),
        ("What is the latest reporting on {topic}?", "1week"),
        ("What happened in {topic} this week?", "1week"),
        ("Show me today's news about {topic}.", "24hours"),
        ("Catch me up on {topic} over the past month.", "1month"),
    )
    for (template, time_range), topic in product(topic_templates, TOPICS):
        yield _sample(
            template.format(topic=topic),
            PlannerOutput(
                "time_sensitive_news",
                [f"{topic} latest news", f"{topic} recent developments"],
                time_range,
                ["news_search", "time_filter"],
            ),
        )

    entity_topic_templates = (
        "What are the newest {entity} developments in {topic}?",
        "What did {entity} do in {topic} this week?",
        "Find recent reporting on {entity} and {topic}.",
        "Any fresh developments from {entity} involving {topic}?",
    )
    for template, entity, topic in product(entity_topic_templates, ENTITIES, TOPICS):
        yield _sample(
            template.format(entity=entity, topic=topic),
            PlannerOutput(
                "time_sensitive_news",
                [f"{entity} {topic} latest news", f"{entity} {topic} recent developments"],
                "1week",
                ["news_search", "time_filter"],
            ),
        )


def _fact_check_candidates() -> Iterable[InstructionSample]:
    templates = (
        "Did {entity} announce a new development in {topic}?",
        "Verify reports about {entity} and {topic}.",
        "Is the reported {entity} development in {topic} accurate?",
        "Check the claim that {entity} made news in {topic}.",
    )
    for template, entity, topic in product(templates, ENTITIES, TOPICS):
        yield _sample(
            template.format(entity=entity, topic=topic),
            PlannerOutput(
                "fact_check",
                [f"{entity} {topic} announcement", f"{entity} {topic} independent reports"],
                None,
                ["news_search", "source_compare", "structured_query"],
            ),
        )


CANDIDATE_BUILDERS = {
    "news_search": _news_candidates,
    "compare_news": _compare_candidates,
    "time_sensitive_news": _time_candidates,
    "fact_check": _fact_check_candidates,
}


def _scaled_counts(total_samples: int) -> dict[str, int]:
    if total_samples < len(DEFAULT_INTENT_COUNTS):
        raise ValueError(f"total_samples must be at least {len(DEFAULT_INTENT_COUNTS)}")
    total_default = sum(DEFAULT_INTENT_COUNTS.values())
    raw = {name: total_samples * count / total_default for name, count in DEFAULT_INTENT_COUNTS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total_samples - sum(counts.values())
    ranked = sorted(raw, key=lambda name: (raw[name] - counts[name], DEFAULT_INTENT_COUNTS[name]), reverse=True)
    for name in ranked[:remainder]:
        counts[name] += 1
    return counts


def generate_samples(total_samples: int = 1000, seed: int = RANDOM_SEED) -> list[InstructionSample]:
    """Generate deterministic samples with the requested intent proportions."""

    rng = random.Random(seed)
    selected: list[InstructionSample] = []
    for intent, requested in _scaled_counts(total_samples).items():
        candidates = list(CANDIDATE_BUILDERS[intent]())
        if requested > len(candidates):
            raise ValueError(f"requested {requested} {intent} samples, only {len(candidates)} are available")
        selected.extend(rng.sample(candidates, requested))
    rng.shuffle(selected)
    return selected


def split_samples(
    samples: list[InstructionSample], validation_ratio: float = 0.1
) -> tuple[list[InstructionSample], list[InstructionSample]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    validation_size = round(len(samples) * validation_ratio)
    return samples[validation_size:], samples[:validation_size]


def write_jsonl(path: Path, samples: Iterable[InstructionSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_dataset(
    output_dir: Path | str | None = None,
    total_samples: int = 1000,
    seed: int = RANDOM_SEED,
) -> tuple[list[InstructionSample], list[InstructionSample]]:
    """Generate and write the train/validation JSONL files."""

    destination = Path(output_dir) if output_dir is not None else Path(__file__).parent / "data"
    train, validation = split_samples(generate_samples(total_samples, seed))
    write_jsonl(destination / "planner_train.jsonl", train)
    write_jsonl(destination / "planner_val.jsonl", validation)
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    train, validation = build_dataset(args.output_dir, args.samples, args.seed)
    intents = Counter(json.loads(item["messages"][2]["content"])["intent"] for item in train + validation)
    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print("Intent distribution:")
    for intent in DEFAULT_INTENT_COUNTS:
        print(f"{intent}: {intents[intent]}")


if __name__ == "__main__":
    main()
