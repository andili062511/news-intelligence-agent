"""CLI for the LangGraph news planner-executor agent."""

import argparse

from llm.qwen import QwenLLM

from .graph import run_agent


def _optional_positive_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer or 'none'")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve news evidence with LangGraph")
    parser.add_argument("--question", required=True, help="News question to investigate")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--grounding-max-items", type=int, default=3)
    parser.add_argument(
        "--max-chunks-per-article", type=_optional_positive_int, default=1
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    llm = QwenLLM(model_name=args.model, max_new_tokens=args.max_new_tokens)
    result = run_agent(
        args.question,
        llm=llm,
        max_retries=args.max_retries,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        grounding_max_items=args.grounding_max_items,
        max_chunks_per_article=args.max_chunks_per_article,
    )

    print(f"Question:\n{result['question']}\n")
    print(f"Intent:\n{result['intent']}\n")
    print("Queries:")
    for query in result.get("search_queries", []):
        print(f"- {query}")
    print(f"\nRetries:\n{result.get('retry_count', 0)}\n")
    grounding_pack = result.get("grounding_pack") or {}
    sufficient = bool(grounding_pack.get("ready"))
    print("Evidence Status:")
    print("SUFFICIENT" if sufficient else "INSUFFICIENT EVIDENCE")
    print("\nCitation Status:")
    print(result.get("citation_status") or "NOT RUN")
    print("\nGrounding Ready:")
    print("YES" if grounding_pack.get("ready") else "NO")
    print(f"\nGeneration Attempts: {result.get('generation_attempts', 0)}")
    if result.get("status") == "completed":
        print("\nAnswer Status:\nVALID")
        print("\nFinal Answer:\n")
        print(result["final_answer"])
    elif result.get("status") == "citation_failed" and result.get("answer_validation"):
        print("\nAnswer Status:\nCITATION FAILED")
        print("\nRaw Model Output:")
        print(result.get("raw_model_output") or result.get("raw_generated_answer") or "NONE")
        print("\nValidation Error:")
        print(result.get("reason", "Answer citation validation failed"))
        print("\nFinal Answer:\nNONE")
    else:
        print("\nReason:")
        print(result.get("reason", "Not enough valid evidence"))
    print("\nCitations:\n")

    for item in grounding_pack.get("evidence", []):
        evidence_id = item.get("evidence_id", "")
        title = " ".join(str(item.get("title", "")).split())
        print(f"[{evidence_id}] {title}")


if __name__ == "__main__":
    main()
