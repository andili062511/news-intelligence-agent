"""Command-line interface for BM25 news retrieval."""

import argparse

from .bm25 import BM25Retriever
from .loader import load_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search news chunks with BM25")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--data", default="data/chunks.jsonl", help="Path to chunks JSONL file"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chunks = load_chunks(args.data)
    results = BM25Retriever(chunks).search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print()
    if not results:
        print("No matching chunks found.")
        return

    for index, result in enumerate(results):
        preview = " ".join(str(result["text"]).split())[:300]
        print(f"[{index + 1}] Score: {result['score']:.2f}")
        print(f"Title: {result['title']}")
        print(f"Source: {result['source']}")
        print(f"Date: {result['published_at']}")
        print(f"URL: {result['url']}")
        print(f"Text: {preview}")
        print()
        print("--------------------------------------------------")


if __name__ == "__main__":
    main()
