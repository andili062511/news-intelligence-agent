"""Command-line interface for hybrid BM25 and FAISS retrieval."""

import argparse

from .embeddings import DEFAULT_MODEL_NAME, EmbeddingModel
from .faiss_retriever import FAISSRetriever
from .hybrid import HybridRetriever
from .loader import load_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search news chunks with hybrid BM25 and FAISS retrieval"
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of candidates requested from each retriever",
    )
    parser.add_argument(
        "--data", default="data/chunks.jsonl", help="Path to chunks JSONL file"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME, help="Sentence Transformer model name"
    )
    parser.add_argument(
        "--max-chunks-per-article",
        type=int,
        default=None,
        help="Maximum number of returned chunks from one article",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chunks = load_chunks(args.data)

    if chunks:
        faiss_retriever = FAISSRetriever(
            chunks,
            encoder=EmbeddingModel(args.model),
        )
        retriever = HybridRetriever(chunks, faiss_retriever=faiss_retriever)
    else:
        retriever = HybridRetriever(chunks)

    results = retriever.search(
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_chunks_per_article=args.max_chunks_per_article,
    )

    print(f"Query: {args.query}")
    print()
    if not results:
        print("No matching chunks found.")
        return

    for result in results:
        retrieval_sources = result.get("retrieval_sources", {})
        bm25_rank = retrieval_sources.get("bm25_rank", "-")
        faiss_rank = retrieval_sources.get("faiss_rank", "-")
        preview = " ".join(str(result.get("text", "")).split())[:300]
        print(f"[{result['rank']}] RRF Score: {result['rrf_score']:.4f}")
        print(f"BM25 Rank: {bm25_rank}")
        print(f"FAISS Rank: {faiss_rank}")
        print(f"Title: {result.get('title', '')}")
        print(f"Source: {result.get('source', '')}")
        print(f"Date: {result.get('published_at', '')}")
        print(f"URL: {result.get('url', '')}")
        print(f"Text: {preview}")
        print()
        print("--------------------------------")


if __name__ == "__main__":
    main()
