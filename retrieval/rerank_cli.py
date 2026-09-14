"""Command-line interface for hybrid retrieval with cross-encoder reranking."""

import argparse

from .embeddings import DEFAULT_MODEL_NAME, EmbeddingModel
from .faiss_retriever import FAISSRetriever
from .hybrid import HybridRetriever
from .loader import load_chunks
from .reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    HybridRerankRetriever,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search news chunks with hybrid retrieval and cross-encoder reranking"
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=15,
        help="Number of hybrid candidates to rerank",
    )
    parser.add_argument(
        "--data", default="data/chunks.jsonl", help="Path to chunks JSONL file"
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL_NAME,
        help="Sentence Transformer embedding model name",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder model name",
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
            encoder=EmbeddingModel(args.embedding_model),
        )
        hybrid_retriever = HybridRetriever(
            chunks,
            faiss_retriever=faiss_retriever,
        )
    else:
        hybrid_retriever = HybridRetriever(chunks)

    retriever = HybridRerankRetriever(
        hybrid_retriever,
        CrossEncoderReranker(args.reranker_model),
    )
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
        print(f"[{result['rank']}]")
        print(f"Rerank Score: {result['rerank_score']:.4f}")
        print(f"RRF Score: {result['rrf_score']:.4f}")
        print(f"BM25 Rank: {bm25_rank if bm25_rank is not None else '-'}")
        print(f"FAISS Rank: {faiss_rank if faiss_rank is not None else '-'}")
        print(f"Title: {result.get('title', '')}")
        print(f"Source: {result.get('source', '')}")
        print(f"Date: {result.get('published_at', '')}")
        print(f"URL: {result.get('url', '')}")
        print(f"Text: {preview}")
        print()
        print("--------------------------------")


if __name__ == "__main__":
    main()
