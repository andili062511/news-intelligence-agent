"""Rank-based fusion utilities for retrieval results."""

from typing import Any


_RESULT_FIELDS = (
    "chunk_id",
    "article_id",
    "title",
    "source",
    "url",
    "published_at",
    "text",
)


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    id_key: str = "chunk_id",
) -> list[dict[str, Any]]:
    """Fuse ranked retrieval results using Reciprocal Rank Fusion.

    Scores from the individual retrievers are intentionally ignored. Each
    result contributes ``1 / (k + rank)`` to the item identified by
    ``id_key``. The first two result lists are treated as BM25 and FAISS for
    the debug metadata exposed in ``retrieval_sources``.
    """

    if not result_lists:
        return []
    if k < 0:
        raise ValueError("RRF k must be non-negative")

    fused: dict[Any, dict[str, Any]] = {}
    source_names = ("bm25", "faiss")

    for list_index, results in enumerate(result_lists):
        source_name = (
            source_names[list_index]
            if list_index < len(source_names)
            else f"retriever_{list_index + 1}"
        )
        source_rank_key = f"{source_name}_rank"
        seen_ids: set[Any] = set()

        for position, result in enumerate(results, start=1):
            chunk_id = result.get(id_key)
            # One retriever can contribute at most once per document. This
            # also prevents malformed duplicate candidates from inflating an
            # item's RRF score.
            if chunk_id is None or chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            rank = result.get("rank", position)
            if not isinstance(rank, (int, float)) or rank <= 0:
                rank = position

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "rrf_score": 0.0,
                    "retrieval_sources": {
                        "bm25_rank": None,
                        "faiss_rank": None,
                    },
                    **{field: result.get(field, "") for field in _RESULT_FIELDS},
                }
                if id_key not in _RESULT_FIELDS:
                    fused[chunk_id][id_key] = chunk_id

            item = fused[chunk_id]
            item["rrf_score"] += 1.0 / (k + rank)
            source_ranks = item["retrieval_sources"]
            existing_rank = source_ranks.get(source_rank_key)
            if existing_rank is None or rank < existing_rank:
                source_ranks[source_rank_key] = rank

            # Prefer the first result's metadata, but fill missing values if
            # another retriever has a more complete copy of the chunk.
            for field in _RESULT_FIELDS:
                if not item.get(field) and result.get(field) is not None:
                    item[field] = result.get(field)

    ranked_items = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )
    for rank, item in enumerate(ranked_items, start=1):
        item["rank"] = rank
    return ranked_items
