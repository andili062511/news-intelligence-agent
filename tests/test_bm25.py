from retrieval.bm25 import BM25Retriever


CHUNKS = [
    {
        "chunk_id": "chunk-1",
        "article_id": "article-1",
        "title": "OpenAI releases new AI model",
        "source": "example.com",
        "url": "https://example.com/ai",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "OpenAI released a new artificial intelligence model for developers.",
    },
    {
        "chunk_id": "chunk-2",
        "article_id": "article-2",
        "title": "Football match results",
        "source": "sports.example.com",
        "url": "https://sports.example.com/football",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "The football club won its league match.",
    },
    {
        "chunk_id": "chunk-3",
        "article_id": "article-3",
        "title": "NVIDIA launches new GPU",
        "source": "tech.example.com",
        "url": "https://tech.example.com/gpu",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "NVIDIA announced a new GPU designed for artificial intelligence workloads.",
    },
]


def test_search_ranks_relevant_news_above_unrelated_news() -> None:
    results = BM25Retriever(CHUNKS).search("artificial intelligence")

    assert len(results) == 3
    assert results[0]["chunk_id"] in {"chunk-1", "chunk-3"}
    assert results[-1]["chunk_id"] == "chunk-2"
    assert [result["rank"] for result in results] == [1, 2, 3]


def test_search_uses_title_keywords_for_ranking() -> None:
    chunks = [
        {
            "chunk_id": "chunk-without-title-match",
            "title": "General market update",
            "text": "The company published its regular announcement.",
        },
        {
            "chunk_id": "chunk-with-title-match",
            "title": "Aurora research breakthrough",
            "text": "The company published its regular announcement.",
        },
        {
            "chunk_id": "another-chunk-without-title-match",
            "title": "Technology company update",
            "text": "The company published its regular announcement.",
        },
    ]

    results = BM25Retriever(chunks).search("Aurora")

    assert results[0]["chunk_id"] == "chunk-with-title-match"


def test_search_handles_empty_query() -> None:
    assert BM25Retriever(CHUNKS).search("") == []


def test_search_respects_top_k() -> None:
    results = BM25Retriever(CHUNKS).search("new", top_k=1)
    assert len(results) == 1
    assert BM25Retriever(CHUNKS).search("new", top_k=0) == []
    assert len(BM25Retriever(CHUNKS).search("new", top_k=10)) == len(CHUNKS)


def test_search_handles_empty_corpus() -> None:
    assert BM25Retriever([]).search("artificial intelligence") == []
