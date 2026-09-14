import numpy as np

from retrieval.faiss_retriever import FAISSRetriever


CHUNKS = [
    {
        "chunk_id": "chunk-1",
        "article_id": "article-1",
        "title": "OpenAI releases AI model",
        "source": "example.com",
        "url": "https://example.com/ai",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "OpenAI released an artificial intelligence model.",
    },
    {
        "chunk_id": "chunk-2",
        "article_id": "article-2",
        "title": "Football league result",
        "source": "sports.example.com",
        "url": "https://sports.example.com/football",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "A football team won the league match.",
    },
    {
        "chunk_id": "chunk-3",
        "article_id": "article-3",
        "title": "NVIDIA AI GPU",
        "source": "tech.example.com",
        "url": "https://tech.example.com/gpu",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "NVIDIA introduced a GPU for machine learning workloads.",
    },
]


class FakeEncoder:
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        assert all(" " in text for text in texts)
        return np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        assert query == "AI safety and advanced models"
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_search_ranks_semantically_relevant_news() -> None:
    results = FAISSRetriever(CHUNKS, encoder=FakeEncoder()).search(
        "AI safety and advanced models"
    )

    assert [result["chunk_id"] for result in results] == [
        "chunk-1",
        "chunk-3",
        "chunk-2",
    ]
    assert [result["rank"] for result in results] == [1, 2, 3]
    assert results[0]["score"] == 1.0


def test_search_respects_top_k() -> None:
    retriever = FAISSRetriever(CHUNKS, encoder=FakeEncoder())

    assert len(retriever.search("AI safety and advanced models", top_k=2)) == 2
    assert len(retriever.search("AI safety and advanced models", top_k=10)) == 3
    assert retriever.search("AI safety and advanced models", top_k=0) == []
    assert retriever.search("AI safety and advanced models", top_k=-1) == []


def test_search_handles_empty_query() -> None:
    assert FAISSRetriever(CHUNKS, encoder=FakeEncoder()).search("   ") == []


def test_search_handles_empty_corpus() -> None:
    assert FAISSRetriever([], encoder=FakeEncoder()).search("AI news") == []


def test_search_does_not_modify_original_chunk() -> None:
    chunks = [dict(CHUNKS[0])]
    original = dict(chunks[0])

    FAISSRetriever(chunks, encoder=FakeEncoder()).search("AI safety and advanced models")

    assert chunks == [original]
