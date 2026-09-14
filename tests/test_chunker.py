from ingestion.chunker import chunk_text, create_chunks


def test_chunk_text_creates_overlapping_chunks() -> None:
    words = [f"word{i}" for i in range(1000)]
    chunks = chunk_text(" ".join(words), chunk_size=400, overlap=50)

    assert len(chunks) == 3
    assert len(chunks[0].split()) == 400
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]
    assert len(chunks[-1].split()) == 300


def test_chunk_text_handles_short_and_empty_text() -> None:
    assert chunk_text("one two", chunk_size=400, overlap=50) == ["one two"]
    assert chunk_text("") == []


def test_create_chunks_carries_article_metadata() -> None:
    article = {
        "article_id": "article-1",
        "title": "A title",
        "source": "example.com",
        "url": "https://example.com/a",
        "published_at": "2026-09-14T00:00:00+00:00",
        "text": "one two three",
    }
    chunks = create_chunks(article)
    assert chunks == [
        {
            "chunk_id": "article-1_0",
            "article_id": "article-1",
            "chunk_index": 0,
            "title": "A title",
            "source": "example.com",
            "url": "https://example.com/a",
            "published_at": "2026-09-14T00:00:00+00:00",
            "text": "one two three",
        }
    ]

