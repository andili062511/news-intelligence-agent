from ingestion.deduplicator import deduplicate_articles


def test_deduplicate_articles_removes_duplicate_urls_and_titles(capsys) -> None:
    articles = [
        {"url": "https://example.com/one", "title": "First story"},
        {"url": "https://example.com/one", "title": "A different title"},
        {"url": "https://example.com/two", "title": " first story!!! "},
        {"url": "https://example.com/three", "title": "Unique story"},
    ]

    result = deduplicate_articles(articles)

    assert result == [articles[0], articles[3]]
    output = capsys.readouterr().out
    assert "Before deduplication: 4" in output
    assert "After deduplication: 2" in output
    assert "Removed duplicates: 2" in output

