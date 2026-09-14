import json

import pytest

from ingestion import pipeline


ARTICLES = [
    {
        "title": "English article",
        "summary": "English summary",
        "language": "English",
    },
    {
        "title": "Turkish article",
        "summary": "Turkish summary",
        "language": "Turkish",
    },
]


def _run_pipeline(monkeypatch, tmp_path, language, articles=ARTICLES):
    monkeypatch.setattr(
        pipeline, "fetch_news", lambda *args, **kwargs: (articles, "GDELT")
    )
    saved_articles, _ = pipeline.run_pipeline(
        "artificial intelligence",
        language=language,
        output_dir=tmp_path,
    )
    on_disk = [
        json.loads(line)
        for line in (tmp_path / "articles.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert on_disk == saved_articles
    return saved_articles


def test_english_article_is_retained_and_turkish_article_is_filtered(
    monkeypatch, tmp_path, capsys
) -> None:
    saved_articles = _run_pipeline(monkeypatch, tmp_path, "English")

    assert [article["title"] for article in saved_articles] == ["English article"]
    output = capsys.readouterr().out
    assert "Fetched: 2" in output
    assert "Language matched: 1" in output
    assert "Valid articles: 2" in output


@pytest.mark.parametrize("article_language", [" English ", "english", "EN", "en-US"])
def test_english_language_matching_is_case_insensitive_and_accepts_common_codes(
    monkeypatch, tmp_path, article_language
) -> None:
    articles = [{"title": "Article", "language": article_language}]

    saved_articles = _run_pipeline(monkeypatch, tmp_path, " eNgLiSh ", articles)

    assert len(saved_articles) == 1


def test_language_all_retains_articles_in_every_language(monkeypatch, tmp_path) -> None:
    saved_articles = _run_pipeline(monkeypatch, tmp_path, " ALL ")

    assert [article["title"] for article in saved_articles] == [
        "English article",
        "Turkish article",
    ]
