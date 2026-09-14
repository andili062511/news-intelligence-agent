import pytest

from ingestion import pipeline, providers


@pytest.mark.parametrize("provider", ["auto", "gdelt", "rss"])
def test_argparse_accepts_provider(provider):
    args = pipeline.build_parser().parse_args(
        ["--query", "artificial intelligence", "--provider", provider]
    )

    assert args.provider == provider


def test_argparse_defaults_to_auto_provider():
    args = pipeline.build_parser().parse_args(["--query", "test"])

    assert args.provider == "auto"


def test_gdelt_mode_does_not_call_rss(monkeypatch):
    monkeypatch.setattr(providers, "fetch_gdelt_articles", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        providers,
        "fetch_rss_articles",
        lambda *args, **kwargs: pytest.fail("RSS must not be called"),
    )

    articles, provider_used = providers.fetch_news("test", provider="gdelt")

    assert articles == []
    assert provider_used == "GDELT"


def test_rss_mode_does_not_call_gdelt(monkeypatch):
    rss_articles = [{"title": "RSS article"}]
    monkeypatch.setattr(
        providers,
        "fetch_gdelt_articles",
        lambda *args, **kwargs: pytest.fail("GDELT must not be called"),
    )
    monkeypatch.setattr(
        providers, "fetch_rss_articles", lambda *args, **kwargs: rss_articles
    )

    articles, provider_used = providers.fetch_news("test", provider="rss")

    assert articles == rss_articles
    assert provider_used == "RSS"


def test_auto_falls_back_to_rss_when_gdelt_is_empty(monkeypatch, capsys):
    rss_articles = [{"title": "Fallback article"}]
    monkeypatch.setattr(providers, "fetch_gdelt_articles", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        providers, "fetch_rss_articles", lambda *args, **kwargs: rss_articles
    )

    assert providers.fetch_news("test", provider="auto") == (rss_articles, "RSS")
    output = capsys.readouterr().out
    assert "Primary provider: GDELT" in output
    assert "GDELT unavailable. Falling back to RSS." in output
    assert "Provider used: RSS" in output


def test_auto_falls_back_to_rss_when_gdelt_raises(monkeypatch):
    rss_articles = [{"title": "Fallback article"}]

    def raise_gdelt_error(*args, **kwargs):
        raise RuntimeError("GDELT unavailable")

    monkeypatch.setattr(providers, "fetch_gdelt_articles", raise_gdelt_error)
    monkeypatch.setattr(
        providers, "fetch_rss_articles", lambda *args, **kwargs: rss_articles
    )

    assert providers.fetch_news("test", provider="auto") == (rss_articles, "RSS")
