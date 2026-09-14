from ingestion import rss


RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <language>en-US</language>
    <item>
      <title>AI &amp; Science</title>
      <link>https://example.com/article</link>
      <pubDate>Sun, 14 Sep 2026 10:00:00 GMT</pubDate>
      <source>Example News</source>
      <description><![CDATA[<p>Article summary.</p>]]></description>
    </item>
  </channel>
</rss>
"""


class FakeResponse:
    content = RSS_XML

    def raise_for_status(self):
        return None


def test_fetch_articles_parses_rss_without_network(monkeypatch):
    request = {}

    def fake_get(url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(rss.requests, "get", fake_get)

    articles = rss.fetch_articles("artificial intelligence", max_records=1)

    assert articles == [
        {
            "title": "AI & Science",
            "url": "https://example.com/article",
            "source": "Example News",
            "published_at": "Sun, 14 Sep 2026 10:00:00 GMT",
            "language": "en-US",
            "summary": "<p>Article summary.</p>",
        }
    ]
    assert request["url"] == rss.GOOGLE_NEWS_RSS_URL
    assert request["params"]["q"] == "artificial intelligence when:7d"
