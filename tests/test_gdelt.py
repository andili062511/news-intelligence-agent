import requests

from ingestion import gdelt


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_429_without_retry_after_uses_exponential_backoff(monkeypatch):
    responses = [
        FakeResponse(429),
        FakeResponse(429),
        FakeResponse(429),
        FakeResponse(200, {"articles": [{"title": "Recovered"}]}),
    ]
    delays = []
    monkeypatch.setattr(gdelt.requests, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(gdelt.time, "sleep", delays.append)

    articles = gdelt.fetch_articles("test")

    assert articles == [{"title": "Recovered"}]
    assert delays == [2.0, 4.0, 8.0]


def test_429_honors_retry_after(monkeypatch):
    responses = [
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(200, {"articles": [{"title": "Recovered"}]}),
    ]
    delays = []
    monkeypatch.setattr(gdelt.requests, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(gdelt.time, "sleep", delays.append)

    gdelt.fetch_articles("test")

    assert delays == [7.0]


def test_retryable_server_errors_use_exponential_backoff(monkeypatch):
    responses = [
        FakeResponse(500),
        FakeResponse(502),
        FakeResponse(503),
        FakeResponse(200, {"articles": [{"title": "Recovered"}]}),
    ]
    delays = []
    monkeypatch.setattr(gdelt.requests, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(gdelt.time, "sleep", delays.append)

    articles = gdelt.fetch_articles("test")

    assert articles == [{"title": "Recovered"}]
    assert delays == [2.0, 4.0, 8.0]


def test_retry_stops_after_three_retries(monkeypatch):
    responses = [FakeResponse(504) for _ in range(4)]
    delays = []
    monkeypatch.setattr(gdelt.requests, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(gdelt.time, "sleep", delays.append)

    assert gdelt.fetch_articles("test") == []
    assert delays == [2.0, 4.0, 8.0]
