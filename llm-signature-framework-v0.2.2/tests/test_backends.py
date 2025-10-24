import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from llm_signature_framework.backends import (
    HybridBackend,
    MockBackend,
    get_backend,
    set_backend,
)
from llm_signature_framework.tools import FatalToolError


def test_get_backend_default_and_override(monkeypatch):
    set_backend(None)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert isinstance(get_backend(), MockBackend)

    mock = MockBackend()
    set_backend(mock)
    assert get_backend() is mock

    set_backend(None)
    monkeypatch.setenv("LLM_BACKEND", "mock")
    assert isinstance(get_backend(), MockBackend)


def test_hybrid_backend_requires_endpoint(monkeypatch):
    monkeypatch.delenv("HYBRID_BACKEND_URL", raising=False)
    hb = HybridBackend()
    with pytest.raises(FatalToolError):
        asyncio.run(hb.run())


def test_hybrid_backend_posts_payload(monkeypatch):
    monkeypatch.setenv("HYBRID_BACKEND_URL", "https://api.example.com/run")
    monkeypatch.setenv("HYBRID_API_KEY", "sekret")
    hb = HybridBackend(headers={"X-Test": "1"})

    captured = {}

    class FakeHeaders(SimpleNamespace):
        def get_content_charset(self):
            return "utf-8"

    class FakeResponse:
        def __init__(self, data: bytes):
            self._data = data
            self.headers = FakeHeaders()

        def read(self, *_args, **_kwargs):
            chunk, self._data = self._data, b""
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=20):  # noqa: N803
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = req.data
        return FakeResponse(json.dumps({"content": "ok"}).encode())

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = asyncio.run(hb.run(messages=[{"role": "user", "content": "hi"}], model="gpt-4o"))
    assert result == "ok"
    body = json.loads(captured["payload"].decode())
    assert body["messages"][0]["content"] == "hi"
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer sekret"
    assert headers["x-test"] == "1"


def test_get_backend_hybrid(monkeypatch):
    set_backend(None)
    monkeypatch.setenv("LLM_BACKEND", "hybrid")
    monkeypatch.setenv("HYBRID_BACKEND_URL", "https://api.example.com/run")
    backend = get_backend()
    assert isinstance(backend, HybridBackend)
    assert backend.endpoint == "https://api.example.com/run"
