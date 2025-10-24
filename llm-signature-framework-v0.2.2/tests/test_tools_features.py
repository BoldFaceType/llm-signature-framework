import asyncio
import base64
import urllib.error
import urllib.request

import pytest

import llm_signature_framework.tools as tools
from llm_signature_framework.tools import ExecutionError, FatalToolError, ImageBlob, Tool, ToolRegistry


def test_image_blob_modes(tmp_path, monkeypatch):
    blob = ImageBlob(data=b"hello", mime="image/jpeg")
    encoded = blob.to_llm_part()
    assert encoded.startswith("data:image/jpeg;base64,")

    blob_b64 = ImageBlob(mode="b64", data=base64.b64encode(b"hello").decode())
    assert blob_b64.to_llm_part().endswith("aGVsbG8=")

    file_path = tmp_path / "img.bin"
    file_path.write_bytes(b"hello")
    monkeypatch.setattr(tools, "_SAFE_MEDIA_ROOT", str(tmp_path))
    blob_path = ImageBlob(mode="path", data=file_path, mime="text/plain")
    assert blob_path.to_llm_part().startswith("data:text/plain;base64,")

    monkeypatch.setattr(tools, "_SAFE_MEDIA_ROOT", str(tmp_path / "other"))
    with pytest.raises(FatalToolError):
        ImageBlob(mode="path", data=file_path).to_llm_part()

    monkeypatch.setattr(tools, "_SAFE_MEDIA_ROOT", None)
    assert ImageBlob(mode="url", data="https://example.com").to_llm_part() == "https://example.com"


def test_tool_retry_and_logging(monkeypatch):
    monkeypatch.setattr(tools.ToolRegistry, "_reg", {}, raising=False)

    events = []

    def record(event):
        events.append(event)

    monkeypatch.setattr(tools.ToolRegistry._state, "log_execution", record)

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(tools.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(tools.random, "uniform", lambda a, b: 0.0)

    attempt = {"count": 0}

    @Tool(name="flaky", retries=1, backoff=0.0)
    async def flaky(x: int) -> int:
        attempt["count"] += 1
        if attempt["count"] == 1:
            raise RuntimeError("boom")
        return x * 2

    assert asyncio.run(ToolRegistry.call("flaky", x=3)) == 6
    assert events[0]["ok"] is True

    @Tool(name="always_fail", retries=1, backoff=0.0)
    async def always_fail(x: int) -> int:
        raise RuntimeError("bad")

    with pytest.raises(ExecutionError):
        asyncio.run(ToolRegistry.call("always_fail", x=1))

    assert events[-1]["ok"] is False
    assert "bad" in events[-1]["error"]


def test_list_tools_openai_format(monkeypatch):
    monkeypatch.setattr(tools.ToolRegistry, "_reg", {}, raising=False)

    @Tool(name="hello", desc="Say hello")
    def hello(name: str) -> str:
        return f"hi {name}"

    tools_info = tools.list_tools_openai()
    assert tools_info[0]["function"]["name"] == "hello"
    assert "parameters" in tools_info[0]["function"]


def test_fetch_url_allowlist_and_parsing(monkeypatch):
    monkeypatch.setenv("SAFE_FETCH_ALLOWLIST", "example.com")

    with pytest.raises(FatalToolError):
        asyncio.run(ToolRegistry.call("fetch_url", url="https://other.com", timeout=0.1, max_bytes=10))

    monkeypatch.setenv("SAFE_FETCH_ALLOWLIST", "")

    class FakeHeaders:
        def get_content_charset(self):
            return "utf-8"

    class FakeResponse:
        def __init__(self):
            self.headers = FakeHeaders()
            self._reads = [
                b"<html><body>hello <b>world</b></body></html>",
                b"",
            ]

        def read(self, *_args, **_kwargs):
            return self._reads.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=6.0):
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    text = asyncio.run(ToolRegistry.call("fetch_url", url="https://example.com", timeout=0.1, max_bytes=50))
    assert "hello" in text

    def raising_urlopen(req, timeout=6.0):
        raise urllib.error.URLError("nope")

    monkeypatch.setattr(urllib.request, "urlopen", raising_urlopen)

    with pytest.raises(ExecutionError):
        asyncio.run(ToolRegistry.call("fetch_url", url="https://example.com", timeout=0.1, max_bytes=50))
