from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from .tools import FatalToolError


class Backend:
    async def run(
        self,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> Any:
        raise NotImplementedError


class MockBackend(Backend):
    async def run(
        self,
        *,
        messages=None,
        prompt=None,
        model="gpt-4",
        temperature=0.7,
        seed: Optional[int] = None,
    ) -> Any:
        if messages:
            last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            return (last or {}).get("content", "mock-reply") or "mock-reply"
        return "mock-reply"


class OpenAIBackend(Backend):
    def __init__(self):
        try:
            import openai  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise FatalToolError("OpenAI backend requested but openai not installed") from e
        self._openai = openai

    async def run(
        self,
        *,
        messages=None,
        prompt=None,
        model="gpt-4",
        temperature=0.7,
        seed: Optional[int] = None,
    ):
        client = self._openai.OpenAI()
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        resp = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=model, temperature=temperature, messages=messages
            )
        )
        return resp.choices[0].message.content


class AnthropicBackend(Backend):
    def __init__(self):
        try:
            import anthropic  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise FatalToolError("Anthropic backend requested but anthropic not installed") from e
        self._anth = anthropic

    async def run(
        self,
        *,
        messages=None,
        prompt=None,
        model="claude-3-opus-20240229",
        temperature=0.7,
        seed: Optional[int] = None,
    ):
        client = self._anth.Anthropic()
        text = prompt
        if messages:
            text = "\n\n".join(
                m.get("content", "") for m in messages if m.get("role") == "user"
            )
        resp = await asyncio.to_thread(
            lambda: client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": text or ""}],
            )
        )
        return resp.content[0].text


class HybridBackend(Backend):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.endpoint = (
            endpoint or os.getenv("HYBRID_BACKEND_URL") or os.getenv("HYBRID_ENDPOINT")
        )
        self.api_key = api_key or os.getenv("HYBRID_API_KEY") or os.getenv("AGENT_API_KEY")
        self.headers = headers or {}

    async def run(
        self,
        *,
        messages=None,
        prompt=None,
        model="gpt-4",
        temperature=0.7,
        seed: Optional[int] = None,
    ):
        if not self.endpoint:
            raise FatalToolError("Hybrid backend requires HYBRID_BACKEND_URL")
        import urllib.request

        payload = {"model": model, "temperature": temperature}
        if messages is not None:
            payload["messages"] = messages
        if prompt is not None:
            payload["prompt"] = prompt
        if seed is not None:
            payload["seed"] = seed
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        for k, v in self.headers.items():
            req.add_header(k, v)

        def _do():
            with urllib.request.urlopen(req, timeout=20) as resp:
                out = json.loads(resp.read().decode("utf-8", errors="ignore"))
                return out.get("content") or out.get("text") or out

        return await asyncio.to_thread(_do)


_current_backend: Optional[Backend] = None


def set_backend(backend: Optional[Backend]):
    global _current_backend
    _current_backend = backend


def get_backend() -> Backend:
    if _current_backend is not None:
        return _current_backend
    name = (os.getenv("LLM_BACKEND") or "mock").lower()
    if name == "openai":
        return OpenAIBackend()
    if name == "anthropic":
        return AnthropicBackend()
    if name == "hybrid":
        return HybridBackend()
    return MockBackend()
