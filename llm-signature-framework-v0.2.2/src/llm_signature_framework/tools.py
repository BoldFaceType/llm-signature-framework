from __future__ import annotations

import asyncio
import functools
import inspect
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, conbytes

from .state import StateManager


class ToolError(Exception):
    fatal: bool = False


class InputValidationError(ToolError):
    ...


class OutputValidationError(ToolError):
    ...


class ExecutionError(ToolError):
    ...


class FatalToolError(ToolError):
    fatal = True


_SAFE_MEDIA_ROOT = os.getenv("SAFE_MEDIA_ROOT")


class ImageBlob(BaseModel):
    mode: str = Field("bytes", pattern=r"^(bytes|b64|path|url)$")
    data: Union[conbytes(strict=True), str, Path]
    mime: str = "image/png"

    def to_llm_part(self) -> str:
        from base64 import b64encode

        if self.mode == "bytes":
            return f"data:{self.mime};base64,{b64encode(self.data).decode()}"
        if self.mode == "b64":
            return f"data:{self.mime};base64,{self.data}"
        if self.mode == "path":
            p = Path(self.data).expanduser().resolve()
            if _SAFE_MEDIA_ROOT:
                root = Path(_SAFE_MEDIA_ROOT).expanduser().resolve()
                try:
                    if not p.is_relative_to(root):
                        raise FatalToolError(f"Path {p} outside SAFE_MEDIA_ROOT")
                except AttributeError:
                    if not (p == root or root in p.parents):
                        raise FatalToolError(f"Path {p} outside SAFE_MEDIA_ROOT")
            b = p.read_bytes()
            return f"data:{self.mime};base64,{b64encode(b).decode()}"
        return str(self.data)


class Tool:
    def __init__(self, name: Optional[str] = None, desc: Optional[str] = None, retries: int = 2, backoff: float = 0.4):
        self.name, self.desc, self.retries, self.backoff = name, desc, retries, backoff

    def __call__(self, func):
        sig, hints = inspect.signature(func), get_type_hints(func)

        class _Input(BaseModel):
            __annotations__ = {p.name: hints.get(p.name, Any) for p in sig.parameters.values()}

        Output = TypeAdapter(hints.get("return", Any))

        @functools.wraps(func)
        async def wrapper(**kwargs):
            try:
                parsed = _Input(**kwargs).model_dump()
            except ValidationError as e:
                raise InputValidationError(e) from e
            delay = self.backoff
            for attempt in range(self.retries + 1):
                try:
                    return Output.validate_python(await _maybe_await(func)(**parsed))
                except ValidationError as e:
                    raise OutputValidationError(e) from e
                except FatalToolError:
                    raise
                except Exception as e:
                    if attempt == self.retries:
                        raise ExecutionError(e) from e
                    await asyncio.sleep(delay + random.uniform(0, delay))
                    delay *= 2

        wrapper._tool_meta = {
            "name": self.name or func.__name__,
            "description": (self.desc or (func.__doc__ or "")).strip(),
            "parameters": _Input.model_json_schema(),
        }
        ToolRegistry.register(wrapper)
        return wrapper


def _maybe_await(fn):
    if asyncio.iscoroutinefunction(fn):
        return fn

    async def _run(**kw):
        return await asyncio.to_thread(fn, **kw)

    return _run


class ToolRegistry:
    _reg: Dict[str, Any] = {}
    _state = StateManager()

    @classmethod
    def register(cls, tool_fn):
        cls._reg[tool_fn._tool_meta["name"]] = tool_fn

    @classmethod
    async def call(cls, name: str, **kwargs):
        if name not in cls._reg:
            raise FatalToolError(f"Unknown tool '{name}'")
        start = time.perf_counter()
        ok = True
        err = None
        try:
            return await cls._reg[name](**kwargs)
        except Exception as e:
            ok, err = False, str(e)
            raise
        finally:
            try:
                cls._state.log_execution(
                    {
                        "function": f"tool:{name}",
                        "ts": datetime.now().isoformat(),
                        "ok": ok,
                        "error": err,
                        "args": {
                            k: ("<blob>" if isinstance(v, (bytes, bytearray, ImageBlob)) else v)
                            for k, v in list(kwargs.items())[:12]
                        },
                        "duration_s": round(time.perf_counter() - start, 6),
                    }
                )
            except Exception:
                pass

    @classmethod
    def list_tools(cls) -> List[Dict[str, Any]]:
        return [t._tool_meta for t in cls._reg.values()]


def list_tools_for_planner() -> List[Dict[str, Any]]:
    return ToolRegistry.list_tools()


def list_tools_openai() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": m["name"],
                "description": m.get("description", ""),
                "parameters": m["parameters"],
            },
        }
        for m in ToolRegistry.list_tools()
    ]


async def call_tool(name: str, arguments: Dict[str, Any]):
    return await ToolRegistry.call(name, **arguments)


@Tool(name="fetch_url", desc="Fetch a URL and return plain text (best-effort)")
def fetch_url(url: str, timeout: float = 6.0, max_bytes: int = 2_000_000) -> str:
    import urllib.request, urllib.error, html.parser
    from urllib.parse import urlparse

    allow = os.getenv("SAFE_FETCH_ALLOWLIST")
    if allow:
        allowed = {d.strip().lower() for d in allow.split(',') if d.strip()}
        host = (urlparse(url).hostname or "").lower()
        if host not in allowed:
            raise FatalToolError(f"Domain '{host}' not in SAFE_FETCH_ALLOWLIST")

    class _TextExtractor(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts: List[str] = []

        def handle_data(self, data):
            data = data.strip()
            if data:
                self.parts.append(data)

        def get_text(self):
            return " ".join(self.parts)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            total = 0
            chunks = []
            while True:
                chunk = resp.read(min(65536, max_bytes - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
            html = b"".join(chunks).decode(charset, errors="ignore")
            p = _TextExtractor()
            p.feed(html)
            return p.get_text()
    except urllib.error.URLError as e:
        raise ExecutionError(e)
