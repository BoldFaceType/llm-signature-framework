from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import os
import random
import re
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints

from pydantic import BaseModel, TypeAdapter, ValidationError

from .backends import get_backend
from .state import StateManager
from .tools import ImageBlob, InputValidationError, OutputValidationError, ExecutionError

__version__ = "0.2.2"

_tiktoken_enabled = False
try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
    _tiktoken_enabled = True
except Exception:  # pragma: no cover - import guard
    pass

_yaml_enabled = False
try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
    _yaml_enabled = True
except Exception:  # pragma: no cover
    pass

try:
    _PRICING = json.loads(os.getenv("LLM_PRICING_JSON", "{}"))
except Exception:
    _PRICING = {}


def _usd_cost(model: str, prompt_toks: int, completion_toks: int) -> Optional[float]:
    p = _PRICING.get(model)
    if not p:
        return None
    return round(
        (
            prompt_toks * float(p.get("input", 0.0))
            + completion_toks * float(p.get("output", 0.0))
        )
        / 1000.0,
        6,
    )


def _count_tokens(text: str, model: str = "gpt-4") -> int:
    if _tiktoken_enabled:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, int(len(text.split()) * 0.75))


class _MiniTemplate:
    _var_rx = re.compile(r"{(\w+)}")
    _tag_rx = re.compile(r"{%(.*?)%}", re.S)
    _endfor_rx = re.compile(r"{%-?\s*endfor\s*-?%}")
    _endif_rx = re.compile(r"{%-?\s*endif\s*-?%}")
    _elif_else_rx = re.compile(r"{%-?\s*(elif|else)(.*?)%}")

    def __init__(self, text: str):
        self.text = text

    def render(self, ctx: Dict[str, Any]) -> str:
        return self._render_block(self.text, ctx)

    def _render_block(self, block: str, ctx: Dict[str, Any]) -> str:
        out: List[str] = []
        i = 0
        while i < len(block):
            m = self._tag_rx.search(block, i)
            if not m:
                out.append(self._vars(block[i:], ctx))
                break
            start, end = m.span()
            out.append(self._vars(block[i:start], ctx))
            tag = m.group(1).strip()
            i = end
            if tag.startswith("for"):
                txt, i = self._handle_for(block, i, tag, ctx)
                out.append(txt)
            elif tag.startswith("if"):
                txt, i = self._handle_if(block, i, tag, ctx)
                out.append(txt)
            else:
                out.append(m.group(0))
        return "".join(out)

    def _vars(self, s: str, ctx: Dict[str, Any]) -> str:
        def sub(mm):
            v = ctx.get(mm.group(1), "")
            return v.to_llm_part() if isinstance(v, ImageBlob) else str(v)

        return self._var_rx.sub(sub, s)

    def _handle_for(self, text: str, pos: int, tag: str, ctx: Dict[str, Any]):
        try:
            _, var, _, it_name = tag.split()
        except ValueError:  # pragma: no cover - defensive
            raise ValueError("Malformed {% for %} tag")
        end = self._endfor_rx.search(text, pos)
        if not end:
            raise ValueError("Unclosed {% for %} block")
        body = text[pos : end.start()]
        it = ctx.get(it_name, []) or []
        return (
            "".join(self._render_block(body, {**ctx, var: x}) for x in it),
            end.end(),
        )

    def _handle_if(self, text: str, pos: int, tag: str, ctx: Dict[str, Any]):
        cond_var = tag.split()[1]
        end = self._endif_rx.search(text, pos)
        if not end:
            raise ValueError("Unclosed {% if %} block")
        block = text[pos : end.start()]
        parts: List[tuple[str, str]] = []
        last = 0
        for m in self._elif_else_rx.finditer(block):
            parts.append(("body", block[last : m.start()]))
            parts.append((m.group(1).strip(), m.group(2).strip()))
            last = m.end()
        parts.append(("body", block[last:]))
        active = bool(ctx.get(cond_var))
        idx = 0
        while idx < len(parts):
            typ, content = parts[idx]
            if typ == "body" and active:
                return self._render_block(content, ctx), end.end()
            if typ == "elif":
                var = content.split()[0]
                active = bool(ctx.get(var))
            if typ == "else":
                active = True
            idx += 1
        return "", end.end()


class LLMFunction:
    def __init__(
        self,
        *,
        template: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        retries: int = 1,
        track: bool = True,
        enable_repair: bool = True,
        seed: Optional[int] = None,
    ):
        self.template = template
        self.model = model
        self.temperature = temperature
        self.retries = max(0, retries)
        self.track = track
        self.enable_repair = enable_repair
        self.seed = seed
        self.state = StateManager() if track else None

    def __call__(self, func):
        sig, hints = inspect.signature(func), get_type_hints(func)
        template_text = self.template or self._auto_template(func, sig)
        renderer = _MiniTemplate(template_text)

        class _Input(BaseModel):
            __annotations__ = {n: hints.get(n, Any) for n in sig.parameters}

        Output = TypeAdapter(hints.get("return", Any))
        is_messages_mode = "messages" in sig.parameters

        @functools.wraps(func)
        def sync_wrap(*a, **k):
            return asyncio.run(async_wrap(*a, **k))

        @functools.wraps(func)
        async def async_wrap(*a, **k):
            bound = sig.bind(*a, **k)
            bound.apply_defaults()
            try:
                args = _Input.model_validate(bound.arguments).model_dump()
            except ValidationError as e:
                raise InputValidationError(e) from e
            backend = get_backend()
            if is_messages_mode and args.get("messages"):
                messages, prompt_text = args.get("messages"), None
            else:
                prompt_text, messages = renderer.render(args), None
            ptoks = (
                _count_tokens(prompt_text, self.model)
                if prompt_text
                else sum(_count_tokens(m.get("content", ""), self.model) for m in (messages or []))
            )
            last_err = None
            for attempt in range(self.retries + 1):
                _kwargs = {
                    "messages": messages,
                    "prompt": prompt_text,
                    "model": self.model,
                    "temperature": self.temperature,
                }
                try:
                    if "seed" in inspect.signature(backend.run).parameters and self.seed is not None:
                        _kwargs["seed"] = self.seed
                except Exception:
                    pass
                result = await backend.run(**_kwargs)
                try:
                    out = Output.validate_python(result)
                    ctoks = _count_tokens(str(result), self.model)
                    if self.state:
                        rec = {
                            "function": func.__name__,
                            "ts": datetime.now().isoformat(),
                            "model": self.model,
                            "backend": os.getenv("LLM_BACKEND", "mock"),
                            "prompt_tokens": ptoks,
                            "completion_tokens": ctoks,
                            "attempt": attempt + 1,
                            "temperature": self.temperature,
                            "seed": self.seed,
                            "backend_details": {
                                "name": os.getenv("LLM_BACKEND", "mock"),
                                "endpoint": getattr(backend, "endpoint", None),
                            },
                        }
                        usd = _usd_cost(self.model, ptoks, ctoks)
                        if usd is not None:
                            rec["usd"] = usd
                        self.state.log_execution(rec)
                        manifest = {
                            "function": func.__name__,
                            "template_hash": hashlib.sha1(template_text.encode()).hexdigest(),
                            "inputs_summary": {
                                k: ("<blob>" if isinstance(v, (bytes, bytearray, ImageBlob)) else v)
                                for k, v in list(args.items())[:8]
                            },
                            "model": self.model,
                            "backend": os.getenv("LLM_BACKEND", "mock"),
                            "prompt_tokens": ptoks,
                            "completion_tokens": ctoks,
                            "attempts": attempt + 1,
                            "validation_error": last_err,
                            "git_commit": os.getenv("GIT_COMMIT"),
                            "temperature": self.temperature,
                            "seed": self.seed,
                            "backend_details": rec["backend_details"],
                            "render": {"prompt": prompt_text}
                            if prompt_text
                            else {"messages": messages},
                            "prompt_source": getattr(func, "_prompt_source", None),
                            "version": __version__,
                            "pricing_version": os.getenv("LLM_PRICING_VERSION"),
                        }
                        if usd is not None:
                            manifest["usd"] = usd
                        self.state.write_manifest(manifest)
                    return out
                except ValidationError as ve:
                    last_err = str(ve)
                    if not self.enable_repair or attempt == self.retries:
                        raise OutputValidationError(ve) from ve
                    schema = getattr(Output, "json_schema", lambda: {})()
                    repair_instruction = textwrap.dedent(
                        f"Please return a response that validates against this JSON schema:\n{json.dumps(schema, indent=2)}\nOnly return the JSON object—no prose."
                    ).strip()
                    if messages is None:
                        messages = [
                            {"role": "user", "content": prompt_text or ""},
                            {"role": "system", "content": repair_instruction},
                        ]
                        prompt_text = None
                    else:
                        messages = messages + [
                            {"role": "system", "content": repair_instruction}
                        ]
                    await asyncio.sleep(0.05)
            raise ExecutionError("Exhausted retries without valid output")

        return async_wrap if asyncio.iscoroutinefunction(func) else sync_wrap

    def _auto_template(self, func, sig):
        base = func.__doc__ or func.__name__.replace("_", " ")
        for p in sig.parameters:
            if f"{{{p}}}" not in base:
                base += f"\n{p}: {{{p}}}"
        return base


_PROMPT_DIR = Path(__file__).with_suffix("").parent / "prompts"
llm = LLMFunction()


def _load_prompt_files():
    if not _PROMPT_DIR.exists():
        return []
    for file in _PROMPT_DIR.iterdir():
        try:
            if file.suffix == ".json":
                yield from json.loads(file.read_text(encoding="utf-8"))
            elif file.suffix in {".yml", ".yaml"} and _yaml_enabled:
                yield from yaml.safe_load(file.read_text(encoding="utf-8"))
            elif file.suffix == ".xml":
                import xml.etree.ElementTree as ET

                root = ET.parse(file).getroot()
                for p in root.findall("prompt"):
                    yield {"name": p.get("name"), "template": (p.text or "").strip()}
        except Exception as e:
            print(f"[prompt-store] skip {file.name}: {e}")


def _make_external_prompt_function(
    name: str,
    tpl: str,
    params: Dict[str, Any],
    ret: Any,
    source_path: Optional[str] = None,
):
    def _fn(**kwargs):
        pass

    _fn.__name__ = name
    _fn.__doc__ = f"External prompt\n{tpl}"
    _fn.__annotations__ = {**params, "return": ret}
    if source_path:
        _fn._prompt_source = source_path
    from inspect import Signature, Parameter

    parameters = [Parameter(k, kind=Parameter.KEYWORD_ONLY) for k in params.keys()]
    _fn.__signature__ = Signature(parameters=parameters)  # type: ignore[attr-defined]
    return llm(template=tpl)(_fn)


for prm in _load_prompt_files():
    try:
        name, tpl = prm["name"], prm["template"]
        params, ret = prm.get("params", {"text": str}), prm.get("returns", str)
        globals()[name] = _make_external_prompt_function(
            name, tpl, params, ret, source_path=str(_PROMPT_DIR)
        )
        print(f"[prompt-store] loaded: {name}")
    except Exception as e:
        print(f"[prompt-store] error for {prm}: {e}")


llm_demo = LLMFunction()


@llm_demo
def demo_loop_and_if(things: List[str], flag: bool) -> str:
    """Demo control flow:
    {% if flag %}Flag true!{% else %}Flag false{% endif %}

    {% for t in things %}- {t}{% endfor %}
    """
    pass


@llm_demo
def chat_demo(messages: List[Dict[str, Any]]) -> str:
    """Message-array mode demo (no template used)."""
    pass
