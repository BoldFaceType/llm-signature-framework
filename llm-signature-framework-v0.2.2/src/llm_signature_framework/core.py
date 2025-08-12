#!/usr/bin/env python3
"""
Signature-Driven Prompt Framework — Vetting-Ready Edition (v0.2.2)
"""
from __future__ import annotations

import argparse, asyncio, contextlib, functools, hashlib, inspect, json, os, platform, random, re, textwrap, sys, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, conbytes

__version__ = "0.2.2"

# optional extras
_tiktoken_enabled = False
try:
    import tiktoken  # type: ignore
    _tiktoken_enabled = True
except Exception:
    pass

_yaml_enabled = False
try:
    import yaml  # type: ignore
    _yaml_enabled = True
except Exception:
    pass

# pricing via env JSON
try:
    _PRICING = json.loads(os.getenv("LLM_PRICING_JSON", "{}"))
except Exception:
    _PRICING = {}

def _usd_cost(model: str, prompt_toks: int, completion_toks: int) -> Optional[float]:
    p = _PRICING.get(model)
    if not p:
        return None
    return round((prompt_toks * float(p.get("input", 0.0)) + completion_toks * float(p.get("output", 0.0))) / 1000.0, 6)

# token counting
def _count_tokens(text: str, model: str = "gpt-4") -> int:
    if _tiktoken_enabled:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, int(len(text.split()) * 0.75))

# atomic writes
@contextlib.contextmanager
def _locked_file(path: Path, mode="r+"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with open(path, mode, encoding="utf-8") as fh:
        yield fh

def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def _atomic_write_json(path: Path, obj: Any):
    _atomic_write_text(path, json.dumps(obj, indent=2))

# errors
class ToolError(Exception): fatal: bool = False
class InputValidationError(ToolError): ...
class OutputValidationError(ToolError): ...
class ExecutionError(ToolError): ...
class FatalToolError(ToolError): fatal = True

# multimodal
_SAFE_MEDIA_ROOT = os.getenv("SAFE_MEDIA_ROOT")

class ImageBlob(BaseModel):
    mode: str = Field("bytes", regex=r"^(bytes|b64|path|url)$")
    data: Union[conbytes(strict=True), str, Path]
    mime: str = "image/png"
    def to_llm_part(self) -> str:
        from base64 import b64encode
        if self.mode == "bytes": return f"data:{self.mime};base64,{b64encode(self.data).decode()}"
        if self.mode == "b64": return f"data:{self.mime};base64,{self.data}"
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

# mini template
class _MiniTemplate:
    _var_rx = re.compile(r"{(\w+)}")
    _tag_rx = re.compile(r"{%(.*?)%}", re.S)
    _endfor_rx = re.compile(r"{%-?\s*endfor\s*-?%}")
    _endif_rx = re.compile(r"{%-?\s*endif\s*-?%}")
    _elif_else_rx = re.compile(r"{%-?\s*(elif|else)(.*?)%}")
    def __init__(self, text: str): self.text = text
    def render(self, ctx: Dict[str, Any]) -> str: return self._render_block(self.text, ctx)
    def _render_block(self, block: str, ctx: Dict[str, Any]) -> str:
        out: List[str] = []; i = 0
        while i < len(block):
            m = self._tag_rx.search(block, i)
            if not m: out.append(self._vars(block[i:], ctx)); break
            start, end = m.span(); out.append(self._vars(block[i:start], ctx)); tag = m.group(1).strip(); i = end
            if tag.startswith("for"): txt, i = self._handle_for(block, i, tag, ctx); out.append(txt)
            elif tag.startswith("if"): txt, i = self._handle_if(block, i, tag, ctx); out.append(txt)
            else: out.append(m.group(0))
        return "".join(out)
    def _vars(self, s: str, ctx: Dict[str, Any]) -> str:
        def sub(mm): v = ctx.get(mm.group(1), ""); return (v.to_llm_part() if isinstance(v, ImageBlob) else str(v))
        return self._var_rx.sub(sub, s)
    def _handle_for(self, text: str, pos: int, tag: str, ctx: Dict[str, Any]):
        try: _, var, _, it_name = tag.split()
        except ValueError: raise ValueError("Malformed {% for %} tag")
        end = self._endfor_rx.search(text, pos)
        if not end: raise ValueError("Unclosed {% for %} block")
        body = text[pos:end.start()]; it = ctx.get(it_name, []) or []
        return "".join(self._render_block(body, {**ctx, var: x}) for x in it), end.end()
    def _handle_if(self, text: str, pos: int, tag: str, ctx: Dict[str, Any]):
        cond_var = tag.split()[1]; end = self._endif_rx.search(text, pos)
        if not end: raise ValueError("Unclosed {% if %} block")
        block = text[pos:end.start()]; parts, last = [], 0
        for m in self._elif_else_rx.finditer(block):
            parts.append(("body", block[last:m.start()])); parts.append((m.group(1).strip(), m.group(2).strip())); last = m.end()
        parts.append(("body", block[last:])); active = bool(ctx.get(cond_var)); idx = 0
        while idx < len(parts):
            typ, content = parts[idx]
            if typ == "body" and active: return self._render_block(content, ctx), end.end()
            if typ == "elif": var = content.split()[0]; active = bool(ctx.get(var))
            if typ == "else": active = True
            idx += 1
        return "", end.end()

# state
class StateManager:
    def __init__(self, file: str = ".llm_state.json"):
        self.file = Path(file)
        try: self.state = json.loads(self.file.read_text(encoding="utf-8"))
        except Exception: self.state = {"executions": []}
    def _save(self): _atomic_write_json(self.file, self.state)
    def log_execution(self, record: Dict[str, Any]): self.state["executions"].append(record); self._save()
    def write_manifest(self, manifest: Dict[str, Any]) -> Path:
        runs = Path("runs"); runs.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = runs / f"run-{ts}.json"
        _atomic_write_json(path, manifest); return path

# backends
class Backend:
    async def run(self, *, messages: Optional[List[Dict[str, Any]]] = None,
                  prompt: Optional[str] = None, model: str = "gpt-4",
                  temperature: float = 0.7, seed: Optional[int] = None) -> Any: raise NotImplementedError

class MockBackend(Backend):
    async def run(self, *, messages=None, prompt=None, model="gpt-4", temperature=0.7, seed: Optional[int] = None) -> Any:
        if messages:
            last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            return (last or {}).get("content", "mock-reply") or "mock-reply"
        return "mock-reply"

class OpenAIBackend(Backend):
    def __init__(self):
        try: import openai  # type: ignore
        except Exception as e: raise FatalToolError("OpenAI backend requested but openai not installed") from e
        self._openai = openai
    async def run(self, *, messages=None, prompt=None, model="gpt-4", temperature=0.7, seed: Optional[int] = None):
        client = self._openai.OpenAI()
        if messages is None: messages = [{"role":"user","content": prompt or ""}]
        resp = await asyncio.to_thread(lambda: client.chat.completions.create(model=model, temperature=temperature, messages=messages))
        return resp.choices[0].message.content

class AnthropicBackend(Backend):
    def __init__(self):
        try: import anthropic  # type: ignore
        except Exception as e: raise FatalToolError("Anthropic backend requested but anthropic not installed") from e
        self._anth = anthropic
    async def run(self, *, messages=None, prompt=None, model="claude-3-opus-20240229", temperature=0.7, seed: Optional[int] = None):
        client = self._anth.Anthropic()
        text = prompt
        if messages: text = "\n\n".join(m.get("content","") for m in messages if m.get("role")=="user")
        resp = await asyncio.to_thread(lambda: client.messages.create(model=model, max_tokens=1024, temperature=temperature, messages=[{"role":"user","content": text or ""}]))
        return resp.content[0].text

class HybridBackend(Backend):
    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None, headers: Optional[Dict[str,str]] = None):
        self.endpoint = endpoint or os.getenv("HYBRID_BACKEND_URL") or os.getenv("HYBRID_ENDPOINT")
        self.api_key = api_key or os.getenv("HYBRID_API_KEY") or os.getenv("AGENT_API_KEY")
        self.headers = headers or {}
    async def run(self, *, messages=None, prompt=None, model="gpt-4", temperature=0.7, seed: Optional[int] = None):
        if not self.endpoint: raise FatalToolError("Hybrid backend requires HYBRID_BACKEND_URL")
        import urllib.request
        payload = {"model": model, "temperature": temperature}
        if messages is not None: payload["messages"] = messages
        if prompt is not None: payload["prompt"] = prompt
        if seed is not None: payload["seed"] = seed
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, method="POST"); req.add_header("Content-Type","application/json")
        if self.api_key: req.add_header("Authorization", f"Bearer {self.api_key}")
        for k,v in self.headers.items(): req.add_header(k, v)
        def _do(): 
            with urllib.request.urlopen(req, timeout=20) as resp:
                out = json.loads(resp.read().decode("utf-8", errors="ignore"))
                return out.get("content") or out.get("text") or out
        return await asyncio.to_thread(_do)

_current_backend: Optional[Backend] = None
def set_backend(backend: Optional[Backend]): 
    global _current_backend; _current_backend = backend
def get_backend() -> Backend:
    if _current_backend is not None: return _current_backend
    name = (os.getenv("LLM_BACKEND") or "mock").lower()
    if name == "openai": return OpenAIBackend()
    if name == "anthropic": return AnthropicBackend()
    if name == "hybrid": return HybridBackend()
    return MockBackend()

# tools
class Tool:
    def __init__(self, name: Optional[str] = None, desc: Optional[str] = None, retries: int = 2, backoff: float = 0.4):
        self.name, self.desc, self.retries, self.backoff = name, desc, retries, backoff
    def __call__(self, func):
        sig, hints = inspect.signature(func), get_type_hints(func)
        class _Input(BaseModel): __annotations__ = {p.name: hints.get(p.name, Any) for p in sig.parameters.values()}
        Output = TypeAdapter(hints.get("return", Any))
        @functools.wraps(func)
        async def wrapper(**kwargs):
            try: parsed = _Input(**kwargs).model_dump()
            except ValidationError as e: raise InputValidationError(e) from e
            delay = self.backoff
            for attempt in range(self.retries + 1):
                try: return Output.validate_python(await _maybe_await(func)(**parsed))
                except ValidationError as e: raise OutputValidationError(e) from e
                except FatalToolError: raise
                except Exception as e:
                    if attempt == self.retries: raise ExecutionError(e) from e
                    await asyncio.sleep(delay + random.uniform(0, delay)); delay *= 2
        wrapper._tool_meta = {"name": self.name or func.__name__, "description": (self.desc or (func.__doc__ or "")).strip(), "parameters": _Input.model_json_schema()}
        ToolRegistry.register(wrapper); return wrapper

def _maybe_await(fn):
    if asyncio.iscoroutinefunction(fn): return fn
    async def _run(**kw): return await asyncio.to_thread(fn, **kw)
    return _run

class ToolRegistry:
    _reg: Dict[str, Any] = {}
    _state = StateManager()
    @classmethod
    def register(cls, tool_fn): cls._reg[tool_fn._tool_meta["name"]] = tool_fn
    @classmethod
    async def call(cls, name: str, **kwargs):
        if name not in cls._reg: raise FatalToolError(f"Unknown tool '{name}'")
        start = time.perf_counter(); ok = True; err = None
        try: return await cls._reg[name](**kwargs)
        except Exception as e: ok, err = False, str(e); raise
        finally:
            try:
                cls._state.log_execution({
                    "function": f"tool:{name}", "ts": datetime.now().isoformat(),
                    "ok": ok, "error": err,
                    "args": {k: ("<blob>" if isinstance(v,(bytes,bytearray,ImageBlob)) else v) for k, v in list(kwargs.items())[:12]},
                    "duration_s": round(time.perf_counter() - start, 6),
                })
            except Exception: pass
    @classmethod
    def list_tools(cls) -> List[Dict[str, Any]]: return [t._tool_meta for t in cls._reg.values()]

def list_tools_for_planner() -> List[Dict[str, Any]]: return ToolRegistry.list_tools()
def list_tools_openai() -> List[Dict[str, Any]]:
    return [{"type":"function","function":{"name":m["name"],"description":m.get("description",""),"parameters":m["parameters"]}} for m in ToolRegistry.list_tools()]
async def call_tool(name: str, arguments: Dict[str, Any]): return await ToolRegistry.call(name, **arguments)

# llm function
class LLMFunction:
    def __init__(self, *, template: Optional[str] = None, model: str = "gpt-4", temperature: float = 0.7, retries: int = 1, track: bool = True, enable_repair: bool = True, seed: Optional[int] = None):
        self.template, self.model, self.temperature, self.retries, self.track, self.enable_repair = template, model, temperature, max(0, retries), track, enable_repair
        self.seed = seed; self.state = StateManager() if track else None
    def __call__(self, func):
        sig, hints = inspect.signature(func), get_type_hints(func)
        template_text = self.template or self._auto_template(func, sig); renderer = _MiniTemplate(template_text)
        class _Input(BaseModel): __annotations__ = {n: hints.get(n, Any) for n in sig.parameters}
        Output = TypeAdapter(hints.get("return", Any)); is_messages_mode = "messages" in sig.parameters
        @functools.wraps(func)
        def sync_wrap(*a, **k): return asyncio.run(async_wrap(*a, **k))
        @functools.wraps(func)
        async def async_wrap(*a, **k):
            bound = sig.bind(*a, **k); bound.apply_defaults()
            try: args = _Input.model_validate(bound.arguments).model_dump()
            except ValidationError as e: raise InputValidationError(e) from e
            backend = get_backend()
            if is_messages_mode and args.get("messages"): messages, prompt_text = args.get("messages"), None
            else: prompt_text, messages = renderer.render(args), None
            ptoks = _count_tokens(prompt_text, self.model) if prompt_text else sum(_count_tokens(m.get("content",""), self.model) for m in (messages or []))
            last_err = None
            for attempt in range(self.retries + 1):
                _kwargs = {"messages": messages, "prompt": prompt_text, "model": self.model, "temperature": self.temperature}
                try:
                    if "seed" in inspect.signature(backend.run).parameters and self.seed is not None:
                        _kwargs["seed"] = self.seed
                except Exception: pass
                result = await backend.run(**_kwargs)
                try:
                    out = Output.validate_python(result); ctoks = _count_tokens(str(result), self.model)
                    if self.state:
                        rec = {
                            "function": func.__name__, "ts": datetime.now().isoformat(),
                            "model": self.model, "backend": os.getenv("LLM_BACKEND","mock"),
                            "prompt_tokens": ptoks, "completion_tokens": ctoks, "attempt": attempt+1,
                            "temperature": self.temperature, "seed": self.seed,
                            "backend_details": {"name": os.getenv("LLM_BACKEND","mock"), "endpoint": getattr(backend, "endpoint", None)},
                        }
                        usd = _usd_cost(self.model, ptoks, ctoks)
                        if usd is not None: rec["usd"] = usd
                        self.state.log_execution(rec)
                        manifest = {
                            "function": func.__name__, "template_hash": hashlib.sha1(template_text.encode()).hexdigest(),
                            "inputs_summary": {k: ("<blob>" if isinstance(v,(bytes,bytearray,ImageBlob)) else v) for k, v in list(args.items())[:8]},
                            "model": self.model, "backend": os.getenv("LLM_BACKEND","mock"),
                            "prompt_tokens": ptoks, "completion_tokens": ctoks, "attempts": attempt+1,
                            "validation_error": last_err, "git_commit": os.getenv("GIT_COMMIT"),
                            "temperature": self.temperature, "seed": self.seed,
                            "backend_details": rec["backend_details"],
                            "render": {"prompt": prompt_text} if prompt_text else {"messages": messages},
                            "prompt_source": getattr(func, "_prompt_source", None),
                            "version": __version__, "pricing_version": os.getenv("LLM_PRICING_VERSION"),
                        }
                        if usd is not None: manifest["usd"] = usd
                        self.state.write_manifest(manifest)
                    return out
                except ValidationError as ve:
                    last_err = str(ve)
                    if not self.enable_repair or attempt == self.retries: raise OutputValidationError(ve) from ve
                    schema = getattr(Output, "json_schema", lambda: {})()
                    repair_instruction = textwrap.dedent(f"Please return a response that validates against this JSON schema:\n{json.dumps(schema, indent=2)}\nOnly return the JSON object—no prose.").strip()
                    if messages is None:
                        messages = [{"role": "user", "content": prompt_text or ""}, {"role": "system", "content": repair_instruction}]; prompt_text = None
                    else:
                        messages = messages + [{"role":"system","content": repair_instruction}]
                    await asyncio.sleep(0.05)
            raise ExecutionError("Exhausted retries without valid output")
        return async_wrap if asyncio.iscoroutinefunction(func) else sync_wrap
    def _auto_template(self, func, sig):
        base = func.__doc__ or func.__name__.replace("_"," ")
        for p in sig.parameters:
            if f"{{{p}}}" not in base: base += f"\n{p}: {{{p}}}"
        return base

# external prompt store
_PROMPT_DIR = Path(__file__).with_suffix("").parent / "prompts"
llm = LLMFunction()
def _load_prompt_files():
    if not _PROMPT_DIR.exists(): return []
    for file in _PROMPT_DIR.iterdir():
        try:
            if file.suffix == ".json": yield from json.loads(file.read_text(encoding="utf-8"))
            elif file.suffix in {".yml",".yaml"} and _yaml_enabled: yield from yaml.safe_load(file.read_text(encoding="utf-8"))
            elif file.suffix == ".xml":
                import xml.etree.ElementTree as ET
                root = ET.parse(file).getroot()
                for p in root.findall("prompt"):
                    yield {"name": p.get("name"), "template": (p.text or "").strip()}
        except Exception as e:
            print(f"[prompt-store] skip {file.name}: {e}")
def _make_external_prompt_function(name: str, tpl: str, params: Dict[str, Any], ret: Any, source_path: Optional[str] = None):
    def _fn(**kwargs): pass
    _fn.__name__ = name
    _fn.__doc__ = f"External prompt\n{tpl}"
    _fn.__annotations__ = {**params, "return": ret}
    if source_path: _fn._prompt_source = source_path
    from inspect import Signature, Parameter
    parameters = [Parameter(k, kind=Parameter.KEYWORD_ONLY) for k in params.keys()]
    _fn.__signature__ = Signature(parameters=parameters)  # type: ignore[attr-defined]
    return llm(template=tpl)(_fn)
for prm in _load_prompt_files():
    try:
        name, tpl = prm["name"], prm["template"]
        params, ret = prm.get("params", {"text": str}), prm.get("returns", str)
        globals()[name] = _make_external_prompt_function(name, tpl, params, ret, source_path=str(_PROMPT_DIR))
        print(f"[prompt-store] loaded: {name}")
    except Exception as e:
        print(f"[prompt-store] error for {prm}: {e}")

# stdlib tool
@Tool(name="fetch_url", desc="Fetch a URL and return plain text (best-effort)")
def fetch_url(url: str, timeout: float = 6.0, max_bytes: int = 2_000_000) -> str:
    import urllib.request, urllib.error, html.parser
    from urllib.parse import urlparse
    allow = os.getenv("SAFE_FETCH_ALLOWLIST")
    if allow:
        allowed = {d.strip().lower() for d in allow.split(',') if d.strip()}
        host = (urlparse(url).hostname or "").lower()
        if host not in allowed: raise FatalToolError(f"Domain '{host}' not in SAFE_FETCH_ALLOWLIST")
    class _TextExtractor(html.parser.HTMLParser):
        def __init__(self): super().__init__(); self.parts: List[str] = []
        def handle_data(self, data): data = data.strip(); self.parts.append(data) if data else None
        def get_text(self): return " ".join(self.parts)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            total = 0; chunks = []
            while True:
                chunk = resp.read(min(65536, max_bytes - total))
                if not chunk: break
                chunks.append(chunk); total += len(chunk)
                if total >= max_bytes: break
            html = b"".join(chunks).decode(charset, errors="ignore")
            p = _TextExtractor(); p.feed(html); return p.get_text()
    except urllib.error.URLError as e: raise ExecutionError(e)

# studio
class _Studio(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/index.html"): self._html()
        elif self.path == "/state": self._json(Path(".llm_state.json"))
        elif self.path.startswith("/runs"):
            p = Path(self.path.lstrip("/"))
            if p.is_dir(): self._send(200, "application/json", json.dumps(sorted([f.name for f in p.glob("*.json")])))
            else: self._json(p)
        else: self.send_error(404)
    def _send(self, code, ctype, body):
        b = body.encode("utf-8"); self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def _json(self, path: Path):
        try: self._send(200, "application/json", path.read_text(encoding="utf-8"))
        except Exception: self.send_error(404)
    def _html(self):
        body = """<!doctype html>
<title>Prompt Studio (Read-only)</title>
<style>body{font:14px system-ui;margin:1rem} pre{white-space:pre-wrap}</style>
<h1>Prompt Studio</h1>
<p><button onclick="load()">Refresh</button></p>
<div id="summary"></div>
<script>
async function load(){
  const st = await (await fetch('/state')).json().catch(()=>({}));
  const runs = await (await fetch('/runs')).json().catch(()=>[]);
  const total = st.executions?.length||0;
  const tokens = st.executions?.reduce((a,e)=>a+(e.prompt_tokens||0)+(e.completion_tokens||0),0)||0;
  document.getElementById('summary').innerHTML = `<p>Total execs: ${total}. Tokens: ${tokens}.</p>`+
    `<p>Runs: ${runs.map(r=>` + "`" + '<a href="/runs/' + "${r}" + '" target="_blank">' + "${r}" + '</a>' + "`" + ").join(' | ')}</p>`;
}
load();
</script>
"""
        self._send(200, "text/html; charset=utf-8", body)

def run_prompt_studio(host: str = "127.0.0.1", port: int = 8000):
    httpd = HTTPServer((host, port), _Studio); print(f"Prompt Studio running at http://{host}:{port}")
    try: httpd.serve_forever()
    except KeyboardInterrupt: print("Stopping Prompt Studio…")

# examples
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

# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prompt Framework CLI")
    ap.add_argument("cmd", nargs="?", choices=["studio","tools","run","version"], help="Run UI, list tools, run a prompt, or print version")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", choices=["mock","openai","anthropic","hybrid"])
    ap.add_argument("--endpoint"); ap.add_argument("--api-key")
    ap.add_argument("--name"); ap.add_argument("--json")
    ap.add_argument("--print-version", action="store_true")
    args = ap.parse_args()

    if args.print_version or args.cmd == "version": print(__version__); sys.exit(0)
    if args.backend:
        if args.backend == "hybrid": set_backend(HybridBackend(endpoint=args.endpoint, api_key=args.api_key))
        else: os.environ["LLM_BACKEND"] = args.backend

    if args.cmd == "studio": run_prompt_studio(args.host, args.port)
    elif args.cmd == "tools": print(json.dumps(ToolRegistry.list_tools(), indent=2))
    elif args.cmd == "run":
        if not args.name: raise SystemExit("--name required for 'run'")
        fn = globals().get(args.name); 
        if not callable(fn): raise SystemExit(f"No such prompt function: {args.name}")
        kwargs = json.loads(args.json or "{}"); out = fn(**kwargs)
        print(json.dumps(out, indent=2) if isinstance(out, (dict, list)) else str(out))
    else:
        print(demo_loop_and_if(["alpha","beta"], True))
