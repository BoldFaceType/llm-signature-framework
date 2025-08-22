from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .backends import HybridBackend, set_backend
from .tools import ToolRegistry
from .templates import demo_loop_and_if, __version__


class _Studio(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/index.html"):
            self._html()
        elif self.path == "/state":
            self._json(Path(".llm_state.json"))
        elif self.path.startswith("/runs"):
            p = Path(self.path.lstrip("/"))
            if p.is_dir():
                self._send(
                    200,
                    "application/json",
                    json.dumps(sorted([f.name for f in p.glob("*.json")])),
                )
            else:
                self._json(p)
        else:
            self.send_error(404)

    def _send(self, code, ctype, body):
        b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _json(self, path: Path):
        try:
            self._send(200, "application/json", path.read_text(encoding="utf-8"))
        except Exception:
            self.send_error(404)

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
    `<p>Runs: ${runs.map(r=>`+"`"+"<a href='/runs/"+"${r}"+"' target='_blank'>"+"${r}"+"</a>"+"`").join(' | ')}.</p>`;
}
load();
</script>
"""
        self._send(200, "text/html; charset=utf-8", body)


def run_prompt_studio(host: str = "127.0.0.1", port: int = 8000):
    httpd = HTTPServer((host, port), _Studio)
    print(f"Prompt Studio running at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Prompt Studio…")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prompt Framework CLI")
    ap.add_argument(
        "cmd",
        nargs="?",
        choices=["studio", "tools", "run", "version"],
        help="Run UI, list tools, run a prompt, or print version",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", choices=["mock", "openai", "anthropic", "hybrid"])
    ap.add_argument("--endpoint")
    ap.add_argument("--api-key")
    ap.add_argument("--name")
    ap.add_argument("--json")
    ap.add_argument("--print-version", action="store_true")
    args = ap.parse_args()

    if args.print_version or args.cmd == "version":
        print(__version__)
        sys.exit(0)

    if args.backend:
        if args.backend == "hybrid":
            set_backend(HybridBackend(endpoint=args.endpoint, api_key=args.api_key))
        else:
            os.environ["LLM_BACKEND"] = args.backend

    if args.cmd == "studio":
        run_prompt_studio(args.host, args.port)
    elif args.cmd == "tools":
        print(json.dumps(ToolRegistry.list_tools(), indent=2))
    elif args.cmd == "run":
        if not args.name:
            raise SystemExit("--name required for 'run'")
        import llm_signature_framework.templates as tpl

        fn = getattr(tpl, args.name, None)
        if not callable(fn):
            raise SystemExit(f"No such prompt function: {args.name}")
        kwargs = json.loads(args.json or "{}")
        out = fn(**kwargs)
        print(json.dumps(out, indent=2) if isinstance(out, (dict, list)) else str(out))
    else:
        print(demo_loop_and_if(["alpha", "beta"], True))


if __name__ == "__main__":
    main()
