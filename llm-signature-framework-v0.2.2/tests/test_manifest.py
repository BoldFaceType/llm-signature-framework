import json
from pathlib import Path
from llm_signature_framework.core import LLMFunction, Tool, ToolRegistry, __version__

def test_manifest_and_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_BACKEND", "mock")
    llm = LLMFunction(temperature=0.42, seed=123)

    @llm
    def summarize(text: str) -> str:
        """Summarize: {text}"""
        pass

    out = summarize("hello world")
    assert isinstance(out, str)

    st = json.loads(Path(".llm_state.json").read_text())
    assert st["executions"], "state should have executions"
    rec = st["executions"][-1]
    assert rec["temperature"] == 0.42
    assert rec["backend_details"]["name"] in {"mock","openai","anthropic","hybrid"}
    assert "prompt_tokens" in rec and "completion_tokens" in rec

    run_files = list(Path("runs").glob("run-*.json"))
    assert run_files, "expected a manifest in runs/"
    mf = json.loads(run_files[-1].read_text())
    assert mf["version"] == __version__
    assert "render" in mf and ("prompt" in mf["render"] or "messages" in mf["render"])

def test_tool_logging(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @Tool(desc="add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    import asyncio
    res = asyncio.run(ToolRegistry.call("add", a=2, b=3))
    assert res == 5

    st = json.loads(Path(".llm_state.json").read_text())
    assert any(x for x in st["executions"] if x["function"] == "tool:add")
