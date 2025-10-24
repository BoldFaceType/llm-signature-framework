import asyncio
from types import SimpleNamespace

import pytest

import llm_signature_framework.templates as templates
from llm_signature_framework.templates import LLMFunction


def make_state():
    store = SimpleNamespace(records=[], manifests=[])

    def log_execution(record):
        store.records.append(record)

    def write_manifest(manifest):
        store.manifests.append(manifest)

    store.log_execution = log_execution
    store.write_manifest = write_manifest
    return store


def test_llm_function_success_and_manifest(monkeypatch):
    state = make_state()
    llm = LLMFunction(retries=0, track=True, seed=7)
    llm.state = state

    class DummyBackend:
        def __init__(self):
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            return {"text": "hello"}

    backend = DummyBackend()
    monkeypatch.setattr(templates, "get_backend", lambda: backend)
    monkeypatch.setattr(templates, "_PRICING", {"gpt-4": {"input": 3, "output": 6}})

    @llm
    def greet(name: str) -> dict:
        """Hello {name}"""
        pass

    result = greet("Ada")
    assert result == {"text": "hello"}
    assert backend.calls[0]["prompt"].startswith("Hello Ada")
    assert state.records[0]["seed"] == 7
    assert state.manifests[0]["function"] == "greet"

    with pytest.raises(templates.InputValidationError):
        greet(123)  # type: ignore[arg-type]


def test_llm_function_repair_flow(monkeypatch):
    state = make_state()
    llm = LLMFunction(retries=1, track=True, enable_repair=True)
    llm.state = state

    class RepairBackend:
        def __init__(self):
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return "oops"
            return {"value": 1}

    backend = RepairBackend()
    monkeypatch.setattr(templates, "get_backend", lambda: backend)

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(templates.asyncio, "sleep", fast_sleep)

    @llm
    async def compute(things: list[str]) -> dict:
        """Process {things}"""
        pass

    result = asyncio.run(compute(["a"]))
    assert result == {"value": 1}
    assert len(backend.calls) == 2
    repair_message = backend.calls[1]["messages"][-1]["content"]
    assert "JSON schema" in repair_message
    assert state.records[-1]["attempt"] == 2
