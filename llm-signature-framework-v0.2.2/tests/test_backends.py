import os
from llm_signature_framework.backends import get_backend, set_backend, MockBackend


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
