import sys

import json
import pytest

from llm_signature_framework import cli
from llm_signature_framework.templates import __version__


def run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args):
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setattr(sys, "argv", ["llm-signature-framework", *args])
    exit_code = 0
    try:
        cli.main()
    except SystemExit as exc:  # argparse can exit
        exit_code = exc.code or 0
    result = capsys.readouterr()
    return exit_code, result.out, result.err


def test_cli_print_version(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    exit_code, stdout, stderr = run_cli(monkeypatch, capsys, "--print-version")
    assert exit_code == 0
    assert stdout.strip() == __version__
    assert stderr == ""


def test_cli_run_demo(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    exit_code, stdout, stderr = run_cli(
        monkeypatch,
        capsys,
        "run",
        "--name",
        "demo_loop_and_if",
        "--json",
        '{"things":["x"],"flag": true}',
    )
    assert exit_code == 0
    assert "mock-reply" in stdout
    assert stderr == ""


def test_cli_list_tools(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    exit_code, stdout, stderr = run_cli(monkeypatch, capsys, "tools")
    assert exit_code == 0
    tools = json.loads(stdout)
    assert isinstance(tools, list)
    assert any(t["name"] == "adder" for t in tools)
    assert stderr == ""


def test_cli_run_missing_name(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    exit_code, stdout, stderr = run_cli(monkeypatch, capsys, "run")
    assert "--name required" in str(exit_code)
    assert stdout == ""
    assert stderr == ""


def test_cli_backend_hybrid(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    import llm_signature_framework.backends as backends

    stub_backend = backends.MockBackend()

    monkeypatch.setenv("HYBRID_BACKEND_URL", "https://example.com")
    monkeypatch.setattr(backends, "HybridBackend", lambda **_: stub_backend)
    monkeypatch.setattr(cli, "HybridBackend", lambda **_: stub_backend)

    exit_code, stdout, stderr = run_cli(monkeypatch, capsys, "--backend", "hybrid")
    assert exit_code == 0
    assert "mock-reply" in stdout
    assert stderr == ""
    backends.set_backend(None)
