import os
import subprocess
import sys
from pathlib import Path

from llm_signature_framework.templates import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(PROJECT_ROOT / "src")


def run_cli(*args):
    env = os.environ.copy()
    env.update({"PYTHONPATH": SRC_PATH, "LLM_BACKEND": "mock"})
    cmd = [sys.executable, "-m", "llm_signature_framework.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def test_cli_print_version():
    res = run_cli("--print-version")
    assert res.stdout.strip() == __version__


def test_cli_run_demo():
    res = run_cli("run", "--name", "demo_loop_and_if", "--json", '{"things":["x"],"flag": true}')
    assert "mock-reply" in res.stdout
    assert res.returncode == 0
