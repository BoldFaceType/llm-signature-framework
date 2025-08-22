from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2))


class StateManager:
    def __init__(self, file: str = ".llm_state.json"):
        self.file = Path(file)
        try:
            self.state = json.loads(self.file.read_text(encoding="utf-8"))
        except Exception:
            self.state = {"executions": []}

    def _save(self) -> None:
        _atomic_write_json(self.file, self.state)

    def log_execution(self, record: Dict[str, Any]) -> None:
        self.state["executions"].append(record)
        self._save()

    def write_manifest(self, manifest: Dict[str, Any]) -> Path:
        runs = Path("runs")
        runs.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = runs / f"run-{ts}.json"
        _atomic_write_json(path, manifest)
        return path
