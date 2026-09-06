from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def git_sha(short: int = 7) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", f"--short={short}", "HEAD"],
            cwd=REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def new_run_id() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sha = git_sha()
    return f"{stamp}-{sha}" if sha else stamp


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def write_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_jsonl_index(path: Path, index: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("index", -1)) == index:
                return row
    return None
