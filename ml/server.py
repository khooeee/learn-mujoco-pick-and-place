#!/usr/bin/env python3
"""Local job API for the TwinPick control panel."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from log import read_json, read_jsonl
from replay import render_episode

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OBJECTS = ROOT / "assets" / "objects"
OBJECTS.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_proc: subprocess.Popen | None = None
_run_id: str | None = None
_log: list[str] = []


class TrainReq(BaseModel):
    episodes: int = 200
    seed: int = 0
    resume: bool = False


class RenderReq(BaseModel):
    run_id: str
    episode: int


def _latest_run() -> Path | None:
    if not RUNS.exists():
        return None
    dirs = [p for p in RUNS.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def _status_from_disk() -> dict:
    global _run_id
    run_dir = RUNS / _run_id if _run_id else _latest_run()
    if run_dir is None or not (run_dir / "status.json").exists():
        return {
            "state": "idle",
            "run_id": _run_id,
            "episode": 0,
            "episodes_target": 0,
            "success_rate": 0.0,
            "reward": 0.0,
            "log": "idle",
            "alive": bool(_proc and _proc.poll() is None),
        }
    row = read_json(run_dir / "status.json")
    row["alive"] = bool(_proc and _proc.poll() is None)
    row.setdefault("run_id", run_dir.name)
    return row


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    st = _status_from_disk()
    st["log_tail"] = _log[-40:]
    return st


@app.post("/train/start")
def train_start(req: TrainReq):
    global _proc, _run_id, _log
    if _proc and _proc.poll() is None:
        raise HTTPException(409, "training already running")
    _log = []
    if req.resume:
        latest = _latest_run()
        run_id = _run_id or (latest.name if latest else time.strftime("run-%Y%m%d-%H%M%S"))
        resume = str(RUNS / run_id / "policy.pt")
    else:
        run_id = time.strftime("run-%Y%m%d-%H%M%S")
        resume = ""
    _run_id = run_id
    (RUNS / run_id).mkdir(parents=True, exist_ok=True)
    stop = RUNS / run_id / "STOP"
    if stop.exists():
        stop.unlink()
    cmd = [
        sys.executable,
        str(ROOT / "train_rl.py"),
        "--run",
        run_id,
        "--episodes",
        str(req.episodes),
        "--seed",
        str(req.seed),
    ]
    if resume and Path(resume).exists():
        cmd += ["--resume", resume]
    _proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _reader() -> None:
        assert _proc and _proc.stdout
        for line in _proc.stdout:
            _log.append(line.rstrip())
            del _log[:-200]

    threading.Thread(target=_reader, daemon=True).start()
    return {"run_id": run_id, "state": "running"}


@app.post("/train/stop")
def train_stop():
    global _proc
    run_dir = RUNS / _run_id if _run_id else _latest_run()
    if run_dir:
        (run_dir / "STOP").write_text("1")
    if _proc and _proc.poll() is None:
        _proc.send_signal(signal.SIGINT)
        try:
            _proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _proc.terminate()
    return {"state": "stopping"}


@app.get("/runs")
def runs():
    if not RUNS.exists():
        return []
    out = []
    for p in sorted(RUNS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_dir():
            st = read_json(p / "status.json") if (p / "status.json").exists() else {}
            out.append({"id": p.name, **st})
    return out


@app.get("/runs/{run_id}/episodes")
def episodes(run_id: str):
    path = RUNS / run_id / "episodes.jsonl"
    rows = read_jsonl(path)
    slim = [
        {
            "index": r.get("index"),
            "success": r.get("success"),
            "reward": r.get("reward"),
            "lifted_z": r.get("lifted_z"),
            "has_video": (RUNS / run_id / "videos" / f"ep_{r.get('index')}.mp4").exists(),
        }
        for r in rows
    ]
    return slim


@app.get("/runs/{run_id}/metrics")
def metrics(run_id: str):
    return read_jsonl(RUNS / run_id / "metrics.jsonl")


@app.post("/runs/render")
def render(req: RenderReq):
    dest = RUNS / req.run_id / "videos" / f"ep_{req.episode}.mp4"
    try:
        render_episode(req.run_id, req.episode, dest, view=False)
    except SystemExit as e:
        raise HTTPException(404, str(e)) from e
    return {"url": f"/runs/{req.run_id}/videos/{req.episode}"}


@app.get("/runs/{run_id}/videos/{episode}")
def video(run_id: str, episode: int):
    path = RUNS / run_id / "videos" / f"ep_{episode}.mp4"
    if not path.exists():
        raise HTTPException(404, "render that episode first")
    return FileResponse(path, media_type="video/mp4")


@app.get("/runs/{run_id}/eval")
def latest_eval(run_id: str):
    folder = RUNS / run_id / "videos"
    if not folder.exists():
        raise HTTPException(404, "no videos")
    vids = sorted(folder.glob("ep_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vids:
        raise HTTPException(404, "no videos")
    return FileResponse(vids[0], media_type="video/mp4")


@app.get("/objects")
def objects():
    return [p.name for p in OBJECTS.glob("*") if p.suffix.lower() in {".glb", ".png", ".jpg"}]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("RL_PORT", "8765")))
