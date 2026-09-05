#!/usr/bin/env python3
"""Local job API for the TwinPick control panel."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from log import read_json, read_jsonl
from library import glb_path, import_glb, list_objects, object_dir, selected_id, set_selected, stl_path
from mint import api_key, generate_from_prompt, load_dotenv
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
_mint_lock = threading.Lock()
_mint: dict = {"busy": False, "prompt": "", "error": None, "last": None}


class TrainReq(BaseModel):
    episodes: int = 1000
    seed: int = 0
    resume: bool = False


class MintReq(BaseModel):
    prompt: str


class SelectReq(BaseModel):
    id: str


class RenderReq(BaseModel):
    run_id: str
    episode: int


def _latest_run() -> Path | None:
    if not RUNS.exists():
        return None
    dirs = [p for p in RUNS.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def _safe_run(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise HTTPException(400, "bad id")
    run_dir = (RUNS / run_id).resolve()
    if run_dir.parent != RUNS.resolve() or not run_dir.is_dir():
        raise HTTPException(404, "unknown run")
    return run_dir


def _reveal(path: Path) -> None:
    path = path.resolve()
    if sys.platform == "darwin":
        cmd = ["open", "-R", str(path)] if path.is_file() else ["open", str(path)]
    elif sys.platform == "win32":
        cmd = ["explorer", f"/select,{path}"] if path.is_file() else ["explorer", str(path)]
    else:
        cmd = ["xdg-open", str(path.parent if path.is_file() else path)]
    subprocess.Popen(cmd, start_new_session=True)


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


def _mint_snapshot() -> dict:
    with _mint_lock:
        return {
            "busy": bool(_mint["busy"]),
            "prompt": _mint["prompt"] or "",
            "error": _mint["error"],
            "last": _mint["last"],
        }


def _mint_job(prompt: str) -> None:
    try:
        meta = generate_from_prompt(prompt)
        with _mint_lock:
            _mint["busy"] = False
            _mint["error"] = None
            _mint["last"] = meta
    except Exception as e:
        with _mint_lock:
            _mint["busy"] = False
            _mint["error"] = str(e)
            _mint["last"] = None


@app.get("/health")
def health():
    load_dotenv()
    job = _mint_snapshot()
    return {
        "ok": True,
        "mint": bool(api_key()),
        "mint_busy": job["busy"],
        "mint_prompt": job["prompt"],
        "mint_error": job["error"],
        "mint_last": job["last"],
    }


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
    if not list_objects():
        raise HTTPException(400, "Import or generate an object first")
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
        if not p.is_dir():
            continue
        st = read_json(p / "status.json") if (p / "status.json").exists() else {}
        eps_path = p / "episodes.jsonl"
        n_eps = 0
        if eps_path.exists():
            n_eps = sum(1 for line in eps_path.read_text().splitlines() if line.strip())
        vid_dir = p / "videos"
        n_vid = len(list(vid_dir.glob("ep_*.mp4"))) if vid_dir.exists() else 0
        out.append(
            {
                "id": p.name,
                "state": st.get("state") or "idle",
                "episode": int(st.get("episode") or n_eps),
                "episodes_target": int(st.get("episodes_target") or 0),
                "success_rate": float(st.get("success_rate") or 0.0),
                "reward": float(st.get("reward") or 0.0),
                "episodes": n_eps,
                "videos": n_vid,
                "has_policy": (p / "policy.pt").exists(),
                "mtime": p.stat().st_mtime,
            }
        )
    return out


@app.post("/runs/{run_id}/open")
def open_run_folder(run_id: str):
    run_dir = _safe_run(run_id)
    try:
        _reveal(run_dir)
    except OSError as e:
        raise HTTPException(500, f"could not open folder: {e}") from e
    return {"path": str(run_dir)}


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


@app.post("/runs/{run_id}/episodes/{episode}/open")
def open_episode_folder(run_id: str, episode: int):
    run_dir = _safe_run(run_id)
    video = run_dir / "videos" / f"ep_{episode}.mp4"
    target = video if video.is_file() else run_dir
    try:
        _reveal(target)
    except OSError as e:
        raise HTTPException(500, f"could not open folder: {e}") from e
    return {"path": str(target)}


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
    return {"selected": selected_id(), "items": list_objects()}


@app.get("/objects/{oid}/mesh")
def object_mesh(oid: str):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", oid):
        raise HTTPException(400, "bad id")
    path = stl_path(oid)
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "no mesh")
    if path.resolve().parent != object_dir(oid).resolve():
        raise HTTPException(400, "bad id")
    return FileResponse(path, media_type="model/stl", filename="object.stl")


@app.get("/objects/{oid}/source.glb")
def object_glb(oid: str):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", oid):
        raise HTTPException(400, "bad id")
    path = glb_path(oid)
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "no glb")
    if path.resolve().parent != object_dir(oid).resolve():
        raise HTTPException(400, "bad id")
    return FileResponse(path, media_type="model/gltf-binary", filename="source.glb")


@app.post("/objects/import")
async def objects_import(file: UploadFile = File(...), name: str | None = Form(None)):
    fname = file.filename or "object.glb"
    if not fname.lower().endswith(".glb"):
        raise HTTPException(400, "Upload a .glb file")
    raw = await file.read()
    if len(raw) > 80_000_000:
        raise HTTPException(400, "File too large (80 MB max)")
    if len(raw) < 20:
        raise HTTPException(400, "Empty file")
    label = (name or "").strip() or Path(fname).stem
    try:
        return import_glb(raw, label)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/objects/select")
def objects_select(req: SelectReq):
    try:
        set_selected(req.id)
    except FileNotFoundError as e:
        raise HTTPException(404, f"unknown object {e}") from e
    return {"selected": selected_id(), "items": list_objects()}


@app.post("/mint/generate")
def mint_generate(req: MintReq):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "Enter a prompt")
    if not api_key():
        raise HTTPException(400, "Set MINT_API_KEY in .env (https://platform.mint.gg)")
    with _mint_lock:
        if _mint["busy"]:
            raise HTTPException(409, "Mint is already generating")
        _mint["busy"] = True
        _mint["prompt"] = prompt
        _mint["error"] = None
        _mint["last"] = None
    threading.Thread(target=_mint_job, args=(prompt,), daemon=True).start()
    return {"ok": True, "busy": True, "prompt": prompt}


if __name__ == "__main__":
    import uvicorn

    load_dotenv()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("RL_PORT", "8765")))
