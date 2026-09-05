"""Mint.gg text-to-3D → rescaled STL for the SO-101 table."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from library import object_dir, rescale_glb, save_object

ROOT = Path(__file__).resolve().parent
API = "https://api.mint.gg/v1"


def load_dotenv() -> None:
    for path in (ROOT.parent / ".env", ROOT / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'").strip('"'))


def api_key() -> str | None:
    load_dotenv()
    key = os.environ.get("MINT_API_KEY", "").strip()
    return key or None


def _req(method: str, path: str, body: dict | None = None) -> dict:
    key = api_key()
    if not key:
        raise RuntimeError("Set MINT_API_KEY in .env (https://platform.mint.gg)")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            raw = res.read()
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Mint {e.code}: {err[:400]}") from e
    if not raw:
        return {}
    return json.loads(raw)


def _poll(op_id: str, timeout_s: float = 1800) -> dict:
    wait = 2.0
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        op = _req("GET", f"/operations/{op_id}")
        status = (op.get("status") or op.get("state") or "").lower()
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(op.get("error") or op.get("message") or "Mint generation failed")
        if status in {"billing_required"}:
            billing = op.get("billing") or {}
            url = billing.get("actionUrl") or "https://platform.mint.gg"
            raise RuntimeError(f"Mint needs credits: {url}")
        if status in {"succeeded", "success", "complete", "completed", "partially_succeeded"}:
            return op
        if op.get("resource", {}).get("id") and status in {"", "done"}:
            return op
        time.sleep(wait)
        wait = min(15.0, wait * 1.6)
    raise RuntimeError("Mint timed out")


def _glb_url(model: dict) -> str:
    assets = model.get("assets") or {}
    for key in ("glbUrl", "optimizedGlbUrl", "glb_url"):
        if assets.get(key):
            return str(assets[key])
        if model.get(key):
            return str(model[key])
    files = model.get("files") or []
    for f in files:
        url = f.get("url") or ""
        fmt = (f.get("format") or f.get("type") or "").lower()
        if "glb" in fmt or url.lower().endswith(".glb"):
            return url
    if files and files[0].get("url"):
        return str(files[0]["url"])
    try:
        manifest = _req("GET", f"/assets/model/{model.get('id')}/artifact-manifest")
        for item in manifest.get("files") or manifest.get("artifacts") or []:
            url = item.get("url") or ""
            name = (item.get("name") or item.get("kind") or "").lower()
            if "glb" in name or url.lower().endswith(".glb"):
                return url
    except RuntimeError:
        pass
    raise RuntimeError("Mint model has no GLB url")


def generate_from_prompt(prompt: str) -> dict:
    prompt = prompt.strip()
    if not prompt:
        raise RuntimeError("Enter a prompt")
    started = _req("POST", "/models:generate", {"prompt": prompt, "generationMode": "auto"})
    op_id = started.get("id") or (started.get("operation") or {}).get("id")
    if not op_id:
        raise RuntimeError("Mint did not return an operation id")
    op = _poll(str(op_id))
    resource = op.get("resource") or {}
    model_id = resource.get("id") or op.get("modelId") or op.get("model_id")
    if not model_id:
        raise RuntimeError("Mint finished without a model id")
    model = _req("GET", f"/models/{model_id}")
    url = _glb_url(model)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as res:
        glb = res.read()
    stl, dims = rescale_glb(glb)
    dest = save_object(prompt, stl, {"mint_model_id": str(model_id), "source": "mint", **dims})
    (object_dir(dest["id"]) / "source.glb").write_bytes(glb)
    return dest
