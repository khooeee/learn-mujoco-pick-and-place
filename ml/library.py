"""Saved pick objects: rescaled STL + metadata."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from io import BytesIO

ROOT = Path(__file__).resolve().parent
OBJECTS = ROOT / "assets" / "objects"
SELECTED = OBJECTS / "selected.json"
TARGET_EXTENT = 0.07  # longest edge after rescale (meters)


def _slug(prompt: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", prompt.strip().lower()).strip("-")[:40]
    return s or "object"


def object_dir(oid: str) -> Path:
    return OBJECTS / oid


def list_objects() -> list[dict]:
    OBJECTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(OBJECTS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        meta = p / "meta.json"
        if p.is_dir() and meta.exists():
            rows.append(json.loads(meta.read_text()))
    return rows


def selected_id() -> str | None:
    if not SELECTED.exists():
        return None
    try:
        return json.loads(SELECTED.read_text()).get("id")
    except json.JSONDecodeError:
        return None


def set_selected(oid: str | None) -> None:
    OBJECTS.mkdir(parents=True, exist_ok=True)
    if oid is None:
        if SELECTED.exists():
            SELECTED.unlink()
        return
    if not (object_dir(oid) / "object.stl").exists():
        raise FileNotFoundError(oid)
    SELECTED.write_text(json.dumps({"id": oid}))


def load_meta(oid: str) -> dict:
    path = object_dir(oid) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(oid)
    return json.loads(path.read_text())


def stl_path(oid: str) -> Path:
    return object_dir(oid) / "object.stl"


def save_object(prompt: str, stl_bytes: bytes, extra: dict) -> dict:
    OBJECTS.mkdir(parents=True, exist_ok=True)
    oid = f"{int(time.time())}-{_slug(prompt)}"
    d = object_dir(oid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "object.stl").write_bytes(stl_bytes)
    meta = {
        "id": oid,
        "prompt": prompt,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    set_selected(oid)
    return meta


def rescale_glb(glb: bytes) -> tuple[bytes, dict]:
    import trimesh

    loaded = trimesh.load(BytesIO(glb), file_type="glb", force="scene")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError("GLB had no mesh")
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError("Could not read GLB")
    extent = float(mesh.extents.max())
    if extent <= 1e-6:
        raise RuntimeError("Mesh has zero size")
    scale = TARGET_EXTENT / extent
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounds[0])
    ex, ey, ez = (float(v) for v in mesh.extents)
    stl = mesh.export(file_type="stl")
    if isinstance(stl, str):
        stl = stl.encode()
    return bytes(stl), {
        "scale": scale,
        "source_extent": extent,
        "w": max(ex, ey),
        "h": ez,
        "d": min(ex, ey),
        "target_extent": TARGET_EXTENT,
    }


def import_glb(glb: bytes, name: str) -> dict:
    name = name.strip() or "imported"
    stl, dims = rescale_glb(glb)
    dest = save_object(name, stl, {"source": "import", **dims})
    (object_dir(dest["id"]) / "source.glb").write_bytes(glb)
    return dest
