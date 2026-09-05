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
TARGET_EXTENT = 0.045  # longest edge after rescale (meters); SO-101 jaws open ~7 cm


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
            row = json.loads(meta.read_text())
            row["has_glb"] = (p / "source.glb").exists()
            rows.append(row)
    return rows


def selected_id() -> str | None:
    if not SELECTED.exists():
        return None
    try:
        return json.loads(SELECTED.read_text()).get("id")
    except json.JSONDecodeError:
        return None


def set_selected(oid: str) -> None:
    OBJECTS.mkdir(parents=True, exist_ok=True)
    if not (object_dir(oid) / "object.stl").exists():
        raise FileNotFoundError(oid)
    SELECTED.write_text(json.dumps({"id": oid}))


def require_object_id() -> str:
    oid = selected_id()
    if oid and (object_dir(oid) / "object.stl").exists():
        return oid
    items = list_objects()
    if not items:
        raise RuntimeError("Import or generate an object first")
    set_selected(items[0]["id"])
    return items[0]["id"]


def load_meta(oid: str) -> dict:
    path = object_dir(oid) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(oid)
    return json.loads(path.read_text())


def stl_path(oid: str) -> Path:
    return object_dir(oid) / "object.stl"


def glb_path(oid: str) -> Path:
    return object_dir(oid) / "source.glb"


def obj_path(oid: str) -> Path:
    return object_dir(oid) / "object.obj"


def png_path(oid: str) -> Path:
    return object_dir(oid) / "object.png"


def _trimesh_from_glb(glb: bytes):
    import trimesh

    loaded = trimesh.load(BytesIO(glb), file_type="glb", force="scene")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError("GLB had no mesh")
        return loaded, geoms
    if isinstance(loaded, trimesh.Trimesh):
        return loaded, [loaded]
    raise RuntimeError("Could not read GLB")


def _apply_table_scale(mesh) -> tuple[object, dict]:
    extent = float(mesh.extents.max())
    if extent <= 1e-6:
        raise RuntimeError("Mesh has zero size")
    scale = TARGET_EXTENT / extent
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounds[0])
    ex, ey, ez = (float(v) for v in mesh.extents)
    return mesh, {
        "scale": scale,
        "source_extent": extent,
        "w": max(ex, ey),
        "h": ez,
        "d": min(ex, ey),
        "target_extent": TARGET_EXTENT,
    }


def _diffuse_image(mesh):
    vis = getattr(mesh, "visual", None)
    mat = getattr(vis, "material", None) if vis is not None else None
    img = None
    if mat is not None:
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None:
        img = getattr(vis, "image", None) if vis is not None else None
    return img


def write_visual_from_glb(glb: bytes, dest: Path) -> bool:
    """Rescaled OBJ + PNG for MuJoCo replay. Same frame as the STL."""
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import SimpleMaterial

    _, geoms = _trimesh_from_glb(glb)
    visual = None
    for g in geoms:
        if _diffuse_image(g) is not None and getattr(g.visual, "uv", None) is not None:
            visual = g.copy()
            break
    if visual is None:
        return False
    uv = visual.visual.uv.copy()
    img = _diffuse_image(visual).convert("RGB")
    _apply_table_scale(visual)
    visual.visual = TextureVisuals(uv=uv, material=SimpleMaterial(image=img))
    dest.mkdir(parents=True, exist_ok=True)
    img.save(dest / "object.png")
    visual.export(dest / "object.obj", include_texture=True)
    extra_png = dest / "material_0.png"
    if extra_png.exists() and extra_png.resolve() != (dest / "object.png").resolve():
        extra_png.unlink()
    mtl = dest / "material.mtl"
    if mtl.exists():
        text = mtl.read_text()
        mtl.write_text(text.replace("material_0.png", "object.png"))
    return True


def ensure_visual(oid: str) -> bool:
    if obj_path(oid).exists() and png_path(oid).exists():
        return True
    glb = glb_path(oid)
    if not glb.exists():
        return False
    return write_visual_from_glb(glb.read_bytes(), object_dir(oid))


def rescale_glb(glb: bytes) -> tuple[bytes, dict]:
    import trimesh

    _, geoms = _trimesh_from_glb(glb)
    mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0].copy()
    mesh, dims = _apply_table_scale(mesh)
    stl = mesh.export(file_type="stl")
    if isinstance(stl, str):
        stl = stl.encode()
    return bytes(stl), dims


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


def import_glb(glb: bytes, name: str) -> dict:
    name = name.strip() or "imported"
    stl, dims = rescale_glb(glb)
    dest = save_object(name, stl, {"source": "import", **dims})
    d = object_dir(dest["id"])
    (d / "source.glb").write_bytes(glb)
    write_visual_from_glb(glb, d)
    return dest
