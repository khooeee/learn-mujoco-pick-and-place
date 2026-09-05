"""SO-101 pick env: cameras + joints in, joint targets out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import mujoco
import numpy as np

from library import ensure_visual, load_meta, obj_path, png_path, require_object_id, stl_path

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "so101" / "pick_scene.xml"
SCENE_MESH = ROOT / "so101" / "pick_mesh.xml"
SCENE_TEX = ROOT / "so101" / "pick_tex.xml"
ASSETS = ROOT / "so101" / "assets"
RUNTIME_STL = ASSETS / "runtime_object.stl"
RUNTIME_OBJ = ASSETS / "runtime_object.obj"
RUNTIME_PNG = ASSETS / "runtime_object.png"
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
TABLE_TOP = 0.04
IMG = 84


@dataclass
class ObjectSpec:
    x: float
    y: float
    z: float
    h: float
    w: float
    shape: int  # 0 box, 1 cylinder, 2 mint mesh
    rgba: tuple[float, float, float, float]
    mesh_id: str | None = None


class PickEnv:
    def __init__(self, render: bool = True, seed: int = 0, img_size: int = IMG, textured: bool = False) -> None:
        self.rng = np.random.default_rng(seed)
        self.render_enabled = render
        self.img_size = img_size
        self.textured = textured
        self.n_act = 6
        self.action_repeat = 8
        self.max_actions = 80
        self._t = 0
        self._mesh_key = None
        self.renderer: mujoco.Renderer | None = None
        self.model = None  # type: ignore[assignment]
        self.data = None  # type: ignore[assignment]
        self.spec = ObjectSpec(0.22, 0.0, 0.07, 0.056, 0.044, 0, (0.85, 0.38, 0.16, 1.0))
        self._start_z = TABLE_TOP
        self._load_scene(None)

    def _bind(self) -> None:
        self.act_ids = np.array(
            [self.model.actuator(n).id for n in JOINT_NAMES], dtype=np.int32
        )
        self.ctrl_lo = self.model.actuator_ctrlrange[self.act_ids, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[self.act_ids, 1].copy()
        self.obj_body = self.model.body("object").id
        self.obj_geom = self.model.geom("object_geom").id
        self.grip_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.grip_body = self.model.body("gripper").id
        self.overhead_cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        self.wrist_cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        if self.wrist_cam < 0:
            raise RuntimeError("scene is missing wrist camera on the gripper")
        if self.render_enabled:
            self.renderer = mujoco.Renderer(self.model, height=self.img_size, width=self.img_size)

    def _load_scene(self, mesh_id: str | None) -> None:
        use_tex = bool(self.textured and mesh_id and ensure_visual(mesh_id))
        key = (mesh_id, use_tex)
        if key == self._mesh_key and self.model is not None:
            return
        self.renderer = None
        if mesh_id:
            src = stl_path(mesh_id)
            if not src.exists():
                raise FileNotFoundError(f"object {mesh_id} has no STL")
            shutil.copy(src, RUNTIME_STL)
            if use_tex:
                shutil.copy(obj_path(mesh_id), RUNTIME_OBJ)
                shutil.copy(png_path(mesh_id), RUNTIME_PNG)
                self.model = mujoco.MjModel.from_xml_path(str(SCENE_TEX))
            else:
                self.model = mujoco.MjModel.from_xml_path(str(SCENE_MESH))
        else:
            self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self._mesh_key = key
        self._bind()

    def _place_object(self, spec: ObjectSpec) -> None:
        self.spec = spec
        gid = self.obj_geom
        if spec.shape != 2:
            if spec.shape == 1:
                self.model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_CYLINDER
                self.model.geom_size[gid][:] = np.array([spec.w / 2, spec.h / 2, 0.0])
            else:
                self.model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_BOX
                self.model.geom_size[gid][:] = np.array([spec.w / 2, spec.w / 2, spec.h / 2])
            self.model.geom_rgba[gid][:] = spec.rgba
        self.data.qpos[-7] = spec.x
        self.data.qpos[-6] = spec.y
        self.data.qpos[-5] = spec.z
        self.data.qpos[-4:] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[-6:] = 0.0

    def _set_object(self, spec: ObjectSpec) -> None:
        self._load_scene(spec.mesh_id)
        self._place_object(spec)

    def random_spec(self) -> ObjectSpec:
        x = float(0.16 + self.rng.random() * 0.14)
        y = float(-0.10 + self.rng.random() * 0.20)
        oid = require_object_id()
        meta = load_meta(oid)
        h = float(meta.get("h") or 0.06)
        w = float(meta.get("w") or 0.05)
        return ObjectSpec(
            x, y, TABLE_TOP + 0.002, h, w, 2, (0.85, 0.38, 0.16, 1.0), mesh_id=oid
        )

    def reset(self, seed: int | None = None, spec: ObjectSpec | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        spec = spec or self.random_spec()
        self._load_scene(spec.mesh_id)
        mujoco.mj_resetData(self.model, self.data)
        mid = 0.5 * (self.ctrl_lo + self.ctrl_hi)
        self.data.ctrl[self.act_ids] = mid
        self.data.qpos[:6] = mid
        self._place_object(spec)
        self._start_z = spec.z
        self._t = 0
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def _joints(self) -> np.ndarray:
        return self.data.qpos[:6].astype(np.float32)

    def _priv(self) -> np.ndarray:
        pos = self.data.xpos[self.obj_body]
        return np.array(
            [pos[0], pos[1], pos[2], self.spec.h, self.spec.w], dtype=np.float32
        )

    def _render(self, cam: int | mujoco.MjvCamera) -> np.ndarray:
        assert self.renderer is not None
        self.renderer.update_scene(self.data, camera=cam)
        rgb = self.renderer.render()
        return rgb.astype(np.float32) / 255.0

    def observe(self) -> dict:
        joints = self._joints()
        priv = self._priv()
        if self.render_enabled and self.renderer is not None:
            overhead = np.transpose(self._render(self.overhead_cam), (2, 0, 1))
            wrist = np.transpose(self._render(self.wrist_cam), (2, 0, 1))
            img = np.concatenate([overhead, wrist], axis=0)
        else:
            img = np.zeros((6, self.img_size, self.img_size), dtype=np.float32)
        return {"img": img, "joints": joints, "priv": priv}

    def _scale_action(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a, -1.0, 1.0)
        return 0.5 * (a + 1.0) * (self.ctrl_hi - self.ctrl_lo) + self.ctrl_lo

    def _reward(self) -> tuple[float, bool]:
        obj = self.data.xpos[self.obj_body]
        grip = self.data.site_xpos[self.grip_site]
        dist = float(np.linalg.norm(obj - grip))
        lift = float(obj[2] - TABLE_TOP)
        r = -0.4 * dist + 2.5 * max(0.0, lift - 0.02)
        success = lift > 0.08 and dist < 0.12
        if success:
            r += 4.0
        if obj[2] < 0.01:
            r -= 1.0
        return r, success

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        ctrl = self._scale_action(np.asarray(action, dtype=np.float64))
        reward = 0.0
        success = False
        for _ in range(self.action_repeat):
            self.data.ctrl[self.act_ids] = ctrl
            mujoco.mj_step(self.model, self.data)
        r, success = self._reward()
        reward += r
        self._t += 1
        obj = self.data.xpos[self.obj_body]
        done = self._t >= self.max_actions or obj[2] < -0.02
        info = {
            "success": bool(success),
            "lifted_z": float(obj[2]),
            "spec": {
                "x": self.spec.x,
                "y": self.spec.y,
                "z": self.spec.z,
                "h": self.spec.h,
                "w": self.spec.w,
                "shape": self.spec.shape,
                "rgba": list(self.spec.rgba),
                "mesh_id": self.spec.mesh_id,
            },
        }
        return self.observe(), float(reward), bool(done), info

    def close(self) -> None:
        self.renderer = None
