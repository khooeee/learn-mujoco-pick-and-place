"""SO-101 pick env: cameras + joints in, joint targets out. No IK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "so101" / "pick_scene.xml"
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
    shape: int  # 0 box, 1 cylinder
    rgba: tuple[float, float, float, float]


class PickEnv:
    def __init__(self, render: bool = True, seed: int = 0, img_size: int = IMG) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.render_enabled = render
        self.img_size = img_size
        self.renderer: mujoco.Renderer | None = None
        if render:
            self.renderer = mujoco.Renderer(self.model, height=img_size, width=img_size)

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
        self.n_act = 6
        self.action_repeat = 8
        self.max_actions = 80
        self._t = 0
        self.spec = ObjectSpec(0.22, 0.0, 0.07, 0.056, 0.044, 0, (0.85, 0.38, 0.16, 1.0))
        self._start_z = TABLE_TOP

    def _set_object(self, spec: ObjectSpec) -> None:
        self.spec = spec
        gid = self.obj_geom
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

    def random_spec(self) -> ObjectSpec:
        h = float(0.04 + self.rng.random() * 0.05)
        w = float(0.03 + self.rng.random() * 0.03)
        shape = int(self.rng.integers(0, 2))
        x = float(0.16 + self.rng.random() * 0.14)
        y = float(-0.10 + self.rng.random() * 0.20)
        z = TABLE_TOP + h / 2 + 0.002
        rgba = (
            float(0.4 + self.rng.random() * 0.55),
            float(0.2 + self.rng.random() * 0.5),
            float(0.1 + self.rng.random() * 0.4),
            1.0,
        )
        return ObjectSpec(x, y, z, h, w, shape, rgba)

    def reset(self, seed: int | None = None, spec: ObjectSpec | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        mid = 0.5 * (self.ctrl_lo + self.ctrl_hi)
        self.data.ctrl[self.act_ids] = mid
        self.data.qpos[:6] = mid
        spec = spec or self.random_spec()
        self._set_object(spec)
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

    def _wrist_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.grip_body
        cam.distance = 0.18
        cam.elevation = -25.0
        cam.azimuth = 140.0
        return cam

    def observe(self) -> dict:
        joints = self._joints()
        priv = self._priv()
        if self.render_enabled and self.renderer is not None:
            overhead = np.transpose(self._render(self.overhead_cam), (2, 0, 1))
            wrist = np.transpose(self._render(self._wrist_camera()), (2, 0, 1))
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
            },
        }
        return self.observe(), float(reward), bool(done), info

    def close(self) -> None:
        self.renderer = None
