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
GRIPPER_MAX_OPEN = 0.07  # meters at fully open (SO-101 hinge mapped linearly)
AROUND_DIST = 0.05
TABLE_CX, TABLE_CY = 0.22, 0.0
TABLE_HX, TABLE_HY = 0.28, 0.20
HOLD_STEPS = 16
OBJ_LIN_MAX = 0.15
OBJ_ANG_MAX = 2.0
PINCH_EMA = 0.7
PINCH_ON = 0.35
LIFT_K = 12.0
GRASP_K = 1.0
CONTACT_K = 0.8
REACH_K = 2.0
AROUND_K = 0.8
OPEN_EXTRA = 0.005
STAGE1_ON = 0.5
STAGE1_OFF = 0.2
FIXED_JAW_MESH = "wrist_roll_follower_so101_v1"
MOVING_JAW_MESH = "moving_jaw_so101_v1"


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
    d: float = 0.0

    def __post_init__(self) -> None:
        if self.d <= 0.0:
            self.d = self.w


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
        self._rest_center_z = TABLE_TOP
        self._open_prev = 0.5
        self._dist_prev = 0.0
        self._around_prev = 0.0
        self._lift_prev = 0.0
        self._grasp_prev = 0.0
        self._contact_prev = 0.0
        self._pinch_ema = 0.0
        self._stage1 = False
        self._hold = 0
        self._fixed_jaw_geoms: np.ndarray = np.zeros(0, dtype=np.int32)
        self._moving_jaw_geoms: np.ndarray = np.zeros(0, dtype=np.int32)
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
        self._grip_qadr = int(self.model.joint("gripper").qposadr[0])
        self._grip_lo, self._grip_hi = [float(x) for x in self.model.jnt_range[self.model.joint("gripper").id]]
        self._bind_jaw_geoms()
        self.overhead_cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        self.wrist_cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        if self.wrist_cam < 0:
            raise RuntimeError("scene is missing wrist camera on the gripper")
        if self.render_enabled:
            self.renderer = mujoco.Renderer(self.model, height=self.img_size, width=self.img_size)

    def _bind_jaw_geoms(self) -> None:
        fixed, moving = [], []
        for gid in range(self.model.ngeom):
            if int(self.model.geom_contype[gid]) == 0:
                continue
            if int(self.model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_MESH, int(self.model.geom_dataid[gid])
            )
            if mesh == FIXED_JAW_MESH:
                fixed.append(gid)
            elif mesh == MOVING_JAW_MESH:
                moving.append(gid)
        self._fixed_jaw_geoms = np.array(fixed, dtype=np.int32)
        self._moving_jaw_geoms = np.array(moving, dtype=np.int32)

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
        d = float(meta.get("d") or w)
        return ObjectSpec(
            x, y, TABLE_TOP + 0.002, h, w, 2, (0.85, 0.38, 0.16, 1.0), mesh_id=oid, d=d
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
        self._open_prev = self._gripper_open()
        center = self._obj_center()
        self._rest_center_z = float(center[2])
        grip = self.data.site_xpos[self.grip_site]
        self._dist_prev = float(np.linalg.norm(center - grip))
        self._around_prev = 0.0
        self._lift_prev = 0.0
        self._grasp_prev = 0.0
        self._contact_prev = 0.0
        self._pinch_ema = 0.0
        self._stage1 = False
        self._hold = 0
        return self.observe()

    def _joints(self) -> np.ndarray:
        return self.data.qpos[:6].astype(np.float32)

    def _priv(self) -> np.ndarray:
        pos = self._obj_center()
        return np.array(
            [pos[0], pos[1], pos[2], self.spec.h, self.spec.w], dtype=np.float32
        )

    def _obj_center(self) -> np.ndarray:
        origin = self.data.xpos[self.obj_body]
        rot = self.data.xmat[self.obj_body].reshape(3, 3)
        local = np.array(
            [0.5 * float(self.spec.w), 0.5 * float(self.spec.d), 0.5 * float(self.spec.h)]
        )
        return origin + rot @ local

    def _obj_radius(self) -> float:
        return 0.5 * max(float(self.spec.w), float(self.spec.h), float(self.spec.d))

    def _obj_bottom_z(self) -> float:
        return float(self._obj_center()[2] - self._obj_radius())

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

    def _gripper_open(self) -> float:
        q = float(self.data.qpos[self._grip_qadr])
        span = max(self._grip_hi - self._grip_lo, 1e-6)
        return float(np.clip((q - self._grip_lo) / span, 0.0, 1.0))

    @staticmethod
    def _band(x: float, inner: float, outer: float) -> float:
        ax = abs(x)
        if ax <= inner:
            return 1.0
        if ax >= outer:
            return 0.0
        return (outer - ax) / (outer - inner)

    @staticmethod
    def _squeeze(open_m: float, width: float) -> float:
        tight = 0.85 * width
        loose = width + 0.005
        if open_m <= tight:
            return 1.0
        if open_m >= loose:
            return 0.0
        return (loose - open_m) / max(loose - tight, 1e-6)

    def _tcp_frame(self) -> tuple[np.ndarray, np.ndarray]:
        tcp = self.data.site_xpos[self.grip_site]
        rot = self.data.site_xmat[self.grip_site].reshape(3, 3)
        return tcp, rot

    def _jaw_axis_width(self) -> float:
        _, rot = self._tcp_frame()
        jaw = rot[:, 2]
        body = self.data.xmat[self.obj_body].reshape(3, 3)
        half = np.array(
            [0.5 * float(self.spec.w), 0.5 * float(self.spec.d), 0.5 * float(self.spec.h)]
        )
        return float(2.0 * np.dot(np.abs(body.T @ jaw), half))

    def _around(self, obj: np.ndarray, dist: float) -> float:
        tcp, rot = self._tcp_frame()
        local = rot.T @ (obj - tcp)
        near = self._band(dist, 0.03, AROUND_DIST)
        between = 0.0
        if self._fixed_jaw_geoms.size and self._moving_jaw_geoms.size:
            z_f = float((rot.T @ (self.data.geom_xpos[int(self._fixed_jaw_geoms[0])] - tcp))[2])
            z_m = float((rot.T @ (self.data.geom_xpos[int(self._moving_jaw_geoms[0])] - tcp))[2])
            lo, hi = (z_f, z_m) if z_f < z_m else (z_m, z_f)
            pad = 0.012 + 0.35 * float(self.spec.w)
            if lo - pad <= float(local[2]) <= hi + pad:
                between = 1.0
        else:
            between = self._band(float(local[2]), 0.03, 0.05)
        return near * between

    def _jaw_contacts(self) -> tuple[float, float]:
        touch_fixed = 0.0
        touch_moving = 0.0
        fixed = set(int(g) for g in self._fixed_jaw_geoms)
        moving = set(int(g) for g in self._moving_jaw_geoms)
        obj = int(self.obj_geom)
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 == obj else g1 if g2 == obj else None
            if other is None:
                continue
            if other in fixed:
                touch_fixed = 1.0
            elif other in moving:
                touch_moving = 1.0
        return touch_fixed, touch_moving

    def _obj_speeds(self) -> tuple[float, float]:
        lin = float(np.linalg.norm(self.data.qvel[-6:-3]))
        ang = float(np.linalg.norm(self.data.qvel[-3:]))
        return lin, ang

    def _off_table(self, obj: np.ndarray) -> float:
        dx = abs(float(obj[0]) - TABLE_CX) - TABLE_HX
        dy = abs(float(obj[1]) - TABLE_CY) - TABLE_HY
        return max(0.0, dx, dy)

    def _reward(self) -> tuple[float, bool, dict]:
        obj = self._obj_center()
        grip = self.data.site_xpos[self.grip_site]
        dist = float(np.linalg.norm(obj - grip))
        lift = float(obj[2] - self._rest_center_z)
        open_now = self._gripper_open()
        open_m = open_now * GRIPPER_MAX_OPEN
        around = self._around(obj, dist)
        grasp_w = self._jaw_axis_width()
        opened = float(open_m >= grasp_w + OPEN_EXTRA)
        close_delta = max(0.0, self._open_prev - open_now)
        squeeze = self._squeeze(open_m, grasp_w)
        grasp_q = around * squeeze
        touch_fixed, touch_moving = self._jaw_contacts()
        pinch = touch_fixed * touch_moving
        self._pinch_ema = PINCH_EMA * self._pinch_ema + (1.0 - PINCH_EMA) * pinch
        grasped = self._pinch_ema >= PINCH_ON
        contact_q = 0.5 * touch_fixed + 0.5 * touch_moving
        at_open = around >= STAGE1_ON and opened > 0.0
        phase_lift = grasped and dist < 0.12
        phase_close = (self._stage1 or at_open) and not phase_lift
        phase_reach = not phase_close and not phase_lift

        r_reach = 0.0
        r_close = 0.0
        r_lift = 0.0
        if phase_reach:
            r_reach = opened * (
                REACH_K * (self._dist_prev - dist) + AROUND_K * (around - self._around_prev)
            )
            r_reach -= (1.0 - opened) * close_delta * 0.3
        elif phase_close:
            r_close = GRASP_K * (grasp_q - self._grasp_prev)
            r_close += CONTACT_K * (contact_q - self._contact_prev)
            r_close += around * 0.4 * close_delta
        else:
            r_lift = LIFT_K * (lift - self._lift_prev)

        lin, ang = self._obj_speeds()
        holding = (
            phase_lift
            and lift > 0.08
            and lin < OBJ_LIN_MAX
            and ang < OBJ_ANG_MAX
        )
        self._hold = self._hold + 1 if holding else 0
        success = self._hold >= HOLD_STEPS
        r_hold = 0.4 * (self._hold / HOLD_STEPS) if holding else 0.0
        r_off = -2.0 * self._off_table(obj)
        if self._obj_bottom_z() < 0.01:
            r_off -= 1.0
        r_lift += r_hold + (8.0 if success else 0.0)
        r = r_reach + r_close + r_lift + r_off
        if at_open:
            self._stage1 = True
        elif around < STAGE1_OFF:
            self._stage1 = False
        self._open_prev = open_now
        self._dist_prev = dist
        self._around_prev = around
        self._lift_prev = lift
        self._grasp_prev = grasp_q
        self._contact_prev = contact_q
        return r, success, {
            "r_reach": float(r_reach),
            "r_close": float(r_close),
            "r_lift": float(r_lift),
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        ctrl = self._scale_action(np.asarray(action, dtype=np.float64))
        reward = 0.0
        success = False
        for _ in range(self.action_repeat):
            self.data.ctrl[self.act_ids] = ctrl
            mujoco.mj_step(self.model, self.data)
        r, success, parts = self._reward()
        reward += r
        self._t += 1
        center = self._obj_center()
        done = self._t >= self.max_actions or self._obj_bottom_z() < -0.02
        info = {
            "success": bool(success),
            "lifted_z": float(center[2]),
            "r_reach": parts["r_reach"],
            "r_close": parts["r_close"],
            "r_lift": parts["r_lift"],
            "spec": {
                "x": self.spec.x,
                "y": self.spec.y,
                "z": self.spec.z,
                "h": self.spec.h,
                "w": self.spec.w,
                "d": self.spec.d,
                "shape": self.spec.shape,
                "rgba": list(self.spec.rgba),
                "mesh_id": self.spec.mesh_id,
            },
        }
        return self.observe(), float(reward), bool(done), info

    def close(self) -> None:
        self.renderer = None
