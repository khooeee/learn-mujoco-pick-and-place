#!/usr/bin/env python3
"""Replay a logged episode to mp4 (or open a MuJoCo viewer)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from env import ObjectSpec, PickEnv
from log import read_jsonl_index

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def spec_from_row(row: dict) -> ObjectSpec:
    o = row["object"]
    rgba = tuple(o.get("rgba") or [0.85, 0.38, 0.16, 1.0])
    return ObjectSpec(
        o["x"],
        o["y"],
        o["z"],
        o["h"],
        o["w"],
        int(o["shape"]),
        rgba,  # type: ignore[arg-type]
        mesh_id=o.get("mesh_id"),
        d=float(o.get("d") or o["w"]),
    )


def render_episode(run_id: str, index: int, video: Path | None, view: bool) -> Path | None:
    run_dir = RUNS / run_id
    row = read_jsonl_index(run_dir / "episodes.jsonl", index)
    if row is None:
        raise SystemExit(f"episode {index} not in {run_id}")

    env = PickEnv(render=True, seed=int(row["seed"]), img_size=256, textured=True)
    env.reset(seed=int(row["seed"]), spec=spec_from_row(row))
    frames: list[np.ndarray] = []

    def grab() -> np.ndarray:
        assert env.renderer is not None
        env.renderer.update_scene(env.data, camera=env.overhead_cam)
        overhead = env.renderer.render()
        env.renderer.update_scene(env.data, camera=env.wrist_cam)
        wrist = env.renderer.render()
        h = overhead.shape[0]
        gap = np.zeros((h, 4, 3), dtype=overhead.dtype)
        return np.concatenate([overhead, gap, wrist], axis=1)

    frames.append(grab())
    for a in row["actions"]:
        env.step(np.array(a, dtype=np.float64))
        frames.append(grab())

    out = None
    if video is not None:
        import imageio.v2 as imageio

        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video, frames, fps=20)
        out = video
        print("wrote", video)

    if view:
        import mujoco.viewer

        env.reset(seed=int(row["seed"]), spec=spec_from_row(row))
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            for a in row["actions"]:
                env.step(np.array(a, dtype=np.float64))
                v.sync()
    env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()
    video = RUNS / args.run / "videos" / f"ep_{args.episode}.mp4" if args.mp4 else None
    render_episode(args.run, args.episode, video, args.view)


if __name__ == "__main__":
    main()
