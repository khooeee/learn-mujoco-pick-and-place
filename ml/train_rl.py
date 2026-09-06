#!/usr/bin/env python3
"""Headless PPO on the SO-101 pick env."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from env import PickEnv
from log import append_jsonl, new_run_id, write_json
from ppo import PPO, gae

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def should_stop(run_dir: Path) -> bool:
    return (run_dir / "STOP").exists()


def rollout_episode(env: PickEnv, ppo: PPO, seed: int, deterministic: bool = False):
    obs = env.reset(seed=seed)
    imgs, joints, privs, acts, logps, vals, rews, dones = [], [], [], [], [], [], [], []
    actions_log = []
    success = False
    total = 0.0
    r_reach = 0.0
    r_close = 0.0
    r_lift = 0.0
    lifted = 0.0
    spec = None
    done = False
    while not done:
        a, logp, v = ppo.act(obs, deterministic=deterministic)
        nxt, r, done, info = env.step(a)
        imgs.append(obs["img"])
        joints.append(obs["joints"])
        privs.append(obs["priv"])
        acts.append(a)
        logps.append(logp)
        vals.append(v)
        rews.append(r)
        dones.append(done)
        actions_log.append(a.tolist())
        total += r
        r_reach += float(info.get("r_reach") or 0.0)
        r_close += float(info.get("r_close") or 0.0)
        r_lift += float(info.get("r_lift") or 0.0)
        success = info["success"] or success
        lifted = info["lifted_z"]
        spec = info["spec"]
        obs = nxt
    return {
        "img": np.stack(imgs),
        "joints": np.stack(joints),
        "priv": np.stack(privs),
        "act": np.stack(acts).astype(np.float32),
        "logp": np.array(logps, dtype=np.float32),
        "val": np.array(vals, dtype=np.float32),
        "rew": np.array(rews, dtype=np.float32),
        "done": np.array(dones, dtype=np.float32),
        "actions": actions_log,
        "reward": total,
        "r_reach": r_reach,
        "r_close": r_close,
        "r_lift": r_lift,
        "success": bool(success),
        "lifted_z": lifted,
        "spec": spec,
        "seed": seed,
    }


def to_batch(episodes: list[dict], dev: torch.device) -> dict:
    advs, rets = [], []
    for ep in episodes:
        val = torch.from_numpy(ep["val"])
        rew = torch.from_numpy(ep["rew"])
        done = torch.from_numpy(ep["done"])
        a, r = gae(rew, val, done)
        advs.append(a)
        rets.append(r)
    return {
        "img": torch.from_numpy(np.concatenate([e["img"] for e in episodes])).float(),
        "joints": torch.from_numpy(np.concatenate([e["joints"] for e in episodes])).float(),
        "priv": torch.from_numpy(np.concatenate([e["priv"] for e in episodes])).float(),
        "act": torch.from_numpy(np.concatenate([e["act"] for e in episodes])).float(),
        "logp": torch.from_numpy(np.concatenate([e["logp"] for e in episodes])).float(),
        "adv": torch.cat(advs).float(),
        "ret": torch.cat(rets).float(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--batch-episodes", type=int, default=8)
    args = ap.parse_args()

    run_id = args.run or new_run_id()
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = device()
    print("device:", dev, flush=True)

    env = PickEnv(render=True, seed=args.seed)
    ppo = PPO(dev)
    start_ep = 0
    if args.resume:
        ppo.load(args.resume)
        meta_path = run_dir / "status.json"
        if meta_path.exists():
            import json

            start_ep = int(json.loads(meta_path.read_text()).get("episode", 0))

    ckpt = run_dir / "policy.pt"
    recent = deque(maxlen=40)
    buf: list[dict] = []
    ep = start_ep

    write_json(
        run_dir / "status.json",
        {
            "state": "running",
            "run_id": run_id,
            "episode": ep,
            "episodes_target": args.episodes,
            "success_rate": 0.0,
            "reward": 0.0,
            "r_reach": 0.0,
            "r_close": 0.0,
            "r_lift": 0.0,
            "log": f"start on {dev}",
        },
    )

    try:
        while ep < args.episodes and not should_stop(run_dir):
            seed = args.seed + ep * 17 + 3
            out = rollout_episode(env, ppo, seed=seed)
            buf.append(out)
            recent.append(1.0 if out["success"] else 0.0)
            rate = float(np.mean(recent))
            append_jsonl(
                run_dir / "episodes.jsonl",
                {
                    "index": ep,
                    "seed": seed,
                    "actions": out["actions"],
                    "reward": out["reward"],
                    "r_reach": out["r_reach"],
                    "r_close": out["r_close"],
                    "r_lift": out["r_lift"],
                    "success": out["success"],
                    "lifted_z": out["lifted_z"],
                    "object": out["spec"],
                    "policy_step": ep,
                },
            )
            append_jsonl(
                run_dir / "metrics.jsonl",
                {"episode": ep, "reward": out["reward"], "r_reach": out["r_reach"], "r_close": out["r_close"], "r_lift": out["r_lift"], "success": out["success"], "rate": rate},
            )
            print(
                f"ep {ep:4d}  R {out['reward']:7.2f}  reach {out['r_reach']:5.2f}  close {out['r_close']:5.2f}  lift {out['r_lift']:5.2f}  success {int(out['success'])}  rate {rate:.2f}",
                flush=True,
            )
            write_json(
                run_dir / "status.json",
                {
                    "state": "running",
                    "run_id": run_id,
                    "episode": ep + 1,
                    "episodes_target": args.episodes,
                    "success_rate": rate,
                    "reward": out["reward"],
                    "r_reach": out["r_reach"],
                    "r_close": out["r_close"],
                    "r_lift": out["r_lift"],
                    "log": f"episode {ep} reward {out['reward']:.2f} reach {out['r_reach']:.2f} close {out['r_close']:.2f} lift {out['r_lift']:.2f}",
                },
            )
            ep += 1
            if len(buf) >= args.batch_episodes:
                batch = to_batch(buf, dev)
                stats = ppo.update(batch)
                ppo.save(ckpt)
                buf = []
                print("  ppo", stats, flush=True)
        if buf:
            ppo.update(to_batch(buf, dev))
            ppo.save(ckpt)
    finally:
        ppo.save(ckpt)
        st = "stopped" if should_stop(run_dir) else "idle"
        write_json(
            run_dir / "status.json",
            {
                "state": st,
                "run_id": run_id,
                "episode": ep,
                "episodes_target": args.episodes,
                "success_rate": float(np.mean(recent)) if recent else 0.0,
                "reward": 0.0,
                "r_reach": 0.0,
                "r_close": 0.0,
                "r_lift": 0.0,
                "log": "finished",
            },
        )
        env.close()


if __name__ == "__main__":
    main()
