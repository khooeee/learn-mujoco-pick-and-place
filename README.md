# TwinPick — SO-101 vision RL

**Pick is trained in MuJoCo** on the official SO-101 arm: cameras in, joint targets out, lift reward. Replay any episode as mp4.

## Setup

```bash
npm install
python3 -m venv ml/.venv
ml/.venv/bin/pip install -r ml/requirements.txt
```

## Run (two terminals)

```bash
npm run rl          # API on :8765
npm run dev         # UI on :5173
```

Open http://localhost:5173

1. **Object** — Mint a mesh, or import a `.glb`. Both are rescaled to ~4.5 cm and saved in `ml/assets/objects/`. Pick a saved object for training.
2. **Runs** — click a row, then **Resume** to continue that checkpoint. **Train New** starts a fresh folder.
3. **Episodes → render mp4** — waits, then plays in the stage (overhead | wrist). Replay uses the GLB albedo on the object when a texture exists.

CLI instead of the UI:

```bash
cd ml
.venv/bin/python train_rl.py --run demo --episodes 1000
.venv/bin/python replay.py --run demo --episode 12 --mp4
.venv/bin/python replay.py --run demo --episode 12 --view
```

## What this is

- Robot: `ml/so101/` (TheRobotStudio SO-101 MJCF)
- Actor: small CNN on overhead + wrist views + joints
- Critic: those **plus** object pose (training only)
- Actions: 6 SO-101 actuators, no inverse kinematics
- Reward: reach object and lift it
- Objects: Mint prompt or GLB import → local library. Longest edge is 4.5 cm on import.

Not SmolVLA / ACT. Not the old browser IK toy arm.

## Model architecture

Code: `ml/nets.py` (networks), `ml/ppo.py` (PPO), `ml/env.py` (obs / reward). Weights: `ml/runs/<run_id>/policy.pt` (`actor` + `critic` state dicts).

**Observation (what the policy sees)**

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `img` | `6 × 84 × 84` | Overhead RGB stacked with a wrist-tracking RGB view, values in `[0, 1]` |
| `joints` | `6` | `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` |
| `priv` | `5` | Object AABB-center `x, y, z` plus height and width — **critic only**, not given to the actor |

One policy step is held for **8** MuJoCo steps (`action_repeat`). An episode is at most **80** policy steps.

**Shared visual encoder** (Nature-style CNN, separate weights on actor vs critic)

```
Conv2d  6→32,  k=8, s=4, ReLU     84² → 20²
Conv2d 32→64,  k=4, s=2, ReLU     20² → 9²
Conv2d 64→64,  k=3, s=1, ReLU      9² → 7²
Flatten 64×7×7 = 3136
Linear 3136→256, ReLU
```

**Actor** (used at train and at inference)

```
[256 visual | 6 joints] → Linear 262→128, Tanh → Linear 128→6, Tanh  = μ
log_std  is a learned 6-vector, clamped then exp → σ
action ~ Normal(μ, σ), clipped to [-1, 1]
```

Those 6 values are mapped linearly onto each actuator’s `ctrlrange`. No IK: the network outputs joint targets.

**Critic** (training only — GAE / value loss)

```
[256 visual | 6 joints | 5 priv] → Linear 267→128, Tanh → Linear 128→1  = V(s)
```

**PPO**

- Adam `2.5e-4` on actor + critic together
- GAE `γ=0.99`, `λ=0.95`
- Clip `0.2` (ratio in `[0.8, 1.2]`), 4 epochs / update, grad clip `1.0`
- Loss: `policy + 0.5 × value − 0.01 × entropy`
- Update every **8** finished episodes (`--batch-episodes`)

**Reward** (after each policy step; distances use the object AABB **center**, not the mesh origin)

```
r = 1.0 × (previous XY error − current XY error)   # progress toward the object
  + hover-above bonus while XY is still far
  + gated grasp (center near TCP and between jaws, matching width, closing)
  + 0.05 per jaw contact + 0.15 if both jaws pinch
  + 4.0 × max(0, Δz − 1 cm)  only while both jaws pinch
  + 8.0 if the center rose > 8 cm, still near the gripper, and pinching
  − knock penalty only after ~8 cm of table slide without lift
  − 1.0 if the object falls through the table
```

Lift is how far the center rose from the pose at reset. Closing scores when the center is between the jaws (within ~5 cm of the TCP); closing in free space is a small penalty. Tilt is not penalized (spheres have no upright). Pinch-on-the-table is a weak bonus so the policy cannot farm contact instead of lifting.

Success = center more than 8 cm above its rest height, within 12 cm of the gripper, and both jaws in contact.
