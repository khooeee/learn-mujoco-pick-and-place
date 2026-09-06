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
r = 0.6 × XY progress toward the object, faded out once the center is in the jaws
  + hover-above bonus while XY is still far (same fade)
  + gated grasp (center near TCP and between jaws): squeeze opening
      below ~0.85 × AABB width along the jaw axis, plus a small close-delta
  + 0.05 per jaw contact + 0.15 if both jaws pinch
  + 12 × Δz  only while dual-jaw contact EMA is on  (raise, not linger)
  + hold progress (up to 0.4) while grasped + lift + near + low object speed
  + 8.0 once that hold has lasted 16 policy steps
  − 2.0 × how far the center is past the table edge  (on-table slides are free)
  − 1.0 if the object falls through the table
  − small penalty for closing in free space
```

Lift is how far the center rose from the pose at reset. Approach shaping stops in the jaw volume so pushing the brick around is not a late-episode farm. Grasp width is the object's AABB projected onto the jaw-opening axis, not `min(w, d)` from import. Lift credit requires a dual-jaw pinch (EMA 0.7, threshold 0.35) so stud chatter does not drop the hold, but a flick that never pinches does not count. Tilt is not penalized.

Success = grasped (pinch EMA), center more than 8 cm above rest, within 12 cm of the gripper, object slower than 0.15 m/s and 2 rad/s, **held for 16 policy steps**. Changing this reward means **Train New**, not Resume.
