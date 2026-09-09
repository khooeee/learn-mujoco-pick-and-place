# learn-mujoco-pick-and-place

My learning attempt at training pick & place with a SO-101 arm in a simulation

I learnt enough to know why classical RL has limits lol.

---

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
.venv/bin/python train_rl.py --run demo --episodes 30000
.venv/bin/python replay.py --run demo --episode 12 --mp4
.venv/bin/python replay.py --run demo --episode 12 --view
```

## What this is

- Robot: `ml/so101/` (TheRobotStudio SO-101 MJCF)
- Actor: small CNN on overhead + wrist views + joints
- Critic: those **plus** object pose (training only)
- Actions: 6 SO-101 actuators, no inverse kinematics
- Reward: reach (open) → close → lift
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

**Reward** (three gated phases; distances use the object AABB **center**, not the mesh origin)

```
1. Reach (episode starts with the gripper fully open)
   2.0 × Δ(3D gripper–object distance) always
   + 0.8 × Δ(around) only while the gripper is wider than the brick + 5 mm
   − small penalty for closing in free space
   Stage 1 latches when the center is between the **open** jaws (around ≥ 0.5)

2. Close (only after stage 1; off again if the brick leaves the jaws)
   1.0 × Δ(around × squeeze) + 0.8 × Δ(jaw contact) + close-delta in the jaws

3. Lift (only while pinch EMA is on and dist < 12 cm)
   12 × Δz
   + hold progress (up to 0.4) while 8 cm up, near, slow
   + 8.0 after 16 policy steps of that hold

Always: −2.0 × past table edge, −1.0 if the object falls through the table
```

Only one phase pays at a time. Sitting still scores ~0 after you arrive. Grasp width is the AABB along the jaw axis. Pinch EMA is 0.7 / threshold 0.35. Tilt is not penalized.

Success = phase 3 hold for **16 policy steps**. Changing this reward means **Train New**, not Resume.
