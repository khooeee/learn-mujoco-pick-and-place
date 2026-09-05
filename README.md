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

1. **Object** — prompt Mint for a mesh (optional). It is rescaled to ~7 cm and saved in `ml/assets/objects/`. Pick any saved object, or use random boxes/cylinders.
2. **Train** — headless PPO on MPS. Watch episode / success rate / log. Changing the selected object applies on the next episode.
3. **Episodes → render mp4** — waits, then plays in the stage.

CLI instead of the UI:

```bash
cd ml
.venv/bin/python train_rl.py --run demo --episodes 200
.venv/bin/python replay.py --run demo --episode 12 --mp4
.venv/bin/python replay.py --run demo --episode 12 --view
```

## What this is

- Robot: `ml/so101/` (TheRobotStudio SO-101 MJCF)
- Actor: small CNN on overhead + wrist views + joints
- Critic: those **plus** object pose (training only)
- Actions: 6 SO-101 actuators, no inverse kinematics
- Reward: reach object and lift it
- Objects: Mint prompt → local library, or primitive boxes/cylinders. Longest edge is 7 cm on import.

Not SmolVLA / ACT. Not the old browser IK toy arm.
