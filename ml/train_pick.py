"""Train the same 3→32→3 pick IK clone on Apple GPU (optional).

The website already trains in the browser. Use this if you want to say
"PyTorch on the M5" in the demo.

  pip install torch
  python ml/train_pick.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError as e:
    raise SystemExit("pip install torch") from e

L1, L2 = 0.36, 0.32
BASE = ( -0.62, 0.04, 0.0 )


def ik(x: float, y: float, z: float) -> tuple[float, float, float]:
    dx, dy, dz = x - BASE[0], y - BASE[1], z - BASE[2]
    yaw = math.atan2(dx, dz)
    r = math.hypot(dx, dz)
    d = math.hypot(r, dy)
    d = min(max(d, abs(L1 - L2) + 0.01), L1 + L2 - 0.01)
    cos_e = max(-1.0, min(1.0, (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2)))
    elbow = math.pi - math.acos(cos_e)
    cos_a = max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)))
    shoulder = math.atan2(dy, r) + math.acos(cos_a)
    return yaw, shoulder, elbow


def sample() -> tuple[list[float], list[float]]:
    x = -0.18 + random.random() * 0.56
    z = -0.22 + random.random() * 0.44
    y = 0.10 + random.random() * 0.18
    yaw, sh, el = ik(x, y, z)
    inp = [(x - BASE[0]) / 0.9, y / 0.5, (z - BASE[2]) / 0.7]
    out = [yaw / math.pi, sh / math.pi, el / math.pi]
    return inp, out


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    xs, ys = [], []
    for _ in range(800):
        a, b = sample()
        xs.append(a)
        ys.append(b)
    x = torch.tensor(xs, device=device)
    y = torch.tensor(ys, device=device)
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for epoch in range(200):
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.5f}")

    cpu = {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}
    # Map to the browser MLP layout
    W1 = cpu["net.0.weight"]  # 32 x 3
    b1 = cpu["net.0.bias"]
    W2 = cpu["net.2.weight"]  # 3 x 32
    b2 = cpu["net.2.bias"]
    out = {
        "inSize": 3,
        "hidden": 32,
        "outSize": 3,
        "W1": [w for row in W1 for w in row],
        "b1": b1,
        "W2": [w for row in W2 for w in row],
        "b2": b2,
    }
    dest = Path("ml/policy.json")
    dest.write_text(json.dumps(out))
    print("wrote", dest)


if __name__ == "__main__":
    main()
