from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(64 * 7 * 7, 256)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(self.conv(img)))


class Actor(nn.Module):
    def __init__(self, n_act: int = 6) -> None:
        super().__init__()
        self.enc = Encoder()
        self.mlp = nn.Sequential(nn.Linear(256 + 6, 128), nn.Tanh())
        self.mu = nn.Linear(128, n_act)
        self.log_std = nn.Parameter(torch.zeros(n_act))

    def dist(self, img: torch.Tensor, joints: torch.Tensor) -> torch.distributions.Normal:
        h = torch.cat([self.enc(img), joints], dim=-1)
        mu = torch.tanh(self.mu(self.mlp(h)))
        std = self.log_std.clamp(-1.6, 0.4).exp()
        return torch.distributions.Normal(mu, std)

    def act(self, img: torch.Tensor, joints: torch.Tensor, deterministic: bool = False):
        d = self.dist(img, joints)
        a = d.mean if deterministic else d.sample()
        a = a.clamp(-1, 1)
        logp = d.log_prob(a).sum(-1)
        return a, logp


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = Encoder()
        self.mlp = nn.Sequential(
            nn.Linear(256 + 6 + 5, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, img: torch.Tensor, joints: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.enc(img), joints, priv], dim=-1)
        return self.mlp(h).squeeze(-1)
