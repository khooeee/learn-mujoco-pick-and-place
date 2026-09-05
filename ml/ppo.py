from __future__ import annotations

import torch
from torch import nn

from nets import Actor, Critic


def gae(rew: torch.Tensor, val: torch.Tensor, done: torch.Tensor, gamma=0.99, lam=0.95):
    adv = torch.zeros_like(rew)
    last = 0.0
    for t in reversed(range(len(rew))):
        next_v = 0.0 if t == len(rew) - 1 else val[t + 1]
        next_nonterminal = 0.0 if done[t] else 1.0
        delta = rew[t] + gamma * next_v * next_nonterminal - val[t]
        last = delta + gamma * lam * next_nonterminal * last
        adv[t] = last
    ret = adv + val
    return adv, ret


class PPO:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=2.5e-4,
        )

    def act(self, obs: dict, deterministic: bool = False):
        img = torch.from_numpy(obs["img"]).unsqueeze(0).to(self.device)
        joints = torch.from_numpy(obs["joints"]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a, logp = self.actor.act(img, joints, deterministic=deterministic)
            priv = torch.from_numpy(obs["priv"]).unsqueeze(0).to(self.device)
            v = self.critic(img, joints, priv)
        return a.squeeze(0).cpu().numpy(), float(logp.item()), float(v.item())

    def update(self, batch: dict) -> dict:
        img = batch["img"].to(self.device)
        joints = batch["joints"].to(self.device)
        priv = batch["priv"].to(self.device)
        act = batch["act"].to(self.device)
        old_logp = batch["logp"].to(self.device)
        adv = batch["adv"].to(self.device)
        ret = batch["ret"].to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        metrics = {}
        n = img.shape[0]
        idx = torch.randperm(n, device=self.device)
        mb = max(32, n // 4)
        for _ in range(4):
            for s in range(0, n, mb):
                sl = idx[s : s + mb]
                dist = self.actor.dist(img[sl], joints[sl])
                logp = dist.log_prob(act[sl]).sum(-1)
                ratio = (logp - old_logp[sl]).exp()
                surr1 = ratio * adv[sl]
                surr2 = ratio.clamp(0.8, 1.2) * adv[sl]
                policy_loss = -torch.min(surr1, surr2).mean()
                ent = dist.entropy().sum(-1).mean()
                v = self.critic(img[sl], joints[sl], priv[sl])
                value_loss = nn.functional.mse_loss(v, ret[sl])
                loss = policy_loss + 0.5 * value_loss - 0.01 * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    1.0,
                )
                self.opt.step()
                metrics = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(ent.item()),
                }
        return metrics

    def save(self, path) -> None:
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
