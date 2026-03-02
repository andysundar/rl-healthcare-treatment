"""Minimal Implicit Q-Learning for discrete offline action spaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class IQLConfig:
    state_dim: int
    n_actions: int
    hidden_dim: int = 128
    gamma: float = 0.99
    expectile: float = 0.7
    beta: float = 3.0
    lr: float = 3e-4
    device: str = "cpu"


class _MLP(nn.Module):
    def __init__(self, inp: int, out: int, hid: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, x):
        return self.net(x)


class IQLAgent:
    def __init__(self, config: IQLConfig):
        self.config = config
        d = torch.device(config.device)
        self.device = d
        self.q = _MLP(config.state_dim, config.n_actions, config.hidden_dim).to(d)
        self.v = _MLP(config.state_dim, 1, config.hidden_dim).to(d)
        self.pi = _MLP(config.state_dim, config.n_actions, config.hidden_dim).to(d)
        self.optim_q = torch.optim.Adam(self.q.parameters(), lr=config.lr)
        self.optim_v = torch.optim.Adam(self.v.parameters(), lr=config.lr)
        self.optim_pi = torch.optim.Adam(self.pi.parameters(), lr=config.lr)

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        logits = self.pi(s)
        if deterministic:
            a = torch.argmax(logits, dim=-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        return np.asarray([int(a.item())], dtype=np.float32)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        s = batch['states'].to(self.device)
        a = batch['actions'].long().view(-1).to(self.device)
        r = batch['rewards'].to(self.device).view(-1)
        sp = batch['next_states'].to(self.device)
        d = batch['dones'].to(self.device).view(-1)

        q_all = self.q(s)
        q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            v_next = self.v(sp).squeeze(1)
            target_q = r + self.config.gamma * (1.0 - d) * v_next
        q_loss = F.mse_loss(q_sa, target_q)
        self.optim_q.zero_grad(); q_loss.backward(); self.optim_q.step()

        with torch.no_grad():
            q_detach = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        v_pred = self.v(s).squeeze(1)
        diff = q_detach - v_pred
        w = torch.where(diff > 0, self.config.expectile, 1 - self.config.expectile)
        v_loss = (w * diff.pow(2)).mean()
        self.optim_v.zero_grad(); v_loss.backward(); self.optim_v.step()

        with torch.no_grad():
            adv = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1) - self.v(s).squeeze(1)
            weights = torch.exp(self.config.beta * adv).clamp(max=100.0)
        logp = F.log_softmax(self.pi(s), dim=-1).gather(1, a.unsqueeze(1)).squeeze(1)
        pi_loss = -(weights * logp).mean()
        self.optim_pi.zero_grad(); pi_loss.backward(); self.optim_pi.step()

        return {"q_loss": float(q_loss.item()), "v_loss": float(v_loss.item()), "pi_loss": float(pi_loss.item())}
