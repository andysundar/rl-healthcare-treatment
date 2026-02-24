"""
Policy Transfer / Domain Adaptation
=====================================
Implements PDF §3.3 / §6 step 6: adapt a source policy (trained on MIMIC/synthetic
data) to a new target population by learning a lightweight MLP adapter network.

Architecture
------------
raw_target_state  --[target encoder fϕ]--> target_emb
target_emb        --[PolicyTransferAdapter MLP]--> adapted_source_emb
adapted_source_emb --[source CQLAgent]--> action

Training loss
-------------
  L = w_bc * BCE/MSE(adapter_action, target_action)      # behavioural cloning
    + w_align * (1 - cosine_sim(adapted_emb, source_emb))  # embedding alignment

Classes
-------
TransferConfig          – dataclass for all hyperparameters
PolicyTransferAdapter   – nn.Module MLP: target_dim -> source_dim
PolicyTransferTrainer   – fit / transfer_select_action / save / load
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Type alias
Transition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransferConfig:
    """Hyperparameters for the policy transfer procedure."""
    source_state_dim: int = 64          # encoder output dim (source)
    target_state_dim: int = 64          # encoder output dim (target)
    adaptation_hidden_dim: int = 128
    adaptation_lr: float = 1e-3
    n_adaptation_steps: int = 1000
    adaptation_batch_size: int = 64
    bc_loss_weight: float = 1.0         # behavioural cloning weight
    align_loss_weight: float = 0.5      # cosine alignment weight


# ---------------------------------------------------------------------------
# Adapter network
# ---------------------------------------------------------------------------

class PolicyTransferAdapter(nn.Module):
    """
    Two-layer MLP that maps an encoded target-domain state to the
    source-domain embedding space so the source CQL agent can act on it.
    """

    def __init__(self, config: TransferConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.target_state_dim, config.adaptation_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.adaptation_hidden_dim, config.adaptation_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.adaptation_hidden_dim, config.source_state_dim),
        )

    def forward(self, target_emb: torch.Tensor) -> torch.Tensor:
        """Map target embedding -> source embedding space."""
        return self.net(target_emb)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PolicyTransferTrainer:
    """
    Fits the PolicyTransferAdapter so the source CQL policy can be applied to
    a target population.

    Training objective (per mini-batch):
      1. BC loss  : MSE( source_agent.select_action(adapter(target_emb)),
                         target_action )
      2. Align loss: 1 - cosine_similarity(adapter(target_emb), source_emb)
    """

    def __init__(
        self,
        source_agent,                       # CQLAgent
        source_encoder,                     # StateEncoderWrapper | None
        target_encoder,                     # StateEncoderWrapper | None
        config: TransferConfig,
        device: torch.device,
    ) -> None:
        self.source_agent = source_agent
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.config = config
        self.device = device

        self.adapter = PolicyTransferAdapter(config).to(device)
        self.optimizer = torch.optim.Adam(
            self.adapter.parameters(), lr=config.adaptation_lr
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, wrapper, raw_states: np.ndarray) -> np.ndarray:
        """Encode a batch of raw states; falls back to identity if no wrapper."""
        if wrapper is not None:
            return wrapper._encode_array(raw_states, batch_size=256)
        return raw_states

    def _get_source_actions(self, source_embs: np.ndarray) -> np.ndarray:
        """Query the source CQL agent for deterministic actions."""
        self.source_agent.eval_mode()
        actions = []
        for emb in source_embs:
            a = self.source_agent.select_action(emb, deterministic=True)
            actions.append(float(a) if np.ndim(a) == 0 else float(a[0]))
        return np.array(actions, dtype=np.float32)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        target_transitions: List[Transition],
        source_transitions: List[Transition],
    ) -> Dict:
        """
        Train the adapter.

        Parameters
        ----------
        target_transitions : list of (s, a, r, s', done)  raw target states
        source_transitions : list of (s, a, r, s', done)  raw source states

        Returns
        -------
        history : dict with keys 'bc_loss', 'align_loss', 'total_loss'
        """
        if not target_transitions or not source_transitions:
            raise ValueError("Both target and source transitions must be non-empty.")

        # Extract and encode states
        target_raw = np.array([t[0] for t in target_transitions], dtype=np.float32)
        source_raw = np.array([t[0] for t in source_transitions], dtype=np.float32)

        target_embs = self._encode(self.target_encoder, target_raw)
        source_embs = self._encode(self.source_encoder, source_raw)

        # Reference actions from target data (what the target population did)
        target_actions = np.array([
            float(t[1]) if np.ndim(t[1]) == 0 else float(t[1][0])
            for t in target_transitions
        ], dtype=np.float32)

        # Convert to tensors
        target_embs_t = torch.tensor(target_embs, dtype=torch.float32, device=self.device)
        source_embs_t = torch.tensor(source_embs, dtype=torch.float32, device=self.device)
        target_actions_t = torch.tensor(target_actions, dtype=torch.float32,
                                        device=self.device).unsqueeze(1)

        # Get source-policy reference actions for alignment
        source_ref_actions = self._get_source_actions(source_embs)
        source_ref_t = torch.tensor(source_ref_actions, dtype=torch.float32,
                                    device=self.device).unsqueeze(1)

        n = len(target_embs_t)
        bs = self.config.adaptation_batch_size
        history: Dict[str, List[float]] = {
            'bc_loss': [], 'align_loss': [], 'total_loss': []
        }

        self.adapter.train()
        for step in range(self.config.n_adaptation_steps):
            idx = np.random.choice(n, size=min(bs, n), replace=False)
            t_batch = target_embs_t[idx]
            s_batch = source_embs_t[idx % len(source_embs_t)]
            ta_batch = target_actions_t[idx]

            adapted = self.adapter(t_batch)  # [bs, source_dim]

            # BC loss: adapted embedding should reproduce target actions
            # (proxy: cosine-based action via mean of adapted vector)
            bc_loss = F.mse_loss(adapted.mean(dim=1, keepdim=True), ta_batch)

            # Alignment loss: adapted embedding should be close to source embedding
            cos_sim = F.cosine_similarity(adapted, s_batch, dim=-1)
            align_loss = (1.0 - cos_sim).mean()

            loss = (self.config.bc_loss_weight * bc_loss
                    + self.config.align_loss_weight * align_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            history['bc_loss'].append(float(bc_loss.detach()))
            history['align_loss'].append(float(align_loss.detach()))
            history['total_loss'].append(float(loss.detach()))

            if (step + 1) % max(1, self.config.n_adaptation_steps // 5) == 0:
                logger.info(
                    f"  Transfer step {step+1}/{self.config.n_adaptation_steps}  "
                    f"bc={bc_loss:.4f}  align={align_loss:.4f}"
                )

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transfer_select_action(
        self,
        raw_target_state: np.ndarray,
        deterministic: bool = True,
    ) -> float:
        """
        Select an action for a raw target-domain state using the adapted policy.

        Pipeline: raw_state -> target_encoder -> adapter -> source_agent
        """
        self.adapter.eval()
        with torch.no_grad():
            # Encode target state
            if self.target_encoder is not None:
                target_emb = self.target_encoder.encode_state(raw_target_state)
            else:
                target_emb = raw_target_state

            t = torch.tensor(target_emb, dtype=torch.float32, device=self.device).unsqueeze(0)
            adapted = self.adapter(t).squeeze(0).cpu().numpy()

        return float(self.source_agent.select_action(adapted, deterministic=deterministic))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save adapter weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.adapter.state_dict(), path)
        cfg_path = path.with_suffix('.json')
        with open(cfg_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        logger.info(f"Transfer adapter saved to {path}")

    def load(self, path: Path) -> None:
        """Load adapter weights from a checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.adapter.load_state_dict(state)
        logger.info(f"Transfer adapter loaded from {path}")
