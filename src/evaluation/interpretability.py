"""
Policy Interpretability & Personalization Evaluation
=====================================================
Implements PDF §6 step 8 (interpretability) and §7.2 (personalization score).

Classes
-------
InterpretabilityConfig   - shared configuration dataclass
CounterfactualExplainer  - gradient-based minimal-perturbation explanations
DecisionRuleExtractor    - decision-tree policy approximation + rule extraction
PersonalizationScorer    - (1/T) Σ cos(fθ(st), fϕ(xt))  (PDF §7.2)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InterpretabilityConfig:
    """Shared configuration for all interpretability tools."""
    # Counterfactual settings
    n_counterfactuals: int = 5
    counterfactual_lr: float = 0.01
    counterfactual_steps: int = 200
    counterfactual_l1_weight: float = 0.1
    counterfactual_action_threshold: float = 0.1

    # Decision tree settings
    tree_max_depth: int = 4
    tree_min_samples_leaf: int = 10
    tree_mode: str = 'classifier'        # 'classifier' | 'regressor'
    action_bin_thresholds: Tuple[float, float] = (0.33, 0.67)

    # Feature names matching the 10-dim synthetic state
    feature_names: List[str] = field(default_factory=lambda: [
        'glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max',
        'insulin_mean', 'medication_taken', 'reminder_sent',
        'hypoglycemia', 'hyperglycemia', 'day',
    ])


# ---------------------------------------------------------------------------
# CounterfactualExplainer
# ---------------------------------------------------------------------------

class CounterfactualExplainer:
    """
    For a given raw state s, finds a minimal perturbation s' such that the
    policy changes its action by at least ``action_threshold``.

    Optimises a perturbation vector ``delta`` (nn.Parameter) via:
        loss = l1_weight * ||delta||_1  +  action_similarity_penalty
    where the penalty encourages the policy action on (s + delta) to differ
    from the original action.

    If an encoder_wrapper is provided, states are encoded before being passed
    to the agent; gradients flow back through the encoder into delta.
    """

    def __init__(
        self,
        agent,                              # CQLAgent
        encoder_wrapper,                    # StateEncoderWrapper | None
        config: InterpretabilityConfig,
        device: torch.device,
    ) -> None:
        self.agent = agent
        self.encoder_wrapper = encoder_wrapper
        self.config = config
        self.device = device

    def explain(self, raw_state: np.ndarray) -> List[Dict]:
        """
        Generate up to ``config.n_counterfactuals`` counterfactual explanations
        for the given raw state.

        Returns
        -------
        List of dicts, each with:
            original_state      : np.ndarray
            counterfactual_state: np.ndarray
            original_action     : float
            new_action          : float
            feature_changes     : Dict[str, float]   (non-zero deltas only)
            l1_distance         : float
        """
        original_action = self._get_action(raw_state)
        results = []

        for _ in range(self.config.n_counterfactuals):
            cf = self._optimize_counterfactual(raw_state, original_action)
            if cf is None:
                continue
            cf_state, cf_action, delta_arr = cf

            feature_changes = {
                self.config.feature_names[i]: float(delta_arr[i])
                for i in range(len(delta_arr))
                if abs(delta_arr[i]) > 1e-4
            }
            results.append({
                'original_state': raw_state,
                'counterfactual_state': cf_state,
                'original_action': float(original_action),
                'new_action': float(cf_action),
                'feature_changes': feature_changes,
                'l1_distance': float(np.abs(delta_arr).sum()),
            })

        return results

    def _get_action(self, raw_state: np.ndarray) -> float:
        """Query the agent for its action on a raw state."""
        if self.encoder_wrapper is not None:
            enc_state = self.encoder_wrapper.encode_state(raw_state)
        else:
            enc_state = raw_state
        a = self.agent.select_action(enc_state, deterministic=True)
        return float(a) if np.ndim(a) == 0 else float(np.asarray(a).flat[0])

    def _optimize_counterfactual(
        self,
        raw_state: np.ndarray,
        original_action: float,
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """
        Returns (cf_state, cf_action, delta) or None if optimisation fails.
        """
        state_t = torch.tensor(raw_state, dtype=torch.float32, device=self.device)
        original_action_t = torch.tensor([original_action], dtype=torch.float32,
                                         device=self.device)

        # Random init so each call produces a different counterfactual
        delta = torch.nn.Parameter(
            0.05 * torch.randn_like(state_t), requires_grad=True
        )
        optimizer = torch.optim.Adam([delta], lr=self.config.counterfactual_lr)

        self.agent.eval_mode()
        if self.encoder_wrapper is not None:
            self.encoder_wrapper.encoder.eval()

        for step in range(self.config.counterfactual_steps):
            optimizer.zero_grad()
            perturbed = state_t + delta

            # Encode if wrapper available (gradient flows through encoder)
            if self.encoder_wrapper is not None:
                enc = self.encoder_wrapper.encoder.encode(perturbed.unsqueeze(0)).squeeze(0)
            else:
                enc = perturbed

            # Get Q-value of original action on perturbed state
            q_orig = self.agent.get_q_value(enc.detach().cpu().numpy(),
                                            np.array([original_action], dtype=np.float32))
            q_orig_t = torch.tensor([q_orig], dtype=torch.float32, device=self.device)

            # Sparsity loss + action-change loss (maximise difference from original)
            l1_loss = delta.abs().sum()
            action_loss = -torch.abs(enc.mean() - original_action_t.mean())  # proxy

            loss = self.config.counterfactual_l1_weight * l1_loss + action_loss
            loss.backward()
            optimizer.step()

        # Evaluate result
        delta_arr = delta.detach().cpu().numpy()
        cf_state = raw_state + delta_arr
        cf_action = self._get_action(cf_state)

        if abs(cf_action - original_action) < self.config.counterfactual_action_threshold:
            return None   # did not change action enough

        return cf_state, cf_action, delta_arr


# ---------------------------------------------------------------------------
# DecisionRuleExtractor
# ---------------------------------------------------------------------------

class DecisionRuleExtractor:
    """
    Approximates the RL policy with a shallow sklearn decision tree and
    extracts human-readable if-then rules for clinical audit.

    Continuous CQL actions are binned into {low, medium, high} classes before
    fitting the tree (or left as-is for the regressor variant).
    """

    def __init__(self, config: InterpretabilityConfig) -> None:
        self.config = config
        self._tree = None
        self._feature_names = config.feature_names
        self._resolved_feature_names: List[str] = []

    def _get_feature_names_for_tree(self) -> List[str]:
        """Return feature names aligned with fitted tree input dimensionality."""
        if self._tree is None:
            raise RuntimeError("Call fit() first.")

        n_tree_features = int(self._tree.n_features_in_)
        configured_names = list(self._feature_names)

        if len(configured_names) >= n_tree_features:
            return configured_names[:n_tree_features]

        # Some pipelines (e.g., encoded state vectors) increase dimensionality.
        # Preserve known names and auto-name any additional dimensions.
        missing = n_tree_features - len(configured_names)
        extra_names = [
            f'feature_{len(configured_names) + i}' for i in range(missing)
        ]
        return configured_names + extra_names

    def fit(
        self,
        states: np.ndarray,
        policy_actions: Optional[np.ndarray] = None,
        agent=None,
        encoder_wrapper=None,
    ) -> DecisionRuleExtractor:
        """
        Fit the surrogate decision tree.

        If ``policy_actions`` is None, the agent is queried on ``states``
        (after optional encoding) to get actions.
        """
        if policy_actions is None:
            if agent is None:
                raise ValueError("Provide either policy_actions or an agent.")
            policy_actions = self._query_agent(states, agent, encoder_wrapper)

        if self.config.tree_mode == 'classifier':
            labels = self._bin_actions(policy_actions)
            from sklearn.tree import DecisionTreeClassifier
            self._tree = DecisionTreeClassifier(
                max_depth=self.config.tree_max_depth,
                min_samples_leaf=self.config.tree_min_samples_leaf,
            ).fit(states, labels)
        else:
            from sklearn.tree import DecisionTreeRegressor
            self._tree = DecisionTreeRegressor(
                max_depth=self.config.tree_max_depth,
                min_samples_leaf=self.config.tree_min_samples_leaf,
            ).fit(states, policy_actions)

        logger.info(f"Decision tree fitted on {len(states)} samples "
                    f"(depth={self._tree.get_depth()}, "
                    f"leaves={self._tree.get_n_leaves()})")
        self._resolved_feature_names = self._get_feature_names_for_tree()
        return self

    def _query_agent(self, states, agent, encoder_wrapper) -> np.ndarray:
        agent.eval_mode()
        actions = []
        for s in states:
            if encoder_wrapper is not None:
                enc_s = encoder_wrapper.encode_state(s)
            else:
                enc_s = s
            a = agent.select_action(enc_s, deterministic=True)
            actions.append(float(a) if np.ndim(a) == 0 else float(a[0]))
        return np.array(actions, dtype=np.float32)

    def _bin_actions(self, actions: np.ndarray) -> np.ndarray:
        lo, hi = self.config.action_bin_thresholds
        labels = np.full(len(actions), 'medium', dtype=object)
        labels[actions <= lo] = 'low'
        labels[actions >= hi] = 'high'
        return labels

    def extract_rules(self) -> List[str]:
        """Return human-readable if-then rules from the fitted tree."""
        if self._tree is None:
            raise RuntimeError("Call fit() first.")
        from sklearn.tree import export_text
        rule_text = export_text(
            self._tree,
            feature_names=self._resolved_feature_names,
        )
        # Split into individual rule lines
        rules = [line.strip() for line in rule_text.splitlines() if line.strip()]
        return rules

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance dict keyed by feature name."""
        if self._tree is None:
            raise RuntimeError("Call fit() first.")
        importances = self._tree.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self._resolved_feature_names, importances)
        }

    def fidelity_score(
        self,
        states: np.ndarray,
        agent,
        encoder_wrapper=None,
    ) -> float:
        """
        Fraction of policy decisions the surrogate tree correctly replicates.
        """
        if self._tree is None:
            raise RuntimeError("Call fit() first.")
        policy_actions = self._query_agent(states, agent, encoder_wrapper)
        if self.config.tree_mode == 'classifier':
            labels = self._bin_actions(policy_actions)
            tree_preds = self._tree.predict(states)
            return float(np.mean(labels == tree_preds))
        else:
            tree_preds = self._tree.predict(states)
            # Relative accuracy: 1 - normalised MAE
            mae = float(np.mean(np.abs(policy_actions - tree_preds)))
            scale = float(np.std(policy_actions)) + 1e-8
            return max(0.0, 1.0 - mae / scale)

    def save_rules(self, path: Path) -> None:
        """Save rules as plain text and feature importances as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rules = self.extract_rules()
        with open(path, 'w') as f:
            f.write("POLICY DECISION RULES\n")
            f.write("=" * 60 + "\n")
            f.write("\n".join(rules) + "\n\n")
            f.write("FEATURE IMPORTANCES\n")
            f.write("=" * 60 + "\n")
            for name, imp in sorted(
                self.get_feature_importance().items(), key=lambda x: -x[1]
            ):
                f.write(f"  {name:<25} {imp:.4f}\n")

        json_path = path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.get_feature_importance(), f, indent=2)

        logger.info(f"Rules saved to {path} and {json_path}")


# ---------------------------------------------------------------------------
# PersonalizationScorer  (PDF §7.2)
# ---------------------------------------------------------------------------

class PersonalizationScorer:
    """
    Computes the personalization score from the M.Tech PDF proposal (§7.2):

        score = (1/T) * Σ_t  cosine_similarity( fθ(s_t),  fϕ(x_t) )

    where fθ = source encoder and fϕ = target encoder.

    In single-population mode both encoders can be the same instance; the
    score then measures temporal embedding self-consistency.
    """

    def __init__(self, source_encoder, target_encoder) -> None:
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder

    def compute(
        self,
        source_trajectory: List[np.ndarray],
        target_trajectory: List[np.ndarray],
    ) -> float:
        """
        Compute the personalization score for a single pair of matched
        trajectories (same length T).

        Parameters
        ----------
        source_trajectory : list of np.ndarray  (raw states, source domain)
        target_trajectory : list of np.ndarray  (raw states, target domain)

        Returns
        -------
        float : mean cosine similarity across time steps
        """
        T = min(len(source_trajectory), len(target_trajectory))
        if T == 0:
            return 0.0

        source_states = np.array(source_trajectory[:T], dtype=np.float32)
        target_states = np.array(target_trajectory[:T], dtype=np.float32)

        source_embs = self.source_encoder._encode_array(source_states, batch_size=256)
        target_embs = self.target_encoder._encode_array(target_states, batch_size=256)

        source_t = torch.tensor(source_embs, dtype=torch.float32)
        target_t = torch.tensor(target_embs, dtype=torch.float32)

        cos_sim = F.cosine_similarity(source_t, target_t, dim=-1)  # [T]
        return float(cos_sim.mean().item())

    def compute_batch(
        self,
        source_trajs: List[List[np.ndarray]],
        target_trajs: List[List[np.ndarray]],
    ) -> Dict:
        """
        Compute the personalization score over a batch of trajectory pairs.

        Returns
        -------
        dict with keys: 'mean', 'std', 'per_trajectory'
        """
        n = min(len(source_trajs), len(target_trajs))
        per_traj = [self.compute(source_trajs[i], target_trajs[i]) for i in range(n)]
        arr = np.array(per_traj)
        return {
            'mean': float(arr.mean()) if len(arr) > 0 else 0.0,
            'std':  float(arr.std())  if len(arr) > 0 else 0.0,
            'per_trajectory': per_traj,
        }
