"""Off-policy evaluation utilities with IS/WIS/DR/DM, ESS, clipping, and bootstrap CI."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    def __len__(self):
        return len(self.rewards)


@dataclass
class OPEResult:
    value_estimate: float
    std_error: float
    confidence_interval: Tuple[float, float]
    method: str
    n_trajectories: int
    metadata: Dict


class OffPolicyEvaluator:
    def __init__(self, gamma: float = 0.99, clip_ratio: float = 20.0, n_bootstrap: int = 200, seed: int = 42, q_function=None):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(seed)
        self.q_function = q_function

    def _get_prob(self, policy, state, action, deterministic=False):
        if hasattr(policy, "get_action_probability"):
            p = float(policy.get_action_probability(state, action))
            return max(p, 1e-8)
        # fallback for deterministic policies
        if hasattr(policy, "select_action"):
            pa = policy.select_action(state, deterministic=True)
            return 1.0 if np.allclose(np.asarray(pa).reshape(-1), np.asarray(action).reshape(-1)) else 1e-8
        if callable(policy):
            pa = policy(state, deterministic=True)
            return 1.0 if np.allclose(np.asarray(pa).reshape(-1), np.asarray(action).reshape(-1)) else 1e-8
        raise TypeError("Policy does not provide get_action_probability/select_action/callable interface")

    def _trajectory_weight_and_return(self, policy, behavior_policy, traj):
        w = 1.0
        g = 0.0
        for t in range(len(traj)):
            pi = self._get_prob(policy, traj.states[t], traj.actions[t])
            b = self._get_prob(behavior_policy, traj.states[t], traj.actions[t])
            ratio = np.clip(pi / (b + 1e-8), 0.0, self.clip_ratio)
            w *= ratio
            g += (self.gamma ** t) * float(traj.rewards[t])
        return w, g

    def effective_sample_size(self, weights: np.ndarray) -> float:
        num = np.square(np.sum(weights))
        den = np.sum(np.square(weights)) + 1e-12
        return float(num / den)

    def bootstrap_ci(self, values: np.ndarray) -> Tuple[float, float, float]:
        if len(values) == 0:
            return 0.0, 0.0, 0.0
        idx = np.arange(len(values))
        boots = []
        for _ in range(self.n_bootstrap):
            sample = self.rng.choice(idx, size=len(idx), replace=True)
            boots.append(float(np.mean(values[sample])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(np.std(boots)), float(lo), float(hi)

    def evaluate(self, policy, behavior_policy, trajectories: List[Trajectory], methods=['is', 'wis', 'dr', 'dm']):
        if not trajectories:
            return {}

        weights, returns = [], []
        clipped_steps = 0
        total_steps = 0
        support_issues = 0
        ratios_by_traj: List[List[float]] = []

        for traj in trajectories:
            w = 1.0
            g = 0.0
            traj_ratios = []
            for t in range(len(traj)):
                pi = self._get_prob(policy, traj.states[t], traj.actions[t])
                b = self._get_prob(behavior_policy, traj.states[t], traj.actions[t])
                raw_ratio = pi / (b + 1e-8)
                ratio = np.clip(raw_ratio, 0.0, self.clip_ratio)
                if ratio < raw_ratio:
                    clipped_steps += 1
                total_steps += 1
                traj_ratios.append(float(ratio))
                w *= ratio
                g += (self.gamma ** t) * float(traj.rewards[t])

            weights.append(w)
            returns.append(g)
            ratios_by_traj.append(traj_ratios)
            if w == 0.0 or w >= self.clip_ratio ** max(1, len(traj) // 4):
                support_issues += 1

        weights = np.asarray(weights, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64)
        ess = self.effective_sample_size(weights)
        clip_fraction = float(clipped_steps / max(1, total_steps))

        warnings = []
        unreliable = False
        if ess < max(5.0, 0.1 * len(trajectories)):
            warnings.append(f"Low ESS detected ({ess:.2f}/{len(trajectories)}).")
            unreliable = True
        if support_issues / max(1, len(trajectories)) > 0.2:
            warnings.append("Potential support mismatch: many highly clipped/zero-weight trajectories.")
            unreliable = True
        if np.std(weights) > 10 * (np.mean(weights) + 1e-8):
            warnings.append("High variance in importance weights.")
            unreliable = True
        if clip_fraction > 0.5:
            warnings.append(f"High clipping fraction ({clip_fraction:.2%}).")
            unreliable = True

        base_meta = {
            'ess': ess,
            'clip_ratio': self.clip_ratio,
            'clip_fraction': clip_fraction,
            'reliability_flag': 'unreliable' if unreliable else 'reliable',
            'warnings': warnings,
        }

        results = {}
        if 'is' in methods:
            is_vals = weights * returns
            std, lo, hi = self.bootstrap_ci(is_vals)
            results['is'] = OPEResult(float(np.mean(is_vals)), std, (lo, hi), 'is', len(trajectories), dict(base_meta))

        if 'wis' in methods:
            wis = float(np.sum(weights * returns) / (np.sum(weights) + 1e-8))
            wis_vals = (weights * returns) / (np.mean(weights) + 1e-8)
            std, lo, hi = self.bootstrap_ci(wis_vals)
            results['wis'] = OPEResult(wis, std, (lo, hi), 'wis', len(trajectories), dict(base_meta))

        if 'dm' in methods:
            if self.q_function is None:
                logger.warning("DM requested but q_function is None; skipping DM")
            else:
                dm_vals = []
                for traj in trajectories:
                    v = 0.0
                    for t in range(len(traj)):
                        v += (self.gamma ** t) * float(self.q_function(traj.states[t], traj.actions[t]))
                    dm_vals.append(v)
                dm_vals = np.asarray(dm_vals, dtype=np.float64)
                std, lo, hi = self.bootstrap_ci(dm_vals)
                results['dm'] = OPEResult(float(np.mean(dm_vals)), std, (lo, hi), 'dm', len(trajectories), dict(base_meta))

        if 'dr' in methods:
            if self.q_function is None:
                logger.warning("DR requested but q_function is None; skipping DR")
            else:
                dr_vals = []
                for traj, traj_ratios in zip(trajectories, ratios_by_traj):
                    val = 0.0
                    for t in range(len(traj)):
                        s, a, r = traj.states[t], traj.actions[t], float(traj.rewards[t])
                        rho = float(traj_ratios[t])
                        q_sa = float(self.q_function(s, a))
                        val += (self.gamma ** t) * (rho * (r - q_sa) + q_sa)
                    dr_vals.append(val)
                dr_vals = np.asarray(dr_vals, dtype=np.float64)
                std, lo, hi = self.bootstrap_ci(dr_vals)
                results['dr'] = OPEResult(float(np.mean(dr_vals)), std, (lo, hi), 'dr', len(trajectories), dict(base_meta))
        return results


    def compare_methods(self, results):
        """Backward-compatible pretty print for OPE method results."""
        print("\n" + "=" * 70)
        print("OFF-POLICY EVALUATION COMPARISON")
        print("=" * 70)
        print(f"{'Method':<10} {'Value':<12} {'Std Error':<12} {'95% CI':<25}")
        print("-" * 70)
        for method_name, result in results.items():
            ci = result.confidence_interval
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            print(f"{method_name.upper():<10} {result.value_estimate:<12.3f} {result.std_error:<12.3f} {ci_str:<25}")
        print("=" * 70 + "\n")
