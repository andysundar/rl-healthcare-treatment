"""
Standalone off-policy evaluation module: WIS, DR, and DM estimators.

Loads a saved CQL checkpoint and evaluates the trained policy against a
behaviour policy fitted from the logged training trajectories.

Estimators
----------
WIS  – Weighted Importance Sampling (self-normalised, clipped per-trajectory).
DM   – Direct Method using the CQL Q-network as the value model.
DR   – Doubly Robust, combining DM as control variate with IS correction.

All three return :class:`OPEResult` objects (re-used from off_policy_eval.py)
with point estimates and 95 % bootstrap CIs (default n=1000).

CLI usage
---------
python -m evaluation.off_policy_evaluation \\
    --checkpoint outputs/run/cql/best_model.pt \\
    --pipeline-dir outputs/run/ \\
    [--use-encoder] [--gamma 0.99] [--n-bootstrap 1000]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow direct execution: `python src/evaluation/off_policy_evaluation.py ...`
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluation.off_policy_eval import OffPolicyEvaluator, OPEResult, Trajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Behaviour policy
# ---------------------------------------------------------------------------

class BehaviorPolicy:
    """
    Gaussian behaviour-cloning policy estimated from logged transitions.

    Fits a ridge-regression model  mu(s) = W @ s + b  on the (state, action)
    pairs from the training dataset, then treats the action probability as

        p_b(a | s) = N(a ; mu(s), sigma^2)

    where sigma is estimated from the residuals of the regression.

    Parameters
    ----------
    sigma_floor : float
        Minimum standard deviation (prevents degenerate densities when actions
        are nearly deterministic).
    lam : float
        L2 regularisation weight for ridge regression.
    """

    def __init__(self, sigma_floor: float = 0.05, lam: float = 1e-3):
        self._W: Optional[np.ndarray] = None   # [state_dim]
        self._b: float = 0.0
        self._sigma: float = sigma_floor
        self._sigma_floor = sigma_floor
        self._lam = lam
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, transitions: List[Tuple]) -> "BehaviorPolicy":
        """Fit on a list of (s, a, r, s', done) tuples."""
        states = np.array(
            [np.asarray(t[0], dtype=np.float32).reshape(-1) for t in transitions],
            dtype=np.float32,
        )
        actions = np.array(
            [float(np.asarray(t[1]).reshape(-1)[0]) for t in transitions],
            dtype=np.float32,
        )

        # Design matrix with bias column
        X = np.column_stack(
            [states, np.ones(len(states), dtype=np.float32)]
        )  # [N, state_dim+1]

        # Ridge regression: (X^T X + lam I)^{-1} X^T a
        A = X.T @ X + self._lam * np.eye(X.shape[1], dtype=np.float32)
        b = X.T @ actions
        coef = np.linalg.solve(A, b)

        self._W = coef[:-1]   # [state_dim]
        self._b = float(coef[-1])

        # Residual standard deviation
        residuals = actions - (states @ self._W + self._b)
        self._sigma = float(max(np.std(residuals), self._sigma_floor))
        self._fitted = True

        logger.info(
            "BehaviorPolicy fitted on %d transitions "
            "(sigma=%.4f, b=%.4f)",
            len(transitions), self._sigma, self._b,
        )
        return self

    # ------------------------------------------------------------------
    def _mean(self, state: np.ndarray) -> float:
        if not self._fitted or self._W is None:
            return 0.5
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.shape[0] != self._W.shape[0]:
            return 0.5
        return float(s @ self._W + self._b)

    def get_action_probability(self, state: np.ndarray, action) -> float:
        a   = float(np.asarray(action).reshape(-1)[0])
        mu  = self._mean(state)
        z   = (a - mu) / self._sigma
        # Gaussian pdf value (unnormalised direction: we only care about ratios)
        log_p = -0.5 * z * z - np.log(self._sigma * np.sqrt(2.0 * np.pi))
        return max(float(np.exp(log_p)), 1e-8)

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.array([self._mean(state)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Target-policy wrapper
# ---------------------------------------------------------------------------

class _GaussianPolicyWrapper:
    """
    Wraps a CQLAgent to expose ``get_action_probability``.

    The CQL policy is treated as near-deterministic; we place a narrow
    Gaussian  N(a; pi(s), sigma^2)  around its deterministic output so that
    the evaluator can form well-conditioned importance ratios.
    """

    def __init__(self, agent, sigma: float = 0.05):
        self._agent = agent
        self._sigma = sigma
        self.state_dim: int = int(agent.state_dim)

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._agent.select_action(
            np.asarray(state, dtype=np.float32).reshape(-1),
            deterministic=deterministic,
        )

    def get_action_probability(self, state: np.ndarray, action) -> float:
        a_pi = float(
            np.asarray(self.select_action(state), dtype=np.float32).reshape(-1)[0]
        )
        a    = float(np.asarray(action).reshape(-1)[0])
        z    = (a - a_pi) / self._sigma
        log_p = -0.5 * z * z - np.log(self._sigma * np.sqrt(2.0 * np.pi))
        return max(float(np.exp(log_p)), 1e-8)


# ---------------------------------------------------------------------------
# Estimator classes
# ---------------------------------------------------------------------------

class WISEstimator:
    """
    Weighted Importance Sampling (WIS) estimator.

    Uses per-trajectory cumulative importance ratios (product of step-level
    pi/b ratios, clipped at ``clip_ratio``) and self-normalises across
    trajectories to reduce variance:

        WIS = sum_i(w_i * G_i) / sum_i(w_i)

    Parameters
    ----------
    gamma : float
        Discount factor.
    clip_ratio : float
        Maximum per-trajectory importance weight before normalisation.
    n_bootstrap : int
        Bootstrap resamples for 95 % CI.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        clip_ratio: float = 10.0,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ):
        self._eval = OffPolicyEvaluator(
            gamma=gamma,
            clip_ratio=clip_ratio,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    def estimate(
        self,
        trajectories: List[Trajectory],
        target_policy,
        behavior_policy,
    ) -> OPEResult:
        results = self._eval.evaluate(
            target_policy, behavior_policy, trajectories, methods=["wis"]
        )
        return results["wis"]


class DMEstimator:
    """
    Direct Method (DM) estimator.

    Estimates the policy value purely from a learned Q-model — in this case
    the CQL Q-network:

        DM = E_tau[ sum_t gamma^t * Q(s_t, a_t) ]

    No importance weights are needed, so variance is low; bias depends on
    Q-model quality.

    Parameters
    ----------
    q_function : Callable[[np.ndarray, np.ndarray], float]
        A callable ``(state, action) -> scalar`` Q-value estimate.
    gamma : float
        Discount factor.
    n_bootstrap : int
        Bootstrap resamples for 95 % CI.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        q_function: Callable,
        gamma: float = 0.99,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ):
        self._eval = OffPolicyEvaluator(
            gamma=gamma,
            n_bootstrap=n_bootstrap,
            seed=seed,
            q_function=q_function,
        )

    def estimate(self, trajectories: List[Trajectory]) -> OPEResult:
        # DM needs no policy arguments; pass dummies to satisfy the API.
        _dummy = _NullPolicy()
        results = self._eval.evaluate(_dummy, _dummy, trajectories, methods=["dm"])
        return results["dm"]


class DREstimator:
    """
    Doubly Robust (DR) estimator.

    Combines a Q-model control variate with an IS correction:

        DR = sum_t gamma^t [ rho_t * (r_t - Q(s_t, a_t)) + Q(s_t, a_t) ]

    where ``rho_t = pi(a_t|s_t) / b(a_t|s_t)`` is the per-step ratio.
    Consistent when either the behaviour policy or the Q-model is correct.

    Parameters
    ----------
    q_function : Callable[[np.ndarray, np.ndarray], float]
        A callable ``(state, action) -> scalar``.
    gamma : float
        Discount factor.
    clip_ratio : float
        Clipping threshold applied to per-step importance ratios.
    n_bootstrap : int
        Bootstrap resamples for 95 % CI.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        q_function: Callable,
        gamma: float = 0.99,
        clip_ratio: float = 10.0,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ):
        self._eval = OffPolicyEvaluator(
            gamma=gamma,
            clip_ratio=clip_ratio,
            n_bootstrap=n_bootstrap,
            seed=seed,
            q_function=q_function,
        )

    def estimate(
        self,
        trajectories: List[Trajectory],
        target_policy,
        behavior_policy,
    ) -> OPEResult:
        results = self._eval.evaluate(
            target_policy, behavior_policy, trajectories, methods=["dr"]
        )
        return results["dr"]


class _NullPolicy:
    """Placeholder that satisfies the policy interface without doing anything."""

    def select_action(self, state, deterministic=True):
        return np.array([0.0], dtype=np.float32)

    def get_action_probability(self, state, action) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class OPERunner:
    """
    Orchestrates WIS + DM + DR over a fixed offline dataset.

    Workflow
    --------
    1. Optionally encode both splits if an encoder wrapper is provided.
    2. Fit a :class:`BehaviorPolicy` on the (optionally encoded) training data.
    3. Convert the test transitions to :class:`Trajectory` objects.
    4. Build the Q-function closure from the CQL agent.
    5. Run all three estimators and return a results dict.

    Parameters
    ----------
    gamma, clip_ratio, n_bootstrap, seed
        Passed through to each estimator.
    target_policy_sigma : float
        Width of the Gaussian placed around the target policy's deterministic
        action when computing importance ratios.
    behavior_policy_sigma_floor : float
        Minimum residual sigma for the behaviour policy estimator.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        clip_ratio: float = 10.0,
        n_bootstrap: int = 1000,
        seed: int = 42,
        target_policy_sigma: float = 0.05,
        behavior_policy_sigma_floor: float = 0.05,
    ):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.target_policy_sigma = target_policy_sigma
        self.behavior_policy_sigma_floor = behavior_policy_sigma_floor

    # ------------------------------------------------------------------
    def run(
        self,
        train_transitions: List[Tuple],
        test_transitions: List[Tuple],
        cql_agent,
        encoder_wrapper=None,
    ) -> Dict[str, OPEResult]:
        """
        Run WIS, DM, and DR estimators.

        Parameters
        ----------
        train_transitions : list of (s, a, r, s', done) tuples
            Used only for fitting the behaviour policy.
        test_transitions : list of (s, a, r, s', done) tuples
            Held-out evaluation data.
        cql_agent : CQLAgent
            Trained agent.  Its Q-networks are used for DM/DR; its policy for
            the target-policy importance ratios in WIS/DR.
        encoder_wrapper : StateEncoderWrapper | None
            When provided, both splits are re-encoded in-place before any
            downstream processing so that states match the agent's input dim.

        Returns
        -------
        dict
            Keys ``'wis'``, ``'dm'``, ``'dr'`` → :class:`OPEResult`.
        """
        # 1. Encode both splits once so every downstream component sees states
        #    in the same representation that the CQL agent was trained on.
        if encoder_wrapper is not None:
            logger.info(
                "Encoding %d train + %d test transitions via encoder_wrapper…",
                len(train_transitions), len(test_transitions),
            )
            train_transitions = encoder_wrapper.encode_transitions(train_transitions)
            test_transitions  = encoder_wrapper.encode_transitions(test_transitions)

        # 2. Behaviour policy – fitted on (encoded) training transitions.
        logger.info("Fitting BehaviorPolicy on %d training transitions…", len(train_transitions))
        behavior_policy = BehaviorPolicy(
            sigma_floor=self.behavior_policy_sigma_floor,
        ).fit(train_transitions)

        # 3. Convert flat transitions to episodic Trajectory objects.
        logger.info("Building trajectories from %d test transitions…", len(test_transitions))
        trajectories = _transitions_to_trajectories(test_transitions)
        logger.info(
            "  → %d trajectories, mean length %.1f steps",
            len(trajectories),
            np.mean([len(t) for t in trajectories]) if trajectories else 0.0,
        )

        # 4. Q-function closure – delegates to CQL min-Q.
        q_fn = _build_q_function(cql_agent)

        # 5. Target policy – Gaussian wrapper around CQL deterministic action.
        target_policy = _GaussianPolicyWrapper(
            cql_agent, sigma=self.target_policy_sigma
        )

        # 6. Run estimators (different seeds so bootstrap samples are independent).
        logger.info("Running WIS estimator (n_bootstrap=%d)…", self.n_bootstrap)
        wis_result = WISEstimator(
            gamma=self.gamma,
            clip_ratio=self.clip_ratio,
            n_bootstrap=self.n_bootstrap,
            seed=self.seed,
        ).estimate(trajectories, target_policy, behavior_policy)

        logger.info("Running DM estimator (n_bootstrap=%d)…", self.n_bootstrap)
        dm_result = DMEstimator(
            q_function=q_fn,
            gamma=self.gamma,
            n_bootstrap=self.n_bootstrap,
            seed=self.seed + 1,
        ).estimate(trajectories)

        logger.info("Running DR estimator (n_bootstrap=%d)…", self.n_bootstrap)
        dr_result = DREstimator(
            q_function=q_fn,
            gamma=self.gamma,
            clip_ratio=self.clip_ratio,
            n_bootstrap=self.n_bootstrap,
            seed=self.seed + 2,
        ).estimate(trajectories, target_policy, behavior_policy)

        return {"wis": wis_result, "dm": dm_result, "dr": dr_result}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transitions_to_trajectories(
    transitions: List[Tuple],
    max_ep_steps: int = 500,
) -> List[Trajectory]:
    """
    Group flat ``(s, a, r, s', done)`` tuples into episodic
    :class:`Trajectory` objects.

    Episode boundaries are determined by ``done=True``.  If a long stretch of
    transitions accumulates without a ``done`` flag (e.g. truncated episodes),
    it is flushed as its own trajectory after ``max_ep_steps`` steps.
    """
    episodes: List[Dict] = []
    current: Optional[Dict] = None

    def _flush(ep: Dict) -> None:
        if ep and ep["states"]:
            episodes.append(ep)

    for t in transitions:
        s      = np.asarray(t[0], dtype=np.float32).reshape(-1)
        a      = np.asarray(t[1], dtype=np.float32).reshape(-1)
        r      = float(t[2])
        s_next = np.asarray(t[3], dtype=np.float32).reshape(-1)
        done   = bool(t[4])

        if current is None:
            current = {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []}

        current["states"].append(s)
        current["actions"].append(a)
        current["rewards"].append(r)
        current["next_states"].append(s_next)
        current["dones"].append(done)

        if done or len(current["states"]) >= max_ep_steps:
            _flush(current)
            current = None

    _flush(current)  # open episode at end of dataset

    return [
        Trajectory(
            states=np.stack(ep["states"]),
            actions=np.stack(ep["actions"]),
            rewards=np.asarray(ep["rewards"], dtype=np.float32),
            next_states=np.stack(ep["next_states"]),
            dones=np.asarray(ep["dones"], dtype=bool),
        )
        for ep in episodes
    ]


def _build_q_function(cql_agent) -> Callable:
    """
    Return a ``(state, action) -> float`` closure around ``cql_agent.get_q_value``.

    The CQL agent takes ``min(Q1, Q2)`` which gives a conservative lower-bound
    estimate – exactly what we want for the DM / DR control variate.
    """
    def q_fn(state: np.ndarray, action) -> float:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        return float(cql_agent.get_q_value(s, a))

    return q_fn


def load_cql_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """
    Reconstruct a :class:`CQLAgent` from a ``.pt`` checkpoint file.

    The file must have been written by ``BaseRLAgent.save()``, i.e. contain
    the keys ``state_dim``, ``action_dim``, ``gamma``, and ``model_state``.
    """
    from models.rl import CQLAgent  # lazy import keeps module importable without torch

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim  = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    gamma      = float(ckpt.get("gamma", 0.99))

    agent = CQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        device=device,
    )
    agent._set_model_state(ckpt["model_state"])
    agent.q_network1.eval()
    agent.q_network2.eval()
    agent.policy.eval()
    logger.info(
        "Loaded CQLAgent (state_dim=%d, step=%s) from %s",
        state_dim, ckpt.get("training_step", "?"), checkpoint_path,
    )
    return agent


def load_encoder(checkpoint_path: str, raw_state_dim: int, device: str = "cpu"):
    """
    Load a :class:`StateEncoderWrapper` from an ``encoder_best.pt`` file.

    Returns ``None`` if the file does not exist (encoder is optional).
    """
    if not Path(checkpoint_path).exists():
        logger.warning("Encoder checkpoint not found at %s — running without encoder.", checkpoint_path)
        return None

    from models.encoders import PatientAutoencoder, EncoderConfig
    from models.encoders.state_encoder_wrapper import StateEncoderWrapper

    # Recover latent dim by inspecting the saved state_dict.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    latent_dim = 64  # conservative default
    for key, val in ckpt.items():
        if "weight" in key.lower():
            # The decoder's first linear layer has shape [raw_dim, latent_dim]
            # or the encoder's last has shape [latent_dim, ...]; take the min.
            candidate = int(min(val.shape))
            if 8 <= candidate <= 256:          # sanity bounds
                latent_dim = candidate
                break

    enc_cfg = EncoderConfig(
        lab_dim=raw_state_dim,
        vital_dim=0,
        demo_dim=0,
        state_dim=latent_dim,
    )
    ae      = PatientAutoencoder(enc_cfg)
    wrapper = StateEncoderWrapper(ae, device=torch.device(device), raw_state_dim=raw_state_dim)
    wrapper.load_checkpoint(checkpoint_path)
    wrapper.encoder.eval()
    logger.info(
        "Loaded encoder (raw_dim=%d → latent_dim=%d) from %s",
        raw_state_dim, latent_dim, checkpoint_path,
    )
    return wrapper


def _load_stage_pkl(path: Path):
    """Unpickle a pipeline stage checkpoint and return the inner payload dict."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    # Stage checkpoints are stored as {key: value} dicts by the runner
    return obj if isinstance(obj, dict) else {"data": obj}


def _split_flat_list(all_data: List[Tuple], train_frac: float = 0.85):
    n     = len(all_data)
    split = int(train_frac * n)
    return all_data[:split], all_data[split:]


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary_table(results: Dict[str, OPEResult]) -> None:
    """Print a formatted summary table to stdout."""
    _ORDER  = ["wis", "dm", "dr"]
    _LABELS = {"wis": "WIS", "dm": "DM (CQL Q-net)", "dr": "DR"}

    col_w = [16, 12, 10, 14, 14, 8]
    sep   = "=" * sum(col_w)

    print()
    print(sep)
    print("  OFF-POLICY EVALUATION — SUMMARY")
    print(sep)
    print(
        f"{'Estimator':<{col_w[0]}}"
        f"{'Value':>{col_w[1]}}"
        f"{'Std Err':>{col_w[2]}}"
        f"{'CI lower':>{col_w[3]}}"
        f"{'CI upper':>{col_w[4]}}"
        f"{'N traj':>{col_w[5]}}"
    )
    print("-" * sum(col_w))

    for key in _ORDER:
        if key not in results:
            continue
        r       = results[key]
        lo, hi  = r.confidence_interval
        label   = _LABELS[key]
        print(
            f"{label:<{col_w[0]}}"
            f"{r.value_estimate:>{col_w[1]}.4f}"
            f"{r.std_error:>{col_w[2]}.4f}"
            f"{lo:>{col_w[3]}.4f}"
            f"{hi:>{col_w[4]}.4f}"
            f"{r.n_trajectories:>{col_w[5]}d}"
        )

    print(sep)

    # Reliability annotations
    for key in _ORDER:
        if key not in results:
            continue
        r        = results[key]
        flag     = r.metadata.get("reliability_flag", "unknown")
        ess      = r.metadata.get("ess", float("nan"))
        clip_frc = r.metadata.get("clip_fraction", float("nan"))
        status   = "OK  " if flag == "reliable" else "WARN"
        label    = _LABELS[key]
        line     = f"  [{status}] {label}: ESS={ess:.1f}, clip_frac={clip_frc:.1%}"
        warns    = r.metadata.get("warnings", [])
        if warns:
            line += " | " + "; ".join(warns)
        print(line)

    print()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> Dict[str, OPEResult]:
    """
    CLI entry point — load a CQL checkpoint and run all three estimators.

    Data loading priority
    ~~~~~~~~~~~~~~~~~~~~~
    1. If ``--pipeline-dir`` contains ``checkpoints/pipeline/stage_2b_encoder_training.pkl``
       *and* ``--use-encoder`` is set, the pre-encoded train/test splits are used
       directly (no re-encoding overhead).
    2. Otherwise the raw train/test splits from
       ``checkpoints/pipeline/stage_1_data_preparation.pkl`` are used, and states
       are encoded on-the-fly if ``--use-encoder`` is set.
    3. If neither stage-1 nor stage-2b checkpoints exist, the module tries to
       fall back to ``synthetic_data.pkl`` in ``--pipeline-dir``.
    """
    parser = argparse.ArgumentParser(
        description="Run WIS / DM / DR off-policy evaluation on a saved CQL checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to CQL checkpoint .pt file (e.g., outputs/run/cql/best_model.pt)",
    )
    parser.add_argument(
        "--pipeline-dir", required=True,
        help="Root output directory of the run that produced the checkpoint.",
    )
    parser.add_argument(
        "--gamma",       type=float, default=0.99,  help="Discount factor."
    )
    parser.add_argument(
        "--clip-ratio",  type=float, default=10.0,  help="IS weight clipping threshold."
    )
    parser.add_argument(
        "--n-bootstrap", type=int,   default=1000,  help="Bootstrap resamples for 95 %% CI."
    )
    parser.add_argument(
        "--seed",        type=int,   default=42,    help="Random seed."
    )
    parser.add_argument(
        "--device",      default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device.",
    )
    parser.add_argument(
        "--use-encoder", action="store_true",
        help="Load encoder from <pipeline-dir>/encoder/encoder_best.pt "
             "and encode states before evaluation.",
    )
    parser.add_argument(
        "--raw-state-dim", type=int, default=10,
        help="Dimension of the raw (pre-encoder) state vector. "
             "Used only when loading the encoder (--use-encoder).",
    )
    parser.add_argument(
        "--target-sigma",   type=float, default=0.05,
        help="Gaussian sigma around the CQL policy action (for IS ratios).",
    )
    parser.add_argument(
        "--behavior-sigma-floor", type=float, default=0.05,
        help="Minimum residual sigma for the behaviour policy estimator.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    pipeline_dir = Path(args.pipeline_dir)

    # ------------------------------------------------------------------
    # 1. CQL agent
    # ------------------------------------------------------------------
    agent = load_cql_checkpoint(args.checkpoint, device=args.device)

    # ------------------------------------------------------------------
    # 2. Encoder (optional)
    # ------------------------------------------------------------------
    encoder = None
    if args.use_encoder:
        enc_path = pipeline_dir / "encoder" / "encoder_best.pt"
        encoder  = load_encoder(str(enc_path), raw_state_dim=args.raw_state_dim, device=args.device)

    # ------------------------------------------------------------------
    # 3. Data — check pipeline checkpoints in priority order
    # ------------------------------------------------------------------
    stage2b_pkl = pipeline_dir / "checkpoints" / "pipeline" / "stage_2b_encoder_training.pkl"
    stage1_pkl  = pipeline_dir / "checkpoints" / "pipeline" / "stage_1_data_preparation.pkl"
    syn_pkl     = pipeline_dir / "synthetic_data.pkl"

    train_data: List[Tuple] = []
    test_data:  List[Tuple] = []

    if args.use_encoder and stage2b_pkl.exists():
        logger.info("Loading pre-encoded splits from %s", stage2b_pkl)
        stage2b = _load_stage_pkl(stage2b_pkl)
        enc_data = stage2b.get("encoded_data") or stage2b.get("data")
        if isinstance(enc_data, dict) and "train" in enc_data:
            train_data = enc_data["train"]
            test_data  = enc_data.get("test", enc_data.get("val", []))
            logger.info(
                "Pre-encoded splits loaded (train=%d, test=%d). "
                "Skipping re-encoding.",
                len(train_data), len(test_data),
            )
            encoder = None  # states already encoded

    if not train_data and stage1_pkl.exists():
        logger.info("Loading raw splits from %s", stage1_pkl)
        stage1 = _load_stage_pkl(stage1_pkl)
        raw    = stage1.get("data") or stage1
        if isinstance(raw, dict) and "train" in raw:
            train_data = raw["train"]
            test_data  = raw.get("test", raw.get("val", []))
        else:
            train_data, test_data = _split_flat_list(list(raw))

    if not train_data and syn_pkl.exists():
        logger.info("Falling back to %s", syn_pkl)
        with open(syn_pkl, "rb") as fh:
            raw = pickle.load(fh)
        if isinstance(raw, dict) and "train" in raw:
            train_data = raw["train"]
            test_data  = raw.get("test", raw.get("val", []))
        else:
            train_data, test_data = _split_flat_list(list(raw))

    if not train_data:
        raise FileNotFoundError(
            f"Could not locate training data under '{pipeline_dir}'. "
            "Expected one of:\n"
            f"  {stage2b_pkl}\n  {stage1_pkl}\n  {syn_pkl}"
        )

    logger.info(
        "Dataset: train=%d transitions, test=%d transitions",
        len(train_data), len(test_data),
    )

    # ------------------------------------------------------------------
    # 4. Run OPE
    # ------------------------------------------------------------------
    runner = OPERunner(
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        target_policy_sigma=args.target_sigma,
        behavior_policy_sigma_floor=args.behavior_sigma_floor,
    )
    results = runner.run(train_data, test_data, agent, encoder_wrapper=encoder)

    # ------------------------------------------------------------------
    # 5. Print table
    # ------------------------------------------------------------------
    print_summary_table(results)
    return results


if __name__ == "__main__":
    main()
