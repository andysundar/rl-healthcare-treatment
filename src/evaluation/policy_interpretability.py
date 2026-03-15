"""
policy_interpretability.py
==========================
Function-based interpretability tools for RL treatment policies.

Public API
----------
extract_decision_rules(policy, state_action_pairs, ...)
    Fit a DecisionTreeClassifier (max_depth=5) surrogate and return
    both the tree object and human-readable if-then rules via
    sklearn export_text.

compute_feature_importance(policy, states, feature_names, ...)
    Fit a RandomForestClassifier surrogate, return a descending-sorted
    ``{feature_name: importance}`` dict, and save a horizontal bar chart
    to results/feature_importance.png.

generate_counterfactual(policy, state, feature_names, ...)
    Perturb each feature by ±10 % and ±25 %, record which perturbations
    flip the recommended action, and return a list of descriptive dicts.

Feature names
-------------
All defaults come from the preprocessing pipeline constants
``BASE_STATE_COLS``, ``VITAL_COLS``, and ``MED_HISTORY_COLS`` defined in
``run_integrated_solution.py``.  This module never redefines those lists;
it only imports them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-name defaults — pulled from the canonical pipeline source.
# Fallback to InterpretabilityConfig (which mirrors the same list) if the
# runner module cannot be imported (e.g. standalone unit-test environments).
# ---------------------------------------------------------------------------

try:
    from run_integrated_solution import BASE_STATE_COLS, VITAL_COLS, MED_HISTORY_COLS
    _DEFAULT_FEATURE_NAMES: List[str] = list(BASE_STATE_COLS)
except ImportError:
    try:
        from evaluation.interpretability import InterpretabilityConfig
        _DEFAULT_FEATURE_NAMES = list(InterpretabilityConfig().feature_names)
    except ImportError:
        # Absolute last resort — never used in production runs.
        _DEFAULT_FEATURE_NAMES = [f"feature_{i}" for i in range(10)]

# Perturbation fractions applied to each feature's current value (±10 %, ±25 %).
_PERTURBATION_FRACTIONS: Tuple[float, ...] = (-0.25, -0.10, 0.10, 0.25)

# Action-binning thresholds — match InterpretabilityConfig.action_bin_thresholds.
# (0.33, 0.67) splits CQL's tanh-squashed output into {low, medium, high}.
_BIN_LO: float = 0.33
_BIN_HI: float = 0.67


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _query_policy(policy, states: np.ndarray) -> np.ndarray:
    """
    Query *policy* on every row of *states* and return a 1-D float32 array.

    Handles both scalar and array returns from ``select_action``.
    """
    out = []
    for s in states:
        a = policy.select_action(np.asarray(s, dtype=np.float32), deterministic=True)
        out.append(float(np.asarray(a).flat[0]))
    return np.asarray(out, dtype=np.float32)


def _query_single(policy, state: np.ndarray) -> float:
    """Query *policy* on one state; return a scalar."""
    a = policy.select_action(np.asarray(state, dtype=np.float32), deterministic=True)
    return float(np.asarray(a).flat[0])


def _bin_actions(actions: np.ndarray) -> np.ndarray:
    """
    Map continuous actions → {low, medium, high} object array.

    Thresholds mirror InterpretabilityConfig.action_bin_thresholds = (0.33, 0.67).
    """
    labels = np.full(len(actions), "medium", dtype=object)
    labels[actions <= _BIN_LO] = "low"
    labels[actions >= _BIN_HI] = "high"
    return labels


def _resolve_names(feature_names: Optional[List[str]], n: int) -> List[str]:
    """
    Return a name list of length *n*.

    Uses *feature_names* (trimmed or padded) if provided, otherwise falls
    back to _DEFAULT_FEATURE_NAMES from the pipeline.
    """
    base = list(feature_names) if feature_names is not None else list(_DEFAULT_FEATURE_NAMES)
    if len(base) >= n:
        return base[:n]
    extras = [f"feature_{len(base) + i}" for i in range(n - len(base))]
    return base + extras


def _unpack_pairs(
    pairs: List[Tuple],
    policy,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split (state, action) pairs into stacked arrays.

    ``action`` may be ``None``; in that case *policy* is queried for
    the corresponding state.
    """
    states_list: List[np.ndarray] = []
    actions_list: List[float] = []

    for item in pairs:
        s = np.asarray(item[0], dtype=np.float32).reshape(-1)
        a = item[1] if len(item) > 1 else None
        states_list.append(s)
        if a is None:
            if policy is None:
                raise ValueError(
                    "state_action_pairs contains None actions but policy=None."
                )
            actions_list.append(_query_single(policy, s))
        else:
            actions_list.append(float(np.asarray(a).flat[0]))

    return np.stack(states_list), np.asarray(actions_list, dtype=np.float32)


def _save_importance_plot(
    importance_dict: Dict[str, float],
    save_path: str,
    show: bool,
) -> None:
    """Render and save a horizontal bar chart of feature importances."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping feature importance chart.")
        return

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Most-important feature at the top → reverse the sorted dict.
    names  = list(importance_dict.keys())[::-1]
    values = list(importance_dict.values())[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.42)))
    bars = ax.barh(names, values, color="#4C72B0", edgecolor="white", height=0.7)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title("Feature Importance  (RandomForest surrogate)", fontsize=12)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlim(0, max(values) * 1.18 if values else 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Feature importance chart saved → %s", out)


# ---------------------------------------------------------------------------
# 1. extract_decision_rules
# ---------------------------------------------------------------------------

def extract_decision_rules(
    policy,
    state_action_pairs: List[Tuple],
    *,
    max_depth: int = 5,
    feature_names: Optional[List[str]] = None,
    min_samples_leaf: int = 5,
) -> Tuple:
    """
    Approximate *policy* with a shallow DecisionTreeClassifier and export the
    tree as human-readable if-then rules.

    Parameters
    ----------
    policy :
        Object with ``select_action(state, deterministic=True)``.
        Pass ``None`` only when every pair already contains a pre-computed
        action (no ``None`` actions).
    state_action_pairs : list of (state, action)
        Training supervision.  Actions are binned into {low, medium, high}.
        Supply ``(state, None)`` to have the *policy* queried on-the-fly.
    max_depth : int
        Maximum tree depth (default 5 per task specification).
    feature_names : list of str, optional
        One name per feature dimension.  Falls back to pipeline
        ``BASE_STATE_COLS`` (auto-padded for wider state vectors).
    min_samples_leaf : int
        Minimum samples required in a leaf node.

    Returns
    -------
    tree : sklearn.tree.DecisionTreeClassifier
        The fitted surrogate.
    rules : list of str
        Lines produced by ``sklearn.tree.export_text``, one line per element,
        stripped of leading/trailing whitespace.  Suitable for logging or
        writing to a text file.
    """
    from sklearn.tree import DecisionTreeClassifier, export_text

    states, actions = _unpack_pairs(state_action_pairs, policy)
    names  = _resolve_names(feature_names, states.shape[1])
    labels = _bin_actions(actions)

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    ).fit(states, labels)

    text  = export_text(tree, feature_names=names)
    rules = [line.rstrip() for line in text.splitlines() if line.strip()]

    fidelity = float(np.mean(tree.predict(states) == labels))
    logger.info(
        "Decision tree: depth=%d  leaves=%d  training_fidelity=%.3f",
        tree.get_depth(), tree.get_n_leaves(), fidelity,
    )
    return tree, rules


# ---------------------------------------------------------------------------
# 2. compute_feature_importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    policy,
    states: np.ndarray,
    feature_names: Optional[List[str]] = None,
    *,
    n_estimators: int = 100,
    max_depth: int = 5,
    save_path: str = "results/feature_importance.png",
    show_plot: bool = False,
) -> Dict[str, float]:
    """
    Estimate feature importance via a RandomForestClassifier surrogate and
    save a horizontal bar chart.

    Parameters
    ----------
    policy :
        Object with ``select_action(state, deterministic=True)``.
    states : np.ndarray, shape (N, state_dim)
        States used to build the surrogate training set.
    feature_names : list of str, optional
        Per-dimension labels.  Falls back to pipeline ``BASE_STATE_COLS``.
    n_estimators : int
        Number of trees (default 100).
    max_depth : int
        Maximum depth per tree (default 5).
    save_path : str
        Destination for the PNG chart.  Parent directories are created
        automatically.
    show_plot : bool
        Call ``plt.show()`` after saving (useful in interactive sessions).

    Returns
    -------
    dict
        ``{feature_name: importance_score}`` sorted by importance descending.
        Importances are sklearn's mean decrease in impurity (MDI), summing to 1.
    """
    from sklearn.ensemble import RandomForestClassifier

    states = np.asarray(states, dtype=np.float32)
    names  = _resolve_names(feature_names, states.shape[1])

    actions = _query_policy(policy, states)
    labels  = _bin_actions(actions)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=0,
        n_jobs=-1,
    ).fit(states, labels)

    raw = dict(zip(names, rf.feature_importances_.tolist()))
    sorted_importance = dict(
        sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
    )

    _save_importance_plot(sorted_importance, save_path=save_path, show=show_plot)

    fidelity = float(np.mean(rf.predict(states) == labels))
    logger.info(
        "RandomForest surrogate: training_fidelity=%.3f  top_feature=%s (%.3f)",
        fidelity,
        next(iter(sorted_importance)),
        next(iter(sorted_importance.values())),
    )
    return sorted_importance


# ---------------------------------------------------------------------------
# 3. generate_counterfactual
# ---------------------------------------------------------------------------

def generate_counterfactual(
    policy,
    state: np.ndarray,
    feature_names: Optional[List[str]] = None,
    *,
    perturbation_fractions: Tuple[float, ...] = _PERTURBATION_FRACTIONS,
    absolute_floor: float = 1e-6,
) -> List[Dict]:
    """
    Identify feature perturbations that flip the policy's recommended action.

    For each feature independently, each of the four relative magnitudes
    (−25 %, −10 %, +10 %, +25 %) is applied.  Entries where the resulting
    action differs from the original by more than 1e-4 are returned.

    Parameters
    ----------
    policy :
        Object with ``select_action(state, deterministic=True)``.
    state : np.ndarray
        The state to explain (1-D, or coerced to 1-D).
    feature_names : list of str, optional
        Per-dimension labels.  Falls back to pipeline ``BASE_STATE_COLS``.
    perturbation_fractions : tuple of float
        Relative magnitudes to apply.  Default: (−0.25, −0.10, +0.10, +0.25).
    absolute_floor : float
        Minimum absolute perturbation for zero-valued features, preventing
        no-ops when a feature is exactly 0.

    Returns
    -------
    list of dict
        One entry per (feature × perturbation) that changed the action.
        Each dict contains:

        ``feature``            – feature name
        ``feature_index``      – column index in the state vector
        ``perturbation_frac``  – relative magnitude applied (e.g. −0.10)
        ``perturbation_abs``   – absolute change applied
        ``original_value``     – state[i] before perturbation
        ``perturbed_value``    – state[i] after perturbation
        ``original_action``    – policy output on the original state
        ``new_action``         – policy output on the perturbed state
        ``action_delta``       – new_action − original_action

        Entries where the action did not change are omitted.  If no
        perturbation flips the action, the function returns an empty list.
    """
    state   = np.asarray(state, dtype=np.float32).reshape(-1)
    names   = _resolve_names(feature_names, len(state))
    orig_a  = _query_single(policy, state)
    results = []

    for i, fname in enumerate(names):
        orig_val = float(state[i])

        for frac in perturbation_fractions:
            delta = orig_val * frac
            if abs(delta) < absolute_floor:
                delta = float(np.sign(frac)) * absolute_floor if frac != 0 else absolute_floor

            perturbed    = state.copy()
            perturbed[i] = orig_val + delta
            new_a        = _query_single(policy, perturbed)

            if abs(new_a - orig_a) > 1e-4:
                results.append({
                    "feature":           fname,
                    "feature_index":     i,
                    "perturbation_frac": round(frac, 4),
                    "perturbation_abs":  round(float(delta), 6),
                    "original_value":    round(orig_val, 6),
                    "perturbed_value":   round(float(perturbed[i]), 6),
                    "original_action":   round(orig_a, 6),
                    "new_action":        round(new_a, 6),
                    "action_delta":      round(new_a - orig_a, 6),
                })

    if not results:
        logger.info(
            "generate_counterfactual: no ±10%%/±25%% perturbation changed the "
            "action for this state (original_action=%.4f).",
            orig_a,
        )
    return results
