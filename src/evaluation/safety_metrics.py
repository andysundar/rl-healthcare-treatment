"""
Safety Metrics for Healthcare RL

Implements comprehensive safety evaluation.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers shared by the module-level functions below
# ---------------------------------------------------------------------------

def _get_safe_ranges(
    safe_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return caller-supplied ranges, or fall back to SafetyConfig.safe_ranges."""
    if safe_ranges is not None:
        return safe_ranges
    # Lazy import avoids a hard coupling at module load time.
    from models.safety.config import SafetyConfig
    return SafetyConfig().safe_ranges


def _iter_states(traj):
    """
    Yield individual states from a trajectory regardless of its container format.

    Accepted formats:
    - dict with a ``'states'`` key  (used by SafetyEvaluator / _tuples_to_traj_dicts)
    - object with a ``.states`` attribute  (Trajectory dataclass from off_policy_eval)
    - iterable of states directly  (bare list / array)
    """
    if isinstance(traj, dict):
        yield from traj.get("states", [])
    elif hasattr(traj, "states"):
        yield from traj.states
    else:
        yield from traj


def _check_state_safe(
    state, safe_ranges: Dict[str, Tuple[float, float]]
) -> Tuple[bool, List[str]]:
    """
    Check one state against *safe_ranges* without redefining any thresholds.

    Returns ``(is_safe, violated_variable_names)``.

    State formats handled:
    - ``dict``  — keys must match safe_ranges keys (e.g. ``'glucose'``).
    - array-like — index 0 is mapped to ``'glucose'`` per the BASE_STATE_COLS
      ordering defined in run_integrated_solution.py; all other indices are
      ignored because no unambiguous mapping exists for the remaining
      safe_ranges variables.
    """
    violations: List[str] = []

    if isinstance(state, dict):
        for var, (lo, hi) in safe_ranges.items():
            if var in state:
                if not (lo <= float(state[var]) <= hi):
                    violations.append(var)
    else:
        # Array state: only glucose (index 0) maps unambiguously to a safe range.
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if len(arr) > 0 and "glucose" in safe_ranges:
            lo, hi = safe_ranges["glucose"]
            if not (lo <= float(arr[0]) <= hi):
                violations.append("glucose")

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Public module-level functions
# ---------------------------------------------------------------------------

def compute_safety_index(
    trajectories: List,
    safe_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict:
    """
    Compute per-trajectory and aggregate safety indices.

    Safety Index = 1 - (unsafe_states / total_states)

    Uses ``SafetyConfig.safe_ranges`` when *safe_ranges* is not supplied, so
    physiological thresholds are never redefined here.

    Parameters
    ----------
    trajectories : list
        Trajectory dicts (``{'states': [...]}``), Trajectory dataclass objects,
        or raw lists of states.  Each state may be a dict or a numpy array.
    safe_ranges : dict, optional
        Override the default ``SafetyConfig().safe_ranges``.

    Returns
    -------
    dict
        ``per_trajectory``  – list of per-trajectory safety indices (float).
        ``aggregate``       – mean safety index across all trajectories.
        ``std``             – standard deviation of per-trajectory indices.
        ``n_trajectories``  – number of trajectories evaluated.
        ``total_states``    – total state count.
        ``total_unsafe``    – total unsafe state count.
    """
    ranges = _get_safe_ranges(safe_ranges)

    per_traj: List[float] = []
    total_states = 0
    total_unsafe = 0

    for traj in trajectories:
        n_safe   = 0
        n_states = 0
        for state in _iter_states(traj):
            is_safe, _ = _check_state_safe(state, ranges)
            n_states  += 1
            n_safe    += int(is_safe)

        if n_states == 0:
            continue

        traj_unsafe = n_states - n_safe
        per_traj.append(1.0 - traj_unsafe / n_states)
        total_states += n_states
        total_unsafe += traj_unsafe

    if not per_traj:
        return {
            "per_trajectory": [],
            "aggregate": 0.0,
            "std": 0.0,
            "n_trajectories": 0,
            "total_states": 0,
            "total_unsafe": 0,
        }

    arr = np.asarray(per_traj, dtype=np.float64)
    return {
        "per_trajectory":  per_traj,
        "aggregate":       float(np.mean(arr)),
        "std":             float(np.std(arr, ddof=0)),
        "n_trajectories":  len(per_traj),
        "total_states":    total_states,
        "total_unsafe":    total_unsafe,
    }


def generate_safety_report(
    trajectories: List,
    safe_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    prefix: str = "safety",
) -> Dict[str, float]:
    """
    Produce a flat dict of safety metrics ready for MLflow logging.

    Follows the logging convention in ``src/models/rl/trainer.py``:
    flat ``{key: float}`` pairs, optionally prefixed.  Only numeric values
    are included so every entry can be passed directly to
    ``mlflow.log_metric(key, value, step=…)``.

    Parameters
    ----------
    trajectories : list
        Same format as :func:`compute_safety_index`.
    safe_ranges : dict, optional
        Override the default ``SafetyConfig().safe_ranges``.
    prefix : str
        Metric-name prefix (default ``"safety"``).  An empty string disables
        prefixing.

    Returns
    -------
    dict
        Keys prefixed as ``"<prefix>/<name>"`` (or just ``"<name>"`` when
        *prefix* is empty).  Values are all ``float`` or ``int``.
    """
    idx = compute_safety_index(trajectories, safe_ranges=safe_ranges)

    per = idx["per_trajectory"]
    raw: Dict[str, float] = {
        "safety_index":        idx["aggregate"],
        "safety_index_std":    idx["std"],
        "violation_rate":      1.0 - idx["aggregate"],
        "n_trajectories":      float(idx["n_trajectories"]),
        "total_states":        float(idx["total_states"]),
        "total_unsafe_states": float(idx["total_unsafe"]),
        "min_traj_safety":     float(min(per)) if per else 0.0,
        "max_traj_safety":     float(max(per)) if per else 0.0,
    }

    if prefix:
        return {f"{prefix}/{k}": v for k, v in raw.items()}
    return raw

@dataclass
class SafetyResult:
    safety_index: float
    violation_rate: float
    avg_violation_severity: float
    critical_violations: int
    violation_breakdown: Dict[str, float]
    metadata: Dict = None

class SafeStateChecker:
    """Checks patient state safety."""
    
    def __init__(self, config):
        self.config = config
        self.safe_ranges = {
            'glucose': config.safety.safe_glucose_range,
            'bp_systolic': config.safety.safe_bp_systolic_range,
        }
        
    def is_safe(self, state):
        for variable, (min_val, max_val) in self.safe_ranges.items():
            if variable in state:
                value = state[variable]
                if not (min_val <= value <= max_val):
                    return False
        return True
    
    def get_violations(self, state):
        violations = []
        for variable, (min_val, max_val) in self.safe_ranges.items():
            if variable in state:
                value = state[variable]
                if value < min_val or value > max_val:
                    violations.append((variable, value))
        return violations

class SafetyEvaluator:
    """Comprehensive safety evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.safe_state_checker = SafeStateChecker(config)
        
    def compute_safety_index(self, trajectories):
        total_states = 0
        unsafe_states = 0
        
        for traj in trajectories:
            for state in traj['states']:
                total_states += 1
                if not self.safe_state_checker.is_safe(state):
                    unsafe_states += 1
        
        if total_states == 0:
            return 0.0
        
        return 1.0 - (unsafe_states / total_states)
    
    def compute_violation_rate(self, trajectories):
        total_actions = 0
        violations = 0
        
        for traj in trajectories:
            for state in traj['states']:
                total_actions += 1
                if not self.safe_state_checker.is_safe(state):
                    violations += 1
        
        if total_actions == 0:
            return 0.0
        
        return violations / total_actions
    
    def compute_violation_severity(self, trajectories):
        all_violations = []
        
        for traj in trajectories:
            for state in traj['states']:
                for var, (min_val, max_val) in self.safe_state_checker.safe_ranges.items():
                    if var in state:
                        value = state[var]
                        range_width = max_val - min_val
                        
                        if value < min_val:
                            severity = (min_val - value) / range_width
                            all_violations.append(severity)
                        elif value > max_val:
                            severity = (value - max_val) / range_width
                            all_violations.append(severity)
        
        return np.mean(all_violations) if all_violations else 0.0
    
    def evaluate(self, trajectories):
        safety_index = self.compute_safety_index(trajectories)
        violation_rate = self.compute_violation_rate(trajectories)
        avg_severity = self.compute_violation_severity(trajectories)
        
        critical_count = 0
        violation_breakdown = {'minor': 0, 'moderate': 0, 'severe': 0, 'critical': 0}
        
        result = SafetyResult(
            safety_index=safety_index,
            violation_rate=violation_rate,
            avg_violation_severity=avg_severity,
            critical_violations=critical_count,
            violation_breakdown=violation_breakdown,
            metadata={'n_trajectories': len(trajectories)}
        )
        
        self._print_summary(result)
        return result
    
    def _print_summary(self, result):
        print("\n" + "="*70)
        print("SAFETY EVALUATION SUMMARY")
        print("="*70)
        print(f"Safety Index:           {result.safety_index:.3f}")
        print(f"Violation Rate:         {result.violation_rate:.3f}")
        print(f"Avg Violation Severity: {result.avg_violation_severity:.3f}")
        print("="*70 + "\n")
