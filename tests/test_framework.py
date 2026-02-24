"""
Evaluation Framework Tests

Tests all evaluation components against the actual API.

Author: Anindya Bandopadhyay (M23CSA508)
Date: February 2026
"""

import sys
from pathlib import Path
import numpy as np
import tempfile

# conftest.py (at project root) adds src/ to sys.path when running via pytest.
# For direct execution add it manually.
_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Helpers — produce the dict-per-trajectory format that evaluators expect
# ---------------------------------------------------------------------------

def _make_traj_dicts(n_episodes=20, length=30):
    """
    Return list of trajectory dicts:
        {'states': [...], 'actions': [...], 'rewards': [...],
         'next_states': [...], 'dones': [...]}
    where each state is a dict of clinical features.
    """
    trajs = []
    for _ in range(n_episodes):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for t in range(length):
            states.append({
                "glucose": float(np.random.uniform(80, 180)),
                "blood_pressure_systolic": float(np.random.uniform(100, 140)),
                "adherence_score": float(np.random.uniform(0.5, 1.0)),
            })
            next_states.append({
                "glucose": float(np.random.uniform(80, 160)),
                "blood_pressure_systolic": float(np.random.uniform(100, 130)),
                "adherence_score": float(np.random.uniform(0.6, 1.0)),
            })
            actions.append(np.random.uniform(0, 1, size=1))
            rewards.append(float(np.random.randn()))
            dones.append(t == length - 1)
        trajs.append({"states": states, "actions": actions,
                      "rewards": rewards, "next_states": next_states, "dones": dones})
    return trajs


def _make_trajectory_objects(n_episodes=20, length=30):
    """Return list of Trajectory dataclass objects for OffPolicyEvaluator."""
    from evaluation.off_policy_eval import Trajectory
    trajs = []
    for _ in range(n_episodes):
        states      = np.random.randn(length, 10).astype(np.float32)
        actions     = np.random.uniform(0, 1, (length, 1)).astype(np.float32)
        rewards     = np.random.randn(length).astype(np.float32)
        next_states = np.random.randn(length, 10).astype(np.float32)
        dones       = np.zeros(length, dtype=bool)
        dones[-1]   = True
        trajs.append(Trajectory(states=states, actions=actions, rewards=rewards,
                                next_states=next_states, dones=dones))
    return trajs


class _StubPolicy:
    """Minimal policy with get_action_probability for OPE."""
    def select_action(self, state, deterministic=True):
        return np.array([0.5])

    def get_action_probability(self, state, action):
        return 0.5


class _StubQFunction:
    def predict(self, state, action):
        return float(np.random.randn())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_imports():
    """Test that all evaluation modules can be imported."""
    from evaluation.off_policy_eval import OffPolicyEvaluator, Trajectory
    from evaluation.safety_metrics import SafetyEvaluator, SafeStateChecker
    from evaluation.clinical_metrics import ClinicalEvaluator
    from evaluation.performance_metrics import PerformanceEvaluator, PerformanceResult
    from evaluation.visualizations import EvaluationVisualizer


def test_ope_evaluator():
    """Test OffPolicyEvaluator.evaluate and compare_methods."""
    from evaluation.off_policy_eval import OffPolicyEvaluator

    evaluator = OffPolicyEvaluator(gamma=0.99, q_function=_StubQFunction())
    trajs     = _make_trajectory_objects(10, 20)
    policy    = _StubPolicy()
    behavior  = _StubPolicy()

    # evaluate(policy, behavior_policy, trajectories, methods=[...]) -> dict
    results = evaluator.evaluate(policy, behavior, trajs, methods=["wis"])
    assert isinstance(results, dict)
    assert "wis" in results

    # compare_methods prints a table (returns None)
    evaluator.compare_methods(results)


def test_safety_evaluator():
    """Test SafetyEvaluator.evaluate returns a SafetyResult."""
    from evaluation.safety_metrics import SafetyEvaluator, SafetyResult
    from configs.config import EvaluationConfig

    evaluator = SafetyEvaluator(EvaluationConfig())
    trajs = _make_traj_dicts(10, 20)
    result = evaluator.evaluate(trajs)
    assert isinstance(result, SafetyResult)
    assert 0.0 <= result.safety_index <= 1.0


def test_clinical_evaluator():
    """Test ClinicalEvaluator.compute_time_in_range returns a dict."""
    from evaluation.clinical_metrics import ClinicalEvaluator
    from configs.config import EvaluationConfig

    evaluator = ClinicalEvaluator(EvaluationConfig())
    trajs = _make_traj_dicts(10, 20)
    tir = evaluator.compute_time_in_range(trajs)
    assert isinstance(tir, dict)


def test_performance_metrics():
    """Test PerformanceEvaluator.compute_* helpers."""
    from evaluation.performance_metrics import PerformanceEvaluator
    from configs.config import EvaluationConfig

    evaluator = PerformanceEvaluator(EvaluationConfig())
    trajs = _make_traj_dicts(20, 30)

    mean_ret, std_ret = evaluator.compute_average_return(trajs)
    assert isinstance(mean_ret, float)

    mean_len, std_len = evaluator.compute_episode_lengths(trajs)
    assert mean_len == 30.0   # all episodes have same length

    success = evaluator.compute_success_rate(trajs)
    assert 0.0 <= success <= 1.0


def test_visualizations():
    """Test EvaluationVisualizer can create figures without errors."""
    from evaluation.visualizations import EvaluationVisualizer
    from configs.config import EvaluationConfig

    cfg = EvaluationConfig()
    viz = EvaluationVisualizer(cfg)
    curves = {
        "Policy_A": np.random.randn(50).cumsum().tolist(),
        "Policy_B": np.random.randn(50).cumsum().tolist(),
    }
    viz.plot_learning_curves(curves)

    trajs = _make_traj_dicts(5, 10)
    viz.plot_health_metrics(trajs, metrics=["glucose", "adherence_score"])


def test_performance_evaluator_full():
    """Test PerformanceEvaluator.evaluate returns a PerformanceResult."""
    from evaluation.performance_metrics import PerformanceEvaluator, PerformanceResult
    from configs.config import EvaluationConfig

    evaluator = PerformanceEvaluator(EvaluationConfig())
    trajs = _make_traj_dicts(20, 30)
    result = evaluator.evaluate(trajs)
    assert isinstance(result, PerformanceResult)
    assert hasattr(result, "average_return")


def test_comprehensive_flow():
    """End-to-end: safety + clinical + performance on the same trajectories."""
    from evaluation.safety_metrics import SafetyEvaluator
    from evaluation.clinical_metrics import ClinicalEvaluator
    from evaluation.performance_metrics import PerformanceEvaluator
    from configs.config import EvaluationConfig

    cfg = EvaluationConfig()
    trajs = _make_traj_dicts(20, 30)

    safety = SafetyEvaluator(cfg).evaluate(trajs)
    perf   = PerformanceEvaluator(cfg).evaluate(trajs)
    tir    = ClinicalEvaluator(cfg).compute_time_in_range(trajs)

    assert safety.safety_index >= 0
    assert perf.average_return is not None
    assert isinstance(tir, dict)


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
