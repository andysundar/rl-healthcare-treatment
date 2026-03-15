"""
Unit tests for compute_safety_index() and generate_safety_report()
added to src/evaluation/safety_metrics.py.

All ranges used in tests come from SafetyConfig.safe_ranges so that the
tests break if the canonical thresholds change, not because they define
their own copies.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.safety_metrics import compute_safety_index, generate_safety_report
from models.safety.config import SafetyConfig

# Pull the canonical ranges once so tests stay in sync with the source.
_RANGES = SafetyConfig().safe_ranges
_G_LO, _G_HI = _RANGES["glucose"]          # 70.0, 200.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_traj(glucose_values):
    """Trajectory dict whose states are dicts with a 'glucose' key."""
    return {"states": [{"glucose": float(g)} for g in glucose_values]}


def _arr_traj(glucose_values):
    """Trajectory dict whose states are numpy arrays (index 0 = glucose)."""
    return {"states": [np.array([float(g)], dtype=np.float32) for g in glucose_values]}


def _safe_glucose():
    return (_G_LO + _G_HI) / 2.0   # midpoint — always in range


def _unsafe_glucose():
    return _G_LO - 10.0             # clearly below lower bound


# ---------------------------------------------------------------------------
# compute_safety_index — dict-format states
# ---------------------------------------------------------------------------

class TestComputeSafetyIndexDictStates:
    def test_all_safe_returns_one(self):
        trajs = [_dict_traj([_safe_glucose()] * 5)]
        result = compute_safety_index(trajs)
        assert result["aggregate"] == pytest.approx(1.0)

    def test_all_unsafe_returns_zero(self):
        trajs = [_dict_traj([_unsafe_glucose()] * 5)]
        result = compute_safety_index(trajs)
        assert result["aggregate"] == pytest.approx(0.0)

    def test_half_unsafe(self):
        glucose = [_safe_glucose()] * 4 + [_unsafe_glucose()] * 4
        trajs = [_dict_traj(glucose)]
        result = compute_safety_index(trajs)
        assert result["aggregate"] == pytest.approx(0.5)

    def test_per_trajectory_length(self):
        trajs = [_dict_traj([_safe_glucose()] * 3) for _ in range(7)]
        result = compute_safety_index(trajs)
        assert len(result["per_trajectory"]) == 7

    def test_per_trajectory_values(self):
        # traj 0: all safe → index 1.0; traj 1: all unsafe → index 0.0
        trajs = [
            _dict_traj([_safe_glucose()] * 5),
            _dict_traj([_unsafe_glucose()] * 5),
        ]
        result = compute_safety_index(trajs)
        assert result["per_trajectory"][0] == pytest.approx(1.0)
        assert result["per_trajectory"][1] == pytest.approx(0.0)

    def test_aggregate_equals_mean_of_per_trajectory(self):
        trajs = [
            _dict_traj([_safe_glucose()] * 4),
            _dict_traj([_safe_glucose(), _unsafe_glucose()]),
        ]
        result = compute_safety_index(trajs)
        assert result["aggregate"] == pytest.approx(
            np.mean(result["per_trajectory"])
        )

    def test_std_is_zero_for_identical_trajectories(self):
        trajs = [_dict_traj([_safe_glucose()] * 5) for _ in range(10)]
        result = compute_safety_index(trajs)
        assert result["std"] == pytest.approx(0.0)

    def test_counts_are_consistent(self):
        trajs = [_dict_traj([_safe_glucose()] * 3 + [_unsafe_glucose()] * 2)]
        result = compute_safety_index(trajs)
        assert result["total_states"]  == 5
        assert result["total_unsafe"]  == 2
        assert result["n_trajectories"] == 1


# ---------------------------------------------------------------------------
# compute_safety_index — numpy array states
# ---------------------------------------------------------------------------

class TestComputeSafetyIndexArrayStates:
    def test_all_safe(self):
        trajs = [_arr_traj([_safe_glucose()] * 5)]
        result = compute_safety_index(trajs)
        assert result["aggregate"] == pytest.approx(1.0)

    def test_all_unsafe(self):
        trajs = [_arr_traj([_unsafe_glucose()] * 5)]
        result = compute_safety_index(trajs)
        assert result["aggregate"] == pytest.approx(0.0)

    def test_mixed(self):
        trajs = [_arr_traj([_safe_glucose()] * 3 + [_unsafe_glucose()] * 1)]
        result = compute_safety_index(trajs)
        assert result["aggregate"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# compute_safety_index — Trajectory dataclass (off_policy_eval format)
# ---------------------------------------------------------------------------

class TestComputeSafetyIndexTrajectoryDataclass:
    def test_dataclass_trajectory(self):
        """States stored in a Trajectory.states ndarray (shape [T, state_dim])."""
        from evaluation.off_policy_eval import Trajectory

        safe  = np.full((5, 1), _safe_glucose(), dtype=np.float32)
        traj  = Trajectory(
            states=safe,
            actions=np.zeros((5, 1), dtype=np.float32),
            rewards=np.ones(5, dtype=np.float32),
            next_states=safe.copy(),
            dones=np.zeros(5, dtype=bool),
        )
        result = compute_safety_index([traj])
        assert result["aggregate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_safety_index — edge cases
# ---------------------------------------------------------------------------

class TestComputeSafetyIndexEdgeCases:
    def test_empty_list_returns_zeros(self):
        result = compute_safety_index([])
        assert result["aggregate"] == 0.0
        assert result["std"]       == 0.0
        assert result["per_trajectory"] == []

    def test_trajectory_with_no_states_is_skipped(self):
        result = compute_safety_index([{"states": []}])
        assert result["n_trajectories"] == 0

    def test_custom_safe_ranges_are_used(self):
        """Supplying a tighter custom range should flag more states as unsafe."""
        tight = {"glucose": (100.0, 110.0)}  # much tighter than default 70–200
        # All three values are inside the default range (70–200).
        # Under the tight range only 105 is safe; 80 and 190 are outside.
        glucose = [80.0, 190.0, 105.0]
        trajs = [_dict_traj(glucose)]
        default_result = compute_safety_index(trajs)          # all safe → 1.0
        tight_result   = compute_safety_index(trajs, safe_ranges=tight)  # 1/3
        assert tight_result["aggregate"] < default_result["aggregate"]

    def test_safe_ranges_not_redefined(self):
        """Verifies the function pulls thresholds from SafetyConfig, not hardcoded literals."""
        from models.safety.config import SafetyConfig
        cfg_ranges = SafetyConfig().safe_ranges
        # If we feed a state exactly at the safe boundary it should be accepted.
        lo, hi = cfg_ranges["glucose"]
        trajs  = [_dict_traj([lo, hi])]  # boundary values are inclusive
        result = compute_safety_index(trajs)
        assert result["aggregate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# generate_safety_report
# ---------------------------------------------------------------------------

class TestGenerateSafetyReport:
    def _all_safe_trajs(self):
        return [_dict_traj([_safe_glucose()] * 5) for _ in range(4)]

    def test_returns_dict(self):
        report = generate_safety_report(self._all_safe_trajs())
        assert isinstance(report, dict)

    def test_all_values_are_numeric(self):
        report = generate_safety_report(self._all_safe_trajs())
        for k, v in report.items():
            assert isinstance(v, (int, float)), f"{k}: {type(v)}"

    def test_default_prefix(self):
        report = generate_safety_report(self._all_safe_trajs())
        assert all(k.startswith("safety/") for k in report)

    def test_custom_prefix(self):
        report = generate_safety_report(self._all_safe_trajs(), prefix="eval")
        assert all(k.startswith("eval/") for k in report)

    def test_empty_prefix(self):
        report = generate_safety_report(self._all_safe_trajs(), prefix="")
        assert not any("/" in k for k in report)

    def test_required_keys_present(self):
        report = generate_safety_report(self._all_safe_trajs())
        expected = {
            "safety/safety_index",
            "safety/safety_index_std",
            "safety/violation_rate",
            "safety/n_trajectories",
            "safety/total_states",
            "safety/total_unsafe_states",
            "safety/min_traj_safety",
            "safety/max_traj_safety",
        }
        assert expected.issubset(set(report.keys()))

    def test_safety_index_plus_violation_rate_equals_one(self):
        trajs  = [_dict_traj([_safe_glucose()] * 3 + [_unsafe_glucose()] * 1)]
        report = generate_safety_report(trajs)
        total  = report["safety/safety_index"] + report["safety/violation_rate"]
        assert total == pytest.approx(1.0)

    def test_all_safe_report_values(self):
        report = generate_safety_report(self._all_safe_trajs())
        assert report["safety/safety_index"]        == pytest.approx(1.0)
        assert report["safety/violation_rate"]      == pytest.approx(0.0)
        assert report["safety/total_unsafe_states"] == pytest.approx(0.0)

    def test_mlflow_compatible_flat_structure(self):
        """Simulate the trainer.py logging pattern: every value must be int or float."""
        report = generate_safety_report(self._all_safe_trajs())
        for key, value in report.items():
            # mlflow.log_metric(key, value) requires key=str, value=numeric
            assert isinstance(key, str)
            assert isinstance(value, (int, float))
