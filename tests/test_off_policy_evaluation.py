"""
Unit tests for src/evaluation/off_policy_evaluation.py

All tests are self-contained: they use a tiny fake CQL agent stub and
synthetic transitions so no checkpoint files or MIMIC data are needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.off_policy_evaluation import (
    BehaviorPolicy,
    DMEstimator,
    DREstimator,
    OPERunner,
    WISEstimator,
    _GaussianPolicyWrapper,
    _build_q_function,
    _transitions_to_trajectories,
    print_summary_table,
)
from evaluation.off_policy_eval import OPEResult, Trajectory

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

STATE_DIM = 4
N_TRAIN   = 200
N_TEST    = 100
RNG       = np.random.default_rng(0)


def _make_transitions(n: int, state_dim: int = STATE_DIM, seed: int = 0) -> list:
    """Return n (s, a, r, s', done) tuples with done=True every 10 steps."""
    rng = np.random.default_rng(seed)
    transitions = []
    for i in range(n):
        s      = rng.standard_normal(state_dim).astype(np.float32)
        a      = rng.uniform(0.0, 1.0, size=(1,)).astype(np.float32)
        r      = float(rng.standard_normal())
        s_next = rng.standard_normal(state_dim).astype(np.float32)
        done   = (i % 10 == 9)
        transitions.append((s, a, r, s_next, done))
    return transitions


class _FakeCQLAgent:
    """Minimal stand-in for CQLAgent — no torch required."""

    def __init__(self, state_dim: int = STATE_DIM, q_value: float = 1.5):
        self.state_dim  = state_dim
        self._q_value   = q_value

    def select_action(self, state, deterministic=True):
        return np.array([0.3], dtype=np.float32)

    def get_q_value(self, state, action) -> float:
        return self._q_value


# ---------------------------------------------------------------------------
# BehaviorPolicy
# ---------------------------------------------------------------------------

class TestBehaviorPolicy:
    def test_fit_returns_self(self):
        bp = BehaviorPolicy()
        transitions = _make_transitions(50)
        result = bp.fit(transitions)
        assert result is bp

    def test_fitted_flag(self):
        bp = BehaviorPolicy()
        assert not bp._fitted
        bp.fit(_make_transitions(50))
        assert bp._fitted

    def test_probability_is_positive(self):
        bp = BehaviorPolicy().fit(_make_transitions(100))
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.array([0.5], dtype=np.float32)
        p = bp.get_action_probability(s, a)
        assert p > 0.0

    def test_probability_peaks_near_predicted_mean(self):
        """Probability should be highest at the policy's own predicted action."""
        bp = BehaviorPolicy().fit(_make_transitions(200))
        s  = np.ones(STATE_DIM, dtype=np.float32) * 0.5
        mu = float(bp._mean(s))
        p_at_mean = bp.get_action_probability(s, mu)
        p_far_off = bp.get_action_probability(s, mu + 5.0)
        assert p_at_mean > p_far_off

    def test_select_action_shape(self):
        bp = BehaviorPolicy().fit(_make_transitions(50))
        a = bp.select_action(np.zeros(STATE_DIM, dtype=np.float32))
        assert np.asarray(a).reshape(-1).shape == (1,)

    def test_unfitted_fallback(self):
        """Unfitted policy should not crash — returns a default probability."""
        bp = BehaviorPolicy()
        p = bp.get_action_probability(
            np.zeros(STATE_DIM, dtype=np.float32), np.array([0.5])
        )
        assert p > 0.0

    def test_sigma_not_below_floor(self):
        """Constant-action dataset should not drive sigma to zero."""
        rng = np.random.default_rng(1)
        s   = rng.standard_normal((50, STATE_DIM)).astype(np.float32)
        transitions = [
            (s[i], np.array([0.5], dtype=np.float32), 0.0, s[i], False)
            for i in range(50)
        ]
        bp = BehaviorPolicy(sigma_floor=0.05).fit(transitions)
        assert bp._sigma >= 0.05


# ---------------------------------------------------------------------------
# _GaussianPolicyWrapper
# ---------------------------------------------------------------------------

class TestGaussianPolicyWrapper:
    def test_probability_is_positive(self):
        wrapper = _GaussianPolicyWrapper(_FakeCQLAgent(), sigma=0.05)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        p = wrapper.get_action_probability(s, np.array([0.3]))
        assert p > 0.0

    def test_probability_peaks_at_own_action(self):
        agent   = _FakeCQLAgent()
        wrapper = _GaussianPolicyWrapper(agent, sigma=0.05)
        s       = np.zeros(STATE_DIM, dtype=np.float32)
        own_a   = float(agent.select_action(s)[0])
        p_own   = wrapper.get_action_probability(s, own_a)
        p_other = wrapper.get_action_probability(s, own_a + 1.0)
        assert p_own > p_other

    def test_select_action_delegates(self):
        wrapper = _GaussianPolicyWrapper(_FakeCQLAgent())
        a = wrapper.select_action(np.zeros(STATE_DIM, dtype=np.float32))
        assert float(a[0]) == pytest.approx(0.3)

    def test_state_dim_attribute(self):
        agent = _FakeCQLAgent(state_dim=8)
        assert _GaussianPolicyWrapper(agent).state_dim == 8


# ---------------------------------------------------------------------------
# _transitions_to_trajectories
# ---------------------------------------------------------------------------

class TestTransitionsToTrajectories:
    def test_episode_count(self):
        """10 transitions with done every 10 steps → exactly 1 episode."""
        transitions = _make_transitions(10)
        trajs = _transitions_to_trajectories(transitions)
        assert len(trajs) == 1

    def test_multiple_episodes(self):
        transitions = _make_transitions(30)   # 3 full episodes of 10
        trajs = _transitions_to_trajectories(transitions)
        assert len(trajs) == 3

    def test_open_episode_flushed(self):
        """Transitions without a terminal done should form their own trajectory."""
        transitions = _make_transitions(7)    # 7 steps, none done
        for i, t in enumerate(transitions):
            transitions[i] = (t[0], t[1], t[2], t[3], False)
        trajs = _transitions_to_trajectories(transitions)
        assert len(trajs) == 1
        assert len(trajs[0]) == 7

    def test_trajectory_array_shapes(self):
        transitions = _make_transitions(10)
        traj = _transitions_to_trajectories(transitions)[0]
        assert traj.states.shape      == (10, STATE_DIM)
        assert traj.actions.shape     == (10, 1)
        assert traj.rewards.shape     == (10,)
        assert traj.next_states.shape == (10, STATE_DIM)
        assert traj.dones.shape       == (10,)

    def test_max_ep_steps_splits_long_episode(self):
        transitions = _make_transitions(25)
        for i, t in enumerate(transitions):
            transitions[i] = (t[0], t[1], t[2], t[3], False)
        trajs = _transitions_to_trajectories(transitions, max_ep_steps=10)
        assert len(trajs) == 3   # two full chunks of 10 + one of 5

    def test_empty_input(self):
        assert _transitions_to_trajectories([]) == []


# ---------------------------------------------------------------------------
# _build_q_function
# ---------------------------------------------------------------------------

class TestBuildQFunction:
    def test_returns_scalar(self):
        q_fn = _build_q_function(_FakeCQLAgent(q_value=2.0))
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.array([0.5], dtype=np.float32)
        assert q_fn(s, a) == pytest.approx(2.0)

    def test_delegates_to_agent(self):
        agent = _FakeCQLAgent(q_value=-3.14)
        q_fn  = _build_q_function(agent)
        val   = q_fn(np.ones(STATE_DIM, dtype=np.float32), np.array([1.0]))
        assert val == pytest.approx(-3.14)


# ---------------------------------------------------------------------------
# Estimators (smoke tests — check return type and key fields)
# ---------------------------------------------------------------------------

def _make_trajectories(n_ep: int = 5, ep_len: int = 10) -> list:
    transitions = _make_transitions(n_ep * ep_len)
    return _transitions_to_trajectories(transitions)


def _make_behavior_policy():
    return BehaviorPolicy().fit(_make_transitions(N_TRAIN))


def _make_target_policy():
    return _GaussianPolicyWrapper(_FakeCQLAgent())


class TestWISEstimator:
    def test_returns_ope_result(self):
        wis  = WISEstimator(n_bootstrap=50, seed=0)
        trajs = _make_trajectories()
        res  = wis.estimate(trajs, _make_target_policy(), _make_behavior_policy())
        assert isinstance(res, OPEResult)

    def test_metadata_fields(self):
        wis  = WISEstimator(n_bootstrap=50, seed=1)
        res  = wis.estimate(_make_trajectories(), _make_target_policy(), _make_behavior_policy())
        assert "ess"               in res.metadata
        assert "clip_fraction"     in res.metadata
        assert "reliability_flag"  in res.metadata

    def test_ci_ordered(self):
        res = WISEstimator(n_bootstrap=50).estimate(
            _make_trajectories(), _make_target_policy(), _make_behavior_policy()
        )
        lo, hi = res.confidence_interval
        assert lo <= hi

    def test_n_trajectories_correct(self):
        trajs = _make_trajectories(n_ep=5)
        res   = WISEstimator(n_bootstrap=20).estimate(
            trajs, _make_target_policy(), _make_behavior_policy()
        )
        assert res.n_trajectories == len(trajs)


class TestDMEstimator:
    def test_returns_ope_result(self):
        q_fn = _build_q_function(_FakeCQLAgent(q_value=1.0))
        res  = DMEstimator(q_function=q_fn, n_bootstrap=50).estimate(_make_trajectories())
        assert isinstance(res, OPEResult)

    def test_value_reflects_constant_q(self):
        """When Q always returns C, DM value should be near C * (1/(1-gamma)) ≈ 100 C."""
        q_fn = _build_q_function(_FakeCQLAgent(q_value=1.0))
        res  = DMEstimator(q_function=q_fn, gamma=0.99, n_bootstrap=20).estimate(
            _make_trajectories(n_ep=20, ep_len=10)
        )
        # Sum of discounted Q over 10 steps with Q=1: sum_{t=0}^{9} 0.99^t ≈ 9.56
        expected = sum(0.99 ** t for t in range(10))
        assert abs(res.value_estimate - expected) < 0.5

    def test_ci_ordered(self):
        q_fn = _build_q_function(_FakeCQLAgent(q_value=0.5))
        res  = DMEstimator(q_function=q_fn, n_bootstrap=50).estimate(_make_trajectories())
        lo, hi = res.confidence_interval
        assert lo <= hi


class TestDREstimator:
    def test_returns_ope_result(self):
        q_fn = _build_q_function(_FakeCQLAgent())
        res  = DREstimator(q_function=q_fn, n_bootstrap=50).estimate(
            _make_trajectories(), _make_target_policy(), _make_behavior_policy()
        )
        assert isinstance(res, OPEResult)

    def test_ci_ordered(self):
        q_fn = _build_q_function(_FakeCQLAgent())
        res  = DREstimator(q_function=q_fn, n_bootstrap=50).estimate(
            _make_trajectories(), _make_target_policy(), _make_behavior_policy()
        )
        lo, hi = res.confidence_interval
        assert lo <= hi


# ---------------------------------------------------------------------------
# OPERunner end-to-end
# ---------------------------------------------------------------------------

class TestOPERunner:
    def test_all_three_keys_present(self):
        runner = OPERunner(n_bootstrap=50, seed=0)
        results = runner.run(
            _make_transitions(N_TRAIN),
            _make_transitions(N_TEST, seed=1),
            _FakeCQLAgent(),
        )
        assert set(results.keys()) == {"wis", "dm", "dr"}

    def test_all_results_are_ope_result(self):
        runner  = OPERunner(n_bootstrap=20)
        results = runner.run(
            _make_transitions(N_TRAIN),
            _make_transitions(N_TEST, seed=2),
            _FakeCQLAgent(),
        )
        for key, val in results.items():
            assert isinstance(val, OPEResult), f"{key} is not an OPEResult"

    def test_dm_close_to_expected_with_constant_q(self):
        """DM value ≈ sum_t gamma^t * Q for a constant-Q agent."""
        runner  = OPERunner(gamma=0.99, n_bootstrap=20)
        results = runner.run(
            _make_transitions(N_TRAIN),
            _make_transitions(N_TEST, seed=3),
            _FakeCQLAgent(q_value=2.0),
        )
        expected = sum(0.99 ** t * 2.0 for t in range(10))   # ep_len=10 via done every 10
        assert abs(results["dm"].value_estimate - expected) < 1.0

    def test_no_encoder_does_not_crash(self):
        runner = OPERunner(n_bootstrap=20)
        results = runner.run(
            _make_transitions(N_TRAIN),
            _make_transitions(N_TEST, seed=4),
            _FakeCQLAgent(),
            encoder_wrapper=None,
        )
        assert "wis" in results


# ---------------------------------------------------------------------------
# print_summary_table (smoke)
# ---------------------------------------------------------------------------

class TestPrintSummaryTable:
    def test_runs_without_error(self, capsys):
        runner  = OPERunner(n_bootstrap=20)
        results = runner.run(
            _make_transitions(N_TRAIN),
            _make_transitions(N_TEST, seed=5),
            _FakeCQLAgent(),
        )
        print_summary_table(results)
        out = capsys.readouterr().out
        assert "WIS"       in out
        assert "DM"        in out
        assert "DR"        in out
        assert "SUMMARY"   in out

    def test_empty_results_does_not_crash(self, capsys):
        print_summary_table({})
        # Should produce a table header but no rows — just no crash.
