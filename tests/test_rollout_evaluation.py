from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from src.run_integrated_solution import IntegratedSolutionRunner


class _ConstPolicy:
    def __init__(self, value: float):
        self.value = float(value)

    def select_action(self, state, deterministic=True):
        return np.array([self.value], dtype=np.float32)

    def get_action_bounds(self):
        return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)


def _mk_cfg(tmp_path: Path) -> Namespace:
    return Namespace(
        output_dir=str(tmp_path),
        mode='synthetic',
        seed=42,
        use_vitals=False,
        use_med_history=False,
    )


def _mk_data():
    states = [np.array([100.0, 0.1], dtype=np.float32), np.array([102.0, 0.1], dtype=np.float32), np.array([104.0, 0.1], dtype=np.float32)]
    actions = [np.array([0.3], dtype=np.float32), np.array([0.4], dtype=np.float32), np.array([0.5], dtype=np.float32)]
    rewards = [1.0, 1.2, 1.4]
    next_states = [np.array([101.0, 0.1], dtype=np.float32), np.array([103.0, 0.1], dtype=np.float32), np.array([105.0, 0.1], dtype=np.float32)]
    dones = [0.0, 0.0, 1.0]
    transitions = list(zip(states, actions, rewards, next_states, dones))
    return transitions


def test_rollout_eval_detects_identical_policy_sequences(tmp_path: Path):
    runner = IntegratedSolutionRunner(_mk_cfg(tmp_path))
    runner.results['data'] = {'source': 'synthetic'}
    train = _mk_data()
    eval_data = _mk_data()
    policies = {'P1': _ConstPolicy(0.5), 'P2': _ConstPolicy(0.5)}
    with pytest.raises(RuntimeError, match="Identical policy action sequences"):
        runner._run_policy_rollout_evaluation(policies, train, eval_data, export_prefix='test')


def test_rollout_eval_exports_required_metrics(tmp_path: Path):
    runner = IntegratedSolutionRunner(_mk_cfg(tmp_path))
    runner.results['data'] = {'source': 'synthetic'}
    train = _mk_data()
    eval_data = _mk_data()
    policies = {'P1': _ConstPolicy(0.2), 'P2': _ConstPolicy(0.8)}
    payload = runner._run_policy_rollout_evaluation(policies, train, eval_data, export_prefix='test')
    summary = payload['summary_df']
    for col in [
        'mean_reward', 'std_reward', 'median_reward', 'mean_episode_length',
        'unsafe_action_rate', 'constraint_satisfaction_rate',
        'mean_action_value', 'std_action_value',
    ]:
        assert col in summary.columns
