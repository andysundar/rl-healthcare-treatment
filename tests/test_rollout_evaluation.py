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
        resume=False,
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


def test_rollout_cache_atomic_and_lightweight(tmp_path: Path):
    runner = IntegratedSolutionRunner(_mk_cfg(tmp_path))
    runner.results['data'] = {'source': 'synthetic'}
    train = _mk_data()
    eval_data = _mk_data()
    policies = {'P1': _ConstPolicy(0.2), 'P2': _ConstPolicy(0.8)}

    payload = runner._run_policy_rollout_evaluation(
        policies, train, eval_data, export_prefix='cache_test', use_cache=True
    )
    cache_path = runner._rollout_cache_path('cache_test')
    assert cache_path.exists()
    assert not cache_path.with_suffix(cache_path.suffix + '.tmp').exists()
    assert payload['raw_eval_df'] is not None
    assert payload['episode_summary_df'] is not None

    import pickle
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    assert 'summary_table' in cached
    assert 'raw_eval_df' not in cached
    assert 'episode_summary_df' not in cached
    assert 'evaluation_signature' in cached


def test_corrupt_rollout_cache_falls_back_to_recompute(tmp_path: Path):
    runner = IntegratedSolutionRunner(_mk_cfg(tmp_path))
    runner.results['data'] = {'source': 'synthetic'}
    train = _mk_data()
    eval_data = _mk_data()
    policies = {'P1': _ConstPolicy(0.1), 'P2': _ConstPolicy(0.9)}

    runner._run_policy_rollout_evaluation(
        policies, train, eval_data, export_prefix='corrupt', use_cache=True
    )
    cache_path = runner._rollout_cache_path('corrupt')
    cache_path.write_bytes(b'bad-pickle')

    payload = runner._run_policy_rollout_evaluation(
        policies, train, eval_data, export_prefix='corrupt', use_cache=True
    )
    assert payload['summary_df'] is not None
    assert payload['raw_eval_df'] is not None


def test_stage5_reuse_signature_rejects_dataset_and_eval_changes(tmp_path: Path):
    runner = IntegratedSolutionRunner(_mk_cfg(tmp_path))
    p = {'P1': _ConstPolicy(0.3)}
    train = _mk_data()
    eval_a = _mk_data()
    eval_b = _mk_data() + [(
        np.array([106.0, 0.1], dtype=np.float32),
        np.array([0.6], dtype=np.float32),
        1.0,
        np.array([107.0, 0.1], dtype=np.float32),
        1.0,
    )]

    sig_a = runner._build_rollout_signature(p, train, eval_a, {'fast_eval': False, 'max_eval_samples': 0})
    sig_b = runner._build_rollout_signature(p, train, eval_b, {'fast_eval': False, 'max_eval_samples': 0})
    ok, reason = runner._rollout_reuse_check(sig_a, sig_b)
    assert not ok
    assert reason == "dataset signature changed"

    sig_c = runner._build_rollout_signature(p, train, eval_a, {'fast_eval': True, 'max_eval_samples': 2})
    ok2, reason2 = runner._rollout_reuse_check(sig_a, sig_c)
    assert not ok2
    assert reason2 == "evaluation signature changed"

    p_alt = {'P1': _ConstPolicy(0.7)}
    sig_d = runner._build_rollout_signature(p_alt, train, eval_a, {'fast_eval': False, 'max_eval_samples': 0})
    ok3, reason3 = runner._rollout_reuse_check(sig_a, sig_d)
    assert not ok3
    assert reason3 == "policy artifact fingerprint changed"


def test_rollout_progress_logging_path(tmp_path: Path, caplog):
    cfg = _mk_cfg(tmp_path)
    cfg.rollout_progress_every_episodes = 1
    runner = IntegratedSolutionRunner(cfg)
    runner.results['data'] = {'source': 'synthetic'}
    train = _mk_data()
    # 5 tiny episodes (done every step) to force progress logs
    eval_data = []
    for i in range(5):
        s = np.array([100.0 + i, 0.1], dtype=np.float32)
        ns = np.array([101.0 + i, 0.1], dtype=np.float32)
        eval_data.append((s, np.array([0.4], dtype=np.float32), 1.0, ns, 1.0))

    with caplog.at_level("INFO"):
        runner._run_policy_rollout_evaluation(
            {'P1': _ConstPolicy(0.2)},
            train,
            eval_data,
            export_prefix='progress',
            use_cache=False,
        )
    assert any("rollout progress" in r.message for r in caplog.records)
