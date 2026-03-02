import importlib.util
from pathlib import Path
import numpy as np
import torch

from src.models.rl.iql import IQLAgent, IQLConfig

_spec = importlib.util.spec_from_file_location(
    'off_policy_eval', Path('src/evaluation/off_policy_eval.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
OffPolicyEvaluator = _mod.OffPolicyEvaluator
Trajectory = _mod.Trajectory


def test_iql_outputs_valid_actions_and_updates():
    cfg = IQLConfig(state_dim=4, n_actions=3)
    agent = IQLAgent(cfg)
    batch = {
        'states': torch.randn(32, 4),
        'actions': torch.randint(0, 3, (32, 1)).float(),
        'rewards': torch.randn(32, 1),
        'next_states': torch.randn(32, 4),
        'dones': torch.zeros(32, 1),
    }
    losses = agent.update(batch)
    assert losses['q_loss'] >= 0
    a = agent.select_action(np.zeros(4, dtype=np.float32), deterministic=True)
    assert int(a[0]) in [0, 1, 2]


def test_ope_returns_ess_and_ci():
    class Pol:
        def select_action(self, s, deterministic=True):
            return np.array([0.0], dtype=np.float32)

    trajs = [Trajectory(
        states=np.array([[0.0], [1.0]], dtype=np.float32),
        actions=np.array([[0.0], [0.0]], dtype=np.float32),
        rewards=np.array([1.0, 1.0], dtype=np.float32),
        next_states=np.array([[1.0], [2.0]], dtype=np.float32),
        dones=np.array([0.0, 1.0], dtype=np.float32),
    ) for _ in range(8)]

    ope = OffPolicyEvaluator(seed=0)
    res = ope.evaluate(Pol(), Pol(), trajs, methods=['wis'])
    assert 'wis' in res
    assert 'ess' in res['wis'].metadata
    assert len(res['wis'].confidence_interval) == 2
