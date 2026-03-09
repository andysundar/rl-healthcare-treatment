"""
Comprehensive tests for baseline policies.

Tests all baseline implementations with synthetic data.
"""

import numpy as np
import pytest
from pathlib import Path
import sys
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.baselines import (
    RuleBasedPolicy,
    RandomPolicy,
    SafeRandomPolicy,
    BehaviorCloningPolicy,
    MeanActionPolicy,
    RegressionPolicy,
    KNNPolicy,
    BaselineComparator,
    create_diabetes_rule_policy,
    create_random_policy,
    compare_all_baselines
)
from src.models.baselines.base_baseline import BaselinePolicy, BaselineMetrics


@dataclass
class _CountingPolicy(BaselinePolicy):
    eval_calls: int = 0

    def __init__(self, name: str = "counting", state_dim: int = 10):
        action_space = {'dim': 1, 'low': np.array([0.0]), 'high': np.array([1.0])}
        super().__init__(name=name, action_space=action_space, state_dim=state_dim)
        self.eval_calls = 0

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.array([0.5], dtype=np.float32)

    def evaluate(self, test_data):
        self.eval_calls += 1
        rewards = [float(x[2]) for x in test_data]
        return BaselineMetrics(
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            safety_violations=0,
            total_steps=len(test_data),
            safety_rate=1.0,
            mean_action_value=0.5,
            std_action_value=0.0,
        )


# Test fixtures
@pytest.fixture
def action_space():
    """Standard action space for testing."""
    return {
        'type': 'continuous',
        'low': np.array([0.0]),
        'high': np.array([1.0]),
        'dim': 1
    }


@pytest.fixture
def synthetic_data():
    """Generate synthetic test data."""
    np.random.seed(42)
    n_samples = 100
    state_dim = 10
    
    states = np.random.randn(n_samples, state_dim)
    actions = np.random.uniform(0, 1, (n_samples, 1))
    rewards = np.random.randn(n_samples)
    next_states = np.random.randn(n_samples, state_dim)
    dones = np.zeros(n_samples)
    
    test_data = list(zip(states, actions, rewards, next_states, dones))
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'test_data': test_data,
        'state_dim': state_dim
    }


# Rule-based policy tests
def test_rule_based_policy_creation(action_space):
    """Test rule-based policy initialization."""
    policy = RuleBasedPolicy(
        name="Test-Rule",
        action_space=action_space,
        state_dim=10
    )
    
    assert policy.name == "Test-Rule"
    assert len(policy.rules) == 0
    assert policy.default_action is not None


def test_rule_based_diabetes_rules():
    """Test diabetes rule configuration."""
    policy = create_diabetes_rule_policy(state_dim=10, action_dim=1)
    
    # Should have 3 rules
    assert len(policy.rules) == 3
    
    # Test high glucose
    high_glucose_state = np.zeros(10)
    high_glucose_state[0] = 250  # High glucose
    action = policy.select_action(high_glucose_state)
    
    assert action is not None
    assert 0 <= action[0] <= 1  # Within bounds


def test_rule_based_evaluation(action_space, synthetic_data):
    """Test rule-based policy evaluation."""
    policy = RuleBasedPolicy(
        action_space=action_space,
        state_dim=synthetic_data['state_dim']
    )
    
    # Add simple rule
    policy.add_rule(
        name="test_rule",
        condition=lambda s: True,  # Always applies
        action=lambda s: np.array([0.5]),
        priority=1
    )
    
    metrics = policy.evaluate(synthetic_data['test_data'])
    
    assert metrics.total_steps == len(synthetic_data['test_data'])
    assert 0 <= metrics.safety_rate <= 1


def test_rule_statistics():
    """Test rule usage statistics."""
    policy = create_diabetes_rule_policy()
    
    # Select actions to trigger rules
    for _ in range(10):
        state = np.random.randn(10)
        state[0] = np.random.uniform(50, 300)  # Glucose
        policy.select_action(state)
    
    stats = policy.get_rule_statistics()
    
    assert 'total_decisions' in stats
    assert stats['total_decisions'] == 10


# Random policy tests
def test_random_policy_uniform(action_space):
    """Test uniform random policy."""
    policy = RandomPolicy(
        action_space=action_space,
        state_dim=10,
        seed=42,
        distribution='uniform'
    )
    
    state = np.random.randn(10)
    action1 = policy.select_action(state)
    action2 = policy.select_action(state)
    
    # Should be different (stochastic)
    assert not np.allclose(action1, action2)
    
    # Should be within bounds
    assert 0 <= action1[0] <= 1
    assert 0 <= action2[0] <= 1


def test_random_policy_gaussian(action_space):
    """Test Gaussian random policy."""
    policy = RandomPolicy(
        action_space=action_space,
        state_dim=10,
        seed=42,
        distribution='gaussian'
    )
    
    actions = []
    for _ in range(100):
        state = np.random.randn(10)
        action = policy.select_action(state)
        actions.append(action[0])
    
    # Should be roughly centered around 0.5
    assert 0.3 < np.mean(actions) < 0.7


def test_safe_random_policy(action_space):
    """Test safe random policy."""
    policy = SafeRandomPolicy(
        action_space=action_space,
        state_dim=10,
        num_samples=10,
        seed=42
    )
    
    state = np.random.randn(10)
    action = policy.select_action(state)
    
    assert action is not None
    assert 0 <= action[0] <= 1


# Behavior cloning tests
def test_behavior_cloning_creation():
    """Test behavior cloning policy creation."""
    policy = BehaviorCloningPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=10,
        hidden_dims=[64, 64]
    )
    
    assert policy.network is not None
    assert policy.optimizer is not None


def test_behavior_cloning_training(synthetic_data):
    """Test behavior cloning training."""
    policy = BehaviorCloningPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=synthetic_data['state_dim'],
        hidden_dims=[32, 32]
    )
    
    # Train
    history = policy.train(
        states=synthetic_data['states'],
        actions=synthetic_data['actions'],
        epochs=5,
        batch_size=32,
        verbose=False
    )
    
    assert len(history['loss']) == 5
    assert history['loss'][-1] < history['loss'][0]  # Loss should decrease


def test_behavior_cloning_prediction(synthetic_data):
    """Test behavior cloning prediction."""
    policy = BehaviorCloningPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=synthetic_data['state_dim']
    )
    
    # Train briefly
    policy.train(
        synthetic_data['states'],
        synthetic_data['actions'],
        epochs=2,
        verbose=False
    )
    
    # Predict
    state = synthetic_data['states'][0]
    action = policy.select_action(state)
    
    assert action is not None
    assert 0 <= action[0] <= 1


def test_behavior_cloning_save_load(synthetic_data, tmp_path):
    """Test saving and loading behavior cloning policy."""
    policy = BehaviorCloningPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=synthetic_data['state_dim']
    )
    
    # Train
    policy.train(
        synthetic_data['states'],
        synthetic_data['actions'],
        epochs=2,
        verbose=False
    )
    
    # Save
    save_path = tmp_path / "bc_policy.pt"
    policy.save(str(save_path))
    
    assert save_path.exists()
    
    # Load
    new_policy = BehaviorCloningPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=synthetic_data['state_dim']
    )
    new_policy.load(str(save_path))
    
    # Should give same predictions
    state = synthetic_data['states'][0]
    action1 = policy.select_action(state)
    action2 = new_policy.select_action(state)
    
    assert np.allclose(action1, action2)


# Statistical baseline tests
def test_mean_action_policy(synthetic_data):
    """Test mean action policy."""
    policy = MeanActionPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=synthetic_data['state_dim']
    )
    
    # Fit
    policy.fit(synthetic_data['states'], synthetic_data['actions'])
    
    # All predictions should be the same (mean action)
    state1 = synthetic_data['states'][0]
    state2 = synthetic_data['states'][1]
    
    action1 = policy.select_action(state1)
    action2 = policy.select_action(state2)
    
    assert np.allclose(action1, action2)


def test_regression_policy(synthetic_data):
    """Test regression policy."""
    policy = RegressionPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=synthetic_data['state_dim'],
        regression_type='ridge'
    )
    
    # Fit
    policy.fit(synthetic_data['states'], synthetic_data['actions'])
    
    # Predict
    state = synthetic_data['states'][0]
    action = policy.select_action(state)
    
    assert action is not None
    assert 0 <= action[0] <= 1


def test_knn_policy(synthetic_data):
    """Test KNN policy."""
    policy = KNNPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=synthetic_data['state_dim'],
        k=5
    )
    
    # Fit
    policy.fit(synthetic_data['states'], synthetic_data['actions'])
    
    # Predict
    state = synthetic_data['states'][0]
    action = policy.select_action(state)
    
    assert action is not None
    
    # Get neighbors
    distances, indices = policy.get_neighbors(state)
    assert len(distances) == 5
    assert len(indices) == 5


# Comparison framework tests
def test_baseline_comparator(synthetic_data):
    """Test baseline comparison framework."""
    comparator = BaselineComparator(synthetic_data['test_data'])
    
    # Add baselines
    rule_policy = create_diabetes_rule_policy()
    random_policy = create_random_policy(seed=42)
    
    comparator.add_baseline('rule', rule_policy)
    comparator.add_baseline('random', random_policy)
    
    # Evaluate
    results = comparator.evaluate_all(verbose=False)
    
    assert len(results) == 2
    assert 'rule' in results.index
    assert 'random' in results.index
    assert 'mean_reward' in results.columns


def test_compare_all_baselines_function(synthetic_data):
    """Test convenience comparison function."""
    baselines = {
        'rule': create_diabetes_rule_policy(),
        'random': create_random_policy(seed=42)
    }
    
    results = compare_all_baselines(
        test_data=synthetic_data['test_data'],
        baselines_dict=baselines
    )
    
    assert len(results) == 2
    assert 'mean_reward' in results.columns


def test_generate_report(synthetic_data, tmp_path):
    """Test report generation."""
    comparator = BaselineComparator(synthetic_data['test_data'])
    
    comparator.add_baseline('rule', create_diabetes_rule_policy())
    comparator.add_baseline('random', create_random_policy(seed=42))
    
    # Evaluate first
    comparator.evaluate_all(verbose=False)
    
    # Generate report
    report_path = tmp_path / "report.md"
    report = comparator.generate_report(str(report_path))
    
    assert report_path.exists()
    assert "Baseline Policy Comparison Report" in report
    
    # Check JSON file also created
    json_path = tmp_path / "report.json"
    assert json_path.exists()


# Integration tests
def test_full_comparison_workflow(synthetic_data, tmp_path):
    """Test complete comparison workflow."""
    # Create baselines
    rule_policy = create_diabetes_rule_policy()
    random_policy = create_random_policy(seed=42)
    
    mean_policy = MeanActionPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=synthetic_data['state_dim']
    )
    mean_policy.fit(synthetic_data['states'], synthetic_data['actions'])
    
    bc_policy = BehaviorCloningPolicy(
        action_space={'dim': 1, 'low': np.array([0]), 'high': np.array([1])},
        state_dim=synthetic_data['state_dim'],
        hidden_dims=[32]
    )
    bc_policy.train(
        synthetic_data['states'],
        synthetic_data['actions'],
        epochs=3,
        verbose=False
    )
    
    # Compare
    baselines = {
        'Rule-Based': rule_policy,
        'Random': random_policy,
        'Mean-Action': mean_policy,
        'Behavior-Cloning': bc_policy
    }
    
    results = compare_all_baselines(
        test_data=synthetic_data['test_data'],
        baselines_dict=baselines,
        output_path=str(tmp_path / "comparison_report.md")
    )
    
    # Verify results
    assert len(results) == 4
    assert 'mean_reward' in results.columns
    assert 'safety_rate' in results.columns
    
    # Verify report files
    assert (tmp_path / "comparison_report.md").exists()
    assert (tmp_path / "comparison_report.json").exists()


def test_generate_report_does_not_recompute_when_results_present(synthetic_data, tmp_path, monkeypatch):
    comparator = BaselineComparator(synthetic_data['test_data'])
    policy = _CountingPolicy(state_dim=synthetic_data['state_dim'])
    comparator.add_baseline('counting', policy)
    comparator.evaluate_all(verbose=False)
    assert policy.eval_calls == 1

    def _fail_evaluate(*args, **kwargs):
        raise AssertionError("generate_report should not recompute evaluate_all when cached results exist")

    monkeypatch.setattr(comparator, "evaluate_all", _fail_evaluate)
    report_path = tmp_path / "cached_report.md"
    report = comparator.generate_report(output_path=str(report_path))
    assert "Baseline Policy Comparison Report" in report
    assert report_path.exists()


def test_baseline_cache_reuse_from_disk(synthetic_data, tmp_path):
    comparator = BaselineComparator(synthetic_data['test_data'])
    policy = _CountingPolicy(state_dim=synthetic_data['state_dim'])
    comparator.add_baseline('counting', policy)
    cache_path = tmp_path / "baseline_eval_cache.json"

    comparator.evaluate_all(verbose=False, cache_path=str(cache_path))
    assert policy.eval_calls == 1

    # New comparator instance should load cache and avoid policy.evaluate call
    comparator2 = BaselineComparator(synthetic_data['test_data'])
    policy2 = _CountingPolicy(state_dim=synthetic_data['state_dim'])
    comparator2.add_baseline('counting', policy2)
    comparator2.evaluate_all(verbose=False, cache_path=str(cache_path))
    assert policy2.eval_calls == 0


def test_include_exclude_baseline_selection(synthetic_data):
    comparator = BaselineComparator(synthetic_data['test_data'])
    comparator.add_baseline('rule', create_diabetes_rule_policy())
    comparator.add_baseline('random', create_random_policy(seed=42))
    comparator.add_baseline('mean', _CountingPolicy(name='mean', state_dim=synthetic_data['state_dim']))

    results = comparator.evaluate_all(
        verbose=False,
        include_baselines=['rule', 'random'],
        exclude_baselines=['random'],
    )
    assert list(results.index) == ['rule']


def test_fast_eval_subsampling_is_deterministic(synthetic_data):
    comparator = BaselineComparator(synthetic_data['test_data'])
    comparator.add_baseline('counting', _CountingPolicy(state_dim=synthetic_data['state_dim']))
    max_samples = 20

    subset1 = comparator._subsample_test_data(max_eval_samples=max_samples, seed=123)
    subset2 = comparator._subsample_test_data(max_eval_samples=max_samples, seed=123)
    assert len(subset1) == max_samples
    assert len(subset2) == max_samples
    states1 = np.stack([x[0] for x in subset1], axis=0)
    states2 = np.stack([x[0] for x in subset2], axis=0)
    assert np.allclose(states1, states2)


def test_unknown_baseline_names_fail_clearly(synthetic_data):
    comparator = BaselineComparator(synthetic_data['test_data'])
    comparator.add_baseline('rule', create_diabetes_rule_policy())
    with pytest.raises(ValueError, match="Unknown baseline names"):
        comparator.evaluate_all(verbose=False, include_baselines=['does_not_exist'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
