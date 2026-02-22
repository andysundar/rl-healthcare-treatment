"""
Baseline models for healthcare RL comparison.

This package provides various baseline policies for comparison against
reinforcement learning approaches:

- Rule-based policies (clinical guidelines)
- Random policies (lower bound)
- Behavior cloning (supervised learning from historical data)
- Statistical baselines (mean, regression, KNN)
- Comparison framework

Example usage:
    from src.models.baselines import (
        RuleBasedPolicy,
        RandomPolicy,
        BehaviorCloningPolicy,
        compare_all_baselines
    )
    
    # Create baselines
    rule_policy = RuleBasedPolicy(...)
    random_policy = RandomPolicy(...)
    bc_policy = BehaviorCloningPolicy(...)
    
    # Compare
    baselines = {
        'rule': rule_policy,
        'random': random_policy,
        'bc': bc_policy
    }
    
    results = compare_all_baselines(test_data, baselines)
"""

from .base_baseline import BaselinePolicy, BaselineMetrics
from .rule_based import (
    RuleBasedPolicy, 
    ClinicalRule,
    create_diabetes_rule_policy,
    create_hypertension_rule_policy
)
from .random_policy import (
    RandomPolicy,
    SafeRandomPolicy,
    create_random_policy,
    create_safe_random_policy
)
from .behavior_cloning import (
    BehaviorCloningPolicy,
    BehaviorCloningNetwork,
    create_behavior_cloning_policy
)
from .statistical_baseline import (
    MeanActionPolicy,
    RegressionPolicy,
    KNNPolicy,
    create_mean_action_policy,
    create_regression_policy,
    create_knn_policy
)
from .compare_baselines import (
    BaselineComparator,
    compare_all_baselines,
    compute_action_stability,
    compute_safety_margin,
    compute_expected_return
)

__all__ = [
    # Base classes
    'BaselinePolicy',
    'BaselineMetrics',
    
    # Rule-based
    'RuleBasedPolicy',
    'ClinicalRule',
    'create_diabetes_rule_policy',
    'create_hypertension_rule_policy',
    
    # Random
    'RandomPolicy',
    'SafeRandomPolicy',
    'create_random_policy',
    'create_safe_random_policy',
    
    # Behavior cloning
    'BehaviorCloningPolicy',
    'BehaviorCloningNetwork',
    'create_behavior_cloning_policy',
    
    # Statistical
    'MeanActionPolicy',
    'RegressionPolicy',
    'KNNPolicy',
    'create_mean_action_policy',
    'create_regression_policy',
    'create_knn_policy',
    
    # Comparison
    'BaselineComparator',
    'compare_all_baselines',
    'compute_action_stability',
    'compute_safety_margin',
    'compute_expected_return',
]

__version__ = '1.0.0'
