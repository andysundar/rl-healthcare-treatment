"""
Healthcare RL Evaluation Framework

Comprehensive evaluation tools for offline RL in healthcare settings.
"""

from configs.config import EvaluationConfig
from .off_policy_eval import OffPolicyEvaluator, Trajectory, OPEResult
from .safety_metrics import (
    SafetyEvaluator,
    SafetyResult,
    compute_safety_index,
    generate_safety_report,
)
from .off_policy_evaluation import (
    BehaviorPolicy,
    WISEstimator,
    DMEstimator,
    DREstimator,
    OPERunner,
    load_cql_checkpoint,
    load_encoder,
    print_summary_table,
)
from .safety_metrics import SafetyEvaluator, SafetyResult
from .clinical_metrics import ClinicalEvaluator, ClinicalResult
from .performance_metrics import PerformanceEvaluator, PerformanceResult
from .comparison import PolicyComparator
from .visualizations import EvaluationVisualizer
from .interpretability import (
    InterpretabilityConfig,
    CounterfactualExplainer,
    DecisionRuleExtractor,
    PersonalizationScorer,
)

__version__ = "1.0.0"

__all__ = [
    'EvaluationConfig',
    'OffPolicyEvaluator',
    'SafetyEvaluator',
    'ClinicalEvaluator',
    'PerformanceEvaluator',
    'PolicyComparator',
    'EvaluationVisualizer',
    'Trajectory',
    'OPEResult',
    'SafetyResult',
    'compute_safety_index',
    'generate_safety_report',
    'ClinicalResult',
    'PerformanceResult',
    # Standalone OPE estimators
    'BehaviorPolicy',
    'WISEstimator',
    'DMEstimator',
    'DREstimator',
    'OPERunner',
    'load_cql_checkpoint',
    'load_encoder',
    'print_summary_table',
    # Interpretability
    'InterpretabilityConfig',
    'CounterfactualExplainer',
    'DecisionRuleExtractor',
    'PersonalizationScorer',
]
