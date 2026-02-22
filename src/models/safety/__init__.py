"""
Safety Package
Complete safety system for healthcare RL

Main Components:
- SafetyLayer: Primary interface for safety checks
- SafetyConfig: Configuration for safety system
- Constraint classes: DosageConstraint, PhysiologicalConstraint, etc.
- SafetyCritic: Neural network for learned safety prediction
- SafetyMetrics: Functions for computing safety metrics
- SafeRLAgent: RL agent wrapper with safety enforcement
"""

from .config import SafetyConfig
from .constraints import (
    Constraint,
    DosageConstraint,
    PhysiologicalConstraint,
    ContraindicationConstraint,
    FrequencyConstraint
)
from .safety_critic import SafetyCritic, train_safety_critic
from .constraint_optimizer import ConstrainedActionOptimizer
from .safety_layer import SafetyLayer, SafeRLAgent
from .safety_metrics import (
    safety_index,
    violation_rate,
    violation_severity,
    constraint_satisfaction_rate,
    generate_safety_report,
    print_safety_report
)

__all__ = [
    # Configuration
    'SafetyConfig',
    
    # Constraints
    'Constraint',
    'DosageConstraint',
    'PhysiologicalConstraint',
    'ContraindicationConstraint',
    'FrequencyConstraint',
    
    # Safety Critic
    'SafetyCritic',
    'train_safety_critic',
    
    # Optimizer
    'ConstrainedActionOptimizer',
    
    # Main Interface
    'SafetyLayer',
    'SafeRLAgent',
    
    # Metrics
    'safety_index',
    'violation_rate',
    'violation_severity',
    'constraint_satisfaction_rate',
    'generate_safety_report',
    'print_safety_report',
]

__version__ = '1.0.0'
