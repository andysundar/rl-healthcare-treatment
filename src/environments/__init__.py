"""
Healthcare RL Environments
==========================
Simulation environments for testing reinforcement learning policies in healthcare.

This package provides:
- BaseHealthcareEnv: Abstract base class following Gymnasium interface
- DiabetesManagementEnv: Diabetes treatment optimization with Bergman model
- MedicationAdherenceEnv: Medication adherence optimization
- Disease models: Physiological simulation models
- PatientSimulator: Generate diverse patient populations
- Testing utilities: Comprehensive environment testing tools
"""

from .base_env import BaseHealthcareEnv
from .diabetes_env import DiabetesManagementEnv as DiabetesEnv, DiabetesEnvConfig
from .adherence_env import MedicationAdherenceEnv as AdherenceEnv, AdherenceEnvConfig

# Legacy aliases used by example scripts
DiabetesManagementEnv  = DiabetesEnv
MedicationAdherenceEnv = AdherenceEnv
from .disease_models import (
    BergmanMinimalModel,
    BergmanModelParams,
    AdherenceDynamicsModel,
    AdherenceModelParams,
    BloodPressureModel
)
from .patient_simulator import (
    PatientSimulator,
    DiabetesPatient,
    AdherencePatient,
    PatientDemographics,
    PatientClinical,
    DiseaseSeverity
)
from .test_env import (
    test_environment,
    test_determinism,
    test_action_effects,
    test_safety_constraints,
    benchmark_environment,
    comprehensive_test_suite,
    PolicyEvaluator
)

__version__ = "0.1.0"

__all__ = [
    # Base environment
    'BaseHealthcareEnv',
    
    # Specific environments
    'DiabetesEnv',
    'DiabetesEnvConfig',
    'AdherenceEnv',
    'AdherenceEnvConfig',
    
    # Disease models
    'BergmanMinimalModel',
    'BergmanModelParams',
    'AdherenceDynamicsModel',
    'AdherenceModelParams',
    'BloodPressureModel',
    
    # Patient simulation
    'PatientSimulator',
    'DiabetesPatient',
    'AdherencePatient',
    'PatientDemographics',
    'PatientClinical',
    'DiseaseSeverity',
    
    # Testing utilities
    'test_environment',
    'test_determinism',
    'test_action_effects',
    'test_safety_constraints',
    'benchmark_environment',
    'comprehensive_test_suite',
    'PolicyEvaluator',
]
