"""
Offline Reinforcement Learning Package for Healthcare

This package provides production-ready implementations of offline RL algorithms
for healthcare treatment recommendations, with emphasis on safety and clinical applicability.

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

from .base_agent import BaseRLAgent
from .cql import CQLAgent
from .bcq import BCQAgent
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .trainer import OfflineRLTrainer, EvaluationManager
from .networks import QNetwork, PolicyNetwork, ValueNetwork
from .config import (
    CQLConfig,
    BCQConfig,
    TrainingConfig,
    SafetyConfig,
    HealthcareConfig,
    get_diabetes_management_config,
    get_mimic_experiment_config
)

__version__ = '1.0.0'

__all__ = [
    # Agents
    'BaseRLAgent',
    'CQLAgent',
    'BCQAgent',
    
    # Replay buffers
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    
    # Training
    'OfflineRLTrainer',
    'EvaluationManager',
    
    # Networks
    'QNetwork',
    'PolicyNetwork',
    'ValueNetwork',
    
    # Configuration
    'CQLConfig',
    'BCQConfig',
    'TrainingConfig',
    'SafetyConfig',
    'HealthcareConfig',
    'get_diabetes_management_config',
    'get_mimic_experiment_config'
]
