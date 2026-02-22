"""
Reward Functions Package

This package contains all reward function implementations for the healthcare RL system.

Components:
- BaseRewardFunction: Abstract base class for all reward functions
- AdherenceReward: Rewards medication adherence improvements
- HealthOutcomeReward: Rewards improvements in clinical markers
- SafetyPenalty: Penalizes unsafe states and actions
- CostEffectivenessReward: Penalizes high-cost interventions
- CompositeRewardFunction: Combines multiple reward components
- RewardConfig: Configuration for all reward components

Usage:
    from src.rewards import CompositeRewardFunction, RewardConfig
    from src.rewards import AdherenceReward, HealthOutcomeReward, SafetyPenalty
    
    # Create configuration
    config = RewardConfig(
        w_adherence=1.0,
        w_health=2.0,
        w_safety=5.0,
        w_cost=0.1
    )
    
    # Create composite reward
    reward_fn = CompositeRewardFunction(config)
    
    # Add components
    reward_fn.add_component('adherence', AdherenceReward(config), config.w_adherence)
    reward_fn.add_component('health', HealthOutcomeReward(config), config.w_health)
    reward_fn.add_component('safety', SafetyPenalty(config), config.w_safety)
    
    # Compute reward
    reward = reward_fn.compute_reward(state, action, next_state)
"""

from .base_reward import BaseRewardFunction
from .adherence_reward import AdherenceReward
from .health_reward import HealthOutcomeReward
from .safety_reward import SafetyPenalty
from .cost_reward import CostEffectivenessReward
from .composite_reward import CompositeRewardFunction
from .reward_config import (
    RewardConfig,
    ConservativeRewardConfig,
    AggressiveRewardConfig,
    CostAwareRewardConfig
)
from .reward_shaping import (
    normalize_reward,
    clip_reward,
    smooth_reward,
    sparse_to_dense,
    potential_based_shaping,
    health_potential_function,
    adaptive_reward_scaling,
    reward_curriculum,
    hindsight_reward,
    state_distance_l2,
    curiosity_bonus,
    reward_smoothing_filter,
    RewardNormalizer
)

__all__ = [
    # Base classes
    'BaseRewardFunction',
    
    # Component rewards
    'AdherenceReward',
    'HealthOutcomeReward',
    'SafetyPenalty',
    'CostEffectivenessReward',
    
    # Composite
    'CompositeRewardFunction',
    
    # Configuration
    'RewardConfig',
    'ConservativeRewardConfig',
    'AggressiveRewardConfig',
    'CostAwareRewardConfig',
    
    # Shaping utilities
    'normalize_reward',
    'clip_reward',
    'smooth_reward',
    'sparse_to_dense',
    'potential_based_shaping',
    'health_potential_function',
    'adaptive_reward_scaling',
    'reward_curriculum',
    'hindsight_reward',
    'state_distance_l2',
    'curiosity_bonus',
    'reward_smoothing_filter',
    'RewardNormalizer'
]

__version__ = '1.0.0'
__author__ = 'Andy Bandopadhyay'
