"""
Reward Shaping Utilities
Helper functions for reward normalization, clipping, and shaping.
"""

from typing import Dict, Any, Callable, Tuple
import numpy as np


def normalize_reward(
    reward: float,
    min_val: float,
    max_val: float,
    target_range: Tuple[float, float] = (0.0, 1.0)
) -> float:
    """
    Normalize reward to target range.
    
    Args:
        reward: Raw reward value
        min_val: Minimum expected reward
        max_val: Maximum expected reward
        target_range: Target (min, max) range
        
    Returns:
        Normalized reward in target range
    """
    if abs(max_val - min_val) < 1e-8:
        # Avoid division by zero
        return (target_range[0] + target_range[1]) / 2.0
    
    # Normalize to [0, 1]
    normalized = (reward - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Scale to target range
    target_min, target_max = target_range
    scaled = target_min + normalized * (target_max - target_min)
    
    return scaled


def clip_reward(
    reward: float,
    clip_range: Tuple[float, float]
) -> float:
    """
    Clip reward to specified range.
    
    Args:
        reward: Raw reward value
        clip_range: (min, max) clipping range
        
    Returns:
        Clipped reward
    """
    min_val, max_val = clip_range
    return np.clip(reward, min_val, max_val)


def smooth_reward(
    current_reward: float,
    previous_reward: float,
    smoothing_factor: float = 0.9
) -> float:
    """
    Exponentially smooth reward to reduce variance.
    
    Args:
        current_reward: Current reward value
        previous_reward: Previous smoothed reward
        smoothing_factor: Smoothing factor in [0, 1] (higher = more smoothing)
        
    Returns:
        Smoothed reward
    """
    return smoothing_factor * previous_reward + (1 - smoothing_factor) * current_reward


def sparse_to_dense(
    sparse_reward: float,
    state: Dict[str, Any],
    action: Dict[str, Any],
    shaping_function: Callable[[Dict, Dict], float]
) -> float:
    """
    Convert sparse reward to dense reward using shaping.
    
    Args:
        sparse_reward: Original sparse reward (e.g., only at episode end)
        state: Current state
        action: Current action
        shaping_function: Function that computes shaping reward from state/action
        
    Returns:
        Dense reward (sparse + shaping)
    """
    shaping = shaping_function(state, action)
    return sparse_reward + shaping


def potential_based_shaping(
    state: Dict[str, Any],
    next_state: Dict[str, Any],
    potential_function: Callable[[Dict], float],
    gamma: float = 0.99
) -> float:
    """
    Compute potential-based reward shaping.
    
    This is a theoretically sound way to shape rewards without
    changing the optimal policy.
    
    Formula: F(s, s') = γ * Φ(s') - Φ(s)
    
    Args:
        state: Current state
        next_state: Next state
        potential_function: Function that computes potential Φ(s)
        gamma: Discount factor
        
    Returns:
        Shaping reward
    """
    current_potential = potential_function(state)
    next_potential = potential_function(next_state)
    
    shaping = gamma * next_potential - current_potential
    return shaping


def health_potential_function(state: Dict[str, Any]) -> float:
    """
    Example potential function based on health metrics.
    
    Higher potential = healthier state
    
    Args:
        state: Patient state
        
    Returns:
        Potential value (higher is better)
    """
    potential = 0.0
    
    # Glucose potential (closer to 100 = higher potential)
    glucose = state.get('glucose', 100.0)
    glucose_target = 100.0
    glucose_potential = -abs(glucose - glucose_target) / 100.0
    potential += glucose_potential
    
    # Blood pressure potential
    systolic = state.get('systolic_bp', 110.0)
    systolic_target = 110.0
    bp_potential = -abs(systolic - systolic_target) / 50.0
    potential += bp_potential
    
    # Adherence potential
    adherence = state.get('adherence_score', 0.5)
    adherence_potential = adherence - 0.5  # Centered at 0.5
    potential += adherence_potential
    
    return potential


def adaptive_reward_scaling(
    reward: float,
    reward_history: list,
    target_std: float = 1.0
) -> float:
    """
    Adaptively scale rewards to maintain consistent variance.
    
    Args:
        reward: Current reward
        reward_history: List of recent rewards
        target_std: Target standard deviation
        
    Returns:
        Scaled reward
    """
    if len(reward_history) < 10:
        # Not enough history
        return reward
    
    current_std = np.std(reward_history)
    
    if current_std < 1e-8:
        # No variance
        return reward
    
    scaling_factor = target_std / current_std
    return reward * scaling_factor


def reward_curriculum(
    base_reward: float,
    episode: int,
    total_episodes: int,
    curriculum_type: str = 'linear'
) -> float:
    """
    Apply curriculum learning to rewards.
    
    Start with easier (more reward) and gradually make harder.
    
    Args:
        base_reward: Original reward
        episode: Current episode number
        total_episodes: Total number of training episodes
        curriculum_type: Type of curriculum ('linear', 'exponential', 'step')
        
    Returns:
        Curriculum-adjusted reward
    """
    progress = episode / total_episodes
    
    if curriculum_type == 'linear':
        difficulty = progress
    elif curriculum_type == 'exponential':
        difficulty = progress ** 2
    elif curriculum_type == 'step':
        # Step curriculum: easy -> medium -> hard
        if progress < 0.33:
            difficulty = 0.0
        elif progress < 0.67:
            difficulty = 0.5
        else:
            difficulty = 1.0
    else:
        difficulty = progress
    
    # Make rewards more challenging as training progresses
    # Early: bonus rewards, Late: full difficulty
    bonus = (1.0 - difficulty) * abs(base_reward) * 0.5
    
    if base_reward > 0:
        return base_reward + bonus
    else:
        return base_reward - bonus


def hindsight_reward(
    achieved_state: Dict[str, Any],
    desired_state: Dict[str, Any],
    distance_metric: Callable[[Dict, Dict], float]
) -> float:
    """
    Compute hindsight reward based on achieved vs desired state.
    
    Useful for goal-conditioned RL.
    
    Args:
        achieved_state: State that was achieved
        desired_state: Desired target state
        distance_metric: Function to compute distance between states
        
    Returns:
        Reward (higher = closer to goal)
    """
    distance = distance_metric(achieved_state, desired_state)
    
    # Convert distance to reward (closer = higher reward)
    reward = -distance
    return reward


def state_distance_l2(state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
    """
    Compute L2 distance between states.
    
    Args:
        state1: First state
        state2: Second state
        
    Returns:
        L2 distance
    """
    # Extract numeric features
    features = ['glucose', 'systolic_bp', 'diastolic_bp', 'adherence_score']
    
    distance = 0.0
    for feature in features:
        if feature in state1 and feature in state2:
            diff = state1[feature] - state2[feature]
            distance += diff ** 2
    
    return np.sqrt(distance)


def curiosity_bonus(
    state: Dict[str, Any],
    state_visit_counts: Dict[str, int],
    bonus_scale: float = 0.1
) -> float:
    """
    Add curiosity bonus for exploring novel states.
    
    Args:
        state: Current state
        state_visit_counts: Dictionary tracking state visit counts
        bonus_scale: Scale of curiosity bonus
        
    Returns:
        Curiosity bonus
    """
    # Hash state to count visits
    state_key = _hash_state(state)
    
    visit_count = state_visit_counts.get(state_key, 0)
    
    # Bonus inversely proportional to visit count
    bonus = bonus_scale / (1.0 + visit_count)
    
    return bonus


def _hash_state(state: Dict[str, Any], precision: int = 1) -> str:
    """
    Create hash of state for counting visits.
    
    Args:
        state: State dictionary
        precision: Decimal precision for rounding
        
    Returns:
        State hash string
    """
    # Round numeric values and create string representation
    rounded = {}
    
    for key, value in state.items():
        if isinstance(value, (int, float)):
            rounded[key] = round(value, precision)
        else:
            rounded[key] = value
    
    # Create sorted string representation
    items = sorted(rounded.items())
    hash_str = str(items)
    
    return hash_str


def reward_smoothing_filter(
    rewards: list,
    window_size: int = 5
) -> list:
    """
    Apply moving average smoothing to reward sequence.
    
    Args:
        rewards: List of rewards
        window_size: Size of smoothing window
        
    Returns:
        Smoothed rewards
    """
    if len(rewards) < window_size:
        return rewards
    
    smoothed = []
    
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(rewards), i + window_size // 2 + 1)
        
        window = rewards[start_idx:end_idx]
        smoothed_value = np.mean(window)
        smoothed.append(smoothed_value)
    
    return smoothed


class RewardNormalizer:
    """
    Online reward normalization using running statistics.
    """
    
    def __init__(self, clip_range: Tuple[float, float] = (-10.0, 10.0)):
        """
        Initialize reward normalizer.
        
        Args:
            clip_range: Range to clip normalized rewards
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip_range = clip_range
        
    def update(self, reward: float) -> None:
        """
        Update running statistics with new reward.
        
        Args:
            reward: New reward value
        """
        self.count += 1
        
        # Welford's online algorithm for mean and variance
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count
        
    def normalize(self, reward: float) -> float:
        """
        Normalize reward using running statistics.
        
        Args:
            reward: Raw reward
            
        Returns:
            Normalized reward
        """
        if self.count < 2:
            return reward
        
        std = np.sqrt(self.var)
        
        if std < 1e-8:
            normalized = reward - self.mean
        else:
            normalized = (reward - self.mean) / std
        
        # Clip to prevent extreme values
        normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])
        
        return normalized
    
    def reset(self) -> None:
        """Reset statistics."""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
