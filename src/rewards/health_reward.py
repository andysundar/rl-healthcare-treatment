"""
Health Outcome Reward Function
Rewards improvements in clinical health markers (glucose, BP, HbA1c, etc.).
"""

from typing import Dict, Any, Tuple, List
import numpy as np
from .base_reward import BaseRewardFunction


class HealthOutcomeReward(BaseRewardFunction):
    """
    Reward function based on clinical health markers.
    
    Tracks multiple health metrics and rewards:
    1. Being within target clinical ranges
    2. Moving toward target ranges
    3. Avoiding dangerous physiological states
    
    Clinical Rationale:
    - Primary goal is maintaining optimal clinical markers
    - Different metrics have different importance
    - Gradual improvements should be rewarded
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize health outcome reward.
        
        Args:
            config: Configuration with health metric targets and weights
        """
        super().__init__(config)
        
        # Default target ranges (can be overridden by config)
        self.metric_targets = {
            'glucose': getattr(config, 'glucose_target', (80.0, 130.0)),
            'systolic_bp': getattr(config, 'systolic_bp_target', (90.0, 120.0)),
            'diastolic_bp': getattr(config, 'diastolic_bp_target', (60.0, 80.0)),
            'hba1c': getattr(config, 'hba1c_target', (4.0, 6.5)),
            'weight': getattr(config, 'weight_target', None),  # Personalized
            'cholesterol': getattr(config, 'cholesterol_target', (0.0, 200.0))
        }
        
        # Metric weights (importance)
        self.metric_weights = {
            'glucose': getattr(config, 'glucose_weight', 1.0),
            'systolic_bp': getattr(config, 'systolic_bp_weight', 0.8),
            'diastolic_bp': getattr(config, 'diastolic_bp_weight', 0.6),
            'hba1c': getattr(config, 'hba1c_weight', 1.2),
            'weight': getattr(config, 'weight_weight', 0.5),
            'cholesterol': getattr(config, 'cholesterol_weight', 0.7)
        }
        
        # Improvement bonus multiplier
        self.improvement_multiplier = getattr(config, 'health_improvement_multiplier', 1.5)
        
    def compute_reward(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Compute health outcome reward.
        
        Args:
            state: Previous patient state with health metrics
            action: Action taken
            next_state: New state with updated health metrics
            
        Returns:
            Health outcome reward (positive for good/improving health)
        """
        total_reward = 0.0
        total_weight = 0.0
        
        for metric_name, target_range in self.metric_targets.items():
            if target_range is None:
                continue
                
            # Get current and previous values
            current_value = next_state.get(metric_name)
            previous_value = state.get(metric_name)
            
            if current_value is None:
                continue
                
            # Weight for this metric
            weight = self.metric_weights.get(metric_name, 1.0)
            total_weight += weight
            
            # Compute reward for being in/near target range
            range_reward = self._compute_metric_reward(current_value, target_range)
            
            # Compute improvement bonus
            improvement_bonus = 0.0
            if previous_value is not None:
                improvement_bonus = self._compute_improvement_reward(
                    previous_value,
                    current_value,
                    target_range
                )
            
            # Weighted contribution
            metric_reward = (range_reward + improvement_bonus) * weight
            total_reward += metric_reward
        
        # Normalize by total weight
        if total_weight > 0:
            total_reward /= total_weight
            
        return total_reward
    
    def _compute_metric_reward(
        self,
        value: float,
        target_range: Tuple[float, float]
    ) -> float:
        """
        Compute reward for a single metric based on target range.
        
        Args:
            value: Current metric value
            target_range: (min, max) target range
            
        Returns:
            Reward in [-1, 1] range
        """
        min_val, max_val = target_range
        
        if min_val <= value <= max_val:
            # In target range: positive reward
            # Closer to center = higher reward
            center = (min_val + max_val) / 2
            range_width = max_val - min_val
            distance_from_center = abs(value - center)
            normalized_distance = distance_from_center / (range_width / 2)
            return 1.0 - 0.3 * normalized_distance
        
        elif value < min_val:
            # Below target: penalty proportional to distance
            distance = min_val - value
            normalized_distance = distance / min_val if min_val > 0 else distance
            return -min(normalized_distance, 1.0)
        
        else:  # value > max_val
            # Above target: penalty proportional to distance
            distance = value - max_val
            normalized_distance = distance / max_val if max_val > 0 else distance
            return -min(normalized_distance, 1.0)
    
    def _compute_improvement_reward(
        self,
        previous_value: float,
        current_value: float,
        target_range: Tuple[float, float]
    ) -> float:
        """
        Compute reward for improvement toward target.
        
        Args:
            previous_value: Previous metric value
            current_value: Current metric value
            target_range: Target range for metric
            
        Returns:
            Improvement reward (positive for moving toward target)
        """
        min_val, max_val = target_range
        center = (min_val + max_val) / 2
        
        # Distance from target center
        prev_distance = abs(previous_value - center)
        curr_distance = abs(current_value - center)
        
        # Improvement = reduction in distance
        improvement = prev_distance - curr_distance
        
        # Normalize by range width
        range_width = max_val - min_val
        normalized_improvement = improvement / range_width if range_width > 0 else 0.0
        
        # Scale by improvement multiplier
        return normalized_improvement * self.improvement_multiplier
    
    def get_reward_components(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of health reward components.
        
        Returns:
            Dictionary with per-metric reward breakdown
        """
        components = {}
        
        for metric_name, target_range in self.metric_targets.items():
            if target_range is None:
                continue
                
            current_value = next_state.get(metric_name)
            previous_value = state.get(metric_name)
            
            if current_value is None:
                continue
            
            # Range reward
            range_reward = self._compute_metric_reward(current_value, target_range)
            components[f'health_{metric_name}_range'] = range_reward
            
            # Improvement reward
            if previous_value is not None:
                improvement = self._compute_improvement_reward(
                    previous_value,
                    current_value,
                    target_range
                )
                components[f'health_{metric_name}_improvement'] = improvement
        
        # Total
        components['health_total'] = self.compute_reward(state, action, next_state)
        
        return components
    
    def set_personalized_targets(
        self,
        patient_id: str,
        metric_targets: Dict[str, Tuple[float, float]]
    ) -> None:
        """
        Set personalized target ranges for a specific patient.
        
        Args:
            patient_id: Patient identifier
            metric_targets: Dictionary of metric targets for this patient
        """
        # Store patient-specific targets
        # In a real system, this would persist to database
        self.metric_targets.update(metric_targets)
