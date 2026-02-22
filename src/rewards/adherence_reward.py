"""
Adherence Reward Function
Rewards medication adherence and improvements in adherence behavior.
"""

from typing import Dict, Any
import numpy as np
from .base_reward import BaseRewardFunction


class AdherenceReward(BaseRewardFunction):
    """
    Reward function based on medication adherence.
    
    Rewards:
    1. Current adherence level (0-1)
    2. Improvement in adherence over previous state
    3. Bonus for sustained high adherence
    
    Clinical Rationale:
    - Medication adherence is critical for chronic disease management
    - Improving adherence leads to better health outcomes
    - Sustained adherence should be strongly rewarded
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize adherence reward.
        
        Args:
            config: Configuration with adherence-specific parameters
        """
        super().__init__(config)
        
        # Default parameters (can be overridden by config)
        self.improvement_multiplier = getattr(config, 'adherence_improvement_multiplier', 2.0)
        self.high_adherence_threshold = getattr(config, 'high_adherence_threshold', 0.8)
        self.high_adherence_bonus = getattr(config, 'high_adherence_bonus', 1.0)
        self.sustained_adherence_days = getattr(config, 'sustained_adherence_days', 30)
        self.sustained_adherence_bonus = getattr(config, 'sustained_adherence_bonus', 2.0)
        
    def compute_reward(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Compute adherence-based reward.
        
        Args:
            state: Previous patient state with 'adherence_score'
            action: Action taken (may include reminder actions)
            next_state: New state with updated 'adherence_score'
            
        Returns:
            Adherence reward (typically positive for good adherence)
        """
        # Get adherence scores
        current_adherence = next_state.get('adherence_score', 0.0)
        previous_adherence = state.get('adherence_score', 0.0)
        
        # Base reward: proportional to current adherence
        base_reward = current_adherence
        
        # Improvement bonus: reward improvements in adherence
        improvement = max(0.0, current_adherence - previous_adherence)
        improvement_bonus = improvement * self.improvement_multiplier
        
        # High adherence bonus: reward maintaining high adherence
        high_adherence_bonus = 0.0
        if current_adherence >= self.high_adherence_threshold:
            high_adherence_bonus = self.high_adherence_bonus
        
        # Sustained adherence bonus: extra reward for long-term high adherence
        sustained_bonus = 0.0
        adherence_streak = next_state.get('adherence_streak_days', 0)
        if (adherence_streak >= self.sustained_adherence_days and 
            current_adherence >= self.high_adherence_threshold):
            sustained_bonus = self.sustained_adherence_bonus
        
        # Total reward
        total_reward = (
            base_reward + 
            improvement_bonus + 
            high_adherence_bonus + 
            sustained_bonus
        )
        
        return total_reward
    
    def get_reward_components(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of adherence reward components.
        
        Returns:
            Dictionary with breakdown of reward components
        """
        current_adherence = next_state.get('adherence_score', 0.0)
        previous_adherence = state.get('adherence_score', 0.0)
        adherence_streak = next_state.get('adherence_streak_days', 0)
        
        # Compute components
        base = current_adherence
        improvement = max(0.0, current_adherence - previous_adherence)
        improvement_bonus = improvement * self.improvement_multiplier
        
        high_bonus = 0.0
        if current_adherence >= self.high_adherence_threshold:
            high_bonus = self.high_adherence_bonus
            
        sustained = 0.0
        if (adherence_streak >= self.sustained_adherence_days and 
            current_adherence >= self.high_adherence_threshold):
            sustained = self.sustained_adherence_bonus
        
        return {
            'adherence_base': base,
            'adherence_improvement': improvement_bonus,
            'adherence_high_bonus': high_bonus,
            'adherence_sustained_bonus': sustained,
            'adherence_total': base + improvement_bonus + high_bonus + sustained
        }
    
    def _compute_adherence_trend(self, state: Dict[str, Any]) -> float:
        """
        Compute adherence trend over recent history.
        
        Args:
            state: Patient state with adherence history
            
        Returns:
            Trend value (positive = improving, negative = declining)
        """
        adherence_history = state.get('adherence_history', [])
        
        if len(adherence_history) < 2:
            return 0.0
        
        # Simple linear trend using first and last values
        recent_window = adherence_history[-7:]  # Last 7 measurements
        if len(recent_window) < 2:
            return 0.0
            
        trend = recent_window[-1] - recent_window[0]
        return trend
