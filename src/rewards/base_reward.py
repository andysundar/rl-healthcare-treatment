"""
Base Reward Function Interface
All reward functions inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseRewardFunction(ABC):
    """
    Abstract base class for all reward functions.
    
    All reward implementations must inherit from this class and implement
    the compute_reward method. This ensures consistent interface across
    different reward components.
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize reward function.
        
        Args:
            config: Configuration object with reward-specific parameters
        """
        self.config = config
        self._weights = {}
        
    @abstractmethod
    def compute_reward(
        self, 
        state: Dict[str, Any], 
        action: Dict[str, Any], 
        next_state: Dict[str, Any]
    ) -> float:
        """
        Compute reward for a state-action-next_state transition.
        
        Args:
            state: Current patient state (vitals, labs, adherence, etc.)
            action: Action taken (medication dosage, scheduling, etc.)
            next_state: Resulting patient state after action
            
        Returns:
            Scalar reward value
        """
        pass
    
    def get_reward_components(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get breakdown of reward components for interpretability.
        
        Default implementation returns single total reward.
        Override in subclasses to provide detailed breakdown.
        
        Args:
            state: Current patient state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Dictionary mapping component names to their reward values
        """
        total_reward = self.compute_reward(state, action, next_state)
        return {"total": total_reward}
    
    def set_weights(self, weight_dict: Dict[str, float]) -> None:
        """
        Set weights for reward components.
        
        Args:
            weight_dict: Dictionary mapping component names to weights
        """
        self._weights.update(weight_dict)
        
    def get_weights(self) -> Dict[str, float]:
        """
        Get current reward component weights.
        
        Returns:
            Dictionary of component weights
        """
        return self._weights.copy()
    
    def normalize_value(
        self, 
        value: float, 
        min_val: float, 
        max_val: float
    ) -> float:
        """
        Normalize value to [0, 1] range.
        
        Args:
            value: Value to normalize
            min_val: Minimum value in range
            max_val: Maximum value in range
            
        Returns:
            Normalized value in [0, 1]
        """
        if max_val == min_val:
            return 0.5
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def __call__(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Make reward function callable.
        
        Args:
            state: Current patient state
            action: Action taken  
            next_state: Resulting state
            
        Returns:
            Reward value
        """
        return self.compute_reward(state, action, next_state)
