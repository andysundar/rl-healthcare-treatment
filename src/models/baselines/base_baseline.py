"""
Base abstract class for baseline policies.

This module provides the interface that all baseline policies must implement,
ensuring compatibility with the evaluation framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class BaselineMetrics:
    """Metrics computed for baseline evaluation."""
    mean_reward: float
    safety_violations: int
    total_steps: int
    safety_rate: float
    mean_action_value: float
    std_action_value: float


class BaselinePolicy(ABC):
    """
    Abstract base class for all baseline policies.
    
    All baseline implementations should inherit from this class and implement
    the required methods to ensure compatibility with the evaluation framework.
    
    Attributes:
        name: Human-readable name for the baseline
        action_space: Action space specification
        state_dim: Dimension of state space
        device: Torch device for computation
    """
    
    def __init__(self, name: str, action_space: Dict[str, Any], 
                 state_dim: int, device: str = 'cpu'):
        """
        Initialize baseline policy.
        
        Args:
            name: Name of the baseline policy
            action_space: Dictionary specifying action space
            state_dim: Dimension of state space
            device: Device for computation ('cpu' or 'cuda')
        """
        self.name = name
        self.action_space = action_space
        self.state_dim = state_dim
        self.device = device
        
        # Track evaluation statistics
        self.evaluation_history = []
        
    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state (numpy array or dict)
            deterministic: Whether to use deterministic policy
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: List[Tuple]) -> BaselineMetrics:
        """
        Evaluate policy on test data.
        
        Args:
            test_data: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            BaselineMetrics object with evaluation results
        """
        pass
    
    def reset(self):
        """Reset any internal state (for stateful policies)."""
        pass
    
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action space bounds.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if 'low' in self.action_space and 'high' in self.action_space:
            return self.action_space['low'], self.action_space['high']
        elif 'range' in self.action_space:
            return self.action_space['range'][0], self.action_space['range'][1]
        else:
            # Default bounds for continuous actions
            return np.array([0.0]), np.array([1.0])
    
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clip action to valid bounds.
        
        Args:
            action: Raw action
            
        Returns:
            Clipped action within valid bounds
        """
        low, high = self.get_action_bounds()
        return np.clip(action, low, high)
    
    def compute_metrics(self, states: List[np.ndarray], 
                       actions: List[np.ndarray],
                       rewards: List[float],
                       safety_violations: int) -> BaselineMetrics:
        """
        Compute evaluation metrics.
        
        Args:
            states: List of states
            actions: List of actions taken
            rewards: List of rewards received
            safety_violations: Number of safety violations
            
        Returns:
            BaselineMetrics object
        """
        total_steps = len(rewards)
        mean_reward = np.mean(rewards) if rewards else 0.0
        safety_rate = 1.0 - (safety_violations / total_steps) if total_steps > 0 else 1.0
        
        # Action statistics
        actions_array = np.array(actions)
        mean_action = float(np.mean(actions_array))
        std_action = float(np.std(actions_array))
        
        metrics = BaselineMetrics(
            mean_reward=mean_reward,
            safety_violations=safety_violations,
            total_steps=total_steps,
            safety_rate=safety_rate,
            mean_action_value=mean_action,
            std_action_value=std_action
        )
        
        self.evaluation_history.append(metrics)
        return metrics
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get policy information and parameters.
        
        Returns:
            Dictionary with policy information
        """
        return {
            'name': self.name,
            'state_dim': self.state_dim,
            'action_space': self.action_space,
            'device': self.device
        }
    
    def save(self, path: str):
        """
        Save policy to disk.
        
        Args:
            path: Path to save policy
        """
        raise NotImplementedError("Save not implemented for this baseline")
    
    def load(self, path: str):
        """
        Load policy from disk.
        
        Args:
            path: Path to load policy from
        """
        raise NotImplementedError("Load not implemented for this baseline")
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} Baseline Policy"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(name='{self.name}', state_dim={self.state_dim})"
