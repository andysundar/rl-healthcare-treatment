"""
Random policy baseline.

This baseline selects random actions within safe bounds, serving as a lower
bound for performance comparison. Any reasonable policy should outperform
random action selection.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging

from .base_baseline import BaselinePolicy, BaselineMetrics

logger = logging.getLogger(__name__)


class RandomPolicy(BaselinePolicy):
    """
    Random action selection policy.
    
    This policy serves as a lower bound baseline by selecting random actions
    from the valid action space. It respects safety constraints but otherwise
    makes no intelligent decisions.
    
    Attributes:
        seed: Random seed for reproducibility
        action_bounds: Tuple of (lower_bounds, upper_bounds)
        distribution: Distribution type ('uniform', 'gaussian')
    """
    
    def __init__(self,
                 name: str = "Random",
                 action_space: Dict[str, Any] = None,
                 state_dim: int = None,
                 seed: Optional[int] = None,
                 distribution: str = 'uniform',
                 device: str = 'cpu'):
        """
        Initialize random policy.
        
        Args:
            name: Policy name
            action_space: Action space specification
            state_dim: State dimension
            seed: Random seed for reproducibility
            distribution: 'uniform' or 'gaussian'
            device: Computation device
        """
        super().__init__(name, action_space, state_dim, device)
        
        self.seed = seed
        self.distribution = distribution
        self.action_bounds = self.get_action_bounds()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Set random seed to {seed}")
        
        # For Gaussian distribution
        self.gaussian_mean = None
        self.gaussian_std = None
        if distribution == 'gaussian':
            self._init_gaussian_params()
    
    def _init_gaussian_params(self):
        """Initialize parameters for Gaussian distribution."""
        low, high = self.action_bounds
        # Mean at center of action space
        self.gaussian_mean = (low + high) / 2.0
        # Std to cover ~95% of action space
        self.gaussian_std = (high - low) / 4.0
        logger.info(f"Gaussian params: mean={self.gaussian_mean}, std={self.gaussian_std}")
    
    def set_action_bounds(self, 
                         lower_bounds: np.ndarray, 
                         upper_bounds: np.ndarray):
        """
        Set custom action bounds.
        
        Args:
            lower_bounds: Lower bounds for actions
            upper_bounds: Upper bounds for actions
        """
        self.action_bounds = (lower_bounds, upper_bounds)
        
        # Update Gaussian params if using Gaussian distribution
        if self.distribution == 'gaussian':
            self._init_gaussian_params()
        
        logger.info(f"Updated action bounds: [{lower_bounds}, {upper_bounds}]")
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select random action.
        
        Args:
            state: Current state (not used, but kept for interface compatibility)
            deterministic: Not used for random policy
            
        Returns:
            Random action within bounds
        """
        low, high = self.action_bounds
        
        if self.distribution == 'uniform':
            # Uniform random sampling
            action = np.random.uniform(low, high)
        
        elif self.distribution == 'gaussian':
            # Gaussian sampling with clipping
            action = np.random.normal(self.gaussian_mean, self.gaussian_std)
            action = self.clip_action(action)
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        return action
    
    def evaluate(self, test_data: List[Tuple]) -> BaselineMetrics:
        """
        Evaluate random policy on test data.
        
        Args:
            test_data: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            BaselineMetrics with evaluation results
        """
        states = []
        actions = []
        rewards = []
        safety_violations = 0
        
        for state, _, reward, next_state, _ in test_data:
            # Get random action (ignoring actual state)
            action = self.select_action(state)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Check safety
            if self._check_safety_violation(next_state):
                safety_violations += 1
        
        return self.compute_metrics(states, actions, rewards, safety_violations)
    
    def _check_safety_violation(self, state: np.ndarray) -> bool:
        """
        Check if state represents a safety violation.
        
        Args:
            state: State to check
            
        Returns:
            True if safety violation detected
        """
        # Simple check - assumes first feature is glucose
        if len(state) > 0:
            glucose = state[0]
            return glucose < 50 or glucose > 300
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_info()
        info.update({
            'seed': self.seed,
            'distribution': self.distribution,
            'action_bounds': {
                'low': self.action_bounds[0].tolist(),
                'high': self.action_bounds[1].tolist()
            }
        })
        
        if self.distribution == 'gaussian':
            info['gaussian_params'] = {
                'mean': self.gaussian_mean.tolist() if self.gaussian_mean is not None else None,
                'std': self.gaussian_std.tolist() if self.gaussian_std is not None else None
            }
        
        return info
    
    def __str__(self) -> str:
        """String representation."""
        return f"Random Policy ({self.distribution} distribution)"


class SafeRandomPolicy(RandomPolicy):
    """
    Random policy with explicit safety checking.
    
    This variant samples multiple random actions and selects the safest one
    based on a safety scoring function. This provides a slightly stronger
    baseline than pure random selection.
    
    Attributes:
        num_samples: Number of actions to sample per decision
        safety_scorer: Function to score action safety
    """
    
    def __init__(self,
                 name: str = "Safe-Random",
                 action_space: Dict[str, Any] = None,
                 state_dim: int = None,
                 seed: Optional[int] = None,
                 num_samples: int = 10,
                 distribution: str = 'uniform',
                 device: str = 'cpu'):
        """
        Initialize safe random policy.
        
        Args:
            name: Policy name
            action_space: Action space specification
            state_dim: State dimension
            seed: Random seed
            num_samples: Number of actions to sample per decision
            distribution: Distribution type
            device: Computation device
        """
        super().__init__(name, action_space, state_dim, seed, distribution, device)
        self.num_samples = num_samples
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select safest action from multiple random samples.
        
        Args:
            state: Current state
            deterministic: Not used
            
        Returns:
            Safest random action
        """
        # Sample multiple actions
        candidate_actions = []
        for _ in range(self.num_samples):
            action = super().select_action(state, deterministic)
            candidate_actions.append(action)
        
        # Score each action for safety
        safety_scores = [self._score_action_safety(state, action) 
                        for action in candidate_actions]
        
        # Select safest action
        safest_idx = np.argmax(safety_scores)
        return candidate_actions[safest_idx]
    
    def _score_action_safety(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Score an action for safety.
        
        Args:
            state: Current state
            action: Candidate action
            
        Returns:
            Safety score (higher = safer)
        """
        # Simple safety heuristic - prefer moderate actions
        # This can be customized based on domain knowledge
        low, high = self.action_bounds
        center = (low + high) / 2.0
        
        # Distance from center (normalized)
        distance = np.abs(action - center) / (high - low)
        
        # Safety score (closer to center = safer)
        safety_score = 1.0 - np.mean(distance)
        
        return safety_score
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_info()
        info['num_samples'] = self.num_samples
        return info
    
    def __str__(self) -> str:
        """String representation."""
        return f"Safe Random Policy ({self.num_samples} samples)"


# Convenience functions
def create_random_policy(action_dim: int = 1,
                        state_dim: int = 10,
                        seed: Optional[int] = None,
                        distribution: str = 'uniform') -> RandomPolicy:
    """
    Create a random policy with standard configuration.
    
    Args:
        action_dim: Dimension of action space
        state_dim: Dimension of state space
        seed: Random seed for reproducibility
        distribution: Distribution type ('uniform' or 'gaussian')
        
    Returns:
        Configured RandomPolicy
    """
    action_space = {
        'type': 'continuous',
        'low': np.zeros(action_dim),
        'high': np.ones(action_dim),
        'dim': action_dim
    }
    
    return RandomPolicy(
        name=f"Random-{distribution}",
        action_space=action_space,
        state_dim=state_dim,
        seed=seed,
        distribution=distribution
    )


def create_safe_random_policy(action_dim: int = 1,
                              state_dim: int = 10,
                              seed: Optional[int] = None,
                              num_samples: int = 10) -> SafeRandomPolicy:
    """
    Create a safe random policy with standard configuration.
    
    Args:
        action_dim: Dimension of action space
        state_dim: Dimension of state space
        seed: Random seed
        num_samples: Number of actions to sample per decision
        
    Returns:
        Configured SafeRandomPolicy
    """
    action_space = {
        'type': 'continuous',
        'low': np.zeros(action_dim),
        'high': np.ones(action_dim),
        'dim': action_dim
    }
    
    return SafeRandomPolicy(
        name="Safe-Random",
        action_space=action_space,
        state_dim=state_dim,
        seed=seed,
        num_samples=num_samples
    )
