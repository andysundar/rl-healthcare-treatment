"""
Base Healthcare Environment
===========================
Abstract base class for all healthcare simulation environments.
Follows OpenAI Gymnasium interface for compatibility with RL libraries.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseHealthcareEnv(gym.Env, ABC):
    """
    Abstract base class for healthcare simulation environments.
    
    All healthcare environments should inherit from this class and implement:
    - _get_observation_space()
    - _get_action_space()
    - _reset_state()
    - _step_dynamics()
    - _compute_reward()
    - _check_termination()
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Any, render_mode: Optional[str] = None):
        """
        Initialize base healthcare environment.
        
        Args:
            config: Environment configuration object
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        # Define observation and action spaces
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = getattr(config, 'max_steps', 100)
        self.episode_history = []
        self.safety_violations = []
        
        # Random number generator
        self._np_random = None
        
        # Current state
        self.state = None
        
    @abstractmethod
    def _get_observation_space(self) -> spaces.Space:
        """Define the observation space for the environment."""
        pass
    
    @abstractmethod
    def _get_action_space(self) -> spaces.Space:
        """Define the action space for the environment."""
        pass
    
    @abstractmethod
    def _reset_state(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            initial_state: Initial observation
        """
        pass
    
    @abstractmethod
    def _step_dynamics(self, action: np.ndarray) -> np.ndarray:
        """
        Update environment dynamics based on action.
        
        Args:
            action: Action taken by agent
            
        Returns:
            next_state: Next observation
        """
        pass
    
    @abstractmethod
    def _compute_reward(self, state: np.ndarray, action: np.ndarray, 
                       next_state: np.ndarray) -> float:
        """
        Compute reward for transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            reward: Scalar reward value
        """
        pass
    
    @abstractmethod
    def _check_termination(self, state: np.ndarray) -> Tuple[bool, bool]:
        """
        Check if episode should terminate.
        
        Args:
            state: Current state
            
        Returns:
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
        """
        pass
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset tracking variables
        self.current_step = 0
        self.episode_history = []
        self.safety_violations = []
        
        # Reset state
        self.state = self._reset_state(seed=seed)
        
        # Prepare info dict
        info = self._get_info()
        
        return self.state.copy(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward for this transition
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information dictionary
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} not in action space {self.action_space}")
        
        # Store current state for reward computation
        prev_state = self.state.copy()
        
        # Update state
        self.state = self._step_dynamics(action)
        
        # Compute reward
        reward = self._compute_reward(prev_state, action, self.state)
        
        # Check termination
        terminated, truncated = self._check_termination(self.state)
        
        # Check if max steps reached
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Track safety violations
        if self._is_unsafe_state(self.state):
            self.safety_violations.append({
                'step': self.current_step,
                'state': self.state.copy(),
                'action': action.copy()
            })
        
        # Update history
        self.episode_history.append({
            'step': self.current_step,
            'state': prev_state.copy(),
            'action': action.copy(),
            'next_state': self.state.copy(),
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated
        })
        
        # Prepare info dict
        info = self._get_info()
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def _is_unsafe_state(self, state: np.ndarray) -> bool:
        """
        Check if current state is unsafe.
        Override in subclass to define safety constraints.
        
        Args:
            state: Current state
            
        Returns:
            is_unsafe: Whether state violates safety constraints
        """
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information for current state.
        
        Returns:
            info: Dictionary with additional information
        """
        return {
            'step': self.current_step,
            'safety_violations': len(self.safety_violations),
            'episode_length': len(self.episode_history)
        }
    
    def render(self):
        """
        Render the environment.
        Override in subclass for custom visualization.
        """
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, State: {self.state}")
        elif self.render_mode == 'rgb_array':
            # Return RGB array representation
            # Subclasses should implement custom rendering
            pass
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics for completed episode.
        
        Returns:
            metrics: Dictionary of episode metrics
        """
        if not self.episode_history:
            return {}
        
        rewards = [step['reward'] for step in self.episode_history]
        
        return {
            'total_reward': sum(rewards),
            'mean_reward': np.mean(rewards),
            'episode_length': len(self.episode_history),
            'safety_violations': len(self.safety_violations),
            'safety_violation_rate': len(self.safety_violations) / len(self.episode_history)
        }
