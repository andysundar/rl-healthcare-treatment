"""
Base Agent for Offline Reinforcement Learning

This module defines the abstract base class for all RL agents in the healthcare
treatment recommendation system. It provides a common interface for training,
action selection, and model management.


Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRLAgent(ABC):
    """
    Abstract base class for all RL agents.
    
    This class defines the common interface that all RL algorithms must implement,
    including action selection, training, and model persistence.
    
    Attributes:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        gamma (float): Discount factor for future rewards (0 < γ < 1)
        device (torch.device): Device for computation (CPU/GPU)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state representation
            action_dim: Dimension of action space
            gamma: Discount factor (default: 0.99)
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        _req = torch.device(device)
        if _req.type == "cuda" and not torch.cuda.is_available():
            _req = torch.device("cpu")
        elif _req.type == "mps" and not torch.backends.mps.is_available():
            _req = torch.device("cpu")
        self.device = _req
        
        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")
        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}, Gamma: {gamma}")
        
        # Training metrics
        self.training_step = 0
        self.total_updates = 0
    
    @abstractmethod
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action given current state.
        
        Args:
            state: Current patient state (shape: [state_dim])
            deterministic: If True, select action deterministically (no exploration)
                          If False, sample from policy distribution
        
        Returns:
            action: Selected action (shape: [action_dim])
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step on a batch of data.
        
        Args:
            batch: Dictionary containing:
                - 'states': torch.Tensor of shape [batch_size, state_dim]
                - 'actions': torch.Tensor of shape [batch_size, action_dim]
                - 'rewards': torch.Tensor of shape [batch_size, 1]
                - 'next_states': torch.Tensor of shape [batch_size, state_dim]
                - 'dones': torch.Tensor of shape [batch_size, 1]
        
        Returns:
            Dictionary of training metrics (losses, Q-values, etc.)
        """
        pass
    
    @abstractmethod
    def update_target_networks(self) -> None:
        """
        Update target networks using soft or hard update.
        
        Soft update: θ_target = τ * θ + (1 - τ) * θ_target
        Hard update: θ_target = θ
        """
        pass
    
    @abstractmethod
    def get_q_value(
        self, 
        state: np.ndarray, 
        action: np.ndarray
    ) -> float:
        """
        Get Q-value for state-action pair.
        
        Args:
            state: Patient state (shape: [state_dim])
            action: Treatment action (shape: [action_dim])
        
        Returns:
            Q-value: Expected return Q(s, a)
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save agent state to disk.
        
        Args:
            path: Path to save checkpoint
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'training_step': self.training_step,
            'total_updates': self.total_updates,
            'model_state': self._get_model_state()
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    def load(self, path: str) -> None:
        """
        Load agent state from disk.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Verify dimensions match
        assert checkpoint['state_dim'] == self.state_dim, "State dimension mismatch"
        assert checkpoint['action_dim'] == self.action_dim, "Action dimension mismatch"
        
        self.gamma = checkpoint['gamma']
        self.training_step = checkpoint['training_step']
        self.total_updates = checkpoint['total_updates']
        
        self._set_model_state(checkpoint['model_state'])
        logger.info(f"Loaded checkpoint from {path}")
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get state dict of all networks for saving."""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Load state dict into all networks."""
        pass
    
    def to(self, device: str) -> 'BaseRLAgent':
        """
        Move agent to specified device.
        
        Args:
            device: 'cpu' or 'cuda'
        
        Returns:
            self for method chaining
        """
        _req = torch.device(device)
        if _req.type == 'cuda' and not torch.cuda.is_available():
            _req = torch.device('cpu')
        elif _req.type == 'mps' and not torch.backends.mps.is_available():
            _req = torch.device('cpu')
        self.device = _req
        return self
    
    def train_mode(self) -> None:
        """Set agent to training mode."""
        pass
    
    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get agent configuration.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {
            'agent_type': self.__class__.__name__,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'device': str(self.device)
        }
