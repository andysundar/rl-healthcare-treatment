"""
Neural Network Architectures for Offline RL

This module implements the core neural network architectures used in offline
reinforcement learning algorithms:
- Q-Networks: Q(s, a) → scalar value
- Policy Networks: π(a|s) → action distribution
- Value Networks: V(s) → scalar value

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class QNetwork(nn.Module):
    """
    Q-Network for estimating action-value function Q(s, a).
    
    Architecture: [state ⊕ action] → FC(256) → FC(256) → FC(1)
    
    The Q-network takes concatenated state and action as input and outputs
    a scalar Q-value representing the expected return.
    
    Mathematical formulation:
        Q(s, a) = E[Σ_{t=0}^∞ γ^t r_t | s_0=s, a_0=a, π]
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2
    ):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of units in hidden layers (default: 256)
            n_hidden_layers: Number of hidden layers (default: 2)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Input layer
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_hidden_layers - 1)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, 1)
        
        # Initialize weights using Xavier uniform
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            action: Batch of actions [batch_size, action_dim]
        
        Returns:
            Q-values [batch_size, 1]
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Input layer with ReLU
        x = F.relu(self.fc1(x))
        
        # Hidden layers with ReLU
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output layer (no activation)
        q_value = self.fc_out(x)
        
        return q_value


class PolicyNetwork(nn.Module):
    """
    Policy Network for learning action distribution π(a|s).
    
    For continuous actions: Outputs mean μ(s) and log_std log(σ(s))
        Action sampling: a ~ N(μ(s), σ(s))
    
    For discrete actions: Outputs logits for categorical distribution
        Action sampling: a ~ Categorical(softmax(logits(s)))
    
    Architecture: state → FC(256) → FC(256) → FC(action_dim * 2) or FC(action_dim)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_space: str = 'continuous',
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """
        Initialize Policy Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of units in hidden layers
            action_space: 'continuous' or 'discrete'
            log_std_min: Minimum log standard deviation (for continuous)
            log_std_max: Maximum log standard deviation (for continuous)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if action_space == 'continuous':
            # Output mean and log_std for Gaussian policy
            self.mean_layer = nn.Linear(hidden_dim, action_dim)
            self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        elif action_space == 'discrete':
            # Output logits for categorical distribution
            self.logits_layer = nn.Linear(hidden_dim, action_dim)
        else:
            raise ValueError(f"Unknown action_space: {action_space}")
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through policy network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            deterministic: If True, return mean action (continuous) or 
                          argmax action (discrete)
        
        Returns:
            For continuous: (action, log_prob)
            For discrete: (action, log_prob)
        """
        # Shared layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.action_space == 'continuous':
            return self._continuous_action(x, deterministic)
        else:
            return self._discrete_action(x, deterministic)
    
    def _continuous_action(
        self, 
        features: torch.Tensor, 
        deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate continuous action from Gaussian distribution.
        
        Reparameterization trick: a = μ + σ * ε, where ε ~ N(0, I)
        Log probability: log π(a|s) = -0.5 * [(a-μ)/σ]^2 - log(σ) - 0.5*log(2π)
        """
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        if deterministic:
            # Return mean action
            action = torch.tanh(mean)  # Squash to [-1, 1]
            return action, None
        
        # Sample action using reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)  # Squash to [-1, 1]
        
        # Compute log probability
        # log π(a|s) = log π(z|s) - log(1 - tanh^2(z))
        log_prob = normal.log_prob(z)
        
        # Correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def _discrete_action(
        self, 
        features: torch.Tensor, 
        deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate discrete action from categorical distribution.
        
        Categorical distribution: π(a|s) = softmax(logits(s))
        """
        logits = self.logits_layer(features)
        
        if deterministic:
            # Return argmax action
            action = torch.argmax(logits, dim=-1, keepdim=True)
            return action, None
        
        # Sample action from categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.unsqueeze(-1), log_prob.unsqueeze(-1)
    
    def get_action_log_prob(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of given action under current policy.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            action: Batch of actions [batch_size, action_dim]
        
        Returns:
            Log probabilities [batch_size, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.action_space == 'continuous':
            mean = self.mean_layer(x)
            log_std = self.log_std_layer(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            
            # Compute log probability for given action
            normal = torch.distributions.Normal(mean, std)
            
            # Inverse tanh to get z
            z = torch.atanh(torch.clamp(action, -0.999, 0.999))
            log_prob = normal.log_prob(z)
            
            # Correction for tanh
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            return log_prob
        else:
            logits = self.logits_layer(x)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action.squeeze(-1))
            return log_prob.unsqueeze(-1)


class ValueNetwork(nn.Module):
    """
    Value Network for estimating state-value function V(s).
    
    Architecture: state → FC(256) → FC(256) → FC(1)
    
    Mathematical formulation:
        V(s) = E[Σ_{t=0}^∞ γ^t r_t | s_0=s, π]
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256
    ):
        """
        Initialize Value Network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of units in hidden layers
        """
        super().__init__()
        
        self.state_dim = state_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
        
        Returns:
            State values [batch_size, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc_out(x)
        
        return value


class EnsembleQNetwork(nn.Module):
    """
    Ensemble of Q-Networks for uncertainty estimation.
    
    Uses multiple Q-networks and takes minimum Q-value to encourage
    conservatism (useful for offline RL).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_ensemble: int = 2
    ):
        """
        Initialize Ensemble Q-Network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of units in hidden layers
            n_ensemble: Number of Q-networks in ensemble
        """
        super().__init__()
        
        self.q_networks = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dim)
            for _ in range(n_ensemble)
        ])
        self.n_ensemble = n_ensemble
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            state: Batch of states
            action: Batch of actions
            return_all: If True, return all Q-values; else return minimum
        
        Returns:
            Q-values [batch_size, 1] or [batch_size, n_ensemble]
        """
        q_values = torch.stack([
            q_net(state, action) for q_net in self.q_networks
        ], dim=-1)  # [batch_size, 1, n_ensemble]
        
        if return_all:
            return q_values.squeeze(1)  # [batch_size, n_ensemble]
        else:
            # Return minimum Q-value (conservative estimate)
            return q_values.min(dim=-1, keepdim=True)[0]  # [batch_size, 1]


# Utility functions for network operations

def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """
    Soft update target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        source: Source network (current)
        target: Target network (slow-moving average)
        tau: Soft update coefficient (typically 0.001 - 0.01)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(source: nn.Module, target: nn.Module) -> None:
    """
    Hard update target network parameters.
    
    θ_target = θ_source
    
    Args:
        source: Source network
        target: Target network
    """
    target.load_state_dict(source.state_dict())
