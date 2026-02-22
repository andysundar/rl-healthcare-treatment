"""
Conservative Q-Learning (CQL) for Offline Reinforcement Learning

This module implements Conservative Q-Learning (Kumar et al., 2020), which
addresses the distributional shift problem in offline RL by learning conservative
Q-functions that lower-bound the true Q-values for out-of-distribution actions.

Key idea: Minimize Q-values for unseen actions while maximizing for actions
in the dataset, preventing overestimation for novel actions.

Reference:
Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020).
"Conservative Q-Learning for Offline Reinforcement Learning"
NeurIPS 2020

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import numpy as np
from copy import deepcopy
import logging

from .base_agent import BaseRLAgent
from .networks import QNetwork, PolicyNetwork, soft_update

logger = logging.getLogger(__name__)


class CQLAgent(BaseRLAgent):
    """
    Conservative Q-Learning (CQL) Agent for Offline RL.
    
    CQL addresses the problem of overestimation in offline RL by learning
    conservative Q-functions. The key innovation is the CQL regularization term:
    
    L_CQL(Q) = α * (E_s~D [log Σ_a exp(Q(s,a))] - E_(s,a)~D [Q(s,a)])
    
    This term:
    1. Increases Q-values for actions in the dataset: E_(s,a)~D [Q(s,a)]
    2. Decreases Q-values for all possible actions: E_s~D [log Σ_a exp(Q(s,a))]
    
    The net effect is conservative Q-values that avoid overestimation on
    out-of-distribution actions.
    
    Algorithm Components:
    - Double Q-learning with two Q-networks (Q1, Q2)
    - Target Q-networks for stable learning
    - Gaussian policy network for continuous actions
    - CQL regularization to prevent overestimation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        q_lr: float = 3e-4,
        policy_lr: float = 1e-4,
        cql_alpha: float = 1.0,
        target_update_freq: int = 2,
        num_random_actions: int = 10,
        device: str = 'cpu',
        action_space: str = 'continuous'
    ):
        """
        Initialize CQL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension for networks
            gamma: Discount factor (0 < γ < 1)
            tau: Soft update coefficient for target networks
            q_lr: Learning rate for Q-networks
            policy_lr: Learning rate for policy network
            cql_alpha: Weight of CQL regularization term (typically 0.1 - 5.0)
            target_update_freq: Frequency of target network updates
            num_random_actions: Number of random actions for CQL penalty
            device: 'cpu' or 'cuda'
            action_space: 'continuous' or 'discrete'
        """
        super().__init__(state_dim, action_dim, gamma, device)
        
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.target_update_freq = target_update_freq
        self.num_random_actions = num_random_actions
        self.action_space = action_space
        
        # Initialize Q-networks (double Q-learning)
        self.q_network1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_network2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize target Q-networks
        self.target_q1 = deepcopy(self.q_network1)
        self.target_q2 = deepcopy(self.q_network2)
        
        # Initialize policy network
        self.policy = PolicyNetwork(
            state_dim, 
            action_dim, 
            hidden_dim,
            action_space=action_space
        ).to(self.device)
        
        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=q_lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=policy_lr
        )
        
        logger.info(f"Initialized CQLAgent:")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  CQL alpha: {cql_alpha}")
        logger.info(f"  Q learning rate: {q_lr}")
        logger.info(f"  Policy learning rate: {policy_lr}")
        logger.info(f"  Tau: {tau}")
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: Current state [state_dim]
            deterministic: If True, return mean action (no exploration)
        
        Returns:
            action: Selected action [action_dim]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.policy(state_tensor, deterministic=deterministic)
            action = action.cpu().numpy()[0]
        
        return action
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step using CQL.
        
        Training consists of:
        1. Update Q-networks with CQL loss
        2. Update policy to maximize Q-values
        3. Soft update target networks
        
        Args:
            batch: Dictionary containing batched transitions
        
        Returns:
            Dictionary of training metrics
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # ============= Update Q-networks =============
        q1_loss, q2_loss, cql_penalty = self._compute_cql_loss(
            states, actions, rewards, next_states, dones
        )
        
        total_q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            max_norm=1.0
        )
        self.q_optimizer.step()
        
        # ============= Update Policy =============
        policy_loss = self._compute_policy_loss(states)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            max_norm=1.0
        )
        self.policy_optimizer.step()
        
        # ============= Update Target Networks =============
        if self.training_step % self.target_update_freq == 0:
            self.update_target_networks()
        
        self.training_step += 1
        self.total_updates += 1
        
        # Return metrics
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql_penalty': cql_penalty.item(),
            'policy_loss': policy_loss.item(),
            'total_updates': self.total_updates
        }
    
    def _compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> tuple:
        """
        Compute CQL loss for Q-networks.
        
        CQL Loss = Bellman Error + α * CQL Penalty
        
        Where:
        Bellman Error = E[(Q(s,a) - (r + γ * min_i Q_target_i(s', a')))^2]
        CQL Penalty = E[log Σ_a exp(Q(s,a))] - E[Q(s,a_data)]
        
        The CQL penalty encourages:
        - Lower Q-values for unseen actions (first term)
        - Higher Q-values for actions in dataset (second term)
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions from dataset [batch_size, action_dim]
            rewards: Batch of rewards [batch_size, 1]
            next_states: Batch of next states [batch_size, state_dim]
            dones: Batch of done flags [batch_size, 1]
        
        Returns:
            q1_loss: Loss for Q-network 1
            q2_loss: Loss for Q-network 2
            cql_penalty: CQL regularization penalty
        """
        batch_size = states.shape[0]
        
        # ============= Compute Target Q-values =============
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs = self.policy(next_states)
            
            # Compute target Q-values using double Q-learning
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Bellman backup: r + γ * (1 - done) * target_Q
            target_q_value = rewards + (1 - dones) * self.gamma * target_q
        
        # ============= Compute Current Q-values =============
        current_q1 = self.q_network1(states, actions)
        current_q2 = self.q_network2(states, actions)
        
        # ============= Bellman Error =============
        bellman_error1 = F.mse_loss(current_q1, target_q_value)
        bellman_error2 = F.mse_loss(current_q2, target_q_value)
        
        # ============= CQL Penalty =============
        # Sample random actions for computing logsumexp term
        random_actions = self._sample_random_actions(batch_size)
        
        # Compute Q-values for random actions
        random_q1 = self.q_network1(
            states.unsqueeze(1).repeat(1, self.num_random_actions, 1).view(-1, self.state_dim),
            random_actions.view(-1, self.action_dim)
        ).view(batch_size, self.num_random_actions, 1)
        
        random_q2 = self.q_network2(
            states.unsqueeze(1).repeat(1, self.num_random_actions, 1).view(-1, self.state_dim),
            random_actions.view(-1, self.action_dim)
        ).view(batch_size, self.num_random_actions, 1)
        
        # Compute Q-values for policy actions
        policy_actions, _ = self.policy(states)
        policy_q1 = self.q_network1(states, policy_actions)
        policy_q2 = self.q_network2(states, policy_actions)
        
        # CQL penalty: E[log Σ exp(Q(s,a))] - E[Q(s,a_data)]
        # We use logsumexp for numerical stability
        
        # Combine random and policy actions
        cat_q1 = torch.cat([random_q1, policy_q1.unsqueeze(1)], dim=1)
        cat_q2 = torch.cat([random_q2, policy_q2.unsqueeze(1)], dim=1)
        
        # Compute logsumexp
        cql_logsumexp_q1 = torch.logsumexp(cat_q1, dim=1).mean()
        cql_logsumexp_q2 = torch.logsumexp(cat_q2, dim=1).mean()
        
        # Q-values for data actions
        cql_data_q1 = current_q1.mean()
        cql_data_q2 = current_q2.mean()
        
        # CQL penalty
        cql_penalty = (
            (cql_logsumexp_q1 - cql_data_q1) + 
            (cql_logsumexp_q2 - cql_data_q2)
        )
        
        # ============= Total Loss =============
        q1_loss = bellman_error1 + self.cql_alpha * (cql_logsumexp_q1 - cql_data_q1)
        q2_loss = bellman_error2 + self.cql_alpha * (cql_logsumexp_q2 - cql_data_q2)
        
        return q1_loss, q2_loss, cql_penalty
    
    def _sample_random_actions(self, batch_size: int) -> torch.Tensor:
        """
        Sample random actions for CQL penalty computation.
        
        Args:
            batch_size: Number of states
        
        Returns:
            Random actions [batch_size, num_random_actions, action_dim]
        """
        if self.action_space == 'continuous':
            # Sample from uniform distribution [-1, 1]
            random_actions = torch.FloatTensor(
                batch_size, self.num_random_actions, self.action_dim
            ).uniform_(-1, 1).to(self.device)
        else:
            # Sample discrete actions
            random_actions = torch.randint(
                0, self.action_dim,
                (batch_size, self.num_random_actions, 1)
            ).float().to(self.device)
        
        return random_actions
    
    def _compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute policy loss.
        
        Policy is trained to maximize the minimum Q-value:
        L_π = -E_s [min_i Q_i(s, π(s))]
        
        Args:
            states: Batch of states [batch_size, state_dim]
        
        Returns:
            Policy loss (scalar)
        """
        # Sample actions from current policy
        actions, log_probs = self.policy(states)
        
        # Compute Q-values
        q1 = self.q_network1(states, actions)
        q2 = self.q_network2(states, actions)
        min_q = torch.min(q1, q2)
        
        # Policy loss: maximize Q-value (minimize negative Q-value)
        policy_loss = -min_q.mean()
        
        return policy_loss
    
    def update_target_networks(self) -> None:
        """
        Soft update of target networks.
        
        θ_target = τ * θ + (1 - τ) * θ_target
        """
        soft_update(self.q_network1, self.target_q1, self.tau)
        soft_update(self.q_network2, self.target_q2, self.tau)
    
    def get_q_value(
        self, 
        state: np.ndarray, 
        action: np.ndarray
    ) -> float:
        """
        Get Q-value for state-action pair.
        
        Uses minimum of two Q-networks for conservative estimate.
        
        Args:
            state: State [state_dim]
            action: Action [action_dim]
        
        Returns:
            Q-value (scalar)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            q1 = self.q_network1(state_tensor, action_tensor)
            q2 = self.q_network2(state_tensor, action_tensor)
            q_value = torch.min(q1, q2).item()
        
        return q_value
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get state dict of all networks."""
        return {
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'policy': self.policy.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict()
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Load state dict into all networks."""
        self.q_network1.load_state_dict(state['q_network1'])
        self.q_network2.load_state_dict(state['q_network2'])
        self.target_q1.load_state_dict(state['target_q1'])
        self.target_q2.load_state_dict(state['target_q2'])
        self.policy.load_state_dict(state['policy'])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
        self.policy_optimizer.load_state_dict(state['policy_optimizer'])
    
    def train_mode(self) -> None:
        """Set all networks to training mode."""
        self.q_network1.train()
        self.q_network2.train()
        self.policy.train()
    
    def eval_mode(self) -> None:
        """Set all networks to evaluation mode."""
        self.q_network1.eval()
        self.q_network2.eval()
        self.policy.eval()
