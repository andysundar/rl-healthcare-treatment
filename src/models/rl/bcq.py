"""
Batch-Constrained Q-Learning (BCQ) for Offline Reinforcement Learning

This module implements Batch-Constrained Q-Learning (Fujimoto et al., 2019),
which addresses distributional shift in offline RL by constraining the learned
policy to select actions similar to those in the batch dataset.

Key idea: Use a VAE to model the behavior policy, then perturb actions from
the VAE to maximize Q-values while staying close to the data distribution.

Reference:
Fujimoto, S., Meger, D., & Precup, D. (2019).
"Off-Policy Deep Reinforcement Learning without Exploration"
ICML 2019

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np
from copy import deepcopy
import logging

from .base_agent import BaseRLAgent
from .networks import QNetwork, soft_update

logger = logging.getLogger(__name__)


class VAE(nn.Module):
    """
    Variational Autoencoder for modeling behavior policy.
    
    The VAE learns to reconstruct actions from states, effectively modeling
    the data distribution p(a|s). This is used to constrain the learned policy.
    
    Architecture:
    - Encoder: (s, a) → (μ, log_σ) [latent distribution parameters]
    - Decoder: (s, z) → a [reconstructed action]
    
    Loss: Reconstruction loss + KL divergence
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256
    ):
        """
        Initialize VAE.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            latent_dim: Dimension of latent space
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Encoder: (s, a) → (μ, log_σ)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_std_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: (s, z) → a
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Constrain actions to [-1, 1]
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def encode(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state-action pair to latent distribution.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            action: Batch of actions [batch_size, action_dim]
        
        Returns:
            mean: Latent mean [batch_size, latent_dim]
            log_std: Latent log std [batch_size, latent_dim]
        """
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        return mean, log_std
    
    def decode(
        self, 
        state: torch.Tensor, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent code and state to action.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            z: Batch of latent codes [batch_size, latent_dim]
        
        Returns:
            Reconstructed actions [batch_size, action_dim]
        """
        x = torch.cat([state, z], dim=-1)
        action = self.decoder(x)
        return action
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode and decode.
        
        Args:
            state: Batch of states
            action: Batch of actions
        
        Returns:
            reconstructed_action: Decoded action
            mean: Latent mean
            log_std: Latent log std
        """
        mean, log_std = self.encode(state, action)
        
        # Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mean + std * eps
        
        reconstructed_action = self.decode(state, z)
        
        return reconstructed_action, mean, log_std
    
    def sample_action(
        self, 
        state: torch.Tensor, 
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample actions from VAE for given states.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            n_samples: Number of action samples per state
        
        Returns:
            Sampled actions [batch_size, n_samples, action_dim]
        """
        batch_size = state.shape[0]
        
        # Sample latent codes from standard normal
        z = torch.randn(batch_size, n_samples, self.latent_dim).to(state.device)
        
        # Decode to actions
        state_expanded = state.unsqueeze(1).repeat(1, n_samples, 1)
        actions = self.decode(
            state_expanded.view(-1, self.state_dim),
            z.view(-1, self.latent_dim)
        )
        actions = actions.view(batch_size, n_samples, self.action_dim)
        
        return actions


class PerturbationNetwork(nn.Module):
    """
    Perturbation network for BCQ.
    
    Takes actions from VAE and applies small perturbations to maximize Q-value
    while staying close to the data distribution.
    
    ξ(s, a) → Δa (perturbation)
    Final action: a_final = a_sampled + φ * ξ(s, a_sampled)
    where φ is a scaling factor (typically 0.05)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        phi: float = 0.05
    ):
        """
        Initialize perturbation network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            phi: Perturbation scaling factor
        """
        super().__init__()
        
        self.phi = phi
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Perturbation in [-1, 1]
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perturbed action.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            action: Batch of actions from VAE [batch_size, action_dim]
        
        Returns:
            Perturbed actions [batch_size, action_dim]
        """
        x = torch.cat([state, action], dim=-1)
        perturbation = self.network(x)
        
        # Apply perturbation with scaling
        perturbed_action = action + self.phi * perturbation
        
        # Clip to valid range
        perturbed_action = torch.clamp(perturbed_action, -1.0, 1.0)
        
        return perturbed_action


class BCQAgent(BaseRLAgent):
    """
    Batch-Constrained Q-Learning (BCQ) Agent.
    
    BCQ learns policies that stay close to the behavior policy by:
    1. Training a VAE to model the behavior policy
    2. Sampling actions from the VAE
    3. Perturbing sampled actions to maximize Q-value
    4. Selecting action with highest Q-value
    
    This approach ensures the policy doesn't deviate too far from the data
    distribution, preventing extrapolation errors in offline RL.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        q_lr: float = 3e-4,
        vae_lr: float = 3e-4,
        perturbation_lr: float = 3e-4,
        n_action_samples: int = 10,
        phi: float = 0.05,
        device: str = 'cpu'
    ):
        """
        Initialize BCQ agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            latent_dim: Latent dimension for VAE
            gamma: Discount factor
            tau: Soft update coefficient
            q_lr: Q-network learning rate
            vae_lr: VAE learning rate
            perturbation_lr: Perturbation network learning rate
            n_action_samples: Number of actions to sample from VAE
            phi: Perturbation scaling factor
            device: 'cpu' or 'cuda'
        """
        super().__init__(state_dim, action_dim, gamma, device)
        
        self.tau = tau
        self.n_action_samples = n_action_samples
        
        # Initialize VAE
        self.vae = VAE(state_dim, action_dim, latent_dim, hidden_dim).to(self.device)
        
        # Initialize Q-networks
        self.q_network1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_network2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize target Q-networks
        self.target_q1 = deepcopy(self.q_network1)
        self.target_q2 = deepcopy(self.q_network2)
        
        # Initialize perturbation network
        self.perturbation = PerturbationNetwork(
            state_dim, action_dim, hidden_dim, phi
        ).to(self.device)
        
        # Optimizers
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=q_lr
        )
        self.perturbation_optimizer = torch.optim.Adam(
            self.perturbation.parameters(),
            lr=perturbation_lr
        )
        
        logger.info(f"Initialized BCQAgent:")
        logger.info(f"  Latent dim: {latent_dim}")
        logger.info(f"  N action samples: {n_action_samples}")
        logger.info(f"  Phi: {phi}")
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action using BCQ: sample from VAE, perturb, select best Q-value.
        
        Args:
            state: Current state [state_dim]
            deterministic: Ignored for BCQ (always uses best Q-value)
        
        Returns:
            Selected action [action_dim]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Sample actions from VAE
            sampled_actions = self.vae.sample_action(
                state_tensor, 
                n_samples=self.n_action_samples
            )  # [1, n_samples, action_dim]
            
            # Perturb actions
            state_repeated = state_tensor.unsqueeze(1).repeat(1, self.n_action_samples, 1)
            perturbed_actions = self.perturbation(
                state_repeated.view(-1, self.state_dim),
                sampled_actions.view(-1, self.action_dim)
            ).view(1, self.n_action_samples, self.action_dim)
            
            # Compute Q-values for all perturbed actions
            q_values = self.q_network1(
                state_repeated.view(-1, self.state_dim),
                perturbed_actions.view(-1, self.action_dim)
            ).view(1, self.n_action_samples)
            
            # Select action with highest Q-value
            best_action_idx = q_values.argmax(dim=1)
            best_action = perturbed_actions[0, best_action_idx].cpu().numpy()[0]
        
        return best_action
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Training step for BCQ.
        
        Steps:
        1. Train VAE to reconstruct actions
        2. Train Q-networks with Bellman error
        3. Train perturbation network to maximize Q
        4. Update target networks
        
        Args:
            batch: Dictionary of batched transitions
        
        Returns:
            Training metrics
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # ============= Train VAE =============
        vae_loss = self._train_vae(states, actions)
        
        # ============= Train Q-networks =============
        q1_loss, q2_loss = self._train_q_networks(
            states, actions, rewards, next_states, dones
        )
        
        # ============= Train Perturbation Network =============
        perturbation_loss = self._train_perturbation(states)
        
        # ============= Update Target Networks =============
        self.update_target_networks()
        
        self.training_step += 1
        self.total_updates += 1
        
        return {
            'vae_loss': vae_loss,
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'perturbation_loss': perturbation_loss,
            'total_updates': self.total_updates
        }
    
    def _train_vae(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> float:
        """
        Train VAE with reconstruction loss and KL divergence.
        
        Loss = Reconstruction Loss + β * KL Divergence
        where KL(q(z|s,a) || p(z)) measures deviation from prior N(0,I)
        """
        reconstructed_actions, mean, log_std = self.vae(states, actions)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed_actions, actions)
        
        # KL divergence: KL(N(μ, σ^2) || N(0, 1))
        # = 0.5 * Σ(μ^2 + σ^2 - log(σ^2) - 1)
        kl_loss = -0.5 * torch.sum(
            1 + 2 * log_std - mean.pow(2) - torch.exp(2 * log_std),
            dim=-1
        ).mean()
        
        # Total VAE loss
        vae_loss = recon_loss + 0.5 * kl_loss
        
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        return vae_loss.item()
    
    def _train_q_networks(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[float, float]:
        """Train Q-networks with Bellman error."""
        with torch.no_grad():
            # Sample actions from VAE for next states
            sampled_next_actions = self.vae.sample_action(
                next_states, 
                n_samples=self.n_action_samples
            )  # [batch_size, n_samples, action_dim]
            
            # Perturb sampled actions
            batch_size = next_states.shape[0]
            next_states_repeated = next_states.unsqueeze(1).repeat(1, self.n_action_samples, 1)
            perturbed_next_actions = self.perturbation(
                next_states_repeated.view(-1, self.state_dim),
                sampled_next_actions.view(-1, self.action_dim)
            ).view(batch_size, self.n_action_samples, self.action_dim)
            
            # Compute target Q-values
            target_q1 = self.target_q1(
                next_states_repeated.view(-1, self.state_dim),
                perturbed_next_actions.view(-1, self.action_dim)
            ).view(batch_size, self.n_action_samples)
            
            target_q2 = self.target_q2(
                next_states_repeated.view(-1, self.state_dim),
                perturbed_next_actions.view(-1, self.action_dim)
            ).view(batch_size, self.n_action_samples)
            
            # Take maximum over sampled actions, minimum over Q-networks
            target_q = torch.min(target_q1.max(dim=1)[0], target_q2.max(dim=1)[0]).unsqueeze(-1)
            target_q_value = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-values
        current_q1 = self.q_network1(states, actions)
        current_q2 = self.q_network2(states, actions)
        
        # Bellman error
        q1_loss = F.mse_loss(current_q1, target_q_value)
        q2_loss = F.mse_loss(current_q2, target_q_value)
        
        total_q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            max_norm=1.0
        )
        self.q_optimizer.step()
        
        return q1_loss.item(), q2_loss.item()
    
    def _train_perturbation(self, states: torch.Tensor) -> float:
        """Train perturbation network to maximize Q-values."""
        # Sample actions from VAE
        sampled_actions = self.vae.sample_action(states, n_samples=1).squeeze(1)
        
        # Perturb actions
        perturbed_actions = self.perturbation(states, sampled_actions)
        
        # Compute Q-value
        q_value = self.q_network1(states, perturbed_actions)
        
        # Loss: negative Q-value (maximize Q by minimizing -Q)
        perturbation_loss = -q_value.mean()
        
        self.perturbation_optimizer.zero_grad()
        perturbation_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.perturbation.parameters(), max_norm=1.0)
        self.perturbation_optimizer.step()
        
        return perturbation_loss.item()
    
    def update_target_networks(self) -> None:
        """Soft update target networks."""
        soft_update(self.q_network1, self.target_q1, self.tau)
        soft_update(self.q_network2, self.target_q2, self.tau)
    
    def get_q_value(
        self, 
        state: np.ndarray, 
        action: np.ndarray
    ) -> float:
        """Get Q-value for state-action pair."""
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
            'vae': self.vae.state_dict(),
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'perturbation': self.perturbation.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'perturbation_optimizer': self.perturbation_optimizer.state_dict()
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Load state dict into all networks."""
        self.vae.load_state_dict(state['vae'])
        self.q_network1.load_state_dict(state['q_network1'])
        self.q_network2.load_state_dict(state['q_network2'])
        self.target_q1.load_state_dict(state['target_q1'])
        self.target_q2.load_state_dict(state['target_q2'])
        self.perturbation.load_state_dict(state['perturbation'])
        self.vae_optimizer.load_state_dict(state['vae_optimizer'])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
        self.perturbation_optimizer.load_state_dict(state['perturbation_optimizer'])
    
    def train_mode(self) -> None:
        """Set all networks to training mode."""
        self.vae.train()
        self.q_network1.train()
        self.q_network2.train()
        self.perturbation.train()
    
    def eval_mode(self) -> None:
        """Set all networks to evaluation mode."""
        self.vae.eval()
        self.q_network1.eval()
        self.q_network2.eval()
        self.perturbation.eval()
