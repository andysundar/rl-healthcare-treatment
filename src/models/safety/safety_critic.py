"""
Safety Critic Neural Network
Learns to predict safety violations from state-action pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging


class SafetyCritic(nn.Module):
    """
    Neural network to predict safety violations
    
    Architecture: C(s, a) -> safety_score ∈ [0, 1]
    - 1 = safe
    - 0 = unsafe
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize SafetyCritic network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension (default: 256)
        """
        super(SafetyCritic, self).__init__()
        
        input_dim = state_dim + action_dim
        
        # Network architecture: [input -> 256 -> 128 -> 1]
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict safety score
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
        
        Returns:
            safety_score: Score in [0, 1], where 1 = safe, 0 = unsafe
                         Shape: [batch_size, 1]
        """
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)
        
        # Pass through network
        safety_score = self.network(state_action)
        
        return safety_score
    
    def train_step(self, 
                   safe_pairs: Tuple[torch.Tensor, torch.Tensor],
                   unsafe_pairs: Tuple[torch.Tensor, torch.Tensor],
                   optimizer: torch.optim.Optimizer) -> float:
        """
        Training step for safety critic
        
        Args:
            safe_pairs: (safe_states, safe_actions) - examples of safe state-action pairs
            unsafe_pairs: (unsafe_states, unsafe_actions) - examples of unsafe pairs
            optimizer: PyTorch optimizer
        
        Returns:
            loss: Training loss value
        """
        safe_states, safe_actions = safe_pairs
        unsafe_states, unsafe_actions = unsafe_pairs
        
        # Predict safety scores
        safe_scores = self.forward(safe_states, safe_actions)
        unsafe_scores = self.forward(unsafe_states, unsafe_actions)
        
        # Create labels
        safe_labels = torch.ones_like(safe_scores)  # Safe = 1
        unsafe_labels = torch.zeros_like(unsafe_scores)  # Unsafe = 0
        
        # Binary cross-entropy loss
        safe_loss = F.binary_cross_entropy(safe_scores, safe_labels)
        unsafe_loss = F.binary_cross_entropy(unsafe_scores, unsafe_labels)
        
        total_loss = safe_loss + unsafe_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def evaluate(self, 
                 safe_pairs: Tuple[torch.Tensor, torch.Tensor],
                 unsafe_pairs: Tuple[torch.Tensor, torch.Tensor]) -> dict:
        """
        Evaluate safety critic performance
        
        Args:
            safe_pairs: (safe_states, safe_actions)
            unsafe_pairs: (unsafe_states, unsafe_actions)
        
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        self.eval()
        
        with torch.no_grad():
            safe_states, safe_actions = safe_pairs
            unsafe_states, unsafe_actions = unsafe_pairs
            
            # Predict scores
            safe_scores = self.forward(safe_states, safe_actions)
            unsafe_scores = self.forward(unsafe_states, unsafe_actions)
            
            # Compute accuracy (threshold at 0.5)
            safe_correct = (safe_scores > 0.5).float().sum().item()
            unsafe_correct = (unsafe_scores <= 0.5).float().sum().item()
            
            total_correct = safe_correct + unsafe_correct
            total_samples = len(safe_states) + len(unsafe_states)
            
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            
            # Compute mean scores
            mean_safe_score = safe_scores.mean().item()
            mean_unsafe_score = unsafe_scores.mean().item()
            
            # Compute separation (want high separation)
            separation = mean_safe_score - mean_unsafe_score
            
            metrics = {
                'accuracy': accuracy,
                'safe_accuracy': safe_correct / len(safe_states) if len(safe_states) > 0 else 0.0,
                'unsafe_accuracy': unsafe_correct / len(unsafe_states) if len(unsafe_states) > 0 else 0.0,
                'mean_safe_score': mean_safe_score,
                'mean_unsafe_score': mean_unsafe_score,
                'separation': separation
            }
        
        self.train()
        return metrics
    
    def predict_safety(self, state: torch.Tensor, action: torch.Tensor, 
                      threshold: float = 0.8) -> Tuple[bool, float]:
        """
        Predict if state-action pair is safe
        
        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            action: Action tensor [batch_size, action_dim] or [action_dim]
            threshold: Safety threshold (default: 0.8)
        
        Returns:
            (is_safe, safety_score)
        """
        self.eval()
        
        with torch.no_grad():
            # Handle single samples
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            
            # Predict
            safety_score = self.forward(state, action).item()
            is_safe = safety_score >= threshold
        
        self.train()
        return is_safe, safety_score


def train_safety_critic(critic: SafetyCritic,
                        safe_dataset: list,
                        unsafe_dataset: list,
                        num_epochs: int = 100,
                        batch_size: int = 32,
                        lr: float = 1e-4,
                        device: str = 'cpu') -> dict:
    """
    Train safety critic on dataset
    
    Args:
        critic: SafetyCritic network
        safe_dataset: List of (state, action) tuples that are safe
        unsafe_dataset: List of (state, action) tuples that are unsafe
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        history: Training history with losses and metrics
    """
    critic = critic.to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    
    # Convert datasets to tensors
    safe_states = torch.FloatTensor([s for s, a in safe_dataset]).to(device)
    safe_actions = torch.FloatTensor([a for s, a in safe_dataset]).to(device)
    
    unsafe_states = torch.FloatTensor([s for s, a in unsafe_dataset]).to(device)
    unsafe_actions = torch.FloatTensor([a for s, a in unsafe_dataset]).to(device)
    
    history = {
        'losses': [],
        'accuracies': [],
        'separations': []
    }
    
    for epoch in range(num_epochs):
        # Sample batches
        safe_indices = torch.randperm(len(safe_states))[:batch_size]
        unsafe_indices = torch.randperm(len(unsafe_states))[:batch_size]
        
        safe_batch = (safe_states[safe_indices], safe_actions[safe_indices])
        unsafe_batch = (unsafe_states[unsafe_indices], unsafe_actions[unsafe_indices])
        
        # Training step
        loss = critic.train_step(safe_batch, unsafe_batch, optimizer)
        history['losses'].append(loss)
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            metrics = critic.evaluate(
                (safe_states, safe_actions),
                (unsafe_states, unsafe_actions)
            )
            
            history['accuracies'].append(metrics['accuracy'])
            history['separations'].append(metrics['separation'])
            
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Loss: {loss:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Separation: {metrics['separation']:.4f}"
            )
    
    return history
