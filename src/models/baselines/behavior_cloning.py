"""
Behavior cloning policy implementation.

This baseline learns to mimic expert behavior from historical clinical data
using supervised learning. It represents what happens when we simply try to
replicate past clinical decisions.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path

from .base_baseline import BaselinePolicy, BaselineMetrics

logger = logging.getLogger(__name__)


class BehaviorCloningDataset(Dataset):
    """Dataset for behavior cloning."""
    
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            states: State observations
            actions: Expert actions
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        
        assert len(self.states) == len(self.actions), \
            "States and actions must have same length"
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx]


class BehaviorCloningNetwork(nn.Module):
    """
    Neural network for behavior cloning.
    
    Architecture: MLP with residual connections for stability.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.1):
        """
        Initialize network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout: Dropout probability
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Input state
            
        Returns:
            Predicted action
        """
        return self.network(state)


class BehaviorCloningPolicy(BaselinePolicy):
    """
    Behavior cloning policy using supervised learning.
    
    This policy learns to imitate expert behavior from historical (state, action)
    pairs using supervised learning. It serves as a strong baseline representing
    what can be achieved by simply replicating past clinical decisions.
    
    Attributes:
        network: Neural network for action prediction
        optimizer: Optimizer for training
        loss_fn: Loss function (MSE for continuous, CE for discrete)
        training_history: History of training metrics
    """
    
    def __init__(self,
                 name: str = "Behavior-Cloning",
                 action_space: Dict[str, Any] = None,
                 state_dim: int = None,
                 hidden_dims: List[int] = [256, 256],
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        """
        Initialize behavior cloning policy.
        
        Args:
            name: Policy name
            action_space: Action space specification
            state_dim: State dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for training
            device: Computation device
        """
        super().__init__(name, action_space, state_dim, device)
        
        # Get action dimension
        action_dim = action_space.get('dim', 1)
        
        # Create network
        self.network = BehaviorCloningNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Loss function (MSE for continuous actions)
        self.loss_fn = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        logger.info(f"Initialized BC policy with {sum(p.numel() for p in self.network.parameters())} parameters")
    
    def train(self,
             states: np.ndarray,
             actions: np.ndarray,
             val_states: Optional[np.ndarray] = None,
             val_actions: Optional[np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 256,
             verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the behavior cloning policy.
        
        Args:
            states: Training states
            actions: Training actions
            val_states: Validation states (optional)
            val_actions: Validation actions (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        # Create dataset and dataloader
        train_dataset = BehaviorCloningDataset(states, actions)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Validation set
        has_val = val_states is not None and val_actions is not None
        if has_val:
            val_dataset = BehaviorCloningDataset(val_states, val_actions)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Training loop
        self.network.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_states, batch_actions in train_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                
                # Forward pass
                predicted_actions = self.network(batch_states)
                loss = self.loss_fn(predicted_actions, batch_actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.training_history['loss'].append(avg_loss)
            self.training_history['epochs'].append(epoch)
            
            # Validation
            if has_val:
                val_loss = self._validate(val_loader)
                self.training_history['val_loss'].append(val_loss)
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}"
                if has_val:
                    msg += f", Val Loss: {val_loss:.6f}"
                logger.info(msg)
        
        logger.info("Training completed")
        return self.training_history
    
    def _validate(self, val_loader: DataLoader) -> float:
        """
        Compute validation loss.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.network.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                
                predicted_actions = self.network(batch_states)
                loss = self.loss_fn(predicted_actions, batch_actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.network.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select action using trained network.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic prediction
            
        Returns:
            Predicted action
        """
        self.network.eval()
        
        with torch.no_grad():
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Predict action
            action_tensor = self.network(state_tensor)
            action = action_tensor.cpu().numpy().squeeze()
            
            # Clip to valid bounds
            action = self.clip_action(action)
        
        return action
    
    def evaluate(self, test_data: List[Tuple]) -> BaselineMetrics:
        """
        Evaluate policy on test data.
        
        Args:
            test_data: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            BaselineMetrics with evaluation results
        """
        states = []
        actions = []
        rewards = []
        safety_violations = 0
        
        self.network.eval()
        
        for state, _, reward, next_state, _ in test_data:
            # Get action from policy
            action = self.select_action(state)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Check safety
            if self._check_safety_violation(next_state):
                safety_violations += 1
        
        return self.compute_metrics(states, actions, rewards, safety_violations)
    
    def _check_safety_violation(self, state: np.ndarray) -> bool:
        """Check if state represents a safety violation."""
        if len(state) > 0:
            glucose = state[0]
            return glucose < 50 or glucose > 300
        return False
    
    def save(self, path: str):
        """
        Save policy to disk.
        
        Args:
            path: Path to save policy
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'state_dim': self.state_dim,
            'action_space': self.action_space
        }, path)
        
        logger.info(f"Saved policy to {path}")
    
    def load(self, path: str):
        """
        Load policy from disk.
        
        Args:
            path: Path to load policy from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded policy from {path}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_info()
        info.update({
            'num_parameters': sum(p.numel() for p in self.network.parameters()),
            'network_architecture': str(self.network),
            'training_epochs': len(self.training_history['epochs']),
            'final_loss': self.training_history['loss'][-1] if self.training_history['loss'] else None
        })
        return info
    
    def __str__(self) -> str:
        """String representation."""
        n_params = sum(p.numel() for p in self.network.parameters())
        return f"Behavior Cloning Policy ({n_params:,} parameters)"


# Convenience function
def create_behavior_cloning_policy(state_dim: int,
                                   action_dim: int = 1,
                                   hidden_dims: List[int] = [256, 256],
                                   learning_rate: float = 1e-3,
                                   device: str = 'cpu') -> BehaviorCloningPolicy:
    """
    Create a behavior cloning policy with standard configuration.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        device: Computation device
        
    Returns:
        Configured BehaviorCloningPolicy
    """
    action_space = {
        'type': 'continuous',
        'low': np.zeros(action_dim),
        'high': np.ones(action_dim),
        'dim': action_dim
    }
    
    return BehaviorCloningPolicy(
        name="Behavior-Cloning",
        action_space=action_space,
        state_dim=state_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        device=device
    )
