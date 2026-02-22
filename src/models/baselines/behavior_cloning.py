"""
Behavior cloning policy implementation.

This baseline learns to mimic expert behavior from historical clinical data
using supervised learning. It represents what happens when we simply try to
replicate past clinical decisions.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import pandas as pd 

from .base_baseline import BaselinePolicy, BaselineMetrics

logger = logging.getLogger(__name__)


class BehaviorCloningDataset(Dataset):
    """Dataset for behavior cloning."""

    @staticmethod
    def _to_numeric_array(
        x: Union[np.ndarray, "pd.DataFrame", "pd.Series", List[Any]],
        name: str,
        dtype: type = np.float32,
    ) -> np.ndarray:
        """Convert input to a numeric numpy array suitable for Torch tensors.

        This prevents crashes when upstream accidentally passes string/object
        dtypes (e.g., gender/ethnicity, ICD codes, medication names, timestamps)
        into the BC baseline.
        """

        # pandas path (best for mixed dtypes)
        if pd is not None and isinstance(x, (pd.DataFrame, pd.Series)):
            if isinstance(x, pd.Series):
                x = x.to_frame()

            non_numeric_cols = list(x.select_dtypes(exclude=["number"]).columns)
            if non_numeric_cols:
                logger.warning(
                    "[BC] %s contains non-numeric columns (%d). Applying one-hot encoding: %s",
                    name,
                    len(non_numeric_cols),
                    non_numeric_cols[:25],
                )
                x = pd.get_dummies(x, dummy_na=True)

            # Coerce any remaining non-numeric values
            x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            arr = x.to_numpy(dtype=dtype, copy=False)
            logger.info(
                "[BC] %s converted via pandas -> shape=%s dtype=%s",
                name,
                arr.shape,
                arr.dtype,
            )
            return arr

        # numpy path
        arr = np.asarray(x)
        original_dtype = getattr(arr, "dtype", None)

        # If we received a 1D object array where each element is itself a vector/list,
        # stack it into a proper 2D matrix. This happens in some pipelines where
        # states are stored as a column of arrays.
        try:
            if hasattr(arr, 'dtype') and arr.dtype.kind == 'O' and arr.ndim == 1 and len(arr) > 0:
                first = arr[0]
                # 1) element is vector-like (list/tuple/ndarray)
                if isinstance(first, (list, tuple)) or (hasattr(first, 'shape') and hasattr(first, '__array__')):
                    import numpy as _np
                    arr = _np.stack([_np.asarray(v) for v in arr], axis=0)
                    logger.info('[BC] %s detected as sequence-of-vectors -> stacked shape=%s dtype=%s', name, arr.shape, getattr(arr,'dtype',None))
                # 2) element is dict-like (list of feature dicts) -> expand to DataFrame columns
                elif isinstance(first, dict) and pd is not None:
                    df = pd.DataFrame(list(arr))
                    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                    arr = df.to_numpy(copy=False)
                    logger.info('[BC] %s detected as sequence-of-dicts -> expanded shape=%s dtype=%s', name, arr.shape, getattr(arr,'dtype',None))
        except Exception:
            pass

        # Ensure 2D for states/actions if a single feature is provided as 1D
        if hasattr(arr, 'ndim') and arr.ndim == 1:
            arr = arr.reshape(-1, 1)
            
        if hasattr(arr, "dtype") and arr.dtype.kind in ("U", "S", "O"):
            logger.warning(
                "[BC] %s has non-numeric dtype=%s. Attempting numeric coercion.",
                name,
                original_dtype,
            )
            if pd is not None:
                # Use pandas coercion as a robust fallback
                df = pd.DataFrame(arr)
                df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                arr = df.to_numpy(copy=False)
            else:
                # Best-effort conversion without pandas
                try:
                    arr = arr.astype(np.float64)
                except Exception:
                    sample = arr.ravel()[:10].tolist()
                    raise TypeError(
                        f"{name} contains non-numeric values and pandas is unavailable. "
                        f"Sample values: {sample}"
                    )

        arr = arr.astype(dtype, copy=False)

        if original_dtype is not None and original_dtype != arr.dtype:
            logger.info(
                "[BC] %s cast: %s -> %s (shape=%s)",
                name,
                original_dtype,
                arr.dtype,
                arr.shape,
            )
        else:
            logger.info(
                "[BC] %s shape=%s dtype=%s",
                name,
                getattr(arr, "shape", None),
                getattr(arr, "dtype", None),
            )

        return arr
    
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            states: State observations
            actions: Expert actions
        """
        # Defensive conversion: prevents crashes when upstream accidentally passes
        # string/object dtypes (common with EHR categorical features).
        states_np = self._to_numeric_array(states, name="states", dtype=np.float32)
        actions_np = self._to_numeric_array(actions, name="actions", dtype=np.float32)

        self.states = torch.from_numpy(states_np)
        self.actions = torch.from_numpy(actions_np)
        
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
         # Normalize input shape to (batch, features)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() > 2:
            state = state.view(state.size(0), -1)

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
                 action_space: Optional[Dict[str, Any]] = None,
                 state_dim: Optional[int] = None,
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
        if action_space is None:
            logger.warning("action_space not provided; using default continuous space with dim=1")
            raise ValueError("action_space must be provided")
        if state_dim is None:
            logger.warning("state_dim not provided; cannot initialize network without state dimension")
            raise ValueError("state_dim must be provided")
        
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

        # If upstream feature processing changed the effective state dimension (e.g.,
        # categorical one-hot expansion or accidental collapse to a single column),
        # rebuild the network to match the data.
        inferred_state_dim = int(train_dataset.states.shape[1]) if train_dataset.states.ndim >= 2 else 1
        if hasattr(self.network, 'state_dim') and self.network.state_dim != inferred_state_dim:
            logger.warning('[BC] Network state_dim=%s does not match training data dim=%s. Rebuilding network.', self.network.state_dim, inferred_state_dim)
            action_dim = getattr(self.network, 'action_dim', None)
            if action_dim is None:
                action_dim = self.action_space.get('dim', 1)
            hidden_dims = [m.out_features for m in self.network.network if isinstance(m, nn.Linear)][:-1] or [256, 256]
            self.network = BehaviorCloningNetwork(state_dim=inferred_state_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.optimizer.param_groups[0]['lr'])
            logger.info('[BC] Rebuilt network: state_dim=%d action_dim=%d', inferred_state_dim, action_dim)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        

        # Extra safety: infer the *actual* feature dimension from a real batch.
        # This catches edge-cases where dataset-level shape looks OK but individual
        # samples collapse to a scalar/column during collation.
        try:
            sample_states, _ = next(iter(train_loader))
            if sample_states.ndim == 1:
                sample_states = sample_states.unsqueeze(1)
            actual_state_dim = int(sample_states.shape[-1])
            if hasattr(self.network, 'state_dim') and self.network.state_dim != actual_state_dim:
                logger.warning(
                    '[BC] Batch state_dim=%s differs from network state_dim=%s (sample batch shape=%s). Rebuilding network.',
                    actual_state_dim,
                    self.network.state_dim,
                    tuple(sample_states.shape),
                )
                action_dim = getattr(self.network, 'action_dim', None) or self.action_space.get('dim', 1)
                hidden_dims = [m.out_features for m in self.network.network if isinstance(m, nn.Linear)][:-1] or [256, 256]
                self.network = BehaviorCloningNetwork(
                    state_dim=actual_state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims
                ).to(self.device)
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.optimizer.param_groups[0]['lr'])
                logger.info('[BC] Rebuilt network from batch: state_dim=%d action_dim=%d', actual_state_dim, action_dim)
        except StopIteration:
            pass
        except Exception as e:
            logger.warning('[BC] Unable to infer batch state_dim (%s). Continuing with configured network.', e)
        
        # Validation set
        has_val = val_states is not None and val_actions is not None
        if has_val:
            assert val_states is not None and val_actions is not None
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
    
    def save(self, path: Union[str, Path]):
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
    
    def load(self, path: Union[str, Path]):
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
