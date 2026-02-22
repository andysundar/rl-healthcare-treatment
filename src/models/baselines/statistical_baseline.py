"""
Statistical baseline policies.

This module implements simple statistical methods for action prediction,
including mean action, regression-based, and k-nearest neighbors approaches.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import logging

from .base_baseline import BaselinePolicy, BaselineMetrics

logger = logging.getLogger(__name__)


class MeanActionPolicy(BaselinePolicy):
    """
    Baseline that always selects the mean action from historical data.
    
    This represents the simplest possible policy: do what was done on average
    in the past, regardless of current state.
    
    Attributes:
        mean_action: Mean action computed from training data
    """
    
    def __init__(self,
                 name: str = "Mean-Action",
                 action_space: Dict[str, Any] = None,
                 state_dim: int = None,
                 device: str = 'cpu'):
        """
        Initialize mean action policy.
        
        Args:
            name: Policy name
            action_space: Action space specification
            state_dim: State dimension
            device: Computation device
        """
        super().__init__(name, action_space, state_dim, device)
        self.mean_action = None
        self.is_fitted = False
    
    def fit(self, states: np.ndarray, actions: np.ndarray):
        """
        Compute mean action from training data.
        
        Args:
            states: Training states (not used, but kept for API consistency)
            actions: Training actions
        """
        self.mean_action = np.mean(actions, axis=0)
        self.mean_action = self.clip_action(self.mean_action)
        self.is_fitted = True
        
        logger.info(f"Computed mean action: {self.mean_action}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select mean action (independent of state).
        
        Args:
            state: Current state (ignored)
            deterministic: Not used
            
        Returns:
            Mean action
        """
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted. Call fit() first.")
        
        return self.mean_action.copy()
    
    def evaluate(self, test_data: List[Tuple]) -> BaselineMetrics:
        """Evaluate policy on test data."""
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted. Call fit() first.")
        
        states = []
        actions = []
        rewards = []
        safety_violations = 0
        
        for state, _, reward, next_state, _ in test_data:
            action = self.select_action(state)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if self._check_safety_violation(next_state):
                safety_violations += 1
        
        return self.compute_metrics(states, actions, rewards, safety_violations)
    
    def _check_safety_violation(self, state: np.ndarray) -> bool:
        """Check safety violation."""
        if len(state) > 0:
            glucose = state[0]
            return glucose < 50 or glucose > 300
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_info()
        info['mean_action'] = self.mean_action.tolist() if self.mean_action is not None else None
        info['is_fitted'] = self.is_fitted
        return info


class RegressionPolicy(BaselinePolicy):
    """
    Regression-based policy that predicts actions from states.
    
    Uses linear or ridge regression to learn a mapping from states to actions.
    This is similar to behavior cloning but with simpler linear models.
    
    Attributes:
        model: Sklearn regression model
        scaler: Feature scaler for states
        regression_type: 'linear' or 'ridge'
    """
    
    def __init__(self,
                 name: str = "Regression",
                 action_space: Dict[str, Any] = None,
                 state_dim: int = None,
                 regression_type: str = 'linear',
                 alpha: float = 1.0,
                 normalize: bool = True,
                 device: str = 'cpu'):
        """
        Initialize regression policy.
        
        Args:
            name: Policy name
            action_space: Action space specification
            state_dim: State dimension
            regression_type: 'linear' or 'ridge'
            alpha: Regularization strength for ridge regression
            normalize: Whether to normalize features
            device: Computation device
        """
        super().__init__(name, action_space, state_dim, device)
        
        self.regression_type = regression_type
        self.normalize = normalize
        self.is_fitted = False
        
        # Create model
        if regression_type == 'linear':
            self.model = LinearRegression()
        elif regression_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")
        
        # Feature scaler
        self.scaler = StandardScaler() if normalize else None
    
    def fit(self, states: np.ndarray, actions: np.ndarray):
        """
        Fit regression model to training data.
        
        Args:
            states: Training states
            actions: Training actions
        """
        # Normalize features if requested
        if self.normalize:
            states = self.scaler.fit_transform(states)
        
        # Fit model
        self.model.fit(states, actions)
        self.is_fitted = True
        
        # Compute training score
        train_score = self.model.score(states, actions)
        logger.info(f"Fitted {self.regression_type} regression, R² = {train_score:.4f}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action using regression model.
        
        Args:
            state: Current state
            deterministic: Not used (regression is deterministic)
            
        Returns:
            Predicted action
        """
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted. Call fit() first.")
        
        # Reshape if needed
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Normalize if needed
        if self.normalize:
            state = self.scaler.transform(state)
        
        # Predict
        action = self.model.predict(state).squeeze()
        
        # Clip to bounds
        action = self.clip_action(action)
        
        return action
    
    def evaluate(self, test_data: List[Tuple]) -> BaselineMetrics:
        """Evaluate policy on test data."""
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted. Call fit() first.")
        
        states = []
        actions = []
        rewards = []
        safety_violations = 0
        
        for state, _, reward, next_state, _ in test_data:
            action = self.select_action(state)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if self._check_safety_violation(next_state):
                safety_violations += 1
        
        return self.compute_metrics(states, actions, rewards, safety_violations)
    
    def _check_safety_violation(self, state: np.ndarray) -> bool:
        """Check safety violation."""
        if len(state) > 0:
            glucose = state[0]
            return glucose < 50 or glucose > 300
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_info()
        info.update({
            'regression_type': self.regression_type,
            'normalize': self.normalize,
            'is_fitted': self.is_fitted
        })
        
        if self.is_fitted and hasattr(self.model, 'coef_'):
            info['coefficients'] = self.model.coef_.tolist()
            if hasattr(self.model, 'intercept_'):
                info['intercept'] = self.model.intercept_.tolist()
        
        return info


class KNNPolicy(BaselinePolicy):
    """
    K-Nearest Neighbors policy.
    
    Finds k most similar states in training data and averages their actions.
    This represents a non-parametric approach that adapts to local state patterns.
    
    Attributes:
        knn_model: Sklearn KNN regressor
        k: Number of neighbors
        metric: Distance metric
    """
    
    def __init__(self,
                 name: str = "KNN",
                 action_space: Dict[str, Any] = None,
                 state_dim: int = None,
                 k: int = 5,
                 metric: str = 'euclidean',
                 normalize: bool = True,
                 device: str = 'cpu'):
        """
        Initialize KNN policy.
        
        Args:
            name: Policy name
            action_space: Action space specification
            state_dim: State dimension
            k: Number of neighbors
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            normalize: Whether to normalize features
            device: Computation device
        """
        super().__init__(name, action_space, state_dim, device)
        
        self.k = k
        self.metric = metric
        self.normalize = normalize
        self.is_fitted = False
        
        # Create KNN model
        self.knn_model = KNeighborsRegressor(
            n_neighbors=k,
            metric=metric,
            weights='distance'  # Weight by inverse distance
        )
        
        # Feature scaler
        self.scaler = StandardScaler() if normalize else None
    
    def fit(self, states: np.ndarray, actions: np.ndarray):
        """
        Fit KNN model to training data.
        
        Args:
            states: Training states
            actions: Training actions
        """
        # Normalize features if requested
        if self.normalize:
            states = self.scaler.fit_transform(states)
        
        # Fit KNN model
        self.knn_model.fit(states, actions)
        self.is_fitted = True
        
        logger.info(f"Fitted KNN policy with k={self.k}, metric={self.metric}")
        logger.info(f"Training set size: {len(states)}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action using KNN.
        
        Args:
            state: Current state
            deterministic: Not used (KNN is deterministic)
            
        Returns:
            Predicted action (weighted average of k nearest neighbors)
        """
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted. Call fit() first.")
        
        # Reshape if needed
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Normalize if needed
        if self.normalize:
            state = self.scaler.transform(state)
        
        # Predict using KNN
        action = self.knn_model.predict(state).squeeze()
        
        # Clip to bounds
        action = self.clip_action(action)
        
        return action
    
    def get_neighbors(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get k nearest neighbors for a state.
        
        Args:
            state: Query state
            
        Returns:
            Tuple of (distances, indices) of k nearest neighbors
        """
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted. Call fit() first.")
        
        # Reshape and normalize
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        if self.normalize:
            state = self.scaler.transform(state)
        
        # Find neighbors
        distances, indices = self.knn_model.kneighbors(state)
        
        return distances.squeeze(), indices.squeeze()
    
    def evaluate(self, test_data: List[Tuple]) -> BaselineMetrics:
        """Evaluate policy on test data."""
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted. Call fit() first.")
        
        states = []
        actions = []
        rewards = []
        safety_violations = 0
        
        for state, _, reward, next_state, _ in test_data:
            action = self.select_action(state)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if self._check_safety_violation(next_state):
                safety_violations += 1
        
        return self.compute_metrics(states, actions, rewards, safety_violations)
    
    def _check_safety_violation(self, state: np.ndarray) -> bool:
        """Check safety violation."""
        if len(state) > 0:
            glucose = state[0]
            return glucose < 50 or glucose > 300
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_info()
        info.update({
            'k': self.k,
            'metric': self.metric,
            'normalize': self.normalize,
            'is_fitted': self.is_fitted
        })
        return info


# Convenience functions
def create_mean_action_policy(action_dim: int = 1,
                              state_dim: int = 10) -> MeanActionPolicy:
    """Create mean action policy."""
    action_space = {
        'type': 'continuous',
        'low': np.zeros(action_dim),
        'high': np.ones(action_dim),
        'dim': action_dim
    }
    
    return MeanActionPolicy(
        name="Mean-Action",
        action_space=action_space,
        state_dim=state_dim
    )


def create_regression_policy(state_dim: int,
                            action_dim: int = 1,
                            regression_type: str = 'ridge',
                            alpha: float = 1.0) -> RegressionPolicy:
    """Create regression policy."""
    action_space = {
        'type': 'continuous',
        'low': np.zeros(action_dim),
        'high': np.ones(action_dim),
        'dim': action_dim
    }
    
    return RegressionPolicy(
        name=f"{regression_type.capitalize()}-Regression",
        action_space=action_space,
        state_dim=state_dim,
        regression_type=regression_type,
        alpha=alpha
    )


def create_knn_policy(state_dim: int,
                     action_dim: int = 1,
                     k: int = 5,
                     metric: str = 'euclidean') -> KNNPolicy:
    """Create KNN policy."""
    action_space = {
        'type': 'continuous',
        'low': np.zeros(action_dim),
        'high': np.ones(action_dim),
        'dim': action_dim
    }
    
    return KNNPolicy(
        name=f"KNN-{k}",
        action_space=action_space,
        state_dim=state_dim,
        k=k,
        metric=metric
    )
