"""
Statistical baseline policies.

This module implements simple statistical methods for action prediction,
including mean action, regression-based, and k-nearest neighbors approaches.
"""

import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
import inspect
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import ast
from .base_baseline import BaselinePolicy, BaselineMetrics
from collections import Counter

logger = logging.getLogger(__name__)

class MeanActionPolicy(BaselinePolicy):
    """
    Policy that always selects the mean (or mode) action from the training data.
    """

    def __init__(self,
                 name: str = 'Mean-Action',
                 action_space: Optional[Dict[str, Any]] = None,
                 state_dim: Optional[int] = None,
                 device: str = 'cpu'):
        action_space = action_space or {'type': 'continuous', 'dim': 1,
                                        'low': np.array([0.0]), 'high': np.array([1.0])}
        state_dim = state_dim or 10
        super().__init__(name=name, action_space=action_space,
                         state_dim=state_dim, device=device)
        self.is_fitted = False
        self.mean_action = None

    def select_action(self, state: np.ndarray, deterministic: bool = True):
        """
        Select the mean (or mode) action, independent of state.
        """
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted. Call fit() first.")

        # If categorical action (string), return as-is
        if isinstance(self.mean_action, str):
            return self.mean_action

        # Otherwise numeric vector/scalar
        return np.array(self.mean_action, copy=True)

    def _check_safety_violation(self, state: np.ndarray) -> bool:
        """Check safety violation."""
        if len(state) > 0:
            glucose = state[0]
            return glucose < 50 or glucose > 300
        return False


    def evaluate(self, test_data):
        """
        Evaluate policy on test_data of the form:
        (state, action, reward, next_state, done)
        Returns BaselineMetrics via compute_metrics (from BaselinePolicy).
        """
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

            # If your BaselinePolicy already has a safety checker, use that.
            # Otherwise keep this simple rule.
            try:
                if self._check_safety_violation(next_state):
                    safety_violations += 1
            except Exception:
                pass

        return self.compute_metrics(states, actions, rewards, safety_violations)

    def fit(self, states: np.ndarray, actions: np.ndarray):
        actions = np.asarray(actions)

        # If actions are already numeric, just use them
        if actions.dtype.kind not in ("U", "S", "O"):
            actions_f = actions.astype(np.float32, copy=False)
            self.mean_action = self.clip_action(np.mean(actions_f, axis=0))
            self.is_fitted = True
            logger.info(f"Computed mean action: {self.mean_action}")
            return

        # Try direct numeric cast
        try:
            actions_f = actions.astype(np.float32)
            self.mean_action = self.clip_action(np.mean(actions_f, axis=0))
            self.is_fitted = True
            logger.info(f"Computed mean action: {self.mean_action}")
            return
        except Exception:
            pass

        # Parse stringified arrays like "[1, 0]" or "['1','0']"
        parsed = []
        scalar_strings = True
        for a in actions:
            v = a
            if isinstance(v, str):
                v_str = v.strip()
                try:
                    v_eval = ast.literal_eval(v_str)
                    v = v_eval
                except Exception:
                    # Keep as raw string (e.g., "a", "b")
                    v = v_str

            # Detect if we ended up with vector actions or scalars
            if isinstance(v, (list, tuple, np.ndarray)):
                scalar_strings = False
            parsed.append(v)

        # Case A: vector/continuous actions encoded as strings -> convert to float matrix
        if not scalar_strings:
            actions_f = np.asarray(parsed, dtype=np.float32)
            self.mean_action = self.clip_action(np.mean(actions_f, axis=0))
            self.is_fitted = True
            logger.info(f"Computed mean action: {self.mean_action}")
            return

        # Case B: truly categorical discrete actions like "a", "b", ...
        # Mean doesn't make sense; choose the most frequent (mode)
        counts = Counter(parsed)
        mode_action, _ = counts.most_common(1)[0]
        self.mean_action = mode_action
        self.is_fitted = True
        logger.info(f"Computed mode action (categorical): {self.mean_action}")


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
                 action_space: Optional[Dict[str, Any]] = None,
                 state_dim: Optional[int] = None,
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
        if action_space is None:
            action_space = {}
        if state_dim is None:
            state_dim = 10
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
    
    def _fit_state_encoder(self, states: np.ndarray):
        """
        Convert mixed-type states into numeric matrix:
        - numeric cols: keep as float
        - non-numeric cols: factorize to integer codes (stable mapping saved)
        """

        X = np.asarray(states, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        df = pd.DataFrame(X)
        self._state_value_maps = {}

        for col in df.columns:
            # Try numeric conversion first
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().all():
                df[col] = numeric.astype(np.float32)
            else:
                # Factorize non-numeric (and mixed) columns
                # Fill NaN with a marker string so mapping is consistent
                s = df[col].astype(str).fillna("__NA__")
                uniques = pd.Index(s.unique())
                self._state_value_maps[col] = {v: i for i, v in enumerate(uniques)}
                df[col] = s.map(self._state_value_maps[col]).astype(np.float32)

        return df.to_numpy(dtype=np.float32)


    def _transform_states(self, states: np.ndarray):
        import pandas as pd
        import numpy as np

        X = np.asarray(states, dtype=object)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        df = pd.DataFrame(X)

        for col in df.columns:
            if hasattr(self, "_state_value_maps") and col in self._state_value_maps:
                s = df[col].astype(str).fillna("__NA__")
                m = self._state_value_maps[col]
                # unseen categories -> new code = len(mapping)
                df[col] = s.map(lambda v: m.get(v, len(m))).astype(np.float32)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

        return df.to_numpy(dtype=np.float32)
    
    def fit(self, states: np.ndarray, actions: np.ndarray):
        """
        Fit regression (numeric actions) or classification (categorical actions)
        baseline on training data.
        """

        # --- Basic input logging ---
        try:
            states_arr = np.asarray(states)
            actions_arr = np.asarray(actions)
            logger.info(
                "[RegressionPolicy.fit] Start | states shape=%s dtype=%s | actions shape=%s dtype=%s",
                getattr(states_arr, "shape", None), getattr(states_arr, "dtype", None),
                getattr(actions_arr, "shape", None), getattr(actions_arr, "dtype", None),
            )
        except Exception as e:
            logger.warning("[RegressionPolicy.fit] Could not log raw inputs: %s", e)
            actions_arr = np.asarray(actions)

        # --- 1) Convert mixed-type states to numeric ---
        states_num = self._fit_state_encoder(states)
        logger.info(
            "[RegressionPolicy.fit] Encoded states -> states_num shape=%s dtype=%s",
            states_num.shape, states_num.dtype
        )

        # Quick sanity: count non-finite values
        try:
            non_finite = int(np.sum(~np.isfinite(states_num)))
            if non_finite > 0:
                logger.warning("[RegressionPolicy.fit] states_num contains %d non-finite values (NaN/Inf).", non_finite)
        except Exception:
            pass

        # --- 2) Normalize if requested ---
        if self.normalize and self.scaler is not None:
            logger.info("[RegressionPolicy.fit] Normalization enabled: fitting scaler=%s", type(self.scaler).__name__)
            states_num = self.scaler.fit_transform(states_num)
            logger.info("[RegressionPolicy.fit] Normalization done.")
        else:
            logger.info(
                "[RegressionPolicy.fit] Normalization skipped | normalize=%s scaler=%s",
                getattr(self, "normalize", None),
                type(self.scaler).__name__ if getattr(self, "scaler", None) is not None else None
            )

        # --- 3) Decide numeric vs categorical actions ---
        actions_arr = np.asarray(actions)
        try:
            actions_num = actions_arr.astype(np.float32)
            is_numeric_actions = True
            logger.info("[RegressionPolicy.fit] Actions detected as NUMERIC -> regression path.")
        except Exception as e:
            actions_num = None
            is_numeric_actions = False
            # log a few samples to help debugging
            sample = actions_arr[:10] if actions_arr.size > 0 else actions_arr
            logger.info(
                "[RegressionPolicy.fit] Actions detected as CATEGORICAL -> classification path. "
                "Cast-to-float failed: %s | sample=%s",
                e, sample
            )

        # --- 4A) Numeric actions -> regression ---
        if is_numeric_actions:
            self._is_classifier = False
            assert actions_num is not None, "actions_num should not be None when is_numeric_actions is True"
            logger.info(
                "[RegressionPolicy.fit] Fitting regression model=%s | X=%s y=%s",
                type(self.model).__name__, states_num.shape, actions_num.shape if actions_num is not None else "None"
            )
            self.model.fit(states_num, actions_num)
            self.is_fitted = True

            # Optional quick training score (works for many sklearn regressors)
            try:
                score = self.model.score(states_num, actions_num)
                logger.info("[RegressionPolicy.fit] Regression fit complete | train_score=%0.6f", float(score))
            except Exception as e:
                logger.debug("[RegressionPolicy.fit] Could not compute regression train_score: %s", e)

            return

        # --- 4B) Categorical actions -> classification ---
        self._is_classifier = True
        self._label_encoder = LabelEncoder()
        y_cls = np.asarray(self._label_encoder.fit_transform(actions_arr.astype(str)))

        # Log class distribution
        try:
            classes, counts = np.unique(y_cls, return_counts=True)
            topk = sorted(zip(classes.tolist(), counts.tolist()), key=lambda t: t[1], reverse=True)[:10]
            decoded_topk = [(self._label_encoder.inverse_transform([c])[0], n) for c, n in topk]
            logger.info(
                "[RegressionPolicy.fit] Classification labels: num_classes=%d | top_counts=%s",
                len(self._label_encoder.classes_), decoded_topk
            )
        except Exception as e:
            logger.debug("[RegressionPolicy.fit] Could not compute class distribution: %s", e)

        logger.info(
            "[RegressionPolicy.fit] Fitting classifier model=LogisticRegression | X=%s y=%s",
            states_num.shape, y_cls.shape
        )

        base_kwargs = {
            "max_iter": 2000,
            "solver": "lbfgs",
            "multi_class": "auto",  # might not exist in some versions
        }

        # Keep only supported kwargs for the installed LogisticRegression
        sig = inspect.signature(LogisticRegression)
        supported = set(sig.parameters.keys())
        safe_kwargs = {k: v for k, v in base_kwargs.items() if k in supported}

        logger.info(
            "[RegressionPolicy.fit] Creating LogisticRegression with kwargs=%s (filtered from %s)",
            safe_kwargs, base_kwargs
        )

        self.model = LogisticRegression(**safe_kwargs)
        self.model.fit(states_num, y_cls)
        self.is_fitted = True
        return
      
    def select_action(self, state: np.ndarray, deterministic: bool = True):
        """
        Predict action using regression (numeric actions) or classification (categorical actions).

        Args:
            state: Current state
            deterministic: Not used (both paths are deterministic here)

        Returns:
            Predicted action (numeric array/scalar for regression, original label for classification)
        """

        if not self.is_fitted:
            raise RuntimeError("Policy not fitted. Call fit() first.")

        state = np.asarray(state)

        # Reshape if needed
        if state.ndim == 1:
            state = state.reshape(1, -1)

        logger.debug(
            "[RegressionPolicy.select_action] Input state shape=%s dtype=%s",
            state.shape, state.dtype
        )

        # Convert mixed-type state to numeric using fitted mappings
        state_num = self._transform_states(state)
        logger.debug(
            "[RegressionPolicy.select_action] Encoded state -> state_num shape=%s dtype=%s",
            state_num.shape, state_num.dtype
        )

        # Normalize if needed
        if self.normalize and self.scaler is not None:
            state_num = self.scaler.transform(state_num)
            logger.debug("[RegressionPolicy.select_action] Applied scaler transform.")

        # Predict
        pred = self.model.predict(state_num)

        # Classification path: decode label back to original action string
        if getattr(self, "_is_classifier", False):
            try:
                label = self._label_encoder.inverse_transform(pred.astype(int))[0]
                logger.debug("[RegressionPolicy.select_action] Predicted class_id=%s label=%s", pred[0], label)
                return label
            except Exception as e:
                # Fallback: return raw prediction
                logger.warning("[RegressionPolicy.select_action] Failed to decode class prediction: %s", e)
                return pred.squeeze()

        # Regression path
        action = np.asarray(pred).squeeze()
        action = self.clip_action(action)
        logger.debug("[RegressionPolicy.select_action] Predicted action=%s", action)
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
                info['intercept'] = np.atleast_1d(self.model.intercept_).tolist()
        
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
                 action_space: Optional[Dict[str, Any]] = None,
                 state_dim: Optional[int] = None,
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
        if action_space is None:
            action_space = {}
        if state_dim is None:
            state_dim = 10
        super().__init__(name, action_space, state_dim, device)
        
        self.k = k
        self.metric = metric
        self.normalize = normalize
        self.is_fitted = False
        
        # Model will be created in fit() depending on actions type
        self.knn_model = None
        self._is_classifier = False
        self._label_encoder = None
        self._state_value_maps = None

        # Create KNN model
        self.knn_model = KNeighborsRegressor(
            n_neighbors=self.k,
            metric=self.metric,
            weights="distance"
        )
        self._is_classifier = False
        self._label_encoder = None
        self._state_value_maps = None
                
        # Feature scaler
        self.scaler = StandardScaler() if normalize else None
    
    def _fit_state_encoder(self, states: np.ndarray) -> np.ndarray:
        """
        Fit per-column encoders for mixed-type states and return numeric matrix.
        Non-numeric columns are factorized to integer codes.
        """

        X = np.asarray(states, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        df = pd.DataFrame(X)
        self._state_value_maps = {}

        for col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().all():
                df[col] = numeric.astype(np.float32)
            else:
                s = df[col].astype(str).fillna("__NA__")
                uniq = pd.Index(s.unique())
                self._state_value_maps[col] = {v: i for i, v in enumerate(uniq)}
                df[col] = s.map(self._state_value_maps[col]).astype(np.float32)

        return df.to_numpy(dtype=np.float32)


    def _transform_states(self, states: np.ndarray) -> np.ndarray:
        """Transform states using fitted encoders into numeric matrix."""

        X = np.asarray(states, dtype=object)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        df = pd.DataFrame(X)

        for col in df.columns:
            if self._state_value_maps is not None and col in self._state_value_maps:
                s = df[col].astype(str).fillna("__NA__")
                m = self._state_value_maps[col]
                df[col] = s.map(lambda v: m.get(v, len(m))).astype(np.float32)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

        return df.to_numpy(dtype=np.float32)

    def fit(self, states: np.ndarray, actions: np.ndarray):
        """
        Fit KNN model to training data.

        - Encodes mixed-type states to numeric before scaling
        - If actions are numeric -> KNeighborsRegressor
        - If actions are categorical (e.g., 'a','b') -> KNeighborsClassifier
        """
        states_num = self._fit_state_encoder(states)
        actions_arr = np.asarray(actions)

        # Normalize features if requested
        if self.normalize and self.scaler is not None:
            states_num = self.scaler.fit_transform(states_num)

        # Decide regression vs classification
        try:
            actions_num = actions_arr.astype(np.float32)
            self._is_classifier = False
        except Exception:
            actions_num = None
            self._is_classifier = True

        if not self._is_classifier:
            assert actions_num is not None, "actions_num should not be None when is_numeric_actions is False"
            self.knn_model = KNeighborsRegressor(
                n_neighbors=self.k,
                metric=self.metric,
                weights='distance'
            )
            self.knn_model.fit(states_num, actions_num)
            self.is_fitted = True
            logger.info(f"Fitted KNN REGRESSOR policy with k={self.k}, metric={self.metric}")
            logger.info(f"Training set size: {len(states_num)}")
            return

        # Classification
        self._label_encoder = LabelEncoder()
        y_cls = self._label_encoder.fit_transform(actions_arr.astype(str))

        self.knn_model = KNeighborsClassifier(
            n_neighbors=self.k,
            metric=self.metric,
            weights='distance'
        )
        self.knn_model.fit(states_num, y_cls)
        self.is_fitted = True
        logger.info(f"Fitted KNN CLASSIFIER policy with k={self.k}, metric={self.metric}")
        logger.info(f"Training set size: {len(states_num)} | num_classes={len(self._label_encoder.classes_)}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = True):
        """Predict action for given state using KNN model.
        Args:
            state: Current state
            deterministic: Not used (KNN is deterministic here)
        Returns:            
                Predicted action (numeric for regression, original label for classification)
        """
        if not self.is_fitted:
            raise RuntimeError("KNNPolicy not fitted. Call fit() before get_neighbors().")

        if self.knn_model is None:
            raise RuntimeError("KNNPolicy internal error: knn_model is None after fit().")

        if not hasattr(self.knn_model, "kneighbors"):
            raise RuntimeError(f"KNNPolicy internal error: {type(self.knn_model)} has no kneighbors().")


        state = np.asarray(state)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        state_num = self._transform_states(state)

        if self.normalize and self.scaler is not None:
            state_num = self.scaler.transform(state_num)

        pred = self.knn_model.predict(state_num)

        if getattr(self, "_is_classifier", False):
            try:
                if self._label_encoder is None:
                    logger.warning("Label encoder is None in classifier mode; returning raw prediction")
                    return np.asarray(pred).squeeze()
                return self._label_encoder.inverse_transform(pred.astype(int))[0]
            except Exception:
                return np.asarray(pred).squeeze()

        action = np.asarray(pred).squeeze()
        return self.clip_action(action)
    
    def get_neighbors(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get k nearest neighbors for a state.
        
        Args:
            state: Query state
            
        Returns:
            Tuple of (distances, indices) of k nearest neighbors
        """
        if not self.is_fitted:
            raise RuntimeError("KNNPolicy not fitted. Call fit() before get_neighbors().")

        if self.knn_model is None:
            raise RuntimeError("KNNPolicy internal error: knn_model is None after fit().")

        if not hasattr(self.knn_model, "kneighbors"):
            raise RuntimeError(f"KNNPolicy internal error: {type(self.knn_model)} has no kneighbors().")

        
        # Reshape and normalize
        state = np.asarray(state)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        state_num = self._transform_states(state)
        
        if self.normalize and self.scaler is not None:
            state_num = self.scaler.transform(state_num)
        
        # Find neighbors
        distances, indices = self.knn_model.kneighbors(state_num)
        
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
