"""
Composite Reward Function
Combines multiple reward components with configurable weights.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from .base_reward import BaseRewardFunction


class CompositeRewardFunction(BaseRewardFunction):
    """
    Combines multiple reward components into a single reward signal.
    
    Formula:
        R_total = Σ(w_i * R_i)
        
    where:
        w_i = weight for component i
        R_i = reward from component i
    
    Components can be normalized to ensure fair weighting.
    
    Clinical Rationale:
    - Healthcare optimization requires balancing multiple objectives
    - Safety, efficacy, adherence, and cost all matter
    - Weights can be tuned based on clinical priorities
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize composite reward.
        
        Args:
            config: Configuration with component weights
        """
        super().__init__(config)
        
        # Component reward functions
        self.components: Dict[str, BaseRewardFunction] = {}
        
        # Component weights (default values)
        self._weights = {
            'adherence': getattr(config, 'w_adherence', 1.0),
            'health': getattr(config, 'w_health', 2.0),
            'safety': getattr(config, 'w_safety', 5.0),  # Safety most important
            'cost': getattr(config, 'w_cost', 0.1)
        }
        
        # Normalization settings
        self.normalize_components = getattr(config, 'normalize_components', True)
        self.component_ranges = {}  # Track min/max for normalization
        
        # Tracking for statistics
        self.component_history = []
        
    def add_component(
        self,
        name: str,
        reward_function: BaseRewardFunction,
        weight: float = 1.0
    ) -> None:
        """
        Add a reward component.
        
        Args:
            name: Component name (e.g., 'adherence', 'health')
            reward_function: Reward function instance
            weight: Weight for this component
        """
        self.components[name] = reward_function
        self._weights[name] = weight
        
        # Initialize range tracking
        if self.normalize_components:
            self.component_ranges[name] = {
                'min': float('inf'),
                'max': float('-inf')
            }
    
    def compute_reward(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Compute composite reward.
        
        Args:
            state: Previous patient state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Weighted sum of all component rewards
        """
        if not self.components:
            raise ValueError("No reward components added. Use add_component() first.")
        
        total_reward = 0.0
        component_values = {}
        
        # Compute each component
        for name, reward_fn in self.components.items():
            component_reward = reward_fn.compute_reward(state, action, next_state)
            component_values[name] = component_reward
            
            # Update range tracking
            if self.normalize_components:
                self._update_range(name, component_reward)
        
        # Normalize and weight
        for name, raw_value in component_values.items():
            if self.normalize_components:
                normalized_value = self._normalize_component(name, raw_value)
            else:
                normalized_value = raw_value
            
            weight = self._weights.get(name, 1.0)
            weighted_value = weight * normalized_value
            total_reward += weighted_value
        
        # Store for analysis
        self.component_history.append(component_values.copy())
        
        return total_reward
    
    def get_reward_components(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of all reward components.
        
        Returns:
            Dictionary with all component rewards
        """
        breakdown = {}
        
        # Get each component's breakdown
        for name, reward_fn in self.components.items():
            raw_total = reward_fn.compute_reward(state, action, next_state)
            breakdown[f'{name}_total'] = raw_total
            component_breakdown = reward_fn.get_reward_components(state, action, next_state)

            # Prefix with component name to avoid collisions
            for key, value in component_breakdown.items():
                breakdown[f'{name}_{key}'] = value
        
        # Add total
        breakdown['composite_total'] = self.compute_reward(state, action, next_state)
        
        # Add weighted breakdown
        for name, reward_fn in self.components.items():
            raw_value = reward_fn.compute_reward(state, action, next_state)
            
            if self.normalize_components:
                normalized_value = self._normalize_component(name, raw_value)
            else:
                normalized_value = raw_value
                
            weight = self._weights.get(name, 1.0)
            weighted_value = weight * normalized_value
            
            breakdown[f'{name}_weighted'] = weighted_value
        
        return breakdown
    
    def _update_range(self, component_name: str, value: float) -> None:
        """
        Update min/max range for a component.
        
        Args:
            component_name: Name of component
            value: Current value
        """
        if component_name not in self.component_ranges:
            self.component_ranges[component_name] = {
                'min': value,
                'max': value
            }
        else:
            self.component_ranges[component_name]['min'] = min(
                self.component_ranges[component_name]['min'],
                value
            )
            self.component_ranges[component_name]['max'] = max(
                self.component_ranges[component_name]['max'],
                value
            )
    
    def _normalize_component(self, component_name: str, value: float) -> float:
        """
        Normalize component value to [-1, 1] based on observed range.
        
        Args:
            component_name: Name of component
            value: Raw value
            
        Returns:
            Normalized value
        """
        if component_name not in self.component_ranges:
            return value
        
        min_val = self.component_ranges[component_name]['min']
        max_val = self.component_ranges[component_name]['max']
        
        # Not enough range to normalize — pass raw value through
        if abs(max_val - min_val) < 1e-8:
            return value

        # Normalize to [-1, 1]
        normalized = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
        return np.clip(normalized, -1.0, 1.0)
    
    def set_component_weight(self, component_name: str, weight: float) -> None:
        """
        Update weight for a specific component.
        
        Args:
            component_name: Name of component
            weight: New weight value
        """
        if component_name not in self.components:
            raise ValueError(f"Component '{component_name}' not found")
        
        self._weights[component_name] = weight
    
    def get_component_weights(self) -> Dict[str, float]:
        """
        Get current weights for all components.
        
        Returns:
            Dictionary of component weights
        """
        return self._weights.copy()
    
    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics on component values from history.
        
        Returns:
            Dictionary with mean, std, min, max for each component
        """
        if not self.component_history:
            return {}
        
        stats = {}
        
        for name in self.components.keys():
            values = [h[name] for h in self.component_history if name in h]
            
            if values:
                stats[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset component history and range tracking."""
        self.component_history = []
        
        if self.normalize_components:
            for name in self.components.keys():
                self.component_ranges[name] = {
                    'min': float('inf'),
                    'max': float('-inf')
                }
    
    def tune_weights(
        self,
        validation_episodes: List[List[Tuple]],
        target_metrics: Dict[str, float],
        method: str = 'grid_search'
    ) -> Dict[str, float]:
        """
        Automatically tune component weights based on validation data.
        
        Args:
            validation_episodes: List of episode trajectories
            target_metrics: Target values for each metric
            method: Tuning method ('grid_search', 'bayesian', 'evolutionary')
            
        Returns:
            Optimized weight dictionary
        """
        # Simplified implementation - full version would use actual optimization
        # This is a placeholder for the concept
        
        if method == 'grid_search':
            best_weights = self._grid_search_weights(validation_episodes, target_metrics)
        else:
            # Default: keep current weights
            best_weights = self._weights.copy()
        
        return best_weights
    
    def _grid_search_weights(
        self,
        validation_episodes: List[List[Tuple]],
        target_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Grid search for optimal weights.
        
        Args:
            validation_episodes: Validation trajectories
            target_metrics: Target metric values
            
        Returns:
            Best weight configuration
        """
        # Simplified grid search
        # In production, this would be more sophisticated
        
        best_weights = self._weights.copy()
        best_score = float('-inf')
        
        # Define search space
        weight_options = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        # Try different weight combinations (limited for efficiency)
        # In practice, use more sophisticated optimization
        
        return best_weights
    
    def save_weights(self, filepath: str) -> None:
        """
        Save current weights to file.
        
        Args:
            filepath: Path to save weights
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self._weights, f, indent=2)
    
    def load_weights(self, filepath: str) -> None:
        """
        Load weights from file.
        
        Args:
            filepath: Path to load weights from
        """
        import json
        
        with open(filepath, 'r') as f:
            self._weights = json.load(f)
