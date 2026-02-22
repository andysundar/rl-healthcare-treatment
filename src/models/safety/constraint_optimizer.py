"""
Constrained Action Optimizer
Finds nearest safe action to proposed unsafe action
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Any, Tuple
import logging

from .constraints import Constraint


class ConstrainedActionOptimizer:
    """
    Find nearest safe action using constrained optimization
    
    Solves:
        min ||a_safe - a_proposed||^2
        s.t. all constraints satisfied
    """
    
    def __init__(self, 
                 action_bounds: List[Tuple[float, float]],
                 max_iterations: int = 100,
                 tolerance: float = 1e-6):
        """
        Initialize optimizer
        
        Args:
            action_bounds: List of (min, max) bounds for each action dimension
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        self.action_bounds = action_bounds
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def find_safe_action(self, 
                        state: Dict[str, Any],
                        unsafe_action: np.ndarray,
                        constraints: List[Constraint]) -> np.ndarray:
        """
        Find nearest safe action using constrained optimization
        
        Args:
            state: Current patient state
            unsafe_action: Proposed unsafe action (numpy array)
            constraints: List of Constraint objects to satisfy
        
        Returns:
            safe_action: Nearest action that satisfies all constraints
        """
        def objective(action_array):
            """Minimize distance to proposed action"""
            return np.sum((action_array - unsafe_action) ** 2)
        
        def constraint_function(action_array):
            """
            Constraint function for scipy.optimize
            Returns positive value if constraints satisfied, negative otherwise
            """
            # Convert array to action dict
            action_dict = self._array_to_action_dict(action_array)
            
            # Check all constraints
            for constraint in constraints:
                is_safe, _ = constraint.check(state, action_dict)
                if not is_safe:
                    return -1.0  # Constraint violated
            
            return 1.0  # All constraints satisfied
        
        # Set up constraint for scipy
        constraints_scipy = [{
            'type': 'ineq',
            'fun': constraint_function
        }]
        
        try:
            # Run optimization
            result = minimize(
                objective,
                x0=unsafe_action,
                method='SLSQP',
                bounds=self.action_bounds,
                constraints=constraints_scipy,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance
                }
            )
            
            if result.success:
                return result.x
            else:
                # Optimization failed, use fallback
                logging.warning(f"Optimization failed: {result.message}. Using conservative fallback.")
                return self._get_conservative_action(state)
        
        except Exception as e:
            # Optimization error, use fallback
            logging.error(f"Optimization error: {str(e)}. Using conservative fallback.")
            return self._get_conservative_action(state)
    
    def _array_to_action_dict(self, action_array: np.ndarray) -> Dict[str, Any]:
        """
        Convert action array to action dictionary
        
        This mapping is problem-specific and should be customized
        based on your action space definition.
        
        Args:
            action_array: Numpy array of action values
        
        Returns:
            action_dict: Dictionary representation of action
        """
        # Example mapping (customize based on your action space)
        # Assuming action_array has 5 dimensions:
        # [medication_dosage, appointment_days_normalized, reminder_freq_normalized, ...]
        
        action_dict = {}
        
        if len(action_array) >= 1:
            # First dimension: medication dosage (scaled to actual range)
            action_dict['medication_dosage'] = float(action_array[0])
        
        if len(action_array) >= 2:
            # Second dimension: appointment days (0-1 -> 0-90 days)
            action_dict['next_appointment_days'] = int(action_array[1] * 90)
        
        if len(action_array) >= 3:
            # Third dimension: reminder frequency (0-1 -> 0-7 per week)
            action_dict['reminder_frequency'] = int(action_array[2] * 7)
        
        # Add more mappings as needed
        
        return action_dict
    
    def _get_conservative_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Return a conservative safe action as fallback
        
        This is the "do minimal intervention" action
        
        Args:
            state: Current patient state
        
        Returns:
            conservative_action: Safe minimal action
        """
        # Return action with minimal intervention
        # (zeros or lower bounds depending on action space)
        conservative_action = np.array([bound[0] for bound in self.action_bounds])
        
        return conservative_action
    
    def find_safe_action_grid_search(self,
                                     state: Dict[str, Any],
                                     unsafe_action: np.ndarray,
                                     constraints: List[Constraint],
                                     num_samples: int = 100) -> np.ndarray:
        """
        Alternative method: Find safe action using grid search
        
        Useful if gradient-based optimization fails.
        
        Args:
            state: Current patient state
            unsafe_action: Proposed unsafe action
            constraints: List of constraints
            num_samples: Number of samples to try
        
        Returns:
            safe_action: Nearest safe action found
        """
        best_action = None
        best_distance = float('inf')
        
        # Generate random samples within action bounds
        for _ in range(num_samples):
            # Sample random action
            sample_action = np.array([
                np.random.uniform(bound[0], bound[1])
                for bound in self.action_bounds
            ])
            
            # Check if safe
            action_dict = self._array_to_action_dict(sample_action)
            is_safe = True
            
            for constraint in constraints:
                safe, _ = constraint.check(state, action_dict)
                if not safe:
                    is_safe = False
                    break
            
            if is_safe:
                # Compute distance to unsafe action
                distance = np.linalg.norm(sample_action - unsafe_action)
                
                if distance < best_distance:
                    best_distance = distance
                    best_action = sample_action
        
        if best_action is not None:
            return best_action
        else:
            # No safe action found, return conservative
            logging.warning("Grid search found no safe action. Using conservative fallback.")
            return self._get_conservative_action(state)
    
    def project_to_safe_region(self,
                               state: Dict[str, Any],
                               action: np.ndarray,
                               constraints: List[Constraint],
                               max_steps: int = 10,
                               step_size: float = 0.1) -> np.ndarray:
        """
        Project action to safe region using gradient descent
        
        Args:
            state: Current patient state
            action: Current action
            constraints: List of constraints
            max_steps: Maximum projection steps
            step_size: Step size for projection
        
        Returns:
            projected_action: Action projected to safe region
        """
        current_action = action.copy()
        
        for step in range(max_steps):
            # Check if already safe
            action_dict = self._array_to_action_dict(current_action)
            all_safe = True
            
            for constraint in constraints:
                is_safe, _ = constraint.check(state, action_dict)
                if not is_safe:
                    all_safe = False
                    break
            
            if all_safe:
                return current_action
            
            # Move toward bounds (simple heuristic)
            for i in range(len(current_action)):
                min_bound, max_bound = self.action_bounds[i]
                
                # If outside bounds, move toward bound
                if current_action[i] < min_bound:
                    current_action[i] = min(current_action[i] + step_size, min_bound)
                elif current_action[i] > max_bound:
                    current_action[i] = max(current_action[i] - step_size, max_bound)
                else:
                    # Inside bounds, move toward middle
                    middle = (min_bound + max_bound) / 2
                    if current_action[i] < middle:
                        current_action[i] += step_size
                    else:
                        current_action[i] -= step_size
        
        # If still not safe after max_steps, return conservative
        logging.warning("Projection failed to find safe action. Using conservative fallback.")
        return self._get_conservative_action(state)
