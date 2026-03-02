"""
Safety Layer
Primary interface for all safety checks
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Any, Optional
import logging

from .config import SafetyConfig
from .constraints import (
    Constraint,
    DosageConstraint,
    PhysiologicalConstraint,
    ContraindicationConstraint,
    FrequencyConstraint
)
from .safety_critic import SafetyCritic
from .constraint_optimizer import ConstrainedActionOptimizer


class SafetyLayer:
    """
    Primary interface for all safety checks
    
    Methods:
        - check_action_safety(state, action) -> (is_safe, violation_details)
        - enforce_safety(state, proposed_action) -> safe_action
        - get_safe_action_bounds(state) -> (min_action, max_action)
    """
    
    def __init__(self, config: SafetyConfig):
        """
        Initialize SafetyLayer
        
        Args:
            config: SafetyConfig object with all safety parameters
        """
        self.config = config
        
        # Initialize constraints
        self.constraints = self._initialize_constraints()
        
        # Initialize safety critic (will be trained separately)
        self.safety_critic = None
        
        # Initialize action optimizer
        self.action_optimizer = ConstrainedActionOptimizer(
            action_bounds=config.action_bounds,
            max_iterations=config.max_optimization_iterations,
            tolerance=config.optimization_tolerance
        )
        
        # Violation logging
        self.violation_log = []
    
    def _initialize_constraints(self) -> List[Constraint]:
        """Initialize all constraint objects"""
        constraints = [
            DosageConstraint(self.config.drug_limits),
            PhysiologicalConstraint(self.config.safe_ranges),
            ContraindicationConstraint(),
            FrequencyConstraint(
                max_reminders_per_week=self.config.max_reminders_per_week,
                min_appointment_interval_days=self.config.min_appointment_interval_days
            )
        ]
        return constraints
    
    def check_action_safety(self, 
                           state: Dict[str, Any],
                           action: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
        """
        Check if action is safe
        
        Args:
            state: Current patient state dictionary
            action: Proposed action dictionary
        
        Returns:
            (is_safe, violation_details)
            - is_safe: True if action satisfies all constraints
            - violation_details: Dictionary mapping constraint name to violation message
        """
        is_safe = True
        violation_details = {}
        
        # Check all hard constraints
        for constraint in self.constraints:
            satisfied, message = constraint.check(state, action)
            
            if not satisfied:
                is_safe = False
                violation_details[constraint.get_constraint_name()] = message
        
        # Check safety critic (if trained)
        if self.safety_critic is not None:
            critic_is_safe, safety_score = self._check_safety_critic(state, action)
            if not critic_is_safe:
                is_safe = False
                violation_details['SafetyCritic'] = f"Safety score {safety_score:.4f} below threshold {self.config.safety_threshold}"
        
        # Log violations
        if not is_safe:
            self.violation_log.append({
                'state': state.copy(),
                'action': action.copy(),
                'violations': violation_details.copy()
            })
        
        return is_safe, violation_details
    
    def _check_safety_critic(self, 
                            state: Dict[str, Any],
                            action: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check safety using trained safety critic
        
        Args:
            state: Patient state
            action: Proposed action
        
        Returns:
            (is_safe, safety_score)
        """
        # Convert to tensors
        state_array = self._state_dict_to_array(state)
        action_array = self._action_dict_to_array(action)
        
        state_tensor = torch.FloatTensor(state_array)
        action_tensor = torch.FloatTensor(action_array)
        
        # Predict safety
        is_safe, safety_score = self.safety_critic.predict_safety(
            state_tensor, 
            action_tensor,
            threshold=self.config.safety_threshold
        )
        
        return is_safe, safety_score
    
    def enforce_safety(self,
                      state: Dict[str, Any],
                      proposed_action: np.ndarray) -> np.ndarray:
        """
        Enforce safety by finding nearest safe action
        
        Args:
            state: Current patient state
            proposed_action: Proposed unsafe action (numpy array)
        
        Returns:
            safe_action: Nearest safe action (numpy array)
        """
        # Use constrained optimization to find nearest safe action
        safe_action = self.action_optimizer.find_safe_action(
            state,
            proposed_action,
            self.constraints
        )
        
        return safe_action
    
    def get_safe_action_bounds(self, 
                              state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get safe action bounds for current state
        
        Args:
            state: Current patient state
        
        Returns:
            (min_action, max_action): Safe action bounds as numpy arrays
        """
        # Basic bounds from config
        min_action = np.array([bound[0] for bound in self.config.action_bounds])
        max_action = np.array([bound[1] for bound in self.config.action_bounds])
        
        # Could be refined based on state (e.g., tighter bounds for high-risk patients)
        # For now, return config bounds
        
        return min_action, max_action
    
    def get_violation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on safety violations
        
        Returns:
            stats: Dictionary with violation statistics
        """
        if not self.violation_log:
            return {'total_violations': 0}
        
        # Count violations by constraint type
        violation_counts = {}
        for log_entry in self.violation_log:
            for constraint_name in log_entry['violations'].keys():
                violation_counts[constraint_name] = violation_counts.get(constraint_name, 0) + 1
        
        return {
            'total_violations': len(self.violation_log),
            'violation_breakdown': violation_counts,
            'violation_rate': len(self.violation_log) / max(len(self.violation_log), 1)
        }
    
    def clear_violation_log(self):
        """Clear violation log"""
        self.violation_log = []
    
    def set_safety_critic(self, safety_critic: SafetyCritic):
        """
        Set trained safety critic
        
        Args:
            safety_critic: Trained SafetyCritic network
        """
        self.safety_critic = safety_critic
    

    def apply_discrete_action_mask(self, state_vector: np.ndarray, action_idx: int, n_actions: int) -> int:
        """Simple safety mask for discrete insulin buckets using glucose bounds."""
        glucose = float(state_vector[0]) if len(state_vector) > 0 else 120.0
        # conservative rules: low glucose -> avoid high-dose buckets, high glucose -> avoid no-dose
        forbidden = set()
        if glucose < 80:
            forbidden.update(range(max(0, n_actions // 2), n_actions))
        if glucose > 220:
            forbidden.add(0)
        if action_idx in forbidden:
            safe = [a for a in range(n_actions) if a not in forbidden]
            return int(safe[0]) if safe else int(np.clip(action_idx, 0, n_actions - 1))
        return int(action_idx)

    def _state_dict_to_array(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Convert state dictionary to array
        
        This is problem-specific. Customize based on your state space.
        
        Args:
            state: State dictionary
        
        Returns:
            state_array: Numpy array
        """
        # Example: extract key features in consistent order
        features = []
        
        # Physiological variables
        features.append(state.get('glucose', 0.0))
        features.append(state.get('blood_pressure_systolic', 0.0))
        features.append(state.get('blood_pressure_diastolic', 0.0))
        features.append(state.get('heart_rate', 0.0))
        features.append(state.get('temperature', 0.0))
        
        # Patient demographics
        features.append(state.get('age', 0.0))
        features.append(state.get('bmi', 0.0))
        
        # Clinical history
        features.append(state.get('adherence_score', 0.0))
        features.append(state.get('num_comorbidities', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def _action_dict_to_array(self, action: Dict[str, Any]) -> np.ndarray:
        """
        Convert action dictionary to array
        
        Args:
            action: Action dictionary
        
        Returns:
            action_array: Numpy array
        """
        # Example: extract action components
        action_array = []
        
        action_array.append(action.get('medication_dosage', 0.0))
        action_array.append(action.get('next_appointment_days', 0.0) / 90.0)  # Normalize
        action_array.append(action.get('reminder_frequency', 0.0) / 7.0)  # Normalize
        
        # Pad to match action_bounds length
        while len(action_array) < len(self.config.action_bounds):
            action_array.append(0.0)
        
        return np.array(action_array, dtype=np.float32)


class SafeRLAgent:
    """
    RL Agent with integrated safety layer
    Wraps any base RL agent to enforce safety
    """
    
    def __init__(self, 
                 rl_agent: Any,
                 safety_layer: SafetyLayer,
                 safety_threshold: float = 0.8):
        """
        Initialize SafeRLAgent
        
        Args:
            rl_agent: Base RL agent (e.g., CQL agent)
            safety_layer: SafetyLayer instance
            safety_threshold: Minimum safety score to accept action
        """
        self.rl_agent = rl_agent
        self.safety_layer = safety_layer
        self.safety_threshold = safety_threshold
        
        # Statistics
        self.num_actions = 0
        self.num_overrides = 0
    
    def select_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Select action with safety enforcement
        
        Args:
            state: Current patient state
        
        Returns:
            safe_action: Safe action to execute
        """
        self.num_actions += 1
        
        # Get action from base RL agent
        proposed_action = self.rl_agent.select_action(state)
        
        # Convert to dict for constraint checking
        action_dict = self._array_to_dict(proposed_action)
        
        # Check safety
        is_safe, violations = self.safety_layer.check_action_safety(state, action_dict)
        
        if is_safe:
            return proposed_action
        else:
            # Find nearest safe action
            safe_action = self.safety_layer.enforce_safety(state, proposed_action)
            self.num_overrides += 1
            
            logging.warning(
                f"Unsafe action detected (override {self.num_overrides}/{self.num_actions}). "
                f"Violations: {violations}. Using safe alternative."
            )
            
            return safe_action
    
    def _array_to_dict(self, action_array: np.ndarray) -> Dict[str, Any]:
        """
        Convert action array to dictionary
        
        This should match the action space definition
        
        Args:
            action_array: Action as numpy array
        
        Returns:
            action_dict: Action as dictionary
        """
        action_dict = {}
        
        if len(action_array) >= 1:
            action_dict['medication_dosage'] = float(action_array[0])
        
        if len(action_array) >= 2:
            action_dict['next_appointment_days'] = int(action_array[1] * 90)
        
        if len(action_array) >= 3:
            action_dict['reminder_frequency'] = int(action_array[2] * 7)
        
        return action_dict
    
    def get_override_rate(self) -> float:
        """
        Get rate of safety overrides
        
        Returns:
            override_rate: Fraction of actions that were overridden
        """
        if self.num_actions == 0:
            return 0.0
        return self.num_overrides / self.num_actions
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics
        
        Returns:
            stats: Dictionary with statistics
        """
        return {
            'total_actions': self.num_actions,
            'total_overrides': self.num_overrides,
            'override_rate': self.get_override_rate(),
            'safety_violations': self.safety_layer.get_violation_statistics()
        }
