"""
Rule-based policy implementation following clinical guidelines.

This baseline represents current clinical practice using threshold-based rules
for treatment decisions. It serves as a strong baseline as it encodes clinical
expert knowledge.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging
from dataclasses import dataclass

from .base_baseline import BaselinePolicy, BaselineMetrics

logger = logging.getLogger(__name__)


@dataclass
class ClinicalRule:
    """
    Represents a single clinical decision rule.
    
    Attributes:
        name: Rule identifier
        condition: Function that takes state and returns bool
        action: Function that takes state and returns action
        priority: Rule priority (higher = more important)
        description: Human-readable description
    """
    name: str
    condition: callable
    action: callable
    priority: int
    description: str


class RuleBasedPolicy(BaselinePolicy):
    """
    Rule-based policy following clinical guidelines.
    
    This policy implements threshold-based decision rules commonly used in
    clinical practice. Rules are evaluated in priority order and the first
    matching rule determines the action.
    
    Example for diabetes management:
        - If glucose > 200 mg/dL: increase insulin by 10%
        - If glucose < 70 mg/dL: decrease insulin by 20%
        - If 70 <= glucose <= 200: maintain current dosage
    
    Attributes:
        rules: List of clinical rules
        default_action: Action to take if no rule matches
        feature_names: Names of state features for rule conditions
    """
    
    def __init__(self, 
                 name: str = "Rule-Based",
                 action_space: Dict[str, Any] = None,
                 state_dim: int = None,
                 feature_names: Optional[List[str]] = None,
                 device: str = 'cpu'):
        """
        Initialize rule-based policy.
        
        Args:
            name: Policy name
            action_space: Action space specification
            state_dim: State dimension
            feature_names: Names of state features
            device: Computation device
        """
        super().__init__(name, action_space, state_dim, device)
        
        self.rules: List[ClinicalRule] = []
        self.feature_names = feature_names or []
        self.default_action = self._get_default_action()
        
        # Statistics
        self.rule_usage_count = {}
        self.default_action_count = 0
        
    def _get_default_action(self) -> np.ndarray:
        """Get default action (middle of action space)."""
        low, high = self.get_action_bounds()
        return (low + high) / 2.0
    
    def add_rule(self, 
                 name: str,
                 condition: callable,
                 action: callable,
                 priority: int = 0,
                 description: str = "") -> None:
        """
        Add a clinical rule to the policy.
        
        Args:
            name: Rule identifier
            condition: Function(state) -> bool
            action: Function(state) -> action
            priority: Rule priority (higher evaluated first)
            description: Human-readable description
        """
        rule = ClinicalRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            description=description
        )
        
        self.rules.append(rule)
        self.rule_usage_count[name] = 0
        
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"Added rule '{name}' with priority {priority}")
    
    def define_diabetes_rules(self, 
                             glucose_idx: int = 0,
                             insulin_idx: int = 0,
                             high_threshold: float = 200.0,
                             low_threshold: float = 70.0,
                             increase_factor: float = 0.1,
                             decrease_factor: float = 0.2) -> None:
        """
        Define standard diabetes management rules.
        
        Args:
            glucose_idx: Index of glucose in state vector
            insulin_idx: Index of insulin in action vector
            high_threshold: High glucose threshold (mg/dL)
            low_threshold: Low glucose threshold (mg/dL)
            increase_factor: Factor to increase insulin by
            decrease_factor: Factor to decrease insulin by
        """
        # Rule 1: Critical hypoglycemia
        def critical_low_condition(state):
            return state[glucose_idx] < low_threshold
        
        def critical_low_action(state):
            action = self.default_action.copy()
            # Significantly decrease insulin
            current_insulin = action[insulin_idx]
            action[insulin_idx] = max(0.0, current_insulin * (1 - decrease_factor))
            return self.clip_action(action)
        
        self.add_rule(
            name="critical_hypoglycemia",
            condition=critical_low_condition,
            action=critical_low_action,
            priority=100,  # Highest priority - safety critical
            description=f"Glucose < {low_threshold}: decrease insulin by {decrease_factor*100}%"
        )
        
        # Rule 2: Hyperglycemia
        def high_glucose_condition(state):
            return state[glucose_idx] > high_threshold
        
        def high_glucose_action(state):
            action = self.default_action.copy()
            # Increase insulin
            current_insulin = action[insulin_idx]
            action[insulin_idx] = current_insulin * (1 + increase_factor)
            return self.clip_action(action)
        
        self.add_rule(
            name="hyperglycemia",
            condition=high_glucose_condition,
            action=high_glucose_action,
            priority=50,
            description=f"Glucose > {high_threshold}: increase insulin by {increase_factor*100}%"
        )
        
        # Rule 3: Normal range - maintain
        def normal_range_condition(state):
            return low_threshold <= state[glucose_idx] <= high_threshold
        
        def normal_range_action(state):
            # Maintain current dosage
            return self.clip_action(self.default_action.copy())
        
        self.add_rule(
            name="normal_glucose",
            condition=normal_range_condition,
            action=normal_range_action,
            priority=10,
            description=f"{low_threshold} <= Glucose <= {high_threshold}: maintain dosage"
        )
        
        logger.info(f"Defined {len(self.rules)} diabetes management rules")
    
    def define_hypertension_rules(self,
                                  bp_systolic_idx: int = 1,
                                  medication_idx: int = 0,
                                  stage2_threshold: float = 140.0,
                                  stage1_threshold: float = 130.0,
                                  normal_threshold: float = 120.0) -> None:
        """
        Define hypertension management rules based on ACC/AHA guidelines.
        
        Args:
            bp_systolic_idx: Index of systolic BP in state
            medication_idx: Index of BP medication in action
            stage2_threshold: Stage 2 hypertension threshold (mmHg)
            stage1_threshold: Stage 1 hypertension threshold (mmHg)
            normal_threshold: Normal BP threshold (mmHg)
        """
        # Rule 1: Stage 2 hypertension (BP >= 140)
        def stage2_condition(state):
            return state[bp_systolic_idx] >= stage2_threshold
        
        def stage2_action(state):
            action = self.default_action.copy()
            # Increase medication significantly
            action[medication_idx] = min(1.0, action[medication_idx] * 1.3)
            return self.clip_action(action)
        
        self.add_rule(
            name="stage2_hypertension",
            condition=stage2_condition,
            action=stage2_action,
            priority=90,
            description=f"BP >= {stage2_threshold}: increase medication significantly"
        )
        
        # Rule 2: Stage 1 hypertension (130 <= BP < 140)
        def stage1_condition(state):
            return stage1_threshold <= state[bp_systolic_idx] < stage2_threshold
        
        def stage1_action(state):
            action = self.default_action.copy()
            # Moderate increase
            action[medication_idx] = min(1.0, action[medication_idx] * 1.15)
            return self.clip_action(action)
        
        self.add_rule(
            name="stage1_hypertension",
            condition=stage1_condition,
            action=stage1_action,
            priority=70,
            description=f"{stage1_threshold} <= BP < {stage2_threshold}: increase medication moderately"
        )
        
        # Rule 3: Normal BP
        def normal_bp_condition(state):
            return state[bp_systolic_idx] < normal_threshold
        
        def normal_bp_action(state):
            action = self.default_action.copy()
            # Gradually reduce if possible
            action[medication_idx] = max(0.0, action[medication_idx] * 0.95)
            return self.clip_action(action)
        
        self.add_rule(
            name="normal_bp",
            condition=normal_bp_condition,
            action=normal_bp_action,
            priority=30,
            description=f"BP < {normal_threshold}: maintain or reduce medication"
        )
        
        logger.info(f"Defined {len(self.rules)} hypertension management rules")
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select action based on rules.
        
        Args:
            state: Current state
            deterministic: Not used (rule-based is always deterministic)
            
        Returns:
            Action selected by first matching rule
        """
        # Convert to numpy if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Evaluate rules in priority order
        for rule in self.rules:
            try:
                if rule.condition(state):
                    action = rule.action(state)
                    self.rule_usage_count[rule.name] += 1
                    return action
            except Exception as e:
                logger.warning(f"Error evaluating rule '{rule.name}': {e}")
                continue
        
        # No rule matched - use default action
        self.default_action_count += 1
        logger.debug("No rule matched, using default action")
        return self.default_action.copy()
    
    def get_applicable_rules(self, state: np.ndarray) -> List[ClinicalRule]:
        """
        Get all rules that apply to the given state.
        
        Args:
            state: Patient state
            
        Returns:
            List of applicable rules
        """
        applicable = []
        for rule in self.rules:
            try:
                if rule.condition(state):
                    applicable.append(rule)
            except Exception as e:
                logger.warning(f"Error checking rule '{rule.name}': {e}")
        
        return applicable
    
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
        
        for state, _, reward, next_state, _ in test_data:
            # Get action from policy
            action = self.select_action(state)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Check safety (simple check - can be customized)
            if self._check_safety_violation(next_state):
                safety_violations += 1
        
        return self.compute_metrics(states, actions, rewards, safety_violations)
    
    def _check_safety_violation(self, state: np.ndarray) -> bool:
        """
        Check if state represents a safety violation.
        
        Args:
            state: State to check
            
        Returns:
            True if safety violation detected
        """
        # Simple check - assumes first feature is glucose
        # Can be customized based on actual state representation
        if len(state) > 0:
            glucose = state[0]
            return glucose < 50 or glucose > 300  # Critical ranges
        return False
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on rule usage.
        
        Returns:
            Dictionary with rule usage statistics
        """
        total_decisions = sum(self.rule_usage_count.values()) + self.default_action_count
        
        stats = {
            'total_decisions': total_decisions,
            'default_action_count': self.default_action_count,
            'default_action_rate': self.default_action_count / total_decisions if total_decisions > 0 else 0,
            'rule_usage': {}
        }
        
        for rule_name, count in self.rule_usage_count.items():
            stats['rule_usage'][rule_name] = {
                'count': count,
                'rate': count / total_decisions if total_decisions > 0 else 0
            }
        
        return stats
    
    def reset_statistics(self):
        """Reset usage statistics."""
        for rule_name in self.rule_usage_count:
            self.rule_usage_count[rule_name] = 0
        self.default_action_count = 0
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_info()
        info.update({
            'num_rules': len(self.rules),
            'rules': [
                {
                    'name': rule.name,
                    'priority': rule.priority,
                    'description': rule.description
                }
                for rule in self.rules
            ],
            'statistics': self.get_rule_statistics()
        })
        return info
    
    def __str__(self) -> str:
        """String representation."""
        return f"Rule-Based Policy ({len(self.rules)} rules)"


# Example usage and configuration
def create_diabetes_rule_policy(state_dim: int = 10, 
                               action_dim: int = 1) -> RuleBasedPolicy:
    """
    Create pre-configured diabetes management rule-based policy.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        
    Returns:
        Configured RuleBasedPolicy
    """
    action_space = {
        'type': 'continuous',
        'low': np.zeros(action_dim),
        'high': np.ones(action_dim),
        'dim': action_dim
    }
    
    policy = RuleBasedPolicy(
        name="Diabetes-Rule-Based",
        action_space=action_space,
        state_dim=state_dim
    )
    
    # Define standard diabetes rules
    policy.define_diabetes_rules(
        glucose_idx=0,
        insulin_idx=0,
        high_threshold=200.0,
        low_threshold=70.0,
        increase_factor=0.1,
        decrease_factor=0.2
    )
    
    return policy


def create_hypertension_rule_policy(state_dim: int = 10,
                                   action_dim: int = 1) -> RuleBasedPolicy:
    """
    Create pre-configured hypertension management rule-based policy.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        
    Returns:
        Configured RuleBasedPolicy
    """
    action_space = {
        'type': 'continuous',
        'low': np.zeros(action_dim),
        'high': np.ones(action_dim),
        'dim': action_dim
    }
    
    policy = RuleBasedPolicy(
        name="Hypertension-Rule-Based",
        action_space=action_space,
        state_dim=state_dim
    )
    
    # Define hypertension rules
    policy.define_hypertension_rules(
        bp_systolic_idx=1,
        medication_idx=0
    )
    
    return policy
