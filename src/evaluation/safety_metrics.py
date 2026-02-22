"""
Safety Metrics for Healthcare RL

Implements comprehensive safety evaluation.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SafetyResult:
    safety_index: float
    violation_rate: float
    avg_violation_severity: float
    critical_violations: int
    violation_breakdown: Dict[str, float]
    metadata: Dict = None

class SafeStateChecker:
    """Checks patient state safety."""
    
    def __init__(self, config):
        self.config = config
        self.safe_ranges = {
            'glucose': config.safety.safe_glucose_range,
            'bp_systolic': config.safety.safe_bp_systolic_range,
        }
        
    def is_safe(self, state):
        for variable, (min_val, max_val) in self.safe_ranges.items():
            if variable in state:
                value = state[variable]
                if not (min_val <= value <= max_val):
                    return False
        return True
    
    def get_violations(self, state):
        violations = []
        for variable, (min_val, max_val) in self.safe_ranges.items():
            if variable in state:
                value = state[variable]
                if value < min_val or value > max_val:
                    violations.append((variable, value))
        return violations

class SafetyEvaluator:
    """Comprehensive safety evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.safe_state_checker = SafeStateChecker(config)
        
    def compute_safety_index(self, trajectories):
        total_states = 0
        unsafe_states = 0
        
        for traj in trajectories:
            for state in traj['states']:
                total_states += 1
                if not self.safe_state_checker.is_safe(state):
                    unsafe_states += 1
        
        if total_states == 0:
            return 0.0
        
        return 1.0 - (unsafe_states / total_states)
    
    def compute_violation_rate(self, trajectories):
        total_actions = 0
        violations = 0
        
        for traj in trajectories:
            for state in traj['states']:
                total_actions += 1
                if not self.safe_state_checker.is_safe(state):
                    violations += 1
        
        if total_actions == 0:
            return 0.0
        
        return violations / total_actions
    
    def compute_violation_severity(self, trajectories):
        all_violations = []
        
        for traj in trajectories:
            for state in traj['states']:
                for var, (min_val, max_val) in self.safe_state_checker.safe_ranges.items():
                    if var in state:
                        value = state[var]
                        range_width = max_val - min_val
                        
                        if value < min_val:
                            severity = (min_val - value) / range_width
                            all_violations.append(severity)
                        elif value > max_val:
                            severity = (value - max_val) / range_width
                            all_violations.append(severity)
        
        return np.mean(all_violations) if all_violations else 0.0
    
    def evaluate(self, trajectories):
        safety_index = self.compute_safety_index(trajectories)
        violation_rate = self.compute_violation_rate(trajectories)
        avg_severity = self.compute_violation_severity(trajectories)
        
        critical_count = 0
        violation_breakdown = {'minor': 0, 'moderate': 0, 'severe': 0, 'critical': 0}
        
        result = SafetyResult(
            safety_index=safety_index,
            violation_rate=violation_rate,
            avg_violation_severity=avg_severity,
            critical_violations=critical_count,
            violation_breakdown=violation_breakdown,
            metadata={'n_trajectories': len(trajectories)}
        )
        
        self._print_summary(result)
        return result
    
    def _print_summary(self, result):
        print("\n" + "="*70)
        print("SAFETY EVALUATION SUMMARY")
        print("="*70)
        print(f"Safety Index:           {result.safety_index:.3f}")
        print(f"Violation Rate:         {result.violation_rate:.3f}")
        print(f"Avg Violation Severity: {result.avg_violation_severity:.3f}")
        print("="*70 + "\n")
