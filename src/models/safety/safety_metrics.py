"""
Safety Metrics
Functions to compute safety metrics for evaluation
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict

from .constraints import Constraint


def safety_index(trajectories: List[List[Dict[str, Any]]],
                safe_ranges: Dict[str, Tuple[float, float]]) -> float:
    """
    Compute safety index: fraction of time in safe states
    
    Safety Index = 1 - (# unsafe states / # total states)
    
    Args:
        trajectories: List of trajectories, each trajectory is a list of state dicts
        safe_ranges: Dictionary mapping variables to (min_val, max_val)
    
    Returns:
        safety_index: Value in [0, 1], where 1 = always safe
    """
    total_states = 0
    unsafe_states = 0
    
    for trajectory in trajectories:
        for state_dict in trajectory:
            total_states += 1
            
            # Check if state is safe
            for var, (min_val, max_val) in safe_ranges.items():
                if var in state_dict:
                    value = state_dict[var]
                    if not (min_val <= value <= max_val):
                        unsafe_states += 1
                        break  # Count state only once even if multiple violations
    
    if total_states == 0:
        return 1.0
    
    return 1.0 - (unsafe_states / total_states)


def violation_rate(trajectories: List[List[Dict[str, Any]]],
                   constraints: List[Constraint]) -> Dict[str, float]:
    """
    Compute violation rate for each constraint type
    
    Args:
        trajectories: List of trajectories
        constraints: List of Constraint objects
    
    Returns:
        violation_rates: Dictionary mapping constraint name to violation rate
    """
    total_transitions = 0
    violation_counts = {c.get_constraint_name(): 0 for c in constraints}
    
    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            state = trajectory[i]
            action = trajectory[i].get('action', {})
            total_transitions += 1
            
            for constraint in constraints:
                is_safe, _ = constraint.check(state, action)
                if not is_safe:
                    violation_counts[constraint.get_constraint_name()] += 1
    
    if total_transitions == 0:
        return {c.get_constraint_name(): 0.0 for c in constraints}
    
    violation_rates = {
        name: count / total_transitions 
        for name, count in violation_counts.items()
    }
    
    return violation_rates


def violation_severity(trajectories: List[List[Dict[str, Any]]],
                       safe_ranges: Dict[str, Tuple[float, float]]) -> float:
    """
    Compute average severity of violations (how far from safe bounds)
    
    Severity for each violation = distance from bound / bound value
    
    Args:
        trajectories: List of trajectories
        safe_ranges: Dictionary mapping variables to (min_val, max_val)
    
    Returns:
        avg_severity: Average violation severity
    """
    total_severity = 0.0
    num_violations = 0
    
    for trajectory in trajectories:
        for state_dict in trajectory:
            for var, (min_val, max_val) in safe_ranges.items():
                if var in state_dict:
                    value = state_dict[var]
                    
                    if value < min_val:
                        # Below minimum
                        severity = (min_val - value) / min_val if min_val != 0 else (min_val - value)
                        total_severity += severity
                        num_violations += 1
                    elif value > max_val:
                        # Above maximum
                        severity = (value - max_val) / max_val if max_val != 0 else (value - max_val)
                        total_severity += severity
                        num_violations += 1
    
    if num_violations == 0:
        return 0.0
    
    return total_severity / num_violations


def constraint_satisfaction_rate(trajectories: List[List[Dict[str, Any]]],
                                 constraints: List[Constraint]) -> Dict[str, float]:
    """
    Compute satisfaction rate for each constraint (1 - violation rate)
    
    Args:
        trajectories: List of trajectories
        constraints: List of Constraint objects
    
    Returns:
        satisfaction_rates: Dictionary mapping constraint name to satisfaction rate
    """
    violation_rates_dict = violation_rate(trajectories, constraints)
    satisfaction_rates = {
        name: 1.0 - rate 
        for name, rate in violation_rates_dict.items()
    }
    return satisfaction_rates


def temporal_violation_analysis(trajectories: List[List[Dict[str, Any]]],
                                safe_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, List[float]]:
    """
    Analyze violation patterns over time
    
    Args:
        trajectories: List of trajectories
        safe_ranges: Dictionary of safe ranges
    
    Returns:
        temporal_data: Dictionary with timestep-indexed violation rates
    """
    # Find maximum trajectory length
    max_length = max(len(traj) for traj in trajectories) if trajectories else 0
    
    # Track violations at each timestep
    violations_per_timestep = [0] * max_length
    total_per_timestep = [0] * max_length
    
    for trajectory in trajectories:
        for t, state_dict in enumerate(trajectory):
            if t >= max_length:
                break
            
            total_per_timestep[t] += 1
            
            # Check if violated
            is_violated = False
            for var, (min_val, max_val) in safe_ranges.items():
                if var in state_dict:
                    value = state_dict[var]
                    if not (min_val <= value <= max_val):
                        is_violated = True
                        break
            
            if is_violated:
                violations_per_timestep[t] += 1
    
    # Compute violation rate per timestep
    violation_rates_over_time = [
        violations_per_timestep[t] / total_per_timestep[t] if total_per_timestep[t] > 0 else 0.0
        for t in range(max_length)
    ]
    
    return {
        'timesteps': list(range(max_length)),
        'violation_rates': violation_rates_over_time,
        'total_samples': total_per_timestep
    }


def patient_specific_safety_analysis(trajectories: List[List[Dict[str, Any]]],
                                     patient_ids: List[str],
                                     safe_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Dict]:
    """
    Analyze safety metrics for individual patients
    
    Args:
        trajectories: List of trajectories
        patient_ids: List of patient IDs corresponding to trajectories
        safe_ranges: Dictionary of safe ranges
    
    Returns:
        patient_analysis: Dictionary mapping patient_id to safety metrics
    """
    patient_analysis = {}
    
    for patient_id, trajectory in zip(patient_ids, trajectories):
        total_states = len(trajectory)
        unsafe_states = 0
        violations_by_var = defaultdict(int)
        
        for state_dict in trajectory:
            state_is_unsafe = False
            
            for var, (min_val, max_val) in safe_ranges.items():
                if var in state_dict:
                    value = state_dict[var]
                    if not (min_val <= value <= max_val):
                        violations_by_var[var] += 1
                        state_is_unsafe = True
            
            if state_is_unsafe:
                unsafe_states += 1
        
        patient_safety_index = 1.0 - (unsafe_states / total_states) if total_states > 0 else 1.0
        
        patient_analysis[patient_id] = {
            'safety_index': patient_safety_index,
            'total_states': total_states,
            'unsafe_states': unsafe_states,
            'violations_by_variable': dict(violations_by_var)
        }
    
    return patient_analysis


def generate_safety_report(trajectories: List[List[Dict[str, Any]]],
                           constraints: List[Constraint],
                           safe_ranges: Dict[str, Tuple[float, float]],
                           patient_ids: List[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive safety report
    
    Args:
        trajectories: List of trajectories
        constraints: List of Constraint objects
        safe_ranges: Dictionary of safe ranges
        patient_ids: Optional list of patient IDs
    
    Returns:
        report: Comprehensive safety report dictionary
    """
    report = {
        'overall_metrics': {},
        'constraint_metrics': {},
        'temporal_analysis': {},
        'violation_details': {}
    }
    
    # Overall metrics
    report['overall_metrics'] = {
        'safety_index': safety_index(trajectories, safe_ranges),
        'violation_severity': violation_severity(trajectories, safe_ranges),
        'total_trajectories': len(trajectories),
        'total_states': sum(len(traj) for traj in trajectories)
    }
    
    # Constraint-specific metrics
    report['constraint_metrics'] = {
        'violation_rates': violation_rate(trajectories, constraints),
        'satisfaction_rates': constraint_satisfaction_rate(trajectories, constraints)
    }
    
    # Temporal analysis
    report['temporal_analysis'] = temporal_violation_analysis(trajectories, safe_ranges)
    
    # Patient-specific analysis (if patient IDs provided)
    if patient_ids is not None and len(patient_ids) == len(trajectories):
        report['patient_analysis'] = patient_specific_safety_analysis(
            trajectories, patient_ids, safe_ranges
        )
    
    # Violation breakdown
    violation_breakdown = defaultdict(int)
    total_violations = 0
    
    for trajectory in trajectories:
        for state_dict in trajectory:
            for var, (min_val, max_val) in safe_ranges.items():
                if var in state_dict:
                    value = state_dict[var]
                    if not (min_val <= value <= max_val):
                        violation_breakdown[var] += 1
                        total_violations += 1
    
    report['violation_details'] = {
        'total_violations': total_violations,
        'violations_by_variable': dict(violation_breakdown)
    }
    
    return report


def print_safety_report(report: Dict[str, Any], verbose: bool = True):
    """
    Print safety report in human-readable format
    
    Args:
        report: Safety report from generate_safety_report()
        verbose: Whether to print detailed information
    """
    print("=" * 60)
    print("SAFETY REPORT")
    print("=" * 60)
    
    # Overall metrics
    print("\nOVERALL METRICS:")
    print(f"  Safety Index: {report['overall_metrics']['safety_index']:.4f}")
    print(f"  Violation Severity: {report['overall_metrics']['violation_severity']:.4f}")
    print(f"  Total Trajectories: {report['overall_metrics']['total_trajectories']}")
    print(f"  Total States: {report['overall_metrics']['total_states']}")
    
    # Constraint metrics
    print("\nCONSTRAINT SATISFACTION:")
    for constraint_name, rate in report['constraint_metrics']['satisfaction_rates'].items():
        print(f"  {constraint_name}: {rate:.4f} ({rate * 100:.2f}%)")
    
    # Violation details
    print("\nVIOLATION BREAKDOWN:")
    print(f"  Total Violations: {report['violation_details']['total_violations']}")
    if verbose:
        print("  By Variable:")
        for var, count in report['violation_details']['violations_by_variable'].items():
            print(f"    {var}: {count}")
    
    # Patient analysis (if available)
    if 'patient_analysis' in report and verbose:
        print("\nPATIENT-SPECIFIC ANALYSIS:")
        print(f"  Number of patients: {len(report['patient_analysis'])}")
        
        # Show best and worst patients
        patient_safety_scores = {
            pid: metrics['safety_index']
            for pid, metrics in report['patient_analysis'].items()
        }
        
        if patient_safety_scores:
            best_patient = max(patient_safety_scores, key=patient_safety_scores.get)
            worst_patient = min(patient_safety_scores, key=patient_safety_scores.get)
            
            print(f"  Best patient: {best_patient} (Safety: {patient_safety_scores[best_patient]:.4f})")
            print(f"  Worst patient: {worst_patient} (Safety: {patient_safety_scores[worst_patient]:.4f})")
    
    print("=" * 60)
