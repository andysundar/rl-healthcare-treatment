"""
Clinical Metrics for Healthcare RL Evaluation

Implements clinical outcome evaluation including:
1. Health Outcome Improvement
2. Time in Target Range  
3. Adverse Event Rate
4. Clinical Goal Achievement
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClinicalResult:
    """Results from clinical evaluation."""
    health_improvements: Dict[str, float]
    time_in_range: Dict[str, float]
    adverse_event_rate: float
    goal_achievement_rate: float
    clinical_significance: Dict[str, bool]
    metadata: Optional[Dict] = None


class ClinicalEvaluator:
    """Comprehensive clinical evaluation for healthcare RL policies."""
    
    def __init__(self, config):
        self.config = config
        self.target_ranges = {
            'glucose': config.clinical.target_glucose_range,
            'bp_systolic': config.clinical.target_bp_systolic_range,
            'bp_diastolic': config.clinical.target_bp_diastolic_range,
        }
        self.target_hba1c = config.clinical.target_hba1c
        self.adverse_events = config.clinical.adverse_events
        
    def compute_health_improvement(
        self,
        policy_trajectories: List[Dict],
        baseline_trajectories: List[Dict],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compare health metric improvements vs baseline.
        
        Args:
            policy_trajectories: Trajectories from evaluated policy
            baseline_trajectories: Trajectories from baseline (e.g., standard care)
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary of {metric: improvement_rate}
        """
        if metrics is None:
            metrics = ['glucose', 'bp_systolic', 'hba1c']
        
        improvements = {}
        
        for metric in metrics:
            if metric not in self.target_ranges and metric != 'hba1c':
                continue
            
            # Policy outcomes
            policy_in_target = self._compute_target_achievement(
                policy_trajectories, metric
            )
            
            # Baseline outcomes
            baseline_in_target = self._compute_target_achievement(
                baseline_trajectories, metric
            )
            
            # Relative improvement
            improvement = policy_in_target - baseline_in_target
            improvements[metric] = improvement
            
            logger.info(f"{metric} improvement: {improvement:.3f} "
                       f"(Policy: {policy_in_target:.3f}, Baseline: {baseline_in_target:.3f})")
        
        return improvements
    
    def _compute_target_achievement(
        self,
        trajectories: List[Dict],
        metric: str
    ) -> float:
        """Compute fraction of final states in target range."""
        in_target_count = 0
        total_count = 0
        
        for traj in trajectories:
            if len(traj['states']) == 0:
                continue
            
            final_state = traj['states'][-1]
            if metric not in final_state:
                continue
            
            total_count += 1
            value = final_state[metric]
            
            if metric == 'hba1c':
                if value <= self.target_hba1c:
                    in_target_count += 1
            elif metric in self.target_ranges:
                min_val, max_val = self.target_ranges[metric]
                if min_val <= value <= max_val:
                    in_target_count += 1
        
        if total_count == 0:
            return 0.0
        
        return in_target_count / total_count
    
    def compute_time_in_range(
        self,
        trajectories: List[Dict],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute time in target range for each metric.
        
        Args:
            trajectories: Episode trajectories
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary of {metric: time_in_range_fraction}
        """
        if metrics is None:
            metrics = list(self.target_ranges.keys())
        
        time_in_range_results = {}
        
        for metric in metrics:
            if metric not in self.target_ranges:
                continue
            
            min_val, max_val = self.target_ranges[metric]
            in_range_count = 0
            total_count = 0
            
            for traj in trajectories:
                for state in traj['states']:
                    if metric in state:
                        total_count += 1
                        value = state[metric]
                        if min_val <= value <= max_val:
                            in_range_count += 1
            
            if total_count > 0:
                time_in_range = in_range_count / total_count
                time_in_range_results[metric] = time_in_range
                logger.info(f"{metric} time in range: {time_in_range:.3f}")
            else:
                time_in_range_results[metric] = 0.0
        
        return time_in_range_results
    
    def compute_adverse_event_rate(
        self,
        trajectories: List[Dict]
    ) -> float:
        """
        Compute fraction of episodes with adverse events.
        
        Returns:
            Adverse event rate in [0, 1]
        """
        episodes_with_events = 0
        
        for traj in trajectories:
            has_adverse_event = False
            
            for state in traj['states']:
                # Check for any adverse event markers
                for event_type in self.adverse_events:
                    if state.get(f'adverse_{event_type}', False):
                        has_adverse_event = True
                        break
                
                if has_adverse_event:
                    break
            
            if has_adverse_event:
                episodes_with_events += 1
        
        if len(trajectories) == 0:
            return 0.0
        
        adverse_event_rate = episodes_with_events / len(trajectories)
        logger.info(f"Adverse event rate: {adverse_event_rate:.3f}")
        
        return adverse_event_rate
    
    def evaluate(
        self,
        policy_trajectories: List[Dict],
        baseline_trajectories: Optional[List[Dict]] = None
    ) -> ClinicalResult:
        """
        Comprehensive clinical evaluation.
        
        Args:
            policy_trajectories: Trajectories from evaluated policy
            baseline_trajectories: Baseline trajectories for comparison
            
        Returns:
            ClinicalResult with all clinical metrics
        """
        # Time in range
        time_in_range = self.compute_time_in_range(policy_trajectories)
        
        # Adverse events
        adverse_event_rate = self.compute_adverse_event_rate(policy_trajectories)
        
        # Health improvements (if baseline available)
        if baseline_trajectories is not None:
            health_improvements = self.compute_health_improvement(
                policy_trajectories, baseline_trajectories
            )
        else:
            health_improvements = {}
        
        # Goal achievement
        goal_achievement_rate = self._compute_goal_achievement(policy_trajectories)
        
        # Clinical significance
        clinical_significance = self._assess_clinical_significance(
            health_improvements, time_in_range
        )
        
        result = ClinicalResult(
            health_improvements=health_improvements,
            time_in_range=time_in_range,
            adverse_event_rate=adverse_event_rate,
            goal_achievement_rate=goal_achievement_rate,
            clinical_significance=clinical_significance,
            metadata={
                'n_trajectories': len(policy_trajectories),
                'n_baseline': len(baseline_trajectories) if baseline_trajectories else 0
            }
        )
        
        self._print_summary(result)
        return result
    
    def _compute_goal_achievement(self, trajectories: List[Dict]) -> float:
        """Compute fraction of episodes achieving clinical goals."""
        goals_achieved = 0
        
        for traj in trajectories:
            if len(traj['states']) == 0:
                continue
            
            final_state = traj['states'][-1]
            
            # Check if all key metrics are in target
            all_in_target = True
            for metric, (min_val, max_val) in self.target_ranges.items():
                if metric in final_state:
                    value = final_state[metric]
                    if not (min_val <= value <= max_val):
                        all_in_target = False
                        break
            
            if all_in_target:
                goals_achieved += 1
        
        if len(trajectories) == 0:
            return 0.0
        
        return goals_achieved / len(trajectories)
    
    def _assess_clinical_significance(
        self,
        improvements: Dict[str, float],
        time_in_range: Dict[str, float]
    ) -> Dict[str, bool]:
        """Assess if improvements are clinically significant."""
        significance = {}
        
        # Clinical significance thresholds
        thresholds = {
            'glucose': 0.10,  # 10% improvement
            'bp_systolic': 0.05,  # 5% improvement
            'hba1c': 0.10  # 10% improvement
        }
        
        for metric, improvement in improvements.items():
            threshold = thresholds.get(metric, 0.05)
            significance[metric] = improvement >= threshold
        
        return significance
    
    def _print_summary(self, result: ClinicalResult) -> None:
        """Print clinical evaluation summary."""
        print("\n" + "="*70)
        print("CLINICAL EVALUATION SUMMARY")
        print("="*70)
        
        print("\nHealth Improvements:")
        for metric, improvement in result.health_improvements.items():
            sig = "***" if result.clinical_significance.get(metric, False) else ""
            print(f"  {metric:<15}: {improvement:+.3f} {sig}")
        
        print("\nTime in Target Range:")
        for metric, tir in result.time_in_range.items():
            print(f"  {metric:<15}: {tir:.3f}")
        
        print(f"\nAdverse Event Rate:    {result.adverse_event_rate:.3f}")
        print(f"Goal Achievement Rate: {result.goal_achievement_rate:.3f}")
        print("="*70 + "\n")
