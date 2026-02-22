"""
Performance Metrics for RL Evaluation

Standard RL performance metrics including:
1. Average Return
2. Success Rate  
3. Episode Length
4. Convergence Metrics
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    """Results from performance evaluation."""
    average_return: float
    std_return: float
    success_rate: float
    average_episode_length: float
    median_return: float
    min_return: float
    max_return: float
    metadata: Dict = None


class PerformanceEvaluator:
    """Standard RL performance evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.success_thresholds = config.performance.success_thresholds
        
    def compute_average_return(
        self,
        trajectories: List[Dict],
        gamma: float = 0.99
    ) -> Tuple[float, float]:
        """
        Compute average discounted return.
        
        Returns:
            (mean_return, std_return)
        """
        returns = []
        
        for traj in trajectories:
            discounted_return = 0.0
            for t, reward in enumerate(traj['rewards']):
                discounted_return += (gamma ** t) * reward
            returns.append(discounted_return)
        
        mean_return = np.mean(returns) if returns else 0.0
        std_return = np.std(returns) if returns else 0.0
        
        logger.info(f"Average return: {mean_return:.3f} +/- {std_return:.3f}")
        return mean_return, std_return
    
    def compute_success_rate(
        self,
        trajectories: List[Dict],
        success_criterion: str = 'glucose_control'
    ) -> float:
        """
        Compute success rate based on criterion.
        
        Args:
            trajectories: Episode trajectories
            success_criterion: Success criterion to use
            
        Returns:
            Success rate in [0, 1]
        """
        if success_criterion not in self.success_thresholds:
            logger.warning(f"Unknown criterion: {success_criterion}, returning 0")
            return 0.0
        
        threshold = self.success_thresholds[success_criterion]
        successes = 0
        
        for traj in trajectories:
            if self._check_success(traj, success_criterion, threshold):
                successes += 1
        
        if len(trajectories) == 0:
            return 0.0
        
        success_rate = successes / len(trajectories)
        logger.info(f"Success rate ({success_criterion}): {success_rate:.3f}")
        
        return success_rate
    
    def _check_success(
        self,
        trajectory: Dict,
        criterion: str,
        threshold: float
    ) -> bool:
        """Check if trajectory meets success criterion."""
        if criterion == 'glucose_control':
            # Check glucose time in range
            in_range_count = 0
            total_count = 0
            
            for state in trajectory['states']:
                if 'glucose' in state:
                    total_count += 1
                    glucose = state['glucose']
                    if 80 <= glucose <= 130:  # Target range
                        in_range_count += 1
            
            if total_count > 0:
                time_in_range = in_range_count / total_count
                return time_in_range >= threshold
        
        return False
    
    def compute_episode_lengths(
        self,
        trajectories: List[Dict]
    ) -> Tuple[float, float]:
        """
        Compute episode length statistics.
        
        Returns:
            (mean_length, std_length)
        """
        lengths = [len(traj['states']) for traj in trajectories]
        
        mean_length = np.mean(lengths) if lengths else 0.0
        std_length = np.std(lengths) if lengths else 0.0
        
        logger.info(f"Average episode length: {mean_length:.1f} +/- {std_length:.1f}")
        return mean_length, std_length
    
    def evaluate(
        self,
        trajectories: List[Dict],
        gamma: float = 0.99
    ) -> PerformanceResult:
        """
        Comprehensive performance evaluation.
        
        Returns:
            PerformanceResult with all performance metrics
        """
        # Compute returns
        mean_return, std_return = self.compute_average_return(trajectories, gamma)
        
        # Compute success rate
        success_rate = self.compute_success_rate(trajectories)
        
        # Episode lengths
        mean_length, _ = self.compute_episode_lengths(trajectories)
        
        # Additional statistics
        returns = []
        for traj in trajectories:
            discounted_return = sum(
                (gamma ** t) * r for t, r in enumerate(traj['rewards'])
            )
            returns.append(discounted_return)
        
        median_return = np.median(returns) if returns else 0.0
        min_return = np.min(returns) if returns else 0.0
        max_return = np.max(returns) if returns else 0.0
        
        result = PerformanceResult(
            average_return=mean_return,
            std_return=std_return,
            success_rate=success_rate,
            average_episode_length=mean_length,
            median_return=median_return,
            min_return=min_return,
            max_return=max_return,
            metadata={'n_trajectories': len(trajectories), 'gamma': gamma}
        )
        
        self._print_summary(result)
        return result
    
    def _print_summary(self, result: PerformanceResult) -> None:
        """Print performance evaluation summary."""
        print("\n" + "="*70)
        print("PERFORMANCE EVALUATION SUMMARY")
        print("="*70)
        print(f"Average Return:  {result.average_return:.3f} +/- {result.std_return:.3f}")
        print(f"Median Return:   {result.median_return:.3f}")
        print(f"Return Range:    [{result.min_return:.3f}, {result.max_return:.3f}]")
        print(f"Success Rate:    {result.success_rate:.3f}")
        print(f"Avg Episode Len: {result.average_episode_length:.1f}")
        print("="*70 + "\n")
