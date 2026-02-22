"""
Policy Comparison Utilities

Compare multiple RL policies with statistical significance testing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class PolicyComparator:
    """Compare multiple policies on evaluation metrics."""
    
    def __init__(self, config):
        self.config = config
        self.significance_level = 0.05
        
    def compare_policies(
        self,
        policies_dict: Dict[str, any],
        test_data: List[Dict],
        evaluators: List[Callable]
    ) -> pd.DataFrame:
        """
        Compare multiple policies on test data.
        
        Args:
            policies_dict: {'policy_name': policy_object}
            test_data: Test trajectories
            evaluators: List of evaluator functions
            
        Returns:
            DataFrame with comparison results
        """
        results = {}
        
        for policy_name, policy in policies_dict.items():
            logger.info(f"Evaluating policy: {policy_name}")
            
            # Generate trajectories with this policy
            policy_trajectories = self._generate_trajectories(policy, test_data)
            
            # Evaluate with all evaluators
            metrics_dict = {}
            for evaluator in evaluators:
                result = evaluator(policy_trajectories)
                metrics_dict.update(self._extract_metrics(result))
            
            results[policy_name] = metrics_dict
        
        # Create comparison DataFrame
        df = pd.DataFrame(results).T
        
        # Add statistical significance tests
        self._add_significance_tests(df, policies_dict, test_data)
        
        return df
    
    def _generate_trajectories(self, policy, test_data):
        """Generate trajectories using policy on test data."""
        # Placeholder - actual implementation would use the policy
        # to generate new trajectories
        return test_data
    
    def _extract_metrics(self, result) -> Dict[str, float]:
        """Extract numeric metrics from evaluation result."""
        metrics = {}
        
        # Handle different result types
        if hasattr(result, 'value_estimate'):
            metrics['value'] = result.value_estimate
        if hasattr(result, 'safety_index'):
            metrics['safety_index'] = result.safety_index
        if hasattr(result, 'average_return'):
            metrics['avg_return'] = result.average_return
        if hasattr(result, 'adverse_event_rate'):
            metrics['adverse_events'] = result.adverse_event_rate
        
        return metrics
    
    def _add_significance_tests(
        self,
        df: pd.DataFrame,
        policies_dict: Dict,
        test_data: List[Dict]
    ) -> None:
        """Add statistical significance tests to comparison."""
        # For now, just log that tests would be performed
        logger.info("Statistical significance tests would be performed here")
        
    def pairwise_comparison(
        self,
        policy1_results: List[float],
        policy2_results: List[float],
        test: str = 'ttest'
    ) -> Tuple[float, float]:
        """
        Perform pairwise statistical test.
        
        Args:
            policy1_results: Results from policy 1
            policy2_results: Results from policy 2
            test: Statistical test to use
            
        Returns:
            (test_statistic, p_value)
        """
        if test == 'ttest':
            statistic, p_value = stats.ttest_ind(policy1_results, policy2_results)
        elif test == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(policy1_results, policy2_results)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        is_significant = p_value < self.significance_level
        logger.info(f"Pairwise comparison: p={p_value:.4f}, significant={is_significant}")
        
        return statistic, p_value
    
    def rank_policies(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'avg_return',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank policies by a specific metric.
        
        Returns:
            DataFrame sorted by metric
        """
        if metric not in comparison_df.columns:
            logger.warning(f"Metric {metric} not found in results")
            return comparison_df
        
        ranked = comparison_df.sort_values(by=metric, ascending=ascending)
        logger.info(f"Policy ranking by {metric}:")
        for idx, (name, row) in enumerate(ranked.iterrows(), 1):
            logger.info(f"  {idx}. {name}: {row[metric]:.3f}")
        
        return ranked
    
    def create_comparison_table(
        self,
        comparison_df: pd.DataFrame,
        save_path: str = None
    ) -> str:
        """
        Create formatted comparison table.
        
        Returns:
            Formatted table string
        """
        table = "\n" + "="*80 + "\n"
        table += "POLICY COMPARISON\n"
        table += "="*80 + "\n"
        table += comparison_df.to_string() + "\n"
        table += "="*80 + "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(table)
        
        return table
