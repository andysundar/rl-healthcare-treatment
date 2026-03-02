"""
Baseline comparison framework.

This module provides utilities for comparing multiple baseline policies
on the same test data and generating comprehensive comparison reports.
"""

from typing import Dict, List, Any, Optional, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from dataclasses import asdict

from .base_baseline import BaselinePolicy, BaselineMetrics

logger = logging.getLogger(__name__)


class BaselineComparator:
    """
    Framework for comparing multiple baseline policies.
    
    Attributes:
        baselines: Dictionary mapping names to baseline policies
        test_data: Test dataset for evaluation
        results: Dictionary storing evaluation results
    """
    
    def __init__(self, test_data: Optional[List] = None):
        """
        Initialize comparator.
        
        Args:
            test_data: Test dataset (list of tuples)
        """
        self.baselines: Dict[str, BaselinePolicy] = {}
        self.test_data = test_data
        self.results: Dict[str, BaselineMetrics] = {}
        self.custom_metrics: Dict[str, Callable] = {}
    
    def add_baseline(self, name: str, baseline: BaselinePolicy):
        """
        Add a baseline to compare.
        
        Args:
            name: Unique name for the baseline
            baseline: Baseline policy instance
        """
        if name in self.baselines:
            logger.warning(f"Baseline '{name}' already exists, overwriting")
        
        self.baselines[name] = baseline
        logger.info(f"Added baseline: {name}")
    
    def add_custom_metric(self, name: str, metric_fn: Callable):
        """
        Add a custom metric function.
        
        Args:
            name: Metric name
            metric_fn: Function(policy, test_data) -> float
        """
        self.custom_metrics[name] = metric_fn
        logger.info(f"Added custom metric: {name}")
    
    def set_test_data(self, test_data: List):
        """
        Set test dataset.
        
        Args:
            test_data: Test dataset
        """
        self.test_data = test_data
        logger.info(f"Set test data with {len(test_data)} samples")
    
    def evaluate_all(self, verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate all baselines on test data.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            DataFrame with comparison results
        """
        if self.test_data is None or len(self.test_data) == 0:
            raise ValueError("Test data not set or empty. Call set_test_data() with transitions.")
        
        if not self.baselines:
            raise ValueError("No baselines added. Call add_baseline() first.")
        
        results = {}
        
        for name, baseline in self.baselines.items():
            if verbose:
                logger.info(f"Evaluating {name}...")
            
            try:
                if not hasattr(baseline, 'evaluate') or not hasattr(baseline, 'select_action'):
                    raise TypeError(f"Baseline {name} does not implement required BaselinePolicy interface")
                metrics = baseline.evaluate(self.test_data)
                
                # Convert to dict
                result_dict = asdict(metrics)
                
                # Add custom metrics
                for metric_name, metric_fn in self.custom_metrics.items():
                    try:
                        result_dict[metric_name] = metric_fn(baseline, self.test_data)
                    except Exception as e:
                        logger.warning(f"Error computing {metric_name} for {name}: {e}")
                        result_dict[metric_name] = np.nan
                
                results[name] = result_dict
                self.results[name] = metrics
                
                if verbose:
                    logger.info(f"  Mean Reward: {metrics.mean_reward:.4f}")
                    logger.info(f"  Safety Rate: {metrics.safety_rate:.4f}")
            
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                results[name] = {
                    'error': str(e)
                }
        
        # Create comparison DataFrame
        df = pd.DataFrame(results).T
        
        # Sort by mean reward (descending)
        if 'mean_reward' in df.columns:
            df = df.sort_values('mean_reward', ascending=False)
        
        return df
    
    def get_best_baseline(self, metric: str = 'mean_reward') -> tuple:
        """
        Get best baseline according to a metric.
        
        Args:
            metric: Metric to use for ranking
            
        Returns:
            Tuple of (name, baseline, score)
        """
        if not self.results:
            raise ValueError("No evaluation results. Call evaluate_all() first.")
        
        best_name = None
        best_score = -float('inf')
        
        for name, metrics in self.results.items():
            score = getattr(metrics, metric, -float('inf'))
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_name is None:
            logger.warning(f"No valid results found for metric '{metric}'.")
            raise ValueError(f"No valid results found for metric '{metric}'.")
        
        return best_name, self.baselines[best_name], best_score
    
    def compare_against_rl(self, 
                          rl_results: Dict[str, float],
                          rl_name: str = "RL-Policy") -> pd.DataFrame:
        """
        Compare baselines against RL policy results.
        
        Args:
            rl_results: Dictionary of RL policy metrics
            rl_name: Name for RL policy in comparison
            
        Returns:
            DataFrame with baselines and RL policy
        """
        # Get baseline results
        baseline_df = self.evaluate_all(verbose=False)
        
        # Add RL results as a row
        rl_series = pd.Series(rl_results, name=rl_name)
        
        # Combine
        comparison_df = pd.concat([baseline_df, rl_series.to_frame().T])
        
        # Highlight best values
        if 'mean_reward' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('mean_reward', ascending=False)
        
        return comparison_df
    
    def generate_report(self, 
                       output_path: Optional[str] = None,
                       include_plots: bool = True) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_path: Path to save report (None for return only)
            include_plots: Whether to include plots
            
        Returns:
            Report as markdown string
        """
        if not self.results:
            logger.warning("No evaluation results to report.")
            raise ValueError("No evaluation results. Call evaluate_all() first.")
        
        report_lines = []
        report_lines.append("# Baseline Policy Comparison Report\n")
        
        # Summary statistics
        report_lines.append("## Summary Statistics\n")
        df = self.evaluate_all(verbose=False)
        report_lines.append(df.to_markdown())
        report_lines.append("\n")
        
        # Best baseline
        best_name, _, best_score = self.get_best_baseline()
        report_lines.append(f"## Best Baseline: {best_name}\n")
        report_lines.append(f"Mean Reward: {best_score:.4f}\n\n")
        
        # Individual baseline details
        report_lines.append("## Individual Baseline Details\n")
        for name, baseline in self.baselines.items():
            report_lines.append(f"### {name}\n")
            info = baseline.get_info()
            
            # Format info as list
            for key, value in info.items():
                if key not in ['network_architecture', 'rules']:  # Skip verbose fields
                    report_lines.append(f"- **{key}**: {value}\n")
            
            report_lines.append("\n")
        
        # Performance comparison
        report_lines.append("## Performance Comparison\n")
        
        # Rank by different metrics
        metrics_to_rank = ['mean_reward', 'safety_rate', 'mean_action_value']
        
        for metric in metrics_to_rank:
            if metric in df.columns:
                report_lines.append(f"### Ranked by {metric}\n")
                ranked = df.sort_values(metric, ascending=False)
                report_lines.append(ranked[[metric]].to_markdown())
                report_lines.append("\n")
        
        # Compile report
        report = '\n'.join(report_lines)
        
        # Save if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Saved report to {output_file}")
            
            # Save raw results as JSON
            json_path = output_file.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(
                    {name: asdict(metrics) for name, metrics in self.results.items()},
                    f,
                    indent=2
                )
            logger.info(f"Saved raw results to {json_path}")
        
        return report
    
    def export_results(self, output_path: str):
        """
        Export results to CSV.
        
        Args:
            output_path: Path to save CSV file
        """
        df = self.evaluate_all(verbose=False)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file)
        logger.info(f"Exported results to {output_file}")


def compare_all_baselines(test_data: List,
                         baselines_dict: Dict[str, BaselinePolicy],
                         custom_metrics: Optional[Dict[str, Callable]] = None,
                         output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to compare multiple baselines.
    
    Args:
        test_data: Test dataset
        baselines_dict: Dictionary mapping names to baseline policies
        custom_metrics: Optional custom metric functions
        output_path: Optional path to save report
        
    Returns:
        DataFrame with comparison results
    """
    comparator = BaselineComparator(test_data)
    
    # Add all baselines
    for name, baseline in baselines_dict.items():
        comparator.add_baseline(name, baseline)
    
    # Add custom metrics if provided
    if custom_metrics:
        for name, metric_fn in custom_metrics.items():
            comparator.add_custom_metric(name, metric_fn)
    
    # Evaluate
    results = comparator.evaluate_all(verbose=True)
    
    # Generate report if path provided
    if output_path:
        comparator.generate_report(output_path)
    
    return results


# Useful custom metrics
def compute_action_stability(policy: BaselinePolicy, test_data: List) -> float:
    """
    Compute action stability (variance of actions).
    
    Lower variance = more stable policy.
    
    Args:
        policy: Baseline policy
        test_data: Test data
        
    Returns:
        Action variance
    """
    actions = []
    for state, _, _, _, _ in test_data:
        action = policy.select_action(state)
        actions.append(action)
    
    return float(np.var(actions))


def compute_safety_margin(policy: BaselinePolicy, 
                         test_data: List,
                         safety_threshold: float = 0.95) -> float:
    """
    Compute safety margin (how far above safety threshold).
    
    Args:
        policy: Baseline policy
        test_data: Test data
        safety_threshold: Target safety rate
        
    Returns:
        Safety margin (positive = above threshold)
    """
    metrics = policy.evaluate(test_data)
    return metrics.safety_rate - safety_threshold


def compute_expected_return(policy: BaselinePolicy,
                           test_data: List,
                           gamma: float = 0.99) -> float:
    """
    Compute discounted expected return.
    
    Args:
        policy: Baseline policy
        test_data: Test data
        gamma: Discount factor
        
    Returns:
        Expected discounted return
    """
    total_return = 0.0
    
    for t, (state, _, reward, _, _) in enumerate(test_data):
        total_return += (gamma ** t) * reward
    
    return total_return / len(test_data)
