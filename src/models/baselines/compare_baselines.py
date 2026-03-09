"""
Baseline comparison framework.

This module provides utilities for comparing multiple baseline policies
on the same test data and generating comprehensive comparison reports.
"""

from typing import Dict, List, Any, Optional, Callable, Sequence
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from dataclasses import asdict
import time
import hashlib

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
        self.results_df: Optional[pd.DataFrame] = None
        self.custom_metrics: Dict[str, Callable] = {}
        self._evaluation_computed = False
        self.last_eval_signature: Optional[str] = None
    
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
        self._evaluation_computed = False
        self.results = {}
        self.results_df = None
        logger.info(f"Set test data with {len(test_data)} samples")

    def _compute_signature(
        self,
        baseline_names: Sequence[str],
        n_samples: int,
        fast_eval: bool,
        max_eval_samples: Optional[int],
    ) -> str:
        payload = {
            'baselines': list(baseline_names),
            'n_samples': int(n_samples),
            'fast_eval': bool(fast_eval),
            'max_eval_samples': int(max_eval_samples) if max_eval_samples is not None else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()

    def _subsample_test_data(
        self,
        max_eval_samples: Optional[int],
        seed: int = 42,
    ) -> List:
        if self.test_data is None:
            return []
        if max_eval_samples is None or max_eval_samples <= 0 or len(self.test_data) <= max_eval_samples:
            return self.test_data
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self.test_data), size=max_eval_samples, replace=False))
        subset = [self.test_data[i] for i in idx]
        logger.info(
            "Fast eval enabled in BaselineComparator: reducing samples from %s to %s (seed=%s).",
            len(self.test_data), len(subset), seed,
        )
        return subset

    @staticmethod
    def _select_baseline_names(
        all_names: Sequence[str],
        include_baselines: Optional[Sequence[str]] = None,
        exclude_baselines: Optional[Sequence[str]] = None,
        skip_slow_baselines: bool = False,
        slow_names: Optional[Sequence[str]] = None,
        max_rollout_policies: Optional[int] = None,
    ) -> List[str]:
        selected = list(all_names)
        if include_baselines:
            unknown = sorted(set(include_baselines) - set(all_names))
            if unknown:
                raise ValueError(f"Unknown baseline names in include_baselines: {unknown}")
            selected = [n for n in selected if n in include_baselines]
        if exclude_baselines:
            unknown = sorted(set(exclude_baselines) - set(all_names))
            if unknown:
                raise ValueError(f"Unknown baseline names in exclude_baselines: {unknown}")
            selected = [n for n in selected if n not in set(exclude_baselines)]
        if skip_slow_baselines:
            slow = set(slow_names or ['knn', 'behavior_cloning', 'KNN-5', 'Behavior-Cloning'])
            selected = [n for n in selected if n not in slow]
        if max_rollout_policies is not None and max_rollout_policies > 0:
            selected = selected[:max_rollout_policies]
        return selected

    @staticmethod
    def _load_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _save_cache(cache_path: Path, payload: Dict[str, Any]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(payload, f, indent=2)

    def evaluate_all(
        self,
        verbose: bool = True,
        force_recompute: bool = False,
        fast_eval: bool = False,
        max_eval_samples: Optional[int] = None,
        include_baselines: Optional[Sequence[str]] = None,
        exclude_baselines: Optional[Sequence[str]] = None,
        skip_slow_baselines: bool = False,
        slow_baselines: Optional[Sequence[str]] = None,
        max_rollout_policies: Optional[int] = None,
        cache_path: Optional[str] = None,
        force_recompute_cache: bool = False,
    ) -> pd.DataFrame:
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
        
        selected_names = self._select_baseline_names(
            list(self.baselines.keys()),
            include_baselines=include_baselines,
            exclude_baselines=exclude_baselines,
            skip_slow_baselines=skip_slow_baselines,
            slow_names=slow_baselines,
            max_rollout_policies=max_rollout_policies,
        )
        if not selected_names:
            raise ValueError("No baselines selected after include/exclude filtering.")

        eval_data = self._subsample_test_data(max_eval_samples if fast_eval else None)
        signature = self._compute_signature(
            baseline_names=selected_names,
            n_samples=len(eval_data),
            fast_eval=fast_eval,
            max_eval_samples=max_eval_samples if fast_eval else None,
        )

        if self._evaluation_computed and not force_recompute and self.last_eval_signature == signature and self.results_df is not None:
            logger.info("Using cached baseline evaluation results")
            return self.results_df.copy()

        cache_file = Path(cache_path) if cache_path else None
        if cache_file is not None and not force_recompute_cache:
            payload = self._load_cache(cache_file)
            if payload and payload.get('signature') == signature:
                logger.info("Found cached baseline evaluation results at %s", cache_file)
                df = pd.DataFrame(payload.get('results_table', {}))
                if not df.empty and 'policy' in df.columns:
                    df = df.set_index('policy')
                self.results_df = df
                self._evaluation_computed = True
                self.last_eval_signature = signature
                return df.copy()
            if payload:
                logger.info("Cache miss: evaluation parameters changed")

        logger.info("Computing baseline evaluation results")
        results = {}
        t0_all = time.perf_counter()
        logger.info("Baseline evaluation start: baselines=%s, samples=%s", len(selected_names), len(eval_data))

        for name in selected_names:
            baseline = self.baselines[name]
            if verbose:
                logger.info("Evaluating baseline: %s (samples=%s)", name, len(eval_data))
            t0 = time.perf_counter()
            
            try:
                if not hasattr(baseline, 'evaluate') or not hasattr(baseline, 'select_action'):
                    raise TypeError(f"Baseline {name} does not implement required BaselinePolicy interface")
                metrics = baseline.evaluate(eval_data)
                
                # Convert to dict
                result_dict = asdict(metrics)
                
                # Add custom metrics
                for metric_name, metric_fn in self.custom_metrics.items():
                    try:
                        result_dict[metric_name] = metric_fn(baseline, eval_data)
                    except Exception as e:
                        logger.warning(f"Error computing {metric_name} for {name}: {e}")
                        result_dict[metric_name] = np.nan
                
                results[name] = result_dict
                self.results[name] = metrics
                
                if verbose:
                    logger.info(f"  Mean Reward: {metrics.mean_reward:.4f}")
                    logger.info(f"  Safety Rate: {metrics.safety_rate:.4f}")
                logger.info("Completed baseline: %s in %.2fs", name, time.perf_counter() - t0)
            
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                results[name] = {
                    'error': str(e)
                }
                logger.info("Completed baseline: %s in %.2fs (with error)", name, time.perf_counter() - t0)
        
        # Create comparison DataFrame
        df = pd.DataFrame(results).T
        
        # Sort by mean reward (descending)
        if 'mean_reward' in df.columns:
            df = df.sort_values('mean_reward', ascending=False)
        elapsed = time.perf_counter() - t0_all
        logger.info("Completed all baseline evaluations in %.2fs", elapsed)

        self.results_df = df.copy()
        self._evaluation_computed = True
        self.last_eval_signature = signature

        if cache_file is not None:
            self._save_cache(cache_file, {
                'signature': signature,
                'results_table': df.reset_index().rename(columns={'index': 'policy'}).to_dict(orient='list'),
            })
            logger.info("Saved baseline evaluation cache to %s", cache_file)

        return df.copy()
    
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
                       include_plots: bool = True,
                       force_recompute: bool = False,
                       **eval_kwargs) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_path: Path to save report (None for return only)
            include_plots: Whether to include plots
            
        Returns:
            Report as markdown string
        """
        if self.results_df is None and not self._evaluation_computed:
            logger.info("Computing baseline evaluation results")
            self.evaluate_all(verbose=False, force_recompute=force_recompute, **eval_kwargs)
        elif self._evaluation_computed:
            logger.info("Using cached baseline evaluation results")
        
        report_lines = []
        report_lines.append("# Baseline Policy Comparison Report\n")
        
        # Summary statistics
        report_lines.append("## Summary Statistics\n")
        df = self.results_df.copy() if self.results_df is not None else self.evaluate_all(
            verbose=False,
            force_recompute=force_recompute,
            **eval_kwargs,
        )
        report_lines.append(df.to_markdown())
        report_lines.append("\n")
        
        # Best baseline
        if self.results:
            best_name, _, best_score = self.get_best_baseline()
        else:
            best_name = str(df['mean_reward'].astype(float).idxmax()) if 'mean_reward' in df.columns else str(df.index[0])
            best_score = float(df.loc[best_name, 'mean_reward']) if 'mean_reward' in df.columns else float('nan')
        report_lines.append(f"## Best Baseline: {best_name}\n")
        report_lines.append(f"Mean Reward: {best_score:.4f}\n\n")
        
        # Individual baseline details
        report_lines.append("## Individual Baseline Details\n")
        for name in df.index.tolist():
            baseline = self.baselines[name]
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
                    (
                        {name: asdict(metrics) for name, metrics in self.results.items()}
                        if self.results else
                        df.reset_index().rename(columns={'index': 'policy'}).to_dict(orient='records')
                    ),
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
        df = self.results_df.copy() if self.results_df is not None else self.evaluate_all(verbose=False)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file)
        logger.info(f"Exported results to {output_file}")


def compare_all_baselines(test_data: List,
                         baselines_dict: Dict[str, BaselinePolicy],
                         custom_metrics: Optional[Dict[str, Callable]] = None,
                         output_path: Optional[str] = None,
                         **eval_kwargs) -> pd.DataFrame:
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
    results = comparator.evaluate_all(verbose=True, **eval_kwargs)
    
    # Generate report if path provided
    if output_path:
        comparator.generate_report(output_path, **eval_kwargs)
    
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
