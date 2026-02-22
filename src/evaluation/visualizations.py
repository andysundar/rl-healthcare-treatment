"""
Visualization Utilities for Evaluation Results

Provides plotting functions for:
1. Policy comparison charts
2. Safety violation heatmaps
3. Health metric trajectories
4. Learning curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EvaluationVisualizer:
    """Visualization tools for RL evaluation results."""
    
    def __init__(self, config):
        self.config = config
        
    def plot_comparison(
        self,
        comparison_df,
        metrics: Optional[List[str]] = None,
        save_path: str = None
    ) -> None:
        """
        Create comparison bar chart for multiple policies.
        
        Args:
            comparison_df: DataFrame with policy comparison results
            metrics: List of metrics to plot
            save_path: Path to save figure
        """
        if metrics is None:
            metrics = list(comparison_df.columns)
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax = axes[idx]
                comparison_df[metric].plot(kind='bar', ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        plt.show()
    
    def plot_safety_violations(
        self,
        trajectories: List[Dict],
        variables: List[str] = None,
        save_path: str = None
    ) -> None:
        """
        Visualize safety violations over time.
        
        Args:
            trajectories: Episode trajectories
            variables: Variables to plot
            save_path: Path to save figure
        """
        if variables is None:
            variables = ['glucose', 'bp_systolic']
        
        fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4*len(variables)))
        
        if len(variables) == 1:
            axes = [axes]
        
        for idx, var in enumerate(variables):
            ax = axes[idx]
            
            # Plot all trajectories
            for traj in trajectories[:50]:  # Plot first 50 trajectories
                values = [s.get(var, np.nan) for s in traj['states']]
                ax.plot(values, alpha=0.3, color='blue', linewidth=0.5)
            
            # Add safe range
            if var == 'glucose':
                ax.axhspan(70, 180, alpha=0.2, color='green', label='Safe Range')
                ax.axhline(y=54, color='red', linestyle='--', label='Critical Low')
                ax.axhline(y=250, color='red', linestyle='--', label='Critical High')
            elif var == 'bp_systolic':
                ax.axhspan(90, 140, alpha=0.2, color='green', label='Safe Range')
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(var.replace('_', ' ').title())
            ax.set_title(f'{var} Trajectory Visualization')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved safety violations plot to {save_path}")
        
        plt.show()
    
    def plot_health_metrics(
        self,
        trajectories: List[Dict],
        metrics: List[str],
        save_path: str = None
    ) -> None:
        """
        Plot health metric trajectories.
        
        Args:
            trajectories: Episode trajectories
            metrics: Metrics to plot
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Aggregate trajectories
            all_values = []
            for traj in trajectories:
                values = [s.get(metric, np.nan) for s in traj['states']]
                if values:
                    all_values.append(values)
            
            if all_values:
                # Compute mean and std
                max_len = max(len(v) for v in all_values)
                padded = [v + [np.nan]*(max_len - len(v)) for v in all_values]
                arr = np.array(padded)
                
                mean_vals = np.nanmean(arr, axis=0)
                std_vals = np.nanstd(arr, axis=0)
                
                # Plot
                x = np.arange(len(mean_vals))
                ax.plot(x, mean_vals, label='Mean', color='blue', linewidth=2)
                ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                               alpha=0.3, color='blue', label='+/- 1 std')
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved health metrics plot to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(
        self,
        training_history: Dict[str, List[float]],
        save_path: str = None
    ) -> None:
        """
        Plot learning curves from training history.
        
        Args:
            training_history: Dict of {metric_name: [values over time]}
            save_path: Path to save figure
        """
        n_metrics = len(training_history)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(training_history.items()):
            ax = axes[idx]
            ax.plot(values, linewidth=2)
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name} Learning Curve')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curves to {save_path}")
        
        plt.show()
    
    def create_evaluation_report(
        self,
        results: Dict,
        save_path: str = 'evaluation_report.pdf'
    ) -> None:
        """
        Create comprehensive evaluation report with all visualizations.
        
        Args:
            results: Dictionary containing all evaluation results
            save_path: Path to save PDF report
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(save_path) as pdf:
            # Page 1: Policy comparison
            if 'comparison' in results:
                self.plot_comparison(results['comparison'], save_path=None)
                pdf.savefig()
                plt.close()
            
            # Page 2: Safety violations
            if 'trajectories' in results:
                self.plot_safety_violations(results['trajectories'], save_path=None)
                pdf.savefig()
                plt.close()
            
            # Page 3: Health metrics
            if 'trajectories' in results:
                self.plot_health_metrics(
                    results['trajectories'],
                    ['glucose', 'bp_systolic'],
                    save_path=None
                )
                pdf.savefig()
                plt.close()
        
        logger.info(f"Saved evaluation report to {save_path}")
