#!/usr/bin/env python3
"""
Integrated Solution Runner for RL Healthcare Treatment Project

This script runs the complete end-to-end pipeline including:
1. Data preparation (MIMIC-III or synthetic)
2. Environment setup
3. Baseline training and evaluation
4. CQL training
5. Off-policy evaluation
6. Results generation

Author: Anindya Bandopadhyay (M23CSA508)
Supervisor: Dr. Pradip Sasmal, IIT Jodhpur
Date: February 2026
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch

from rewards.health_reward import HealthOutcomeReward
from rewards.composite_reward import CompositeRewardFunction
from rewards.safety_reward import SafetyPenalty
from rewards.reward_config import RewardConfig

from environments import DiabetesEnv, AdherenceEnv
from rewards import (
    CompositeRewardFunction, RewardConfig,
    AdherenceReward, HealthOutcomeReward, SafetyPenalty, CostEffectivenessReward
)

from evaluation import (
    OffPolicyEvaluator,
    SafetyEvaluator,
    ClinicalEvaluator
)

from data import SyntheticDataGenerator

from models.rl import CQLConfig, CQLAgent


from models.baselines import (
            create_diabetes_rule_policy,
            create_random_policy,
            create_safe_random_policy,
            create_mean_action_policy,
            create_regression_policy,
            create_knn_policy,
            create_behavior_cloning_policy,
            compare_all_baselines
        )

from environments.diabetes_env import DiabetesEnvConfig
from environments.adherence_env import AdherenceEnvConfig


# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntegratedSolutionRunner:
    """Main class for running the integrated RL healthcare solution."""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("RL HEALTHCARE TREATMENT - INTEGRATED SOLUTION")
        logger.info("=" * 80)
        logger.info(f"Mode: {config.mode}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.get_device()}")
        
    def get_device(self):
        """Get available device (MPS/CUDA/CPU)."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def run_full_pipeline(self):
        """Run complete end-to-end pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 80)
        
        # Stage 1: Data Preparation
        if self.config.mode in ['full', 'data-only']:
            self.stage_1_data_preparation()
        
        # Stage 2: Environment Setup
        if self.config.mode in ['full', 'train-eval', 'synthetic']:
            self.stage_2_environment_setup()
        
        # Stage 3: Baseline Training
        if self.config.mode in ['full', 'train-eval', 'synthetic']:
            self.stage_3_baseline_training()
        
        # Stage 4: CQL Training
        if self.config.mode in ['full', 'train-eval']:
            self.stage_4_cql_training()
        
        # Stage 5: Evaluation
        if self.config.mode in ['full', 'train-eval', 'eval-only']:
            self.stage_5_evaluation()
        
        # Stage 6: Generate Reports
        self.stage_6_generate_reports()
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 80)
        
        return self.results
    
    def stage_1_data_preparation(self):
        """Stage 1: Prepare data from MIMIC-III or synthetic sources."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: DATA PREPARATION")
        logger.info("=" * 80)
        
        try:
            if self.config.use_synthetic or self.config.mode == 'synthetic':
                data = self.prepare_synthetic_data()
            else:
                data = self.prepare_mimic_data()
            
            self.results['data'] = data
            logger.info("✓ Data preparation complete")
            
        except Exception as e:
            logger.error(f"✗ Data preparation failed: {e}")
            if self.config.mode != 'synthetic':
                logger.info("Falling back to synthetic data...")
                data = self.prepare_synthetic_data()
                self.results['data'] = data
            else:
                raise
    
    def prepare_synthetic_data(self):
        """Generate synthetic patient data."""
        logger.info("Generating synthetic diabetes patient data...")
        
        
        
        generator = SyntheticDataGenerator(
            random_seed=42
        )
        
        # Generate patient cohort and trajectories
        patients = generator.generate_diabetes_population(
            n_patients=self.config.n_synthetic_patients
        )

        trajectories = []
        for patient in patients:
            patient_trajectories = generator.simulate_patient_trajectory(
                patient=patient,
                time_horizon_days=self.config.trajectory_length,
                treatment_policy=None  # Use default conservative policy
            )
            trajectories.extend(patient_trajectories)
        
       
        
        logger.info(f"✓ Generated {len(patients)} synthetic patients")
        
        logger.info(f"✓ Generated {len(trajectories)} transitions")
        
        # Split into train/val/test
        n_train = int(0.7 * len(trajectories))
        n_val = int(0.15 * len(trajectories))
        
        train_data = trajectories[:n_train]
        val_data = trajectories[n_train:n_train + n_val]
        test_data = trajectories[n_train + n_val:]
        
        data = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'patients': patients,
            'source': 'synthetic'
        }
        
        # Save data
        output_path = self.output_dir / 'synthetic_data.pkl'
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"✓ Saved synthetic data to {output_path}")
        
        return data
    
    def prepare_mimic_data(self):
        """Prepare data from MIMIC-III."""
        logger.info("Loading MIMIC-III data...")
        
        from data import (
            MIMICLoader, CohortBuilder, FeatureEngineer,
            DataPreprocessor, split_train_val_test
        )
        
        # Load MIMIC data
        loader = MIMICLoader(
            data_dir=self.config.mimic_dir,
            use_cache=True
        )
        
        patients = loader.load_patients()
        admissions = loader.load_admissions()
        diagnoses = loader.load_diagnoses_icd()
        
        logger.info(f"✓ Loaded {len(patients):,} patients")
        
        # Build cohort
        builder = CohortBuilder(patients, admissions, diagnoses)
        diabetes_patients = builder.define_diabetes_cohort()
        
        filtered_patients = builder.apply_inclusion_criteria(
            diabetes_patients,
            min_age=18,
            max_age=80,
            min_admissions=2
        )
        
        final_cohort = builder.apply_exclusion_criteria(
            filtered_patients,
            exclude_pregnancy=True,
            exclude_pediatric=True
        )
        
        logger.info(f"✓ Final cohort: {len(final_cohort):,} patients")
        
        # Use sample if requested
        if self.config.use_sample:
            import random
            random.seed(42)
            final_cohort = random.sample(
                final_cohort,
                min(self.config.sample_size, len(final_cohort))
            )
            logger.info(f" Using sample of {len(final_cohort)} patients")
        
        # Load patient data
        labs = loader.load_lab_events(subject_ids=final_cohort)
        prescriptions = loader.load_prescriptions(subject_ids=final_cohort)
        
        # Feature engineering
        engineer = FeatureEngineer()
        
        demographics = engineer.extract_demographics(patients, admissions)
        demographics = demographics[demographics['subject_id'].isin(final_cohort)]
        
        lab_sequence = engineer.extract_lab_sequence(labs, final_cohort)
        temporal_features = engineer.create_temporal_features(
            lab_sequence,
            time_column='charttime'
        )
        
        # Preprocessing
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_missing_values(temporal_features)
        clean_data = preprocessor.handle_outliers(clean_data)
        normalized_data = preprocessor.normalize_labs(clean_data)
        
        # Split data
        train, val, test = split_train_val_test(
            normalized_data,
            ratios=(0.7, 0.15, 0.15),
            random_state=42
        )
        
        data = {
            'train': train,
            'val': val,
            'test': test,
            'demographics': demographics,
            'cohort': final_cohort,
            'source': 'mimic'
        }
        
        logger.info(f"✓ Train: {len(train):,} samples")
        logger.info(f"  Val:   {len(val):,} samples")
        logger.info(f"  Test:  {len(test):,} samples")
        
        return data
    
    def stage_2_environment_setup(self):
        """Stage 2: Set up healthcare environments."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: ENVIRONMENT SETUP")
        logger.info("=" * 80)
    
        
        # Create diabetes environment
        logger.info("Creating diabetes management environment...")
        # If you want custom settings, set them here; otherwise you can just do DiabetesEnv()
        diabetes_cfg = DiabetesEnvConfig(
            # safe defaults; tweak later if you want
            max_steps=288,                 # e.g., 24h with dt=0.0833 (5 min) -> ~288 steps
            dt=1/12,                       # 5 minutes in hours
            patient_variability=0.1,       # modest heterogeneity
            initial_glucose_range=(120.0, 220.0),
            target_glucose_range=(80.0, 140.0),
            max_insulin_dose=10.0
        )

        diabetes_env = DiabetesEnv(config=diabetes_cfg)
        
        # Create adherence environment
        logger.info("Creating medication adherence environment...")
        adherence_cfg = AdherenceEnvConfig()   # uses defaults, including max_steps
        adherence_env = AdherenceEnv(config=adherence_cfg)
        
        # Configure rewards
        logger.info("Configuring composite reward function...")

        # Map runner's "w_adverse" to RewardConfig's "w_safety"
        reward_config = RewardConfig(
          w_adherence=0.3,
          w_health=0.4,
          w_safety=0.2,
          w_cost=0.1
        )

        composite_reward = CompositeRewardFunction(reward_config)
        composite_reward.add_component("adherence", AdherenceReward(reward_config), reward_config.w_adherence)
        composite_reward.add_component("health", HealthOutcomeReward(reward_config), reward_config.w_health)
        composite_reward.add_component("safety", SafetyPenalty(reward_config), reward_config.w_safety)
        composite_reward.add_component("cost", CostEffectivenessReward(reward_config), reward_config.w_cost)
        
        self.results['environments'] = {
            'diabetes_env': diabetes_env,
            'adherence_env': adherence_env,
            'reward_function': composite_reward
        }
        
        logger.info("✓ Environment setup complete")
    
    def stage_3_baseline_training(self):
        """Stage 3: Train and evaluate baseline policies."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: BASELINE TRAINING")
        logger.info("=" * 80)
        
        
        # Get data
        if 'data' in self.results:
            data = self.results['data']
            if data['source'] == 'synthetic':
                test_data = data['test']
                train_data = data['train']
            else:
                # Convert MIMIC data to trajectory format
                test_data = self._convert_to_trajectory_format(data['test'])
                train_data = self._convert_to_trajectory_format(data['train'])
        else:
            # Generate synthetic data for baseline testing
            logger.info("Generating synthetic data for baselines...")
            data = self.prepare_synthetic_data()
            test_data = data['test']
            train_data = data['train']
        
        state_dim = 10
        action_dim = 1
        
        # Extract states and actions for supervised baselines
        train_states = np.array([d[0] for d in train_data])
        train_actions = np.array([d[1] for d in train_data])
        
        logger.info(f"Training data: {len(train_data)} transitions")
        logger.info(f"Test data: {len(test_data)} transitions")
        
        # Create baseline policies
        logger.info("Creating baseline policies...")
        
        baselines = {}
        
        # 1. Rule-based
        logger.info("  1/7 Rule-based policy...")
        baselines['Rule-Based'] = create_diabetes_rule_policy(state_dim, action_dim)
        
        # 2. Random policies
        logger.info("  2/7 Random policy...")
        baselines['Random-Uniform'] = create_random_policy(
            action_dim, state_dim, seed=42, distribution='uniform'
        )
        
        logger.info("  3/7 Safe random policy...")
        baselines['Random-Safe'] = create_safe_random_policy(
            action_dim, state_dim, seed=42, num_samples=10
        )
        
        # 3. Statistical baselines
        logger.info("  4/7 Mean action policy...")
        mean_policy = create_mean_action_policy(action_dim, state_dim)
        mean_policy.fit(train_states, train_actions)
        baselines['Mean-Action'] = mean_policy
        
        logger.info("  5/7 Ridge regression policy...")
        ridge_policy = create_regression_policy(
            state_dim, action_dim, regression_type='ridge', alpha=1.0
        )
        ridge_policy.fit(train_states, train_actions)
        baselines['Ridge-Regression'] = ridge_policy
        
        logger.info("  6/7 KNN policy...")
        knn_policy = create_knn_policy(state_dim, action_dim, k=5)
        knn_policy.fit(train_states, train_actions)
        baselines['KNN-5'] = knn_policy
        
        # 4. Behavior cloning
        logger.info("  7/7 Behavior cloning policy...")
        bc_policy = create_behavior_cloning_policy(
            state_dim, action_dim, hidden_dims=[64, 64], learning_rate=1e-3
        )
        
        val_states = np.array([d[0] for d in data['val']])
        val_actions = np.array([d[1] for d in data['val']])
        
        bc_policy.train(
            train_states, train_actions,
            val_states, val_actions,
            epochs=20, batch_size=128, verbose=False
        )
        baselines['Behavior-Cloning'] = bc_policy
        
        # Evaluate all baselines
        logger.info("\nEvaluating all baselines...")
        
        comparison_path = self.output_dir / 'baseline_comparison_report.md'
        results_df = compare_all_baselines(
            test_data=test_data,
            baselines_dict=baselines,
            output_path=str(comparison_path)
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("BASELINE COMPARISON RESULTS")
        logger.info("=" * 60)
        logger.info("\n" + results_df.to_string())
        
        self.results['baselines'] = {
            'policies': baselines,
            'comparison': results_df,
            'test_data': test_data
        }
        
        logger.info(f"\n✓ Baseline training complete")
        logger.info(f"  Report saved to {comparison_path}")
    
    def stage_4_cql_training(self):
        """Stage 4: Train Conservative Q-Learning agent."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: CQL TRAINING")
        logger.info("=" * 80)
        
        if not self.config.train_cql:
            logger.info("⚠ CQL training skipped (use --train-cql to enable)")
            return
        
        
        
        # Get data
        if 'data' not in self.results:
            logger.warning("No data available, skipping CQL training")
            return
        
        data = self.results['data']
        
        # Configure CQL
        logger.info("Configuring CQL...")
        state_dim = 10
        action_dim = 1
        config = CQLConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            q_lr=3e-4,
            policy_lr=1e-4,
            cql_alpha=5.0,
            gamma=0.99
        )
        
        # Initialize agent
        agent = CQLAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            q_lr=config.q_lr,
            policy_lr=config.policy_lr,
            cql_alpha=config.cql_alpha,
            gamma=config.gamma
        )
        device = self.get_device()
        
        logger.info(f"Training on device: {device}")
        
        # Train offline
        logger.info("Training CQL agent...")
        
        if data['source'] == 'synthetic':
            train_data = data['train']
        else:
            train_data = self._convert_to_trajectory_format(data['train'])
        
        # Training loop (simplified - full implementation in src/models/rl)
        logger.info(f"Training dataset size: {len(train_data)} transitions")
        logger.info("Note: Full CQL training requires src/models/rl module")
        logger.info("This is a placeholder - implement full training in next iteration")
        
        # Save placeholder results
        self.results['cql'] = {
            'config': config,
            'agent': agent,
            'trained': False,
            'note': 'CQL training requires full implementation'
        }
        
        logger.info("✓ CQL setup complete (training placeholder)")
    
    def stage_5_evaluation(self):
        """Stage 5: Comprehensive evaluation."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: EVALUATION")
        logger.info("=" * 80)
        
        
        # Get test data
        if 'baselines' not in self.results:
            logger.warning("No baselines available for evaluation")
            return
        
        test_data = self.results['baselines']['test_data']
        baselines = self.results['baselines']['policies']
        
        # Off-policy evaluation
        logger.info("Running off-policy evaluation...")
        evaluator = OffPolicyEvaluator()
        
        # Use behavior cloning as behavior policy (the policy that generated trajectories)
        behavior_policy = baselines.get('Behavior-Cloning')
        if behavior_policy is None:
            logger.warning("Behavior policy not available, using first baseline as behavior policy")
            behavior_policy = next(iter(baselines.values()))
        
        ope_results = {}
        for name, policy in baselines.items():
            logger.info(f"  Evaluating {name}...")
            try:
                results = evaluator.evaluate(
                    trajectories=test_data,
                    policy=policy,
                    behavior_policy=behavior_policy,
                    methods=['WIS', 'DR', 'DM']
                )
                ope_results[name] = results
            except Exception as e:
                logger.warning(f"    Evaluation failed: {e}")
                ope_results[name] = None
        
        # Safety metrics
        logger.info("\nComputing safety metrics...")

        safety = SafetyEvaluator(config=safety_config)
        
        safety_results = {}
        for name, policy in baselines.items():
            logger.info(f"  {name}...")
            try:
                safety_index = safety.compute_safety_index(
                    policy=policy,
                    test_data=test_data
                )
                safety_results[name] = {
                    'safety_index': safety_index,
                    'safety_level': 'HIGH' if safety_index > 0.95 else 'MEDIUM' if safety_index > 0.85 else 'LOW'
                }
            except Exception as e:
                logger.warning(f"    Safety computation failed: {e}")
                safety_results[name] = None
        
        # Clinical metrics (simplified)
        logger.info("\nComputing clinical metrics...")
        clinical = ClinicalEvaluator()
        
        clinical_results = {}
        for name, policy in baselines.items():
            logger.info(f"  {name}...")
            try:
                # Compute guideline compliance
                compliance = clinical.compute_health_improvement(
                    policy_trajectories=policy,
                    baseline_trajectories=test_data
                )
                clinical_results[name] = {
                    'guideline_compliance': compliance
                }
            except Exception as e:
                logger.warning(f"    Clinical metrics failed: {e}")
                clinical_results[name] = None
        
        self.results['evaluation'] = {
            'ope': ope_results,
            'safety': safety_results,
            'clinical': clinical_results
        }
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        for name in baselines.keys():
            logger.info(f"\n{name}:")
            
            if safety_results.get(name):
                safety_idx = safety_results[name]['safety_index']
                safety_lvl = safety_results[name]['safety_level']
                logger.info(f"  Safety Index: {safety_idx:.4f} ({safety_lvl})")
            
            if clinical_results.get(name):
                compliance = clinical_results[name]['guideline_compliance']
                logger.info(f"  Guideline Compliance: {compliance:.2%}")
        
        logger.info("\n✓ Evaluation complete")
    
    def stage_6_generate_reports(self):
        """Stage 6: Generate final reports and visualizations."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 6: GENERATING REPORTS")
        logger.info("=" * 80)
        
        # Save results summary
        summary_path = self.output_dir / 'results_summary.json'
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.config.mode,
            'data_source': self.results.get('data', {}).get('source', 'unknown'),
            'baselines_evaluated': list(self.results.get('baselines', {}).get('policies', {}).keys()),
            'output_directory': str(self.output_dir)
        }
        
        # Add evaluation metrics
        if 'evaluation' in self.results:
            summary['evaluation'] = {
                'safety_metrics': {
                    name: res['safety_index'] if res else None
                    for name, res in self.results['evaluation']['safety'].items()
                },
                'clinical_metrics': {
                    name: res['guideline_compliance'] if res else None
                    for name, res in self.results['evaluation']['clinical'].items()
                }
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Results summary saved to {summary_path}")
        
        # Generate LaTeX table
        self._generate_latex_table()
        
        # Generate visualizations
        self._generate_visualizations()
        
        logger.info("✓ Report generation complete")
    
    def _generate_latex_table(self):
        """Generate LaTeX table for thesis."""
        if 'baselines' not in self.results:
            return
        
        latex_path = self.output_dir / 'results_table.tex'
        
        comparison = self.results['baselines']['comparison']
        
        latex_content = "\\begin{table}[h]\n"
        latex_content += "\\centering\n"
        latex_content += "\\caption{Baseline Policy Comparison Results}\n"
        latex_content += "\\label{tab:baseline_comparison}\n"
        latex_content += "\\begin{tabular}{lccc}\n"
        latex_content += "\\hline\n"
        latex_content += "Policy & Mean Reward & Safety Rate & Avg Episode Length \\\\\n"
        latex_content += "\\hline\n"
        
        for idx, row in comparison.iterrows():
            latex_content += f"{row['policy']} & "
            latex_content += f"{row['mean_reward']:.4f} & "
            latex_content += f"{row['safety_rate']:.2%} & "
            latex_content += f"{row['avg_episode_length']:.1f} \\\\\n"
        
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"
        latex_content += "\\end{table}\n"
        
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"✓ LaTeX table saved to {latex_path}")
    
    def _generate_visualizations(self):
        """Generate visualization plots."""
        if 'baselines' not in self.results:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            comparison = self.results['baselines']['comparison']
            
            # Set style
            sns.set_style('whitegrid')
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Mean Reward
            axes[0].barh(comparison['policy'], comparison['mean_reward'])
            axes[0].set_xlabel('Mean Reward')
            axes[0].set_title('Baseline Policy Comparison - Mean Reward')
            axes[0].grid(axis='x', alpha=0.3)
            
            # Plot 2: Safety Rate
            axes[1].barh(comparison['policy'], comparison['safety_rate'])
            axes[1].set_xlabel('Safety Rate')
            axes[1].set_title('Baseline Policy Comparison - Safety Rate')
            axes[1].grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = self.output_dir / 'baseline_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Visualization saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
    
    def _convert_to_trajectory_format(self, df):
        """Convert DataFrame to trajectory format (state, action, reward, next_state, done)."""
        # Placeholder - implement based on actual MIMIC data structure
        logger.warning("Trajectory conversion not implemented - using placeholder")
        return []


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RL Healthcare Treatment - Integrated Solution Runner'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'synthetic', 'data-only', 'train-eval', 'eval-only', 'test'],
        default='synthetic',
        help='Execution mode (default: synthetic)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/integration_run',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--mimic-dir',
        type=str,
        default='data/raw/mimic-iii',
        help='Path to MIMIC-III data directory'
    )
    
    parser.add_argument(
        '--use-synthetic',
        action='store_true',
        help='Use synthetic data instead of MIMIC'
    )
    
    parser.add_argument(
        '--n-synthetic-patients',
        type=int,
        default=1000,
        help='Number of synthetic patients to generate'
    )
    
    parser.add_argument(
        '--trajectory-length',
        type=int,
        default=30,
        help='Length of patient trajectories'
    )
    
    parser.add_argument(
        '--use-sample',
        action='store_true',
        help='Use sample of MIMIC data for quick testing'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Size of MIMIC data sample'
    )
    
    parser.add_argument(
        '--train-cql',
        action='store_true',
        help='Train CQL agent (time-intensive)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Create runner
    runner = IntegratedSolutionRunner(args)
    
    try:
        # Run pipeline
        results = runner.run_full_pipeline()
        
        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS!")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {runner.output_dir}")
        logger.info("\nGenerated files:")
        
        for file_path in runner.output_dir.glob('*'):
            logger.info(f"  - {file_path.name}")
        
        logger.info("\nNext steps:")
        logger.info("  1. Review results_summary.json")
        logger.info("  2. Check baseline_comparison_report.md")
        logger.info("  3. View visualizations: baseline_comparison.png")
        logger.info("  4. Use results_table.tex in thesis")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"\n\nPipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
