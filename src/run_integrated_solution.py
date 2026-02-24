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

# --- path setup MUST come before any local imports ---
sys.path.insert(0, str(Path(__file__).parent))

from rewards.health_reward import HealthOutcomeReward
from rewards.composite_reward import CompositeRewardFunction
from rewards.safety_reward import SafetyPenalty
from rewards.cost_reward import CostEffectivenessReward
from rewards.adherence_reward import AdherenceReward
from rewards.reward_config import RewardConfig

from environments import  DiabetesEnv, AdherenceEnv
from environments.diabetes_env import DiabetesEnvConfig
from environments.adherence_env import AdherenceEnvConfig

from evaluation import (
    OffPolicyEvaluator,
    SafetyEvaluator,
    ClinicalEvaluator,
    EvaluationConfig,
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
    compare_all_baselines,
)


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

        if self.config.mode in ['full', 'data-only']:
            self.stage_1_data_preparation()

        if self.config.mode in ['full', 'train-eval', 'synthetic']:
            self.stage_2_environment_setup()

        if self.config.mode in ['full', 'train-eval', 'synthetic']:
            self.stage_3_baseline_training()

        if self.config.mode in ['full', 'train-eval']:
            self.stage_4_cql_training()

        if self.config.mode in ['full', 'train-eval', 'eval-only']:
            self.stage_5_evaluation()

        self.stage_6_generate_reports()

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 80)

        return self.results

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

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
            logger.info("Data preparation complete")

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            if self.config.mode != 'synthetic':
                logger.info("Falling back to synthetic data...")
                data = self.prepare_synthetic_data()
                self.results['data'] = data
            else:
                raise

    def prepare_synthetic_data(self):
        """Generate synthetic patient data."""
        logger.info("Generating synthetic diabetes patient data...")

        generator = SyntheticDataGenerator(random_seed=42)

        patients = generator.generate_diabetes_population(
            n_patients=self.config.n_synthetic_patients
        )

        state_cols = [
            'glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max',
            'insulin_mean', 'medication_taken', 'reminder_sent',
            'hypoglycemia', 'hyperglycemia', 'day',
        ]

        trajectories = []
        for patient in patients:
            traj_df = generator.simulate_patient_trajectory(
                patient=patient,
                time_horizon_days=self.config.trajectory_length,
                treatment_policy=None
            )
            for i in range(len(traj_df) - 1):
                row = traj_df.iloc[i]
                next_row = traj_df.iloc[i + 1]
                state = row[state_cols].values.astype(np.float32)
                next_state = next_row[state_cols].values.astype(np.float32)
                action = np.array([float(row['medication_taken'])], dtype=np.float32)
                reward = 0.0
                if 80 <= row['glucose_mean'] <= 180:
                    reward += 1.0
                reward -= 2.0 * float(row['hypoglycemia'])
                reward -= 1.0 * float(row['hyperglycemia'])
                reward += 0.5 * float(row['medication_taken'])
                done = (i == len(traj_df) - 2)
                trajectories.append((state, action, float(reward), next_state, done))

        logger.info(f"Generated {len(patients)} synthetic patients")
        logger.info(f"Generated {len(trajectories)} transitions")

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

        output_path = self.output_dir / 'synthetic_data.pkl'
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved synthetic data to {output_path}")
        return data

    def prepare_mimic_data(self):
        """Prepare data from MIMIC-III."""
        logger.info("Loading MIMIC-III data...")

        from data import (
            MIMICLoader, CohortBuilder, FeatureEngineer,
            DataPreprocessor, split_train_val_test
        )

        loader = MIMICLoader(data_dir=self.config.mimic_dir, use_cache=True)

        patients = loader.load_patients()
        admissions = loader.load_admissions()
        diagnoses = loader.load_diagnoses_icd()

        logger.info(f"Loaded {len(patients):,} patients")

        builder = CohortBuilder(patients, admissions, diagnoses)
        diabetes_patients = builder.define_diabetes_cohort()

        filtered_patients = builder.apply_inclusion_criteria(
            diabetes_patients, min_age=18, max_age=80, min_admissions=2
        )
        final_cohort = builder.apply_exclusion_criteria(
            filtered_patients, exclude_pregnancy=True, exclude_pediatric=True
        )

        logger.info(f"Final cohort: {len(final_cohort):,} patients")

        if self.config.use_sample:
            import random
            random.seed(42)
            final_cohort = random.sample(
                final_cohort, min(self.config.sample_size, len(final_cohort))
            )
            logger.info(f"Using sample of {len(final_cohort)} patients")

        labs = loader.load_lab_events(subject_ids=final_cohort)
        prescriptions = loader.load_prescriptions(subject_ids=final_cohort)

        engineer = FeatureEngineer()
        demographics = engineer.extract_demographics(patients, admissions)
        demographics = demographics[demographics['subject_id'].isin(final_cohort)]

        lab_sequence = engineer.extract_lab_sequence(labs, final_cohort)
        temporal_features = engineer.create_temporal_features(
            lab_sequence, time_column='charttime'
        )

        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_missing_values(temporal_features)
        clean_data = preprocessor.handle_outliers(clean_data)
        normalized_data = preprocessor.normalize_labs(clean_data)

        train, val, test = split_train_val_test(
            normalized_data, ratios=(0.7, 0.15, 0.15), random_state=42
        )

        data = {
            'train': train,
            'val': val,
            'test': test,
            'demographics': demographics,
            'cohort': final_cohort,
            'source': 'mimic'
        }

        logger.info(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        return data

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def stage_2_environment_setup(self):
        """Stage 2: Set up healthcare environments."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: ENVIRONMENT SETUP")
        logger.info("=" * 80)

        logger.info("Creating diabetes management environment...")
        diabetes_cfg = DiabetesEnvConfig(
            max_steps=288,
            dt=1 / 12,
            patient_variability=0.1,
            initial_glucose_range=(120.0, 220.0),
            target_glucose_range=(80.0, 140.0),
            max_insulin_dose=10.0,
        )
        diabetes_env = DiabetesEnv(config=diabetes_cfg)

        logger.info("Creating medication adherence environment...")
        adherence_cfg = AdherenceEnvConfig()
        adherence_env = AdherenceEnv(config=adherence_cfg)

        logger.info("Configuring composite reward function...")
        reward_config = RewardConfig(
            w_adherence=0.3,
            w_health=0.4,
            w_safety=0.2,
            w_cost=0.1,
        )
        composite_reward = CompositeRewardFunction(reward_config)
        composite_reward.add_component("adherence", AdherenceReward(reward_config), reward_config.w_adherence)
        composite_reward.add_component("health", HealthOutcomeReward(reward_config), reward_config.w_health)
        composite_reward.add_component("safety", SafetyPenalty(reward_config), reward_config.w_safety)
        composite_reward.add_component("cost", CostEffectivenessReward(reward_config), reward_config.w_cost)

        self.results['environments'] = {
            'diabetes_env': diabetes_env,
            'adherence_env': adherence_env,
            'reward_function': composite_reward,
        }

        logger.info("Environment setup complete")

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------

    def stage_3_baseline_training(self):
        """Stage 3: Train and evaluate baseline policies."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: BASELINE TRAINING")
        logger.info("=" * 80)

        if 'data' in self.results:
            data = self.results['data']
            if data['source'] == 'synthetic':
                test_data = data['test']
                train_data = data['train']
            else:
                test_data = self._convert_to_trajectory_format(data['test'])
                train_data = self._convert_to_trajectory_format(data['train'])
        else:
            logger.info("Generating synthetic data for baselines...")
            data = self.prepare_synthetic_data()
            self.results['data'] = data
            test_data = data['test']
            train_data = data['train']

        state_dim = 10
        action_dim = 1

        train_states = np.array([d[0] for d in train_data])
        train_actions = np.array([d[1] for d in train_data])

        logger.info(f"Training data: {len(train_data)} transitions")
        logger.info(f"Test data: {len(test_data)} transitions")

        logger.info("Creating baseline policies...")
        baselines = {}

        logger.info("  1/7 Rule-based policy...")
        baselines['Rule-Based'] = create_diabetes_rule_policy(state_dim, action_dim)

        logger.info("  2/7 Random policy...")
        baselines['Random-Uniform'] = create_random_policy(
            action_dim, state_dim, seed=42, distribution='uniform'
        )

        logger.info("  3/7 Safe random policy...")
        baselines['Random-Safe'] = create_safe_random_policy(
            action_dim, state_dim, seed=42, num_samples=10
        )

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
            'test_data': test_data,
        }

        logger.info(f"Baseline training complete. Report saved to {comparison_path}")

    # ------------------------------------------------------------------
    # Stage 4
    # ------------------------------------------------------------------

    def stage_4_cql_training(self):
        """Stage 4: Train Conservative Q-Learning agent."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: CQL TRAINING")
        logger.info("=" * 80)

        if not self.config.train_cql:
            logger.info("CQL training skipped (use --train-cql to enable)")
            return

        if 'data' not in self.results:
            logger.warning("No data available, skipping CQL training")
            return

        data = self.results['data']

        logger.info("Configuring CQL...")
        config = CQLConfig(
            state_dim=10,
            action_dim=1,
            hidden_dim=256,
            q_lr=3e-4,
            policy_lr=1e-4,
            cql_alpha=5.0,
            gamma=0.99,
        )

        agent = CQLAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            q_lr=config.q_lr,
            policy_lr=config.policy_lr,
            cql_alpha=config.cql_alpha,
            gamma=config.gamma,
        )
        device = self.get_device()

        logger.info(f"Training on device: {device}")

        if data['source'] == 'synthetic':
            train_data = data['train']
        else:
            train_data = self._convert_to_trajectory_format(data['train'])

        logger.info(f"Training dataset size: {len(train_data)} transitions")
        logger.info("Full CQL training requires src/models/rl — placeholder logged")

        self.results['cql'] = {
            'config': config,
            'agent': agent,
            'trained': False,
            'note': 'CQL training placeholder',
        }

        logger.info("CQL setup complete (training placeholder)")

    # ------------------------------------------------------------------
    # Stage 5
    # ------------------------------------------------------------------

    def stage_5_evaluation(self):
        """Stage 5: Comprehensive evaluation."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: EVALUATION")
        logger.info("=" * 80)

        if 'baselines' not in self.results:
            logger.warning("No baselines available for evaluation")
            return

        test_data = self.results['baselines']['test_data']
        baselines = self.results['baselines']['policies']

        # Off-policy evaluation
        logger.info("Running off-policy evaluation...")
        ope_evaluator = OffPolicyEvaluator()
        behavior_policy = baselines.get('Behavior-Cloning', next(iter(baselines.values())))
        ope_results = {}
        for name, policy in baselines.items():
            logger.info(f"  Evaluating {name}...")
            try:
                results = ope_evaluator.evaluate(
                    policy=policy,
                    behavior_policy=behavior_policy,
                    trajectories=test_data,
                    methods=['wis'],
                )
                ope_results[name] = results
            except Exception as e:
                logger.warning(f"    OPE failed for {name}: {e}")
                ope_results[name] = None

        # Safety metrics
        logger.info("\nComputing safety metrics...")
        eval_config = EvaluationConfig()
        safety_evaluator = SafetyEvaluator(eval_config)
        safety_results = {}
        for name, policy in baselines.items():
            logger.info(f"  {name}...")
            try:
                safety_result = safety_evaluator.evaluate(test_data)
                safety_index = safety_result.safety_index
                safety_results[name] = {
                    'safety_index': safety_index,
                    'safety_level': (
                        'HIGH' if safety_index > 0.95
                        else 'MEDIUM' if safety_index > 0.85
                        else 'LOW'
                    ),
                }
            except Exception as e:
                logger.warning(f"    Safety computation failed for {name}: {e}")
                safety_results[name] = None

        # Clinical metrics
        logger.info("\nComputing clinical metrics...")
        clinical_evaluator = ClinicalEvaluator(eval_config)
        clinical_results = {}
        for name, policy in baselines.items():
            logger.info(f"  {name}...")
            try:
                tir = clinical_evaluator.compute_time_in_range(test_data)
                compliance = float(np.mean(list(tir.values()))) if tir else 0.0
                clinical_results[name] = {'guideline_compliance': compliance}
            except Exception as e:
                logger.warning(f"    Clinical metrics failed for {name}: {e}")
                clinical_results[name] = None

        self.results['evaluation'] = {
            'ope': ope_results,
            'safety': safety_results,
            'clinical': clinical_results,
        }

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)

        for name in baselines.keys():
            logger.info(f"\n{name}:")
            if safety_results.get(name):
                s = safety_results[name]
                logger.info(f"  Safety Index: {s['safety_index']:.4f} ({s['safety_level']})")
            if clinical_results.get(name):
                c = clinical_results[name]
                logger.info(f"  Guideline Compliance: {c['guideline_compliance']:.2%}")

        logger.info("\nEvaluation complete")

    # ------------------------------------------------------------------
    # Stage 6
    # ------------------------------------------------------------------

    def stage_6_generate_reports(self):
        """Stage 6: Generate final reports and visualizations."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 6: GENERATING REPORTS")
        logger.info("=" * 80)

        summary_path = self.output_dir / 'results_summary.json'

        summary = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.config.mode,
            'data_source': self.results.get('data', {}).get('source', 'unknown'),
            'baselines_evaluated': list(
                self.results.get('baselines', {}).get('policies', {}).keys()
            ),
            'output_directory': str(self.output_dir),
        }

        if 'evaluation' in self.results:
            summary['evaluation'] = {
                'safety_metrics': {
                    name: res['safety_index'] if res else None
                    for name, res in self.results['evaluation']['safety'].items()
                },
                'clinical_metrics': {
                    name: res['guideline_compliance'] if res else None
                    for name, res in self.results['evaluation']['clinical'].items()
                },
            }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results summary saved to {summary_path}")

        self._generate_latex_table()
        self._generate_visualizations()

        logger.info("Report generation complete")

    def _generate_latex_table(self):
        if 'baselines' not in self.results:
            return

        latex_path = self.output_dir / 'results_table.tex'
        comparison = self.results['baselines']['comparison']

        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Baseline Policy Comparison Results}",
            "\\label{tab:baseline_comparison}",
            "\\begin{tabular}{lccc}",
            "\\hline",
            "Policy & Mean Reward & Safety Rate & Avg Episode Length \\\\",
            "\\hline",
        ]

        for policy_name, row in comparison.iterrows():
            mean_reward = row.get('mean_reward', float('nan'))
            safety_rate = row.get('safety_rate', float('nan'))
            total_steps = row.get('total_steps', 0)
            lines.append(
                f"{policy_name} & {mean_reward:.4f} & "
                f"{safety_rate:.2%} & {int(total_steps)} \\\\"
            )

        lines += ["\\hline", "\\end{tabular}", "\\end{table}"]

        with open(latex_path, 'w') as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"LaTeX table saved to {latex_path}")

    def _generate_visualizations(self):
        if 'baselines' not in self.results:
            return

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            comparison = self.results['baselines']['comparison']
            sns.set_style('whitegrid')

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].barh(comparison.index, comparison['mean_reward'])
            axes[0].set_xlabel('Mean Reward')
            axes[0].set_title('Baseline Policy Comparison - Mean Reward')
            axes[0].grid(axis='x', alpha=0.3)

            axes[1].barh(comparison.index, comparison['safety_rate'])
            axes[1].set_xlabel('Safety Rate')
            axes[1].set_title('Baseline Policy Comparison - Safety Rate')
            axes[1].grid(axis='x', alpha=0.3)

            plt.tight_layout()

            plot_path = self.output_dir / 'baseline_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Visualization saved to {plot_path}")

        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

    def _convert_to_trajectory_format(self, df):
        logger.warning("Trajectory conversion not implemented - using placeholder")
        return []


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='RL Healthcare Treatment - Integrated Solution Runner'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'synthetic', 'data-only', 'train-eval', 'eval-only', 'test'],
        default='synthetic',
    )
    parser.add_argument('--output-dir', type=str, default='outputs/integration_run')
    parser.add_argument('--mimic-dir', type=str, default='data/raw/mimic-iii')
    parser.add_argument('--use-synthetic', action='store_true')
    parser.add_argument('--n-synthetic-patients', type=int, default=1000)
    parser.add_argument('--trajectory-length', type=int, default=30)
    parser.add_argument('--use-sample', action='store_true')
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--train-cql', action='store_true')

    return parser.parse_args()


def main():
    args = parse_arguments()
    runner = IntegratedSolutionRunner(args)

    try:
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
        logger.warning("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\nPipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())