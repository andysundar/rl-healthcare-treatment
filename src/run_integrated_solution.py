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

# Optional imports for new stages (imported lazily inside stage methods to
# keep the default run path unaffected)
# from models.encoders import PatientAutoencoder, EncoderConfig
# from models.encoders.state_encoder_wrapper import StateEncoderWrapper
# from models.policy_transfer import PolicyTransferTrainer, TransferConfig
# from evaluation.interpretability import (InterpretabilityConfig,
#     CounterfactualExplainer, DecisionRuleExtractor, PersonalizationScorer)


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

# ---------------------------------------------------------------------------
# State-column constants — shared by all pipeline stages
# ---------------------------------------------------------------------------
BASE_STATE_COLS = [
    'glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max',
    'insulin_mean', 'medication_taken', 'reminder_sent',
    'hypoglycemia', 'hyperglycemia', 'day',
]
VITAL_COLS       = ['heart_rate', 'sbp', 'respiratory_rate', 'spo2']
MED_HISTORY_COLS = ['adherence_rate_7d', 'medication_count']


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

    def _get_state_cols(self) -> list:
        """Return ordered state column names based on active feature flags."""
        cols = list(BASE_STATE_COLS)
        if getattr(self.config, 'use_vitals', False):
            cols.extend(VITAL_COLS)
        if getattr(self.config, 'use_med_history', False):
            cols.extend(MED_HISTORY_COLS)
        return cols

    def _get_state_dim(self) -> int:
        """Return scalar state dimension based on active feature flags."""
        return len(self._get_state_cols())

    def run_full_pipeline(self):
        """Run complete end-to-end pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 80)

        if self.config.mode in ['full', 'data-only']:
            self.stage_1_data_preparation()

        if self.config.mode in ['full', 'train-eval', 'synthetic']:
            self.stage_2_environment_setup()

        # Encoder pre-training (opt-in)
        self.stage_2b_encoder_training()

        if self.config.mode in ['full', 'train-eval', 'synthetic']:
            self.stage_3_baseline_training()

        if self.config.mode in ['full', 'train-eval']:
            self.stage_4_cql_training()

        if self.config.mode in ['full', 'train-eval', 'eval-only']:
            self.stage_5_evaluation()

        # Policy transfer (opt-in)
        self.stage_6b_transfer()

        # Interpretability (opt-in)
        self.stage_7_interpretability()

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

        # Derive state columns from active feature flags
        state_cols = self._get_state_cols()
        use_vitals      = getattr(self.config, 'use_vitals', False)
        use_med_history = getattr(self.config, 'use_med_history', False)
        if use_vitals or use_med_history:
            logger.info(
                f"Extended state: {len(state_cols)}-dim "
                f"(vitals={use_vitals}, med_history={use_med_history})"
            )

        trajectories = []
        for patient in patients:
            traj_df = generator.simulate_patient_trajectory(
                patient=patient,
                time_horizon_days=self.config.trajectory_length,
                treatment_policy=None,
                include_vitals=use_vitals,
                include_med_history=use_med_history,
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
            'source': 'synthetic',
            'state_cols': state_cols,
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

        # --- Vital signs (opt-in via --use-vitals) ---
        vitals_df = None
        if getattr(self.config, 'use_vitals', False):
            logger.info("Loading CHARTEVENTS for vital signs...")
            # All item IDs used by FeatureEngineer.extract_vitals_sequence()
            all_vital_ids = [
                211, 220045,                         # heart_rate
                51, 442, 455, 6701, 220050, 220179,  # sbp
                618, 615, 220210, 224690,             # respiratory_rate
                646, 220277,                          # spo2
            ]
            chartevents = loader.load_chartevents(
                subject_ids=final_cohort,
                item_ids=all_vital_ids,
            )
            if len(chartevents) > 0:
                engineer_for_vitals = FeatureEngineer()
                vitals_df = engineer_for_vitals.extract_vitals_sequence(
                    chartevents, subject_ids=final_cohort
                )
                logger.info(
                    f"Extracted vitals: {len(vitals_df):,} timepoints, "
                    f"cols={[c for c in vitals_df.columns if c not in ('subject_id','hadm_id','charttime')]}"
                )

        # --- Medication history (opt-in via --use-med-history) ---
        med_history_df = None
        if getattr(self.config, 'use_med_history', False):
            logger.info("Extracting medication history from prescriptions...")
            engineer_for_meds = FeatureEngineer()
            med_history_df = engineer_for_meds.extract_medication_history(
                prescriptions, encoding='binary'
            )
            logger.info(
                f"Medication history: {len(med_history_df.columns)-1} drug columns, "
                f"{len(med_history_df):,} patients"
            )

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
            'source': 'mimic',
            'state_cols': self._get_state_cols(),
            'vitals': vitals_df,
            'med_history': med_history_df,
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
    # Stage 2b  (opt-in)
    # ------------------------------------------------------------------

    def stage_2b_encoder_training(self):
        """Stage 2b: Pre-train state autoencoder (only if --use-encoder)."""
        if not getattr(self.config, 'use_encoder', False):
            return

        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2b: ENCODER PRE-TRAINING")
        logger.info("=" * 80)

        if 'data' not in self.results:
            logger.info("No data yet — generating synthetic data for encoder training...")
            data = self.prepare_synthetic_data()
            self.results['data'] = data
        else:
            data = self.results['data']

        from models.encoders import PatientAutoencoder, EncoderConfig
        from models.encoders.state_encoder_wrapper import StateEncoderWrapper

        # StateEncoderWrapper always passes the full flat state as "labs";
        # vital_dim / demo_dim stay 0 to avoid MPS device-mismatch on missing modalities.
        enc_cfg = EncoderConfig(
            lab_dim=self._get_state_dim(),
            vital_dim=0,
            demo_dim=0,
            state_dim=getattr(self.config, 'encoder_state_dim', 64),
        )
        variational = (getattr(self.config, 'encoder_type', 'autoencoder') == 'vae')
        ae = PatientAutoencoder(enc_cfg, variational=variational)
        device = self.get_device()
        wrapper = StateEncoderWrapper(ae, device=device, raw_state_dim=self._get_state_dim())

        checkpoint = getattr(self.config, 'encoder_checkpoint', None)
        if checkpoint:
            logger.info(f"Loading encoder checkpoint from {checkpoint}")
            wrapper.load_checkpoint(checkpoint)
        else:
            logger.info("Training encoder...")
            enc_save_dir = self.output_dir / 'encoder'
            history = wrapper.train_on_transitions(
                train_transitions=data['train'],
                val_transitions=data['val'],
                epochs=getattr(self.config, 'encoder_epochs', 50),
                save_dir=enc_save_dir,
            )
            logger.info(
                f"Encoder training done. "
                f"Final val loss: {history['val_loss'][-1]:.4f}"
            )

        logger.info("Re-encoding all transitions with trained encoder...")
        encoded = {
            split: wrapper.encode_transitions(data[split])
            for split in ('train', 'val', 'test')
        }

        self.results['encoder_wrapper'] = wrapper
        self.results['encoded_data'] = encoded
        logger.info(
            f"Encoded state dim: {wrapper.state_dim}  "
            f"(train={len(encoded['train'])}, "
            f"val={len(encoded['val'])}, "
            f"test={len(encoded['test'])})"
        )

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

        # Derive state_dim from actual data to stay correct with --use-vitals / --use-med-history
        state_dim = len(train_data[0][0]) if train_data else self._get_state_dim()
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
            state_dim=self._get_state_dim(),
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

        # Prefer encoded data if encoder was pre-trained
        if 'encoded_data' in self.results:
            train_data = self.results['encoded_data']['train']
            val_data   = self.results['encoded_data']['val']
            state_dim  = self.results['encoder_wrapper'].state_dim
            logger.info(f"Using encoder embeddings as states (dim={state_dim})")
        elif data['source'] == 'synthetic':
            train_data = data['train']
            val_data   = data['val']
            state_dim  = self._get_state_dim()
        else:
            train_data = self._convert_to_trajectory_format(data['train'])
            val_data   = self._convert_to_trajectory_format(data['val'])
            state_dim  = self._get_state_dim()

        # Rebuild agent with correct state_dim
        agent = CQLAgent(
            state_dim=state_dim,
            action_dim=1,
            hidden_dim=256,
            q_lr=3e-4,
            policy_lr=1e-4,
            cql_alpha=5.0,
            gamma=0.99,
            device=str(device),
        )

        logger.info(f"Training dataset size: {len(train_data)} transitions")

        # Build replay buffer
        from models.rl import ReplayBuffer, OfflineRLTrainer
        from models.rl.trainer import create_simple_eval_function

        buffer = ReplayBuffer(
            capacity=len(train_data),
            state_dim=state_dim,
            action_dim=1,
            device=str(device),
        )
        states      = np.array([t[0] for t in train_data], dtype=np.float32)
        actions     = np.array([t[1] for t in train_data], dtype=np.float32)
        rewards     = np.array([t[2] for t in train_data], dtype=np.float32)
        next_states = np.array([t[3] for t in train_data], dtype=np.float32)
        dones       = np.array([t[4] for t in train_data], dtype=np.float32)
        buffer.load_from_dataset(states, actions, rewards, next_states, dones)

        # Simple eval function from validation transitions
        val_episodes = self._transitions_to_episodes(val_data)
        eval_fn = create_simple_eval_function(val_episodes)

        cql_save_dir = self.output_dir / 'cql'
        n_iter  = getattr(self.config, 'cql_iterations', 10_000)
        batch_sz = getattr(self.config, 'cql_batch_size', 256)

        trainer = OfflineRLTrainer(
            agent=agent,
            replay_buffer=buffer,
            save_dir=str(cql_save_dir),
            eval_freq=max(1, n_iter // 10),
            save_freq=max(1, n_iter // 5),
        )
        logger.info(f"Starting CQL training: {n_iter} iterations, batch={batch_sz}")
        history = trainer.train(
            num_iterations=n_iter,
            batch_size=batch_sz,
            eval_fn=eval_fn,
            verbose=True,
        )

        self.results['cql'] = {
            'config': config,
            'agent': agent,
            'trained': True,
            'history': history,
            'state_dim': state_dim,
        }
        logger.info("CQL training complete")

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
        # Evaluators expect dict-format trajectories; test_data is flat tuples
        traj_dicts = self._tuples_to_traj_dicts(test_data)
        safety_results = {}
        for name, policy in baselines.items():
            logger.info(f"  {name}...")
            try:
                safety_result = safety_evaluator.evaluate(traj_dicts)
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
                tir = clinical_evaluator.compute_time_in_range(traj_dicts)
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

        if 'cql' in self.results:
            summary['cql'] = {
                'trained': self.results['cql'].get('trained', False),
                'state_dim': self.results['cql'].get('state_dim', 10),
            }

        if 'interpretability' in self.results:
            interp = self.results['interpretability']
            summary['interpretability'] = {
                'decision_tree_fidelity': interp.get('decision_tree_fidelity'),
                'n_rules': interp.get('n_rules'),
                'n_counterfactuals': interp.get('n_counterfactuals'),
                'personalization_score': interp.get('personalization_score'),
                'rules_path': interp.get('rules_path'),
                'counterfactuals_path': interp.get('counterfactuals_path'),
            }

        if 'transfer' in self.results:
            summary['policy_transfer'] = {
                'adapter_path': self.results['transfer'].get('adapter_path'),
            }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results summary saved to {summary_path}")

        self._generate_latex_table()
        self._generate_visualizations()

        logger.info("Report generation complete")

    # ------------------------------------------------------------------
    # Stage 6b  (opt-in)
    # ------------------------------------------------------------------

    def stage_6b_transfer(self):
        """Stage 6b: Policy transfer to target population (only if --use-transfer)."""
        if not getattr(self.config, 'use_transfer', False):
            return

        logger.info("\n" + "=" * 80)
        logger.info("STAGE 6b: POLICY TRANSFER")
        logger.info("=" * 80)

        if not self.results.get('cql', {}).get('trained', False):
            logger.warning("CQL agent not trained — skipping policy transfer")
            return

        from models.policy_transfer import PolicyTransferTrainer, TransferConfig

        source_agent   = self.results['cql']['agent']
        source_encoder = self.results.get('encoder_wrapper', None)
        target_encoder = source_encoder   # same domain in synthetic mode

        state_dim = self.results['cql']['state_dim']
        tr_cfg = TransferConfig(
            source_state_dim=state_dim,
            target_state_dim=state_dim,
            n_adaptation_steps=getattr(self.config, 'transfer_steps', 1_000),
        )

        trainer = PolicyTransferTrainer(
            source_agent=source_agent,
            source_encoder=source_encoder,
            target_encoder=target_encoder,
            config=tr_cfg,
            device=self.get_device(),
        )

        data = self.results['data']
        source_transitions = data['train']
        target_transitions = data['test']   # treat test split as "target"

        logger.info(
            f"Fitting transfer adapter: "
            f"{len(source_transitions)} source, "
            f"{len(target_transitions)} target transitions"
        )
        history = trainer.fit(target_transitions, source_transitions)

        adapter_path = self.output_dir / 'transfer_adapter.pt'
        trainer.save(adapter_path)

        self.results['transfer'] = {
            'trainer': trainer,
            'adapter_path': str(adapter_path),
            'history': history,
        }
        logger.info(f"Transfer adapter saved to {adapter_path}")

    # ------------------------------------------------------------------
    # Stage 7  (opt-in)
    # ------------------------------------------------------------------

    def stage_7_interpretability(self):
        """Stage 7: Interpretability analysis (only if --use-interpretability)."""
        if not getattr(self.config, 'use_interpretability', False):
            return

        logger.info("\n" + "=" * 80)
        logger.info("STAGE 7: INTERPRETABILITY")
        logger.info("=" * 80)

        if not self.results.get('cql', {}).get('trained', False):
            logger.warning("CQL agent not trained — skipping interpretability")
            return

        from evaluation.interpretability import (
            InterpretabilityConfig, CounterfactualExplainer,
            DecisionRuleExtractor, PersonalizationScorer,
        )

        agent          = self.results['cql']['agent']
        encoder_wrapper = self.results.get('encoder_wrapper', None)
        device         = self.get_device()

        data      = self.results['data']
        test_data = self.results.get('encoded_data', data)['test']
        raw_test  = data['test']          # always raw for counterfactuals

        n_explain = getattr(self.config, 'explain_n_samples', 50)
        max_depth = getattr(self.config, 'tree_max_depth', 4)
        n_cf      = getattr(self.config, 'n_counterfactuals', 5)

        interp_cfg = InterpretabilityConfig(
            n_counterfactuals=n_cf,
            tree_max_depth=max_depth,
        )

        # ---- Decision rules ------------------------------------------
        logger.info("Fitting surrogate decision tree...")
        test_states = np.array([t[0] for t in test_data], dtype=np.float32)
        rule_extractor = DecisionRuleExtractor(interp_cfg)
        rule_extractor.fit(
            states=np.array([t[0] for t in raw_test], dtype=np.float32),
            agent=agent,
            encoder_wrapper=encoder_wrapper,
        )
        fidelity = rule_extractor.fidelity_score(
            np.array([t[0] for t in raw_test[:500]], dtype=np.float32),
            agent, encoder_wrapper,
        )
        logger.info(f"Decision tree fidelity: {fidelity:.4f}")

        rules_path = self.output_dir / 'decision_rules.txt'
        rule_extractor.save_rules(rules_path)
        rules = rule_extractor.extract_rules()

        # ---- Counterfactuals -----------------------------------------
        logger.info(f"Generating counterfactuals for {n_explain} states...")
        cf_explainer = CounterfactualExplainer(agent, encoder_wrapper, interp_cfg, device)
        sample_states = raw_test[:n_explain]
        all_cf = []
        for transition in sample_states:
            raw_state = transition[0]
            cfs = cf_explainer.explain(raw_state)
            all_cf.extend(cfs)

        # Serialise (convert numpy arrays to lists for JSON)
        cf_serialisable = []
        for cf in all_cf:
            cf_serialisable.append({
                'original_action': cf['original_action'],
                'new_action': cf['new_action'],
                'l1_distance': cf['l1_distance'],
                'feature_changes': cf['feature_changes'],
            })

        cf_path = self.output_dir / 'counterfactuals.json'
        import json as _json
        with open(cf_path, 'w') as fh:
            _json.dump(cf_serialisable, fh, indent=2)
        logger.info(f"Counterfactuals saved to {cf_path} ({len(all_cf)} total)")

        # ---- Personalization score -----------------------------------
        pers_score = None
        if encoder_wrapper is not None:
            logger.info("Computing personalization score...")
            source_trajs = self._to_state_trajectories(data['train'])
            target_trajs = self._to_state_trajectories(data['test'])
            scorer = PersonalizationScorer(encoder_wrapper, encoder_wrapper)
            pers_result = scorer.compute_batch(source_trajs, target_trajs)
            pers_score = pers_result['mean']
            logger.info(
                f"Personalization score: {pers_score:.4f} "
                f"+/- {pers_result['std']:.4f}"
            )

        self.results['interpretability'] = {
            'decision_tree_fidelity': fidelity,
            'n_rules': len(rules),
            'n_counterfactuals': len(all_cf),
            'personalization_score': pers_score,
            'rules_path': str(rules_path),
            'counterfactuals_path': str(cf_path),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transitions_to_episodes(self, transitions):
        """
        Group a flat list of (s, a, r, s', done) tuples into episode dicts
        of the form {'states':[], 'actions':[], 'rewards':[], 'next_states':[]}.
        Falls back to fixed-length chunks (trajectory_length) if no done flags.
        """
        episodes = []
        current = {'states': [], 'actions': [], 'rewards': [], 'next_states': []}
        has_done = any(t[4] for t in transitions)
        chunk = getattr(self.config, 'trajectory_length', 30)

        for i, (s, a, r, ns, done) in enumerate(transitions):
            current['states'].append(s)
            current['actions'].append(float(a) if np.ndim(a) == 0 else float(a[0]))
            current['rewards'].append(float(r))
            current['next_states'].append(ns)

            end = done if has_done else ((i + 1) % chunk == 0)
            if end and current['states']:
                episodes.append(current)
                current = {'states': [], 'actions': [], 'rewards': [], 'next_states': []}

        if current['states']:
            episodes.append(current)
        return episodes

    def _to_state_trajectories(self, transitions, chunk_size: int = 30):
        """
        Convert flat transitions into list-of-lists of raw state arrays,
        grouped by episode (done flag) or fixed chunk_size.
        """
        trajs, current = [], []
        has_done = any(t[4] for t in transitions)
        for i, (s, a, r, ns, done) in enumerate(transitions):
            current.append(s)
            end = done if has_done else ((i + 1) % chunk_size == 0)
            if end and current:
                trajs.append(current)
                current = []
        if current:
            trajs.append(current)
        return trajs

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

            sns.set_style('whitegrid')
            plt.rcParams.update({'font.size': 11})
            comparison = self.results['baselines']['comparison']

            # ----------------------------------------------------------
            # Fig 1: Baseline comparison (2 panels — reward + safety)
            # ----------------------------------------------------------
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            colors = sns.color_palette('muted', len(comparison))

            axes[0].barh(comparison.index, comparison['mean_reward'], color=colors)
            axes[0].set_xlabel('Mean Reward')
            axes[0].set_title('Policy Comparison — Mean Reward')
            axes[0].grid(axis='x', alpha=0.3)

            axes[1].barh(comparison.index, comparison['safety_rate'], color=colors)
            axes[1].set_xlabel('Safety Rate')
            axes[1].set_title('Policy Comparison — Safety Rate')
            axes[1].set_xlim(0, 1.05)
            axes[1].grid(axis='x', alpha=0.3)

            plt.tight_layout()
            path1 = self.output_dir / 'baseline_comparison.png'
            plt.savefig(path1, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved {path1}")

            # ----------------------------------------------------------
            # Fig 2: 4-panel policy dashboard
            # ----------------------------------------------------------
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            metrics = [
                ('mean_reward',       'Mean Reward',        axes[0, 0]),
                ('safety_rate',       'Safety Rate',        axes[0, 1]),
                ('mean_action_value', 'Mean Action Value',  axes[1, 0]),
                ('std_action_value',  'Action Std Dev',     axes[1, 1]),
            ]
            for col, title, ax in metrics:
                if col in comparison.columns:
                    vals = comparison[col]
                    bars = ax.bar(range(len(vals)), vals,
                                  color=sns.color_palette('muted', len(vals)))
                    ax.set_xticks(range(len(vals)))
                    ax.set_xticklabels(comparison.index, rotation=30, ha='right', fontsize=9)
                    ax.set_title(title)
                    ax.grid(axis='y', alpha=0.3)
                    # Label bars
                    for bar, v in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.001 * abs(bar.get_height() + 1e-9),
                                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

            fig.suptitle('Policy Performance Dashboard', fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
            path2 = self.output_dir / 'policy_dashboard.png'
            plt.savefig(path2, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved {path2}")

            # ----------------------------------------------------------
            # Fig 2b: comparison.png  (EvaluationVisualizer style)
            # ----------------------------------------------------------
            try:
                from evaluation.visualizations import EvaluationVisualizer
                from configs.config import EvaluationConfig as _EvalCfg
                viz = EvaluationVisualizer(_EvalCfg())
                path_comp = self.output_dir / 'comparison.png'
                viz.plot_comparison(
                    comparison,
                    metrics=['mean_reward', 'safety_rate'],
                    save_path=str(path_comp),
                )
                plt.close('all')
                logger.info(f"Saved {path_comp}")
            except Exception as _e:
                logger.warning(f"comparison.png skipped: {_e}")

            # ----------------------------------------------------------
            # Fig 2c: health_metrics.png  (glucose + adherence over time)
            # ----------------------------------------------------------
            try:
                test_data = self.results['baselines']['test_data']
                traj_dicts = self._tuples_to_traj_dicts(test_data)
                if traj_dicts:
                    from evaluation.visualizations import EvaluationVisualizer
                    from configs.config import EvaluationConfig as _EvalCfg2
                    viz2 = EvaluationVisualizer(_EvalCfg2())
                    path_hm = self.output_dir / 'health_metrics.png'
                    viz2.plot_health_metrics(
                        traj_dicts,
                        metrics=['glucose', 'adherence_score'],
                        save_path=str(path_hm),
                    )
                    plt.close('all')
                    logger.info(f"Saved {path_hm}")
            except Exception as _e:
                logger.warning(f"health_metrics.png skipped: {_e}")

            # ----------------------------------------------------------
            # Fig 3: CQL training curves (only if CQL was trained)
            # ----------------------------------------------------------
            if self.results.get('cql', {}).get('trained', False):
                history = self.results['cql'].get('history', {})
                train_losses_raw = history.get('train_losses', [])
                # CQL stores losses as dicts; extract total_loss or q_loss scalar
                train_losses = [
                    (v.get('total_loss', v.get('q_loss', 0.0)) if isinstance(v, dict) else float(v))
                    for v in train_losses_raw
                ]
                eval_returns = history.get('eval_returns', [])

                n_panels = (1 if not train_losses else 1) + (1 if not eval_returns else 1)
                plot_items = []
                if train_losses:
                    plot_items.append(('Training Loss', train_losses, 'Iteration', 'Loss'))
                if eval_returns:
                    plot_items.append(('Eval Return', eval_returns, 'Eval Step', 'Return'))

                if plot_items:
                    fig, axes = plt.subplots(1, len(plot_items), figsize=(6 * len(plot_items), 5))
                    if len(plot_items) == 1:
                        axes = [axes]
                    for ax, (title, vals, xlabel, ylabel) in zip(axes, plot_items):
                        ax.plot(vals, linewidth=2, color='steelblue')
                        ax.set_title(title)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        ax.grid(alpha=0.3)
                    fig.suptitle('CQL Training Dynamics', fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    path3 = self.output_dir / 'cql_training_curves.png'
                    plt.savefig(path3, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved {path3}")

            # ----------------------------------------------------------
            # Fig 4: Feature importance (only if interpretability ran)
            # ----------------------------------------------------------
            if 'interpretability' in self.results:
                import json as _json
                rules_json = self.results['interpretability'].get('rules_path', '')
                if rules_json:
                    fi_path = str(rules_json).replace('.txt', '.json')
                    try:
                        with open(fi_path) as fh:
                            fi = _json.load(fh)
                        # Sort descending
                        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
                        if any(v > 0 for v in fi_sorted.values()):
                            fig, ax = plt.subplots(figsize=(9, 5))
                            bars = ax.barh(list(fi_sorted.keys()),
                                           list(fi_sorted.values()),
                                           color=sns.color_palette('viridis', len(fi_sorted)))
                            ax.set_xlabel('Feature Importance')
                            ax.set_title('Surrogate Decision Tree — Feature Importances\n'
                                         f'(Fidelity: {self.results["interpretability"].get("decision_tree_fidelity", 0):.3f})')
                            ax.grid(axis='x', alpha=0.3)
                            plt.tight_layout()
                            path4 = self.output_dir / 'feature_importance.png'
                            plt.savefig(path4, dpi=300, bbox_inches='tight')
                            plt.close()
                            logger.info(f"Saved {path4}")
                    except Exception as e:
                        logger.warning(f"Feature importance chart skipped: {e}")

                # Personalization score card
                ps = self.results['interpretability'].get('personalization_score')
                if ps is not None:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.barh(['Personalization\nScore'], [ps],
                            color='mediumseagreen' if ps > 0.5 else 'salmon')
                    ax.set_xlim(-1, 1)
                    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
                    ax.set_title(f'PDF §7.2 Personalization Score = {ps:.4f}',
                                 fontsize=12, fontweight='bold')
                    ax.set_xlabel('Cosine Similarity')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    path5 = self.output_dir / 'personalization_score.png'
                    plt.savefig(path5, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved {path5}")

            # ----------------------------------------------------------
            # Fig 5: Evaluation summary heatmap (safety + clinical)
            # ----------------------------------------------------------
            if 'evaluation' in self.results:
                safety  = self.results['evaluation']['safety']
                clinical = self.results['evaluation']['clinical']
                policies = [p for p in safety if safety[p] is not None]
                if policies:
                    safety_vals   = [safety[p]['safety_index']           if safety[p]   else 0.0 for p in policies]
                    clinical_vals = [clinical[p]['guideline_compliance']  if clinical[p] else 0.0 for p in policies]
                    heat_data = np.array([safety_vals, clinical_vals])
                    fig, ax = plt.subplots(figsize=(max(8, len(policies) * 1.2), 3.5))
                    im = ax.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
                    ax.set_xticks(range(len(policies)))
                    ax.set_xticklabels(policies, rotation=30, ha='right', fontsize=9)
                    ax.set_yticks([0, 1])
                    ax.set_yticklabels(['Safety Index', 'Guideline Compliance'])
                    for i in range(2):
                        for j, p in enumerate(policies):
                            v = heat_data[i, j]
                            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                                    fontsize=10, color='black' if 0.3 < v < 0.8 else 'white')
                    plt.colorbar(im, ax=ax, fraction=0.03)
                    ax.set_title('Safety & Clinical Metrics Heatmap')
                    plt.tight_layout()
                    path6 = self.output_dir / 'safety_clinical_heatmap.png'
                    plt.savefig(path6, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved {path6}")

            # ----------------------------------------------------------
            # Fig 6: Thesis summary PDF (combines all plots)
            # ----------------------------------------------------------
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                pdf_path = self.output_dir / 'thesis_figures.pdf'
                pngs = [p for p in self.output_dir.glob('*.png')
                        if p.name != 'thesis_figures.pdf']
                if pngs:
                    with PdfPages(pdf_path) as pdf:
                        for png in sorted(pngs):
                            img = plt.imread(str(png))
                            fig, ax = plt.subplots(
                                figsize=(img.shape[1] / 100, img.shape[0] / 100)
                            )
                            ax.imshow(img)
                            ax.axis('off')
                            ax.set_title(png.stem.replace('_', ' ').title(),
                                         fontsize=11, pad=8)
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                    logger.info(f"Saved thesis PDF: {pdf_path}")
            except Exception as e:
                logger.warning(f"Thesis PDF generation failed: {e}")

        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}", exc_info=True)

    def _tuples_to_traj_dicts(self, flat_transitions):
        """Convert flat list of (s, a, r, s', done) tuples to trajectory dicts.

        The evaluators (SafetyEvaluator, ClinicalEvaluator, EvaluationVisualizer)
        expect trajectories in the dict format:
            {'states': [state_dict, ...], 'actions': [...], 'rewards': [...],
             'next_states': [state_dict, ...], 'dones': [...]}

        Each state dict maps feature names to float values.  The synthetic state
        vector uses these 10 columns in order.  The evaluators look for keys
        'glucose' and 'bp_systolic', so we add aliases.
        """
        STATE_COLS = self.results.get('data', {}).get('state_cols', BASE_STATE_COLS)

        def _vec_to_dict(vec):
            d = {k: float(v) for k, v in zip(STATE_COLS, vec)}
            d['glucose']                  = d['glucose_mean']
            d['bp_systolic']              = d.get('sbp', 120.0)  # real if --use-vitals; fallback constant
            d['blood_pressure_systolic']  = d['bp_systolic']
            d['adherence_score']          = d['medication_taken']
            return d

        trajectories = []
        current = {'states': [], 'actions': [], 'rewards': [],
                   'next_states': [], 'dones': []}
        for item in flat_transitions:
            s, a, r, ns, done = item
            current['states'].append(_vec_to_dict(s))
            current['actions'].append(a)
            current['rewards'].append(float(r))
            current['next_states'].append(_vec_to_dict(ns))
            current['dones'].append(bool(done))
            if done:
                trajectories.append(current)
                current = {'states': [], 'actions': [], 'rewards': [],
                           'next_states': [], 'dones': []}
        if current['states']:
            trajectories.append(current)
        return trajectories

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

    # Encoder flags
    parser.add_argument('--use-encoder', action='store_true',
                        help='Pre-train a state autoencoder and use embeddings as RL state')
    parser.add_argument('--encoder-state-dim', type=int, default=64,
                        help='Latent dimension of the state encoder')
    parser.add_argument('--encoder-epochs', type=int, default=50,
                        help='Number of epochs to train the encoder')
    parser.add_argument('--encoder-type', choices=['autoencoder', 'vae'],
                        default='autoencoder',
                        help='Type of encoder: standard autoencoder or VAE')
    parser.add_argument('--encoder-checkpoint', type=str, default=None,
                        help='Path to pre-trained encoder checkpoint (skips training)')

    # CQL training flags
    parser.add_argument('--cql-iterations', type=int, default=10_000,
                        help='Number of CQL training iterations')
    parser.add_argument('--cql-batch-size', type=int, default=256,
                        help='Batch size for CQL training')

    # Policy transfer flags
    parser.add_argument('--use-transfer', action='store_true',
                        help='Adapt source policy to target population')
    parser.add_argument('--transfer-steps', type=int, default=1_000,
                        help='Number of adapter training steps')
    parser.add_argument('--target-data-dir', type=str, default=None,
                        help='Directory with target-population data (optional)')

    # Interpretability flags
    parser.add_argument('--use-interpretability', action='store_true',
                        help='Run counterfactual and decision-tree analysis')
    parser.add_argument('--n-counterfactuals', type=int, default=5,
                        help='Counterfactuals per state')
    parser.add_argument('--tree-max-depth', type=int, default=4,
                        help='Max depth for surrogate decision tree')
    parser.add_argument('--explain-n-samples', type=int, default=50,
                        help='Number of test states to explain')

    # Extended state flags
    parser.add_argument('--use-vitals', action='store_true',
                        help='Extend state with heart_rate, sbp, respiratory_rate, spo2 (+4 dims)')
    parser.add_argument('--use-med-history', action='store_true',
                        help='Extend state with adherence_rate_7d, medication_count (+2 dims)')

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