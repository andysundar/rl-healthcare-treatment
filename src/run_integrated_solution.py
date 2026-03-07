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
import os
from pathlib import Path
from datetime import datetime
import json
import hashlib
import subprocess
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt

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

from data import SyntheticDataGenerator, TrajectoryBuildConfig, DeterministicTrajectoryBuilder
from reporting import (
    ArtifactManager,
    plot_ope_returns_ci,
    plot_safety_vs_performance,
    build_defense_report,
    build_one_page_summary,
    build_figures_index,
)

from models.rl import CQLConfig, CQLAgent, IQLAgent

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


LOG_FILENAME = 'integration_run.log'


def _configure_logging() -> None:
    """Configure console + file logging for the integrated runner."""
    project_root = Path(__file__).resolve().parent.parent
    log_path = project_root / LOG_FILENAME

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(),
        ],
        force=True,
    )


_configure_logging()
logger = logging.getLogger(__name__)


def seed_all(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        seed_all(getattr(config, "seed", 42))
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

    def _get_git_commit(self) -> str:
        """Return current git commit hash, if available."""
        try:
            out = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parent.parent,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out
        except Exception:
            return "unknown"

    def _provenance(self) -> Dict[str, Any]:
        """Standard provenance metadata attached to exported artifacts."""
        data = self.results.get('data', {})
        n_train = len(data.get('train', []))
        n_val = len(data.get('val', []))
        n_test = len(data.get('test', []))
        return {
            'run_timestamp': datetime.now().isoformat(),
            'seed': int(getattr(self.config, 'seed', 42)),
            'git_commit': self._get_git_commit(),
            'data_source': data.get('source', 'unknown'),
            'n_patients': int(len(data.get('patients', data.get('cohort', [])))),
            'n_trajectories': int(n_train + n_val + n_test),
            'state_dimension': int(self._get_state_dim()),
            'action_dimension': 1,
            'reward_definition': 'in_range - 2*hypoglycemia - hyperglycemia + 0.5*medication_taken',
            'output_directory': str(self.output_dir),
        }

    def _write_json_with_provenance(self, path: Path, payload: Dict[str, Any]) -> None:
        out = dict(payload)
        out['provenance'] = self._provenance()
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

    def _normalize_output_dir_by_data_source(self) -> None:
        """Ensure folder naming reflects the actual data source."""
        source = self.results.get('data', {}).get('source', '').lower()
        cur_name = self.output_dir.name.lower()
        if source == 'synthetic' and 'mimic' in cur_name:
            new_name = self.output_dir.name.lower().replace('mimic', 'synthetic')
            new_dir = self.output_dir.parent / new_name
            new_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "Output directory name '%s' mismatches source '%s'. Using '%s' instead.",
                self.output_dir.name, source, new_dir,
            )
            self.output_dir = new_dir

    def run_full_pipeline(self):
        """Run complete end-to-end pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 80)

        if getattr(self.config, "defense_bundle", False):
            return self.run_defense_bundle()

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

        self.stage_4b_iql_training()

        if self.config.mode in ['full', 'train-eval', 'eval-only', 'synthetic']:
            self.stage_5_evaluation()

        # Policy transfer (opt-in)
        self.stage_6b_transfer()

        # Interpretability (opt-in)
        self.stage_7_interpretability()
        self.stage_7b_policy_distillation()

        self.stage_6_generate_reports()

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 80)

        return self.results

    def run_defense_bundle(self):
        """Generate complete thesis defense artifacts bundle."""
        run_id = getattr(self.config, 'run_id', datetime.now().strftime('defense_%Y%m%d_%H%M%S'))
        base_output = Path('outputs')
        am = ArtifactManager(base_output, run_id)

        # fast deterministic settings
        self.config.mode = 'synthetic'
        self.config.use_synthetic = True
        self.config.n_synthetic_patients = min(getattr(self.config, 'n_synthetic_patients', 120), 120)
        self.config.trajectory_length = min(getattr(self.config, 'trajectory_length', 14), 14)
        self.config.train_iql = True
        self.config.train_cql = False
        self.config.iql_updates = min(getattr(self.config, 'iql_updates', 300), 300)
        self.output_dir = am.run_root
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Run core pipeline stages
        self.stage_2_environment_setup()
        self.stage_3_baseline_training()
        self.stage_4b_iql_training()
        self.stage_5_evaluation()
        self.stage_7b_policy_distillation()
        self.stage_6_generate_reports()

        # Metadata artifacts
        import sys
        am.write_metadata(self.config, ' '.join(sys.argv), getattr(self.config, 'seed', 42))

        # Data summary artifacts
        raw = []
        if 'data' in self.results:
            for split in ('train', 'val', 'test'):
                for i, _ in enumerate(self.results['data'][split]):
                    pid = f"{split}_{i // max(1, self.config.trajectory_length)}"
                    raw.append({'split': split, 'subject_id': pid, 'hadm_id': pid})
        raw_df = pd.DataFrame(raw) if raw else pd.DataFrame(columns=['split', 'subject_id', 'hadm_id'])
        split_info = {
            'n_train_transitions': len(self.results.get('data', {}).get('train', [])),
            'n_val_transitions': len(self.results.get('data', {}).get('val', [])),
            'n_test_transitions': len(self.results.get('data', {}).get('test', [])),
            'n_train_subjects': raw_df[raw_df['split'] == 'train']['subject_id'].nunique() if len(raw_df) else 0,
            'n_val_subjects': raw_df[raw_df['split'] == 'val']['subject_id'].nunique() if len(raw_df) else 0,
            'n_test_subjects': raw_df[raw_df['split'] == 'test']['subject_id'].nunique() if len(raw_df) else 0,
        }
        leakage_text = """# Leakage Checks
- Split uses patient-level partitioning from deterministic builder in synthetic path.
- Missingness indicators are appended after within-episode forward-fill.
- Train/val/test transitions are generated after split to avoid cross-split contamination.
- NOTE: real-data train-only normalization is pending hardening and should be audited separately.
"""
        state_cols = self.results.get('data', {}).get('state_cols', BASE_STATE_COLS)
        for c in state_cols:
            raw_df[c] = np.nan
        am.write_data_summary(raw_df, split_info, state_cols, leakage_text)

        # Evaluation artifacts
        baseline_df = self.results.get('baselines', {}).get('comparison', pd.DataFrame()).reset_index().rename(columns={'index': 'policy'})
        if 'policy' not in baseline_df.columns and len(baseline_df):
            baseline_df['policy'] = baseline_df.index.astype(str)

        ope_rows, warnings_lines = [], ['# Support mismatch / variance warnings']
        ope = self.results.get('evaluation', {}).get('ope', {})
        for policy, methods in ope.items():
            if not methods:
                continue
            for method, m in methods.items():
                meta = m.get('metadata', {}) if isinstance(m, dict) else {}
                ope_rows.append({
                    'policy': policy,
                    'method': method,
                    'value_estimate': m.get('value_estimate'),
                    'std_error': m.get('std_error'),
                    'ci_low': m.get('confidence_interval', [None, None])[0],
                    'ci_high': m.get('confidence_interval', [None, None])[1],
                    'ess': meta.get('ess'),
                    'clip_ratio': meta.get('clip_ratio'),
                })
                for w in meta.get('warnings', []):
                    warnings_lines.append(f"- {policy}/{method}: {w}")

        safety_rows = []
        for policy, sres in self.results.get('evaluation', {}).get('safety', {}).items():
            if isinstance(sres, dict):
                safety_rows.append({
                    'policy': policy,
                    'violation_rate': 1.0 - float(sres.get('constraint_satisfaction_rate', sres.get('safety_index', 0.0))),
                    'unsafe_action_rate': float(sres.get('unsafe_action_rate', 1.0 - float(sres.get('safety_index', 0.0)))),
                    'cost_return': float(sres.get('safety_index', 0.0)),
                })

        am.write_eval_tables(
            metrics_df=baseline_df if len(baseline_df) else pd.DataFrame(columns=['policy']),
            ope_rows=ope_rows,
            safety_rows=safety_rows,
            subgroup_rows=[],
            warnings_md='\n'.join(warnings_lines) + '\n',
        )

        # Training artifacts
        train_curves = am.dirs['train'] / 'training_curves'
        (am.dirs['train'] / 'training_log.txt').write_text('Training executed via integrated runner.\n')
        src_base_plot = self.output_dir / 'baseline_comparison.png'
        if src_base_plot.exists():
            import shutil
            shutil.copy2(src_base_plot, train_curves / 'baseline_losses.png')
        src_cql_plot = self.output_dir / 'cql_training_curves.png'
        if src_cql_plot.exists():
            import shutil
            shutil.copy2(src_cql_plot, train_curves / 'cql_losses.png')
        else:
            (am.run_root / 'warnings').mkdir(parents=True, exist_ok=True)
            (am.run_root / 'warnings' / 'TRAINING_ARTIFACTS_WARNING.md').write_text(
                "# TRAINING_ARTIFACTS_WARNING\n"
                "- CQL curve artifact missing; no placeholder emitted.\n"
            )

        # Plots are optional in constrained/headless environments.
        # If unavailable, keep numeric artifacts and emit warning.
        (am.run_root / 'warnings').mkdir(parents=True, exist_ok=True)
        (am.run_root / 'warnings' / 'PLOT_ARTIFACTS_WARNING.md').write_text(
            "# PLOT_ARTIFACTS_WARNING\n"
            "- OPE/safety plots skipped in defense-bundle fast mode.\n"
            "- Use root CSV artifacts for evidence.\n"
        )

        # Interpretability artifacts
        interp_dir = am.dirs['interp']
        src_interp = self.output_dir / 'interpretability'
        rules_text = ''
        if (src_interp / 'policy_rules.txt').exists():
            rules_text = (src_interp / 'policy_rules.txt').read_text()
        if rules_text:
            (interp_dir / 'distilled_tree.txt').write_text(rules_text)
        else:
            (am.run_root / 'warnings').mkdir(parents=True, exist_ok=True)
            (am.run_root / 'warnings' / 'INTERPRETABILITY_ARTIFACTS_WARNING.md').write_text(
                "# INTERPRETABILITY_ARTIFACTS_WARNING\n"
                "- Distilled tree text missing; artifact omitted.\n"
            )

        fidelity = 0.0
        if (src_interp / 'distillation_metrics.json').exists():
            try:
                fidelity = float(json.loads((src_interp / 'distillation_metrics.json').read_text()).get('fidelity', 0.0))
            except Exception:
                fidelity = 0.0
        if fidelity > 0:
            pd.DataFrame([{'policy': 'IQL', 'fidelity': fidelity}]).to_csv(interp_dir / 'distillation_fidelity.csv', index=False)

        # Demo/final reports
        am.write_demo_assets()
        defense_md = build_defense_report(am.run_root, transfer_future_work=True)
        one_page = build_one_page_summary(am.run_root)
        fig_idx = build_figures_index()
        am.write_final_reports(defense_md, one_page, fig_idx)

        self.results['defense_bundle_dir'] = str(am.run_root)
        logger.info(f"DEFENSE BUNDLE READY: {am.run_root}")
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
            self._normalize_output_dir_by_data_source()
            logger.info("Data preparation complete")

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            if self.config.mode != 'synthetic':
                logger.info("Falling back to synthetic data...")
                data = self.prepare_synthetic_data()
                self.results['data'] = data
                self._normalize_output_dir_by_data_source()
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

        rows = []
        for patient in patients:
            traj_df = generator.simulate_patient_trajectory(
                patient=patient,
                time_horizon_days=self.config.trajectory_length,
                treatment_policy=None,
                include_vitals=use_vitals,
                include_med_history=use_med_history,
            )
            traj_df['subject_id'] = patient.patient_id
            traj_df['hadm_id'] = patient.patient_id
            traj_df['action'] = traj_df['medication_taken'].astype(float)
            traj_df['reward'] = (
                (traj_df['glucose_mean'].between(80, 180)).astype(float)
                - 2.0 * traj_df['hypoglycemia'].astype(float)
                - 1.0 * traj_df['hyperglycemia'].astype(float)
                + 0.5 * traj_df['medication_taken'].astype(float)
            )
            rows.append(traj_df)

        all_df = pd.concat(rows, ignore_index=True)
        tb_cfg = TrajectoryBuildConfig(
            patient_keys=('subject_id', 'hadm_id'),
            time_col='day',
            state_cols=state_cols,
            action_col='action',
            reward_col='reward',
            seed=42,
        )
        builder = DeterministicTrajectoryBuilder(tb_cfg)
        train_df, val_df, test_df = builder.patient_split(all_df, ratios=(0.7, 0.15, 0.15))
        train_data = builder.build(train_df)
        val_data = builder.build(val_df)
        test_data = builder.build(test_df)

        logger.info(f"Generated {len(patients)} synthetic patients")
        logger.info(f"Generated transitions Train/Val/Test: {len(train_data)}/{len(val_data)}/{len(test_data)}")

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
        """Prepare MIMIC-III data as RL trajectory tuples (same format as synthetic)."""
        logger.info("Loading MIMIC-III data...")

        from data import MIMICLoader, CohortBuilder, FeatureEngineer, DataPreprocessor

        loader = MIMICLoader(data_dir=self.config.mimic_dir, use_cache=True)

        patients    = loader.load_patients()
        admissions  = loader.load_admissions()
        diagnoses   = loader.load_diagnoses_icd()
        logger.info(f"Loaded {len(patients):,} patients")

        builder          = CohortBuilder(patients, admissions, diagnoses)
        diabetes_pts     = builder.define_diabetes_cohort()
        filtered_pts     = builder.apply_inclusion_criteria(
            diabetes_pts, min_age=18, max_age=80, min_admissions=2
        )
        final_cohort     = builder.apply_exclusion_criteria(
            filtered_pts, exclude_pregnancy=True, exclude_pediatric=True
        )
        logger.info(f"Final cohort: {len(final_cohort):,} patients")

        if self.config.use_sample:
            import random
            random.seed(42)
            final_cohort = random.sample(
                final_cohort, min(self.config.sample_size, len(final_cohort))
            )
            logger.info(f"Using sample of {len(final_cohort)} patients")

        labs          = loader.load_lab_events(subject_ids=final_cohort)
        prescriptions = loader.load_prescriptions(subject_ids=final_cohort)

        # --- Vital signs (opt-in via --use-vitals) ---
        vitals_df = None
        if getattr(self.config, 'use_vitals', False):
            logger.info("Loading CHARTEVENTS for vital signs...")
            all_vital_ids = [
                211, 220045,                         # heart_rate
                51, 442, 455, 6701, 220050, 220179,  # sbp
                618, 615, 220210, 224690,             # respiratory_rate
                646, 220277,                          # spo2
            ]
            chartevents = loader.load_chartevents(
                subject_ids=final_cohort, item_ids=all_vital_ids,
            )
            if len(chartevents) > 0:
                vitals_df = FeatureEngineer().extract_vitals_sequence(
                    chartevents, subject_ids=final_cohort
                )
                logger.info(f"Extracted vitals: {len(vitals_df):,} timepoints")

        engineer     = FeatureEngineer()
        demographics = engineer.extract_demographics(patients, admissions)
        demographics = demographics[demographics['subject_id'].isin(final_cohort)]

        if len(labs) == 0:
            raise ValueError(
                "No lab events found for cohort. "
                "Verify --mimic-dir points to the MIMIC-III CSV directory."
            )

        lab_sequence = engineer.extract_lab_sequence(labs, final_cohort)
        if len(lab_sequence) == 0:
            raise ValueError("Lab sequence extraction produced no rows.")

        # Deduplicate before computing per-row indices
        lab_sequence = (
            lab_sequence
            .drop_duplicates(subset=['subject_id', 'hadm_id', 'charttime'])
            .sort_values(['subject_id', 'hadm_id', 'charttime'])
            .reset_index(drop=True)
        )

        # --- Clinical flags from RAW glucose (before any normalization) ---
        clin_flags = lab_sequence[['subject_id', 'hadm_id', 'charttime']].copy()
        if 'glucose' in lab_sequence.columns:
            clin_flags['hypoglycemia']  = (lab_sequence['glucose'] < 70.0).astype(float)
            clin_flags['hyperglycemia'] = (lab_sequence['glucose'] > 180.0).astype(float)
        else:
            clin_flags['hypoglycemia']  = 0.0
            clin_flags['hyperglycemia'] = 0.0

        # --- Day index per admission (0-indexed step within each hadm_id) ---
        day_idx = lab_sequence[['subject_id', 'hadm_id', 'charttime']].copy()
        day_idx['day_raw'] = (
            lab_sequence.groupby(['subject_id', 'hadm_id']).cumcount().astype(float)
        )

        # --- Insulin features from prescriptions ---
        insulin_daily = self._extract_insulin_daily(prescriptions, set(final_cohort))

        # --- Standard preprocessing pipeline (only on lab values) ---
        temporal_features = engineer.create_temporal_features(
            lab_sequence, time_column='charttime'
        )
        preprocessor   = DataPreprocessor()
        clean_data     = preprocessor.clean_missing_values(temporal_features)
        clean_data     = preprocessor.handle_outliers(clean_data)
        normalized_data = preprocessor.normalize_labs(clean_data)

        # --- Join supplementary features (outside normalizer to protect binary cols) ---
        # Clinical flags
        normalized_data = normalized_data.merge(
            clin_flags, on=['subject_id', 'hadm_id', 'charttime'], how='left'
        )
        # Day index → normalize to [0, 1] within each admission
        normalized_data = normalized_data.merge(
            day_idx, on=['subject_id', 'hadm_id', 'charttime'], how='left'
        )
        max_days = (
            normalized_data.groupby(['subject_id', 'hadm_id'])['day_raw']
            .transform('max').replace(0, 1)
        )
        normalized_data['day'] = (normalized_data['day_raw'] / max_days).fillna(0.0)

        # Insulin (join by subject_id + date)
        normalized_data['_jdate'] = pd.to_datetime(
            normalized_data['charttime'], errors='coerce'
        ).dt.date
        if len(insulin_daily) > 0:
            normalized_data = normalized_data.merge(
                insulin_daily.rename(columns={'_lab_date': '_jdate'}),
                on=['subject_id', '_jdate'], how='left',
            )
        for col in ['insulin_flag', 'insulin_dose']:
            if col not in normalized_data.columns:
                normalized_data[col] = 0.0
            else:
                normalized_data[col] = normalized_data[col].fillna(0.0)
        normalized_data = normalized_data.drop(columns=['_jdate'], errors='ignore')

        # Vitals (daily aggregate → join by subject_id + hadm_id + date)
        if getattr(self.config, 'use_vitals', False) and vitals_df is not None and len(vitals_df) > 0:
            avail_vcols = [c for c in VITAL_COLS if c in vitals_df.columns]
            if avail_vcols:
                vdf        = vitals_df.copy()
                vdf['_vd'] = pd.to_datetime(vdf['charttime'], errors='coerce').dt.date
                vagg       = (
                    vdf.groupby(['subject_id', 'hadm_id', '_vd'])[avail_vcols]
                    .mean().reset_index().rename(columns={'_vd': '_jdate'})
                )
                normalized_data['_jdate'] = pd.to_datetime(
                    normalized_data['charttime'], errors='coerce'
                ).dt.date
                normalized_data = normalized_data.merge(
                    vagg, on=['subject_id', 'hadm_id', '_jdate'], how='left'
                )
                normalized_data = normalized_data.drop(columns=['_jdate'], errors='ignore')
        for col in VITAL_COLS:
            if col not in normalized_data.columns:
                normalized_data[col] = 0.0
            else:
                normalized_data[col] = normalized_data[col].fillna(0.0)

        # Medication history
        if getattr(self.config, 'use_med_history', False):
            med_hist = self._compute_med_history_features(prescriptions, set(final_cohort))
            normalized_data = normalized_data.merge(med_hist, on='subject_id', how='left')
        for col in MED_HISTORY_COLS:
            if col not in normalized_data.columns:
                normalized_data[col] = 0.0
            else:
                normalized_data[col] = normalized_data[col].fillna(0.0)

        # Clean up utility columns
        normalized_data = normalized_data.drop(columns=['day_raw'], errors='ignore')

        # --- Map MIMIC columns → BASE_STATE_COLS naming ---
        normalized_data = self._build_mimic_state_columns(normalized_data)

        state_cols = self._get_state_cols()
        for col in state_cols:
            if col not in normalized_data.columns:
                logger.warning("State column '%s' missing from MIMIC data, filling 0.", col)
                normalized_data[col] = 0.0

        # --- Patient-level train / val / test split ---
        unique_pts = list(set(final_cohort))
        rng        = np.random.default_rng(42)
        rng.shuffle(unique_pts)
        n_train = int(0.70 * len(unique_pts))
        n_val   = int(0.15 * len(unique_pts))
        train_pts = set(unique_pts[:n_train])
        val_pts   = set(unique_pts[n_train:n_train + n_val])
        test_pts  = set(unique_pts[n_train + n_val:])

        train_data = self._build_mimic_trajectories(
            normalized_data[normalized_data['subject_id'].isin(train_pts)], state_cols
        )
        val_data = self._build_mimic_trajectories(
            normalized_data[normalized_data['subject_id'].isin(val_pts)], state_cols
        )
        test_data = self._build_mimic_trajectories(
            normalized_data[normalized_data['subject_id'].isin(test_pts)], state_cols
        )

        data = {
            'train':        train_data,
            'val':          val_data,
            'test':         test_data,
            'demographics': demographics,
            'cohort':       final_cohort,
            'source':       'mimic',
            'state_cols':   state_cols,
        }
        logger.info(
            f"MIMIC trajectories — Train: {len(train_data):,} | "
            f"Val: {len(val_data):,} | Test: {len(test_data):,}"
        )
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
            self._normalize_output_dir_by_data_source()
        else:
            data = self.results['data']

        from models.encoders import PatientAutoencoder, EncoderConfig
        from models.encoders.state_encoder_wrapper import StateEncoderWrapper

        # StateEncoderWrapper always passes the full flat state as "labs";
        # vital_dim / demo_dim stay 0 to avoid MPS device-mismatch on missing modalities.
        configured_state_dim = int(self._get_state_dim())

        def _flat_dim(x) -> int:
            return int(np.asarray(x).reshape(-1).shape[0])

        train_dims = [_flat_dim(t[0]) for t in data.get('train', [])]
        raw_state_dim = max(train_dims) if train_dims else configured_state_dim
        if raw_state_dim != configured_state_dim:
            logger.warning(
                "Encoder input dim mismatch detected (configured=%s, inferred=%s). "
                "Using inferred dimension from training transitions.",
                configured_state_dim,
                raw_state_dim,
            )

        def _normalize_transitions_state_dim(transitions, target_dim: int):
            normalized = []
            adjusted = 0
            for s, a, r, ns, d in transitions:
                s_arr = np.asarray(s, dtype=np.float32).reshape(-1)
                ns_arr = np.asarray(ns, dtype=np.float32).reshape(-1)

                if s_arr.shape[0] != target_dim:
                    adjusted += 1
                    if s_arr.shape[0] < target_dim:
                        s_arr = np.pad(s_arr, (0, target_dim - s_arr.shape[0]), mode='constant')
                    else:
                        s_arr = s_arr[:target_dim]

                if ns_arr.shape[0] != target_dim:
                    adjusted += 1
                    if ns_arr.shape[0] < target_dim:
                        ns_arr = np.pad(ns_arr, (0, target_dim - ns_arr.shape[0]), mode='constant')
                    else:
                        ns_arr = ns_arr[:target_dim]

                normalized.append((s_arr, a, r, ns_arr, d))
            return normalized, adjusted

        for split in ('train', 'val', 'test'):
            if split in data:
                data[split], adjusted = _normalize_transitions_state_dim(data[split], raw_state_dim)
                if adjusted:
                    logger.warning(
                        "Normalized %s state vectors to dim=%s in '%s' split.",
                        adjusted,
                        raw_state_dim,
                        split,
                    )

        enc_cfg = EncoderConfig(
            lab_dim=raw_state_dim,
            vital_dim=0,
            demo_dim=0,
            state_dim=getattr(self.config, 'encoder_state_dim', 64),
        )
        variational = (getattr(self.config, 'encoder_type', 'autoencoder') == 'vae')
        ae = PatientAutoencoder(enc_cfg, variational=variational)
        device = self.get_device()
        wrapper = StateEncoderWrapper(ae, device=device, raw_state_dim=raw_state_dim)

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
            self._normalize_output_dir_by_data_source()
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

        logger.info("\nEvaluating all baselines with per-policy rollouts...")
        rollout_payload = self._run_policy_rollout_evaluation(
            policies=baselines,
            train_data=train_data,
            eval_data=test_data,
            export_prefix='baseline',
        )
        results_df = rollout_payload['summary_df']
        self._export_baseline_reports(results_df)

        logger.info("\n" + "=" * 60)
        logger.info("BASELINE COMPARISON RESULTS")
        logger.info("=" * 60)
        logger.info("\n" + results_df.to_string())

        self.results['baselines'] = {
            'policies': baselines,
            'comparison': results_df,
            'test_data': test_data,
            'raw_eval_path': rollout_payload['raw_eval_path'],
            'episode_summary_path': rollout_payload['episode_summary_path'],
        }
        self.results['baseline_rollouts'] = rollout_payload
        logger.info("Baseline training complete")

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
        state_dim = int(config.state_dim)
        if 'encoded_data' in self.results:
            train_data = self.results['encoded_data']['train']
            val_data   = self.results['encoded_data']['val']
            state_dim  = int(self.results['encoder_wrapper'].state_dim)
            logger.info(f"Using encoder embeddings as states (dim={state_dim})")
        elif data['source'] == 'synthetic':
            train_data = data['train']
            val_data   = data['val']
        else:
            train_data = self._convert_to_trajectory_format(data['train'])
            val_data   = self._convert_to_trajectory_format(data['val'])

        # Always infer dimension from actual transitions to avoid config/data drift
        # (e.g., MIMIC feature builders can emit a wider state than CLI defaults).
        if train_data:
            inferred_state_dim = len(np.asarray(train_data[0][0]).reshape(-1))
            if inferred_state_dim != state_dim:
                logger.warning(
                    "CQL state_dim mismatch detected (configured=%s, inferred=%s). "
                    "Using inferred dimension from training transitions.",
                    state_dim,
                    inferred_state_dim,
                )
            state_dim = inferred_state_dim

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

    def stage_4b_iql_training(self):
        """Optional discrete-action IQL training (enabled in --demo)."""
        if not getattr(self.config, 'train_iql', False):
            return
        if 'data' not in self.results:
            logger.warning("No data available, skipping IQL training")
            return
        data = self.results['data']
        train_data = data['train']
        state_dim = len(train_data[0][0])
        n_actions = int(getattr(self.config, 'discrete_actions', 2))
        from models.rl.iql import IQLConfig
        cfg = IQLConfig(state_dim=state_dim, n_actions=n_actions, device=str(self.get_device()))
        agent = IQLAgent(cfg)

        from models.safety.config import SafetyConfig as SafetyLayerConfig
        from models.safety.safety_layer import SafetyLayer
        safety = SafetyLayer(SafetyLayerConfig())

        rng = np.random.default_rng(42)
        for _ in range(getattr(self.config, 'iql_updates', 200)):
            idx = rng.integers(0, len(train_data), size=min(64, len(train_data)))
            b = [train_data[i] for i in idx]
            states = torch.tensor(np.stack([x[0] for x in b]), dtype=torch.float32)
            actions = np.array([int(np.clip(round(float(x[1][0])), 0, n_actions - 1)) for x in b], dtype=np.int64)
            safe_actions = np.array([safety.apply_discrete_action_mask(s, a, n_actions) for s, a in zip(states.numpy(), actions)], dtype=np.int64)
            batch = {
                'states': states,
                'actions': torch.tensor(safe_actions[:, None], dtype=torch.float32),
                'rewards': torch.tensor(np.array([x[2] for x in b], dtype=np.float32)[:, None]),
                'next_states': torch.tensor(np.stack([x[3] for x in b]), dtype=torch.float32),
                'dones': torch.tensor(np.array([x[4] for x in b], dtype=np.float32)[:, None]),
            }
            agent.update(batch)
        self.results['iql'] = {'agent': agent, 'trained': True, 'state_dim': state_dim, 'n_actions': n_actions}
        logger.info("IQL training complete")

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
        train_data = self.results['data']['train']
        eval_policies = dict(self.results['baselines']['policies'])
        if 'iql' in self.results and self.results['iql'].get('agent') is not None:
            eval_policies['IQL'] = self.results['iql']['agent']

        # Per-policy rollouts used as source of truth for reward/safety summaries.
        rollout_eval = self._run_policy_rollout_evaluation(
            policies=eval_policies,
            train_data=train_data,
            eval_data=test_data,
            export_prefix='evaluation',
        )
        summary_df = rollout_eval['summary_df']

        # Off-policy evaluation
        logger.info("Running off-policy evaluation...")
        q_function = self._build_ope_q_function(train_data)
        ope_evaluator = OffPolicyEvaluator(q_function=q_function, clip_ratio=10.0, n_bootstrap=300, seed=getattr(self.config, 'seed', 42))
        behavior_policy = eval_policies.get('Behavior-Cloning', next(iter(eval_policies.values())))
        ope_results = {}
        ope_trajectories = self._to_ope_trajectories(test_data)

        class _ProbWrapper:
            def __init__(self, base_policy, sigma: float):
                self.base_policy = base_policy
                self.sigma = sigma

            def select_action(self, state, deterministic=True):
                return self.base_policy.select_action(state, deterministic=deterministic)

            def get_action_probability(self, state, action):
                if hasattr(self.base_policy, 'get_action_probability'):
                    return float(self.base_policy.get_action_probability(state, action))
                pred = np.asarray(self.base_policy.select_action(state, deterministic=True), dtype=np.float32).reshape(-1)
                act = np.asarray(action, dtype=np.float32).reshape(-1)
                var = self.sigma ** 2
                logp = -0.5 * np.sum(((act - pred) ** 2) / var + np.log(2 * np.pi * var))
                return float(np.exp(np.clip(logp, -40, 5)))

        action_values = np.asarray([float(np.asarray(t[1]).reshape(-1)[0]) for t in train_data], dtype=np.float32)
        sigma = float(max(0.05, np.std(action_values)))
        wrapped_behavior = _ProbWrapper(behavior_policy, sigma=sigma)
        ope_rows = []
        for name, policy in eval_policies.items():
            logger.info(f"  OPE for {name}...")
            try:
                wrapped_target = _ProbWrapper(policy, sigma=sigma)
                results = ope_evaluator.evaluate(
                    policy=wrapped_target,
                    behavior_policy=wrapped_behavior,
                    trajectories=ope_trajectories,
                    methods=['is', 'wis', 'dr', 'dm'],
                )
                ope_results[name] = {
                    m: {
                        'value_estimate': float(v.value_estimate),
                        'std_error': float(v.std_error),
                        'confidence_interval': [float(v.confidence_interval[0]), float(v.confidence_interval[1])],
                        'metadata': v.metadata,
                    } for m, v in results.items()
                }
                for estimator, payload in ope_results[name].items():
                    meta = payload.get('metadata', {})
                    ope_rows.append({
                        'policy': name,
                        'estimator': estimator.upper(),
                        'value_estimate': payload['value_estimate'],
                        'std_error': payload['std_error'],
                        'ci_lower': payload['confidence_interval'][0],
                        'ci_upper': payload['confidence_interval'][1],
                        'ess': float(meta.get('ess', 0.0)),
                        'reliability_flag': str(meta.get('reliability_flag', 'unknown')),
                        'warning': " | ".join(meta.get('warnings', [])),
                        'clipping_threshold_used': float(meta.get('clip_ratio', np.nan)),
                    })
            except Exception as e:
                logger.warning(f"    OPE failed for {name}: {e}")
                ope_results[name] = None

        ope_df = pd.DataFrame(ope_rows)
        if not ope_df.empty:
            for c in ['data_source', 'seed', 'git_commit']:
                if c == 'data_source':
                    ope_df[c] = self.results.get('data', {}).get('source', 'unknown')
                elif c == 'seed':
                    ope_df[c] = int(getattr(self.config, 'seed', 42))
                else:
                    ope_df[c] = self._get_git_commit()
            ope_df.to_csv(self.output_dir / 'ope_estimates.csv', index=False)

            pivot = ope_df.pivot_table(index='policy', columns='estimator', values='value_estimate', aggfunc='first')
            for col in ['WIS', 'DR']:
                if col in pivot.columns and len(pivot[col].dropna().unique()) <= 1 and len(pivot[col].dropna()) > 1:
                    raise RuntimeError(f"Degenerate OPE detected: all policies share identical {col} estimate.")

        # Safety/clinical metrics from rollout summary (single source of truth).
        safety_results = {}
        clinical_results = {}
        for policy_name, row in summary_df.iterrows():
            safety_results[policy_name] = {
                'unsafe_action_rate': float(row['unsafe_action_rate']),
                'safety_violations_count': int(row['safety_violations_count']),
                'constraint_satisfaction_rate': float(row['constraint_satisfaction_rate']),
                'safety_index': float(row['constraint_satisfaction_rate']),
                'safety_level': (
                    'HIGH' if row['constraint_satisfaction_rate'] > 0.95
                    else 'MEDIUM' if row['constraint_satisfaction_rate'] > 0.85
                    else 'LOW'
                ),
            }
            clinical_results[policy_name] = {
                'guideline_compliance': float(row['constraint_satisfaction_rate'])
            }

        safety_df = summary_df.reset_index()[[
            'policy', 'unsafe_action_rate', 'safety_violations_count', 'constraint_satisfaction_rate'
        ]].copy()
        safety_df['data_source'] = self.results.get('data', {}).get('source', 'unknown')
        safety_df['seed'] = int(getattr(self.config, 'seed', 42))
        safety_df['git_commit'] = self._get_git_commit()
        safety_df.to_csv(self.output_dir / 'safety_summary.csv', index=False)

        self.results['evaluation'] = {
            'ope': ope_results,
            'ope_table': ope_df,
            'safety': safety_results,
            'clinical': clinical_results,
            'rollout_summary': summary_df,
            'rollout_raw_path': rollout_eval['raw_eval_path'],
            'rollout_episode_summary_path': rollout_eval['episode_summary_path'],
        }

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)

        for name in eval_policies.keys():
            logger.info(f"\n{name}:")
            if safety_results.get(name):
                s = safety_results[name]
                logger.info(f"  Safety Index: {s['safety_index']:.4f} ({s['safety_level']})")
            if clinical_results.get(name):
                c = clinical_results[name]
                logger.info(f"  Guideline Compliance: {c['guideline_compliance']:.2%}")

        logger.info("\nEvaluation complete")

    def _build_ope_q_function(self, transitions):
        """Build a callable Q-function for DR OPE.

        Priority:
        1) Reuse trained CQL critic when available.
        2) Reuse trained IQL critic when available.
        3) Fit a lightweight linear reward model as fallback.
        """
        cql_agent = self.results.get('cql', {}).get('agent')
        if cql_agent is not None and hasattr(cql_agent, 'q_network1'):
            logger.info("Using trained CQL critic as DR q_function")
            expected_state_dim = int(getattr(cql_agent, 'state_dim', 0)) if hasattr(cql_agent, 'state_dim') else 0
            encoder_wrapper = self.results.get('encoder_wrapper')

            def _cql_q_fn(state, action):
                with torch.no_grad():
                    s_raw = np.asarray(state, dtype=np.float32).reshape(-1)
                    if expected_state_dim and s_raw.shape[0] != expected_state_dim and encoder_wrapper is not None:
                        try:
                            s_raw = np.asarray(encoder_wrapper.encode_state(s_raw), dtype=np.float32).reshape(-1)
                        except Exception:
                            pass
                    if expected_state_dim and s_raw.shape[0] != expected_state_dim:
                        if s_raw.shape[0] < expected_state_dim:
                            s_raw = np.pad(s_raw, (0, expected_state_dim - s_raw.shape[0]), mode='constant')
                        else:
                            s_raw = s_raw[:expected_state_dim]

                    s = s_raw.reshape(1, -1)
                    a = np.asarray(action, dtype=np.float32).reshape(1, -1)
                    s_t = torch.tensor(s, dtype=torch.float32, device=cql_agent.device)
                    a_t = torch.tensor(a, dtype=torch.float32, device=cql_agent.device)
                    q_val = cql_agent.q_network1(s_t, a_t).detach().cpu().numpy().reshape(-1)[0]
                    return float(q_val)

            return _cql_q_fn

        iql_agent = self.results.get('iql', {}).get('agent')
        if iql_agent is not None and hasattr(iql_agent, 'q'):
            logger.info("Using trained IQL critic as DR q_function")

            def _iql_q_fn(state, action):
                with torch.no_grad():
                    s = np.asarray(state, dtype=np.float32).reshape(1, -1)
                    a = int(np.clip(np.rint(np.asarray(action).reshape(-1)[0]), 0, iql_agent.config.n_actions - 1))
                    s_t = torch.tensor(s, dtype=torch.float32, device=iql_agent.device)
                    q_all = iql_agent.q(s_t)
                    return float(q_all[0, a].item())

            return _iql_q_fn

        logger.info("No trained critic found; fitting linear fallback q_function for DR")
        states, actions, rewards = [], [], []
        for s, a, r, _, _ in transitions:
            s_arr = np.asarray(s, dtype=np.float32).reshape(-1)
            a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
            states.append(s_arr)
            actions.append(a_arr)
            rewards.append(float(r))

        if not states:
            logger.warning("Transitions empty; using zero q_function fallback")
            return lambda state, action: 0.0

        s_mat = np.stack(states)
        a_mat = np.stack(actions)
        x = np.concatenate([s_mat, a_mat, np.ones((s_mat.shape[0], 1), dtype=np.float32)], axis=1)
        y = np.asarray(rewards, dtype=np.float32)

        l2 = 1e-3
        reg = l2 * np.eye(x.shape[1], dtype=np.float32)
        weights = np.linalg.solve(x.T @ x + reg, x.T @ y)

        def _linear_q_fn(state, action):
            s_arr = np.asarray(state, dtype=np.float32).reshape(-1)
            a_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            feat = np.concatenate([s_arr, a_arr, np.array([1.0], dtype=np.float32)], axis=0)
            return float(feat @ weights)

        return _linear_q_fn

    def stage_7b_policy_distillation(self):
        """Distill learned policy to shallow decision tree and save rules/fidelity."""
        if not getattr(self.config, 'run_distillation', True):
            return
        if 'data' not in self.results:
            return
        from sklearn.tree import DecisionTreeClassifier, export_text
        data = self.results['data']
        agent = None
        agent_name = None
        encoder_wrapper = self.results.get('encoder_wrapper')
        if self.results.get('cql', {}).get('agent') is not None:
            agent = self.results['cql']['agent']
            agent_name = 'CQL'
        elif self.results.get('iql', {}).get('agent') is not None:
            agent = self.results['iql']['agent']
            agent_name = 'IQL'
        if agent is None:
            return

        states = np.array([t[0] for t in data['test']], dtype=np.float32)
        if len(states) == 0:
            return
        raw_actions = []
        for s in states:
            s_for_policy = s
            if agent_name == 'CQL' and encoder_wrapper is not None:
                try:
                    s_for_policy = np.asarray(encoder_wrapper.encode_state(s), dtype=np.float32).reshape(-1)
                except Exception:
                    s_for_policy = s
            a = agent.select_action(s_for_policy, deterministic=True)
            raw_actions.append(float(np.asarray(a).reshape(-1)[0]))
        raw_actions = np.asarray(raw_actions, dtype=np.float32)
        q1, q2 = np.quantile(raw_actions, [0.33, 0.67])
        y = np.digitize(raw_actions, bins=[q1, q2]).astype(int)
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(states, y)
        pred = tree.predict(states)
        fidelity = float(np.mean(pred == y))
        out_dir = self.output_dir / 'interpretability'
        out_dir.mkdir(parents=True, exist_ok=True)
        feature_names = list(self.results.get('data', {}).get('state_cols', BASE_STATE_COLS))
        if len(feature_names) < states.shape[1]:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), states.shape[1])])
        rules = export_text(tree, feature_names=list(feature_names[:states.shape[1]]))

        importances = {
            (feature_names[i] if i < len(feature_names) else f'feature_{i}'): float(v)
            for i, v in enumerate(tree.feature_importances_)
        }
        tree_depth = int(tree.get_depth())
        all_zero_importance = all(abs(v) <= 1e-12 for v in importances.values())
        if tree_depth == 0 or all_zero_importance:
            (self.output_dir / 'INTERPRETABILITY_WARNING.md').write_text(
                "# INTERPRETABILITY_WARNING\n"
                "- Surrogate tree is degenerate (single leaf or zero importances).\n"
                "- decision_rules.txt and feature_importances.json omitted.\n"
            )
            for p in [self.output_dir / 'decision_rules.txt', self.output_dir / 'feature_importances.json']:
                if p.exists():
                    p.unlink()
        else:
            (out_dir / 'policy_rules.txt').write_text(rules)
            (self.output_dir / 'decision_rules.txt').write_text(rules)
            (out_dir / 'feature_importances.json').write_text(json.dumps(importances, indent=2))
            (self.output_dir / 'feature_importances.json').write_text(json.dumps(importances, indent=2))
        metrics_payload = {
            'policy': agent_name,
            'fidelity': fidelity,
            'tree_depth': tree_depth,
            'leaf_count': int(tree.get_n_leaves()),
            'n_samples': int(len(states)),
            'class_bins': [float(q1), float(q2)],
        }
        (out_dir / 'distillation_metrics.json').write_text(json.dumps(metrics_payload, indent=2))

        # Counterfactual examples from nearest valid state with different predicted action.
        cf_examples = []
        for i in range(min(len(states), 200)):
            s0 = states[i]
            a0 = int(y[i])
            l1 = np.abs(states - s0).sum(axis=1)
            candidate_idx = np.where(y != a0)[0]
            if len(candidate_idx) == 0:
                continue
            j = int(candidate_idx[np.argmin(l1[candidate_idx])])
            s1 = states[j]
            a1 = int(y[j])
            deltas = s1 - s0
            nz = np.where(np.abs(deltas) > 1e-4)[0]
            if len(nz) == 0:
                continue
            top = nz[np.argsort(np.abs(deltas[nz]))[::-1][:3]]
            changed = {
                (feature_names[k] if k < len(feature_names) else f'feature_{k}'): float(deltas[k])
                for k in top
            }
            explanation = f"Action bin changed from {a0} to {a1} after minimal state shift in top features."
            cf_examples.append({
                'original_state_summary': {feature_names[k] if k < len(feature_names) else f'feature_{k}': float(s0[k]) for k in top},
                'original_action': a0,
                'minimally_changed_features': changed,
                'new_action': a1,
                'explanation_text': explanation,
            })
            if len(cf_examples) >= max(10, int(getattr(self.config, 'n_counterfactuals', 10))):
                break

        cf_path = self.output_dir / 'counterfactuals.json'
        if len(cf_examples) >= 10:
            cf_path.write_text(json.dumps(cf_examples, indent=2))
        else:
            warning = (
                "# INTERPRETABILITY_WARNING\n"
                f"- Could not generate >=10 valid counterfactuals (generated {len(cf_examples)}).\n"
                "- Counterfactual artifact omitted to avoid placeholder output.\n"
            )
            (self.output_dir / 'INTERPRETABILITY_WARNING.md').write_text(warning)
            if cf_path.exists():
                cf_path.unlink()

        if fidelity < 0.6:
            (self.output_dir / 'INTERPRETABILITY_WARNING.md').write_text(
                "# INTERPRETABILITY_WARNING\n"
                f"- Distillation fidelity below threshold: {fidelity:.4f} < 0.60.\n"
                "- Interpretability claims should be treated as unreliable.\n"
            )

        self.results['policy_distillation'] = {
            'fidelity': fidelity,
            'rules_path': str(out_dir / 'policy_rules.txt') if (out_dir / 'policy_rules.txt').exists() else None,
            'feature_importances_path': str(out_dir / 'feature_importances.json') if (out_dir / 'feature_importances.json').exists() else None,
            'tree_depth': int(tree.get_depth()),
            'leaf_count': int(tree.get_n_leaves()),
            'counterfactuals_path': str(cf_path) if cf_path.exists() else None,
        }
        logger.info(f"Policy distillation complete ({agent_name}). fidelity={fidelity:.4f}")

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
            'baselines_evaluated': list(self.results.get('baselines', {}).get('policies', {}).keys()),
            'output_directory': str(self.output_dir),
            'dataset_identity': {
                'data_source': self.results.get('data', {}).get('source', 'unknown'),
                'n_patients': len(self.results.get('data', {}).get('patients', self.results.get('data', {}).get('cohort', []))),
                'n_trajectories': len(self.results.get('data', {}).get('train', [])) + len(self.results.get('data', {}).get('val', [])) + len(self.results.get('data', {}).get('test', [])),
                'state_dimension': self._get_state_dim(),
                'action_dimension': 1,
                'reward_definition': 'in_range - 2*hypoglycemia - hyperglycemia + 0.5*medication_taken',
                'seed': int(getattr(self.config, 'seed', 42)),
                'git_commit': self._get_git_commit(),
            },
        }

        if 'evaluation' in self.results:
            summary['evaluation'] = {
                'safety_metrics': self.results['evaluation'].get('safety', {}),
                'clinical_metrics': {
                    name: res['guideline_compliance'] if res else None
                    for name, res in self.results['evaluation']['clinical'].items()
                },
                'ope_results': self.results['evaluation'].get('ope', {}),
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

        self._write_json_with_provenance(summary_path, summary)

        logger.info(f"Results summary saved to {summary_path}")

        self._generate_latex_table()
        self._generate_master_results()
        self._write_limitations_file()
        if not getattr(self.config, 'demo', False) and not getattr(self.config, 'defense_bundle', False):
            self._generate_visualizations()
        else:
            logger.info("Visualization generation skipped in demo/defense-bundle mode.")
        self._run_artifact_validation()

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

    def _action_bounds_for_policy(self, policy) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(policy, 'get_action_bounds'):
            low, high = policy.get_action_bounds()
            return np.asarray(low, dtype=np.float32).reshape(-1), np.asarray(high, dtype=np.float32).reshape(-1)
        if hasattr(policy, 'config') and hasattr(policy.config, 'n_actions'):
            return np.array([0.0], dtype=np.float32), np.array([float(max(0, policy.config.n_actions - 1))], dtype=np.float32)
        return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

    def _safe_policy_action(self, policy, state: np.ndarray) -> Tuple[float, float]:
        raw_action = policy.select_action(state, deterministic=True)
        raw_scalar = float(np.asarray(raw_action, dtype=np.float32).reshape(-1)[0])
        low, high = self._action_bounds_for_policy(policy)
        clipped = float(np.clip(raw_scalar, low[0], high[0]))
        return raw_scalar, clipped

    def _fit_linear_simulator(self, transitions: List[Tuple]) -> Dict[str, np.ndarray]:
        if not transitions:
            raise ValueError("Cannot fit simulator: transitions are empty.")
        states = np.asarray([np.asarray(t[0], dtype=np.float32).reshape(-1) for t in transitions], dtype=np.float32)
        actions = np.asarray([float(np.asarray(t[1]).reshape(-1)[0]) for t in transitions], dtype=np.float32).reshape(-1, 1)
        next_states = np.asarray([np.asarray(t[3], dtype=np.float32).reshape(-1) for t in transitions], dtype=np.float32)
        rewards = np.asarray([float(t[2]) for t in transitions], dtype=np.float32).reshape(-1, 1)

        x = np.concatenate([states, actions, np.ones((len(states), 1), dtype=np.float32)], axis=1)
        l2 = 1e-2
        reg = l2 * np.eye(x.shape[1], dtype=np.float32)
        xtx_inv = np.linalg.inv(x.T @ x + reg)
        w_next = xtx_inv @ x.T @ next_states
        w_reward = xtx_inv @ x.T @ rewards
        return {'w_next': w_next, 'w_reward': w_reward}

    def _simulate_step_with_model(self, state: np.ndarray, action: float, model: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        feat = np.concatenate([state.reshape(-1), np.array([action, 1.0], dtype=np.float32)], axis=0)
        next_state = feat @ model['w_next']
        reward = float((feat @ model['w_reward']).reshape(-1)[0])
        return np.asarray(next_state, dtype=np.float32), reward

    def _state_hash(self, state: np.ndarray) -> str:
        rounded = np.asarray(state, dtype=np.float32).round(4)
        return hashlib.md5(rounded.tobytes()).hexdigest()[:12]

    def _run_policy_rollout_evaluation(
        self,
        policies: Dict[str, Any],
        train_data: List[Tuple],
        eval_data: List[Tuple],
        export_prefix: str,
    ) -> Dict[str, Any]:
        """Evaluate each policy with independent model-based rollouts and export raw traces."""
        sim_model = self._fit_linear_simulator(train_data)
        episodes = self._transitions_to_episodes(eval_data)
        safety_cfg = EvaluationConfig().safety
        g_low, g_high = safety_cfg.safe_glucose_range
        raw_rows = []
        ep_rows = []
        summary_rows = []
        action_signatures: Dict[str, str] = {}

        for policy_name, policy in policies.items():
            per_policy_actions = []
            episode_returns = []
            episode_lengths = []
            unsafe_count = 0
            total_steps = 0

            for ep_id, ep in enumerate(episodes):
                if not ep['states']:
                    continue
                state = np.asarray(ep['states'][0], dtype=np.float32).reshape(-1)
                ep_return = 0.0

                for step_id in range(len(ep['states'])):
                    raw_action, clipped_action = self._safe_policy_action(policy, state)
                    next_state, reward = self._simulate_step_with_model(state, clipped_action, sim_model)
                    glucose = float(next_state[0]) if len(next_state) > 0 else 0.0
                    unsafe = bool(glucose < g_low or glucose > g_high)
                    done = step_id == (len(ep['states']) - 1)
                    constraint_satisfied = not unsafe
                    unsafe_count += int(unsafe)
                    total_steps += 1
                    per_policy_actions.append(clipped_action)
                    ep_return += reward

                    raw_rows.append({
                        'policy_name': policy_name,
                        'episode_id': ep_id,
                        'step_id': step_id,
                        'state_hash': self._state_hash(state),
                        'action': raw_action,
                        'clipped_action': clipped_action,
                        'reward': reward,
                        'done': done,
                        'unsafe_flag': unsafe,
                        'constraint_satisfied_flag': constraint_satisfied,
                        'data_source': self.results.get('data', {}).get('source', 'unknown'),
                        'seed': int(getattr(self.config, 'seed', 42)),
                        'git_commit': self._get_git_commit(),
                    })
                    state = next_state

                episode_returns.append(ep_return)
                episode_lengths.append(len(ep['states']))
                ep_rows.append({
                    'policy_name': policy_name,
                    'episode_id': ep_id,
                    'episode_return': ep_return,
                    'episode_length': len(ep['states']),
                    'unsafe_steps': int(np.sum(
                        [r['unsafe_flag'] for r in raw_rows if r['policy_name'] == policy_name and r['episode_id'] == ep_id]
                    )),
                    'data_source': self.results.get('data', {}).get('source', 'unknown'),
                    'seed': int(getattr(self.config, 'seed', 42)),
                    'git_commit': self._get_git_commit(),
                })

            if total_steps == 0:
                raise RuntimeError(f"No rollout steps generated for policy {policy_name}.")

            action_sig = hashlib.md5(np.asarray(per_policy_actions, dtype=np.float32).round(6).tobytes()).hexdigest()
            action_signatures[policy_name] = action_sig
            unsafe_action_rate = float(unsafe_count / total_steps)
            summary_rows.append({
                'policy': policy_name,
                'mean_reward': float(np.mean(episode_returns)),
                'std_reward': float(np.std(episode_returns)),
                'median_reward': float(np.median(episode_returns)),
                'mean_episode_length': float(np.mean(episode_lengths)),
                'unsafe_action_rate': unsafe_action_rate,
                'safety_violations_count': int(unsafe_count),
                'constraint_satisfaction_rate': float(1.0 - unsafe_action_rate),
                'safety_rate': float(1.0 - unsafe_action_rate),
                'mean_action_value': float(np.mean(per_policy_actions)),
                'std_action_value': float(np.std(per_policy_actions)),
            })

        sig_groups: Dict[str, List[str]] = {}
        for p, sig in action_signatures.items():
            sig_groups.setdefault(sig, []).append(p)
        duplicates = [v for v in sig_groups.values() if len(v) > 1]
        if duplicates and len(sig_groups) == 1:
            raise RuntimeError(f"Identical policy action sequences detected: {duplicates}")

        summary_df = pd.DataFrame(summary_rows).set_index('policy').sort_values('mean_reward', ascending=False)
        if len(summary_df) > 1:
            cols = ['mean_reward', 'std_reward', 'median_reward', 'unsafe_action_rate', 'mean_action_value']
            n_unique = summary_df[cols].drop_duplicates().shape[0]
            if n_unique == 1:
                raise RuntimeError("All policies produced identical rollout summaries; evaluation likely degenerate.")

        raw_df = pd.DataFrame(raw_rows)
        ep_df = pd.DataFrame(ep_rows)
        raw_eval_path = self.output_dir / f'{export_prefix}_policy_raw_evaluation.csv'
        episode_summary_path = self.output_dir / f'{export_prefix}_policy_episode_summary.csv'
        raw_df.to_csv(raw_eval_path, index=False)
        ep_df.to_csv(episode_summary_path, index=False)
        return {
            'summary_df': summary_df,
            'raw_eval_df': raw_df,
            'episode_summary_df': ep_df,
            'raw_eval_path': str(raw_eval_path),
            'episode_summary_path': str(episode_summary_path),
        }

    def _export_baseline_reports(self, summary_df: pd.DataFrame) -> None:
        required_cols = [
            'mean_reward', 'std_reward', 'median_reward', 'mean_episode_length',
            'unsafe_action_rate', 'constraint_satisfaction_rate',
            'mean_action_value', 'std_action_value',
        ]
        for col in required_cols:
            if col not in summary_df.columns:
                raise RuntimeError(f"Baseline summary missing required column: {col}")

        mean_rewards = summary_df['mean_reward'].to_numpy(dtype=np.float64)
        if len(np.unique(mean_rewards)) == 1 and len(mean_rewards) > 1:
            raise RuntimeError("All policies have identical mean_reward to full precision; stopping export.")

        md_path = self.output_dir / 'baseline_comparison_report.md'
        json_path = self.output_dir / 'baseline_comparison_report.json'
        tex_path = self.output_dir / 'results_table.tex'

        summary_reset = summary_df.reset_index()
        md_lines = [
            "# Baseline Comparison Report",
            "",
            f"- Generated: {datetime.now().isoformat()}",
            f"- Data source: {self.results.get('data', {}).get('source', 'unknown')}",
            "",
            summary_reset.to_markdown(index=False),
            "",
        ]
        md_path.write_text("\n".join(md_lines))
        self._write_json_with_provenance(
            json_path,
            {'baseline_metrics': summary_reset.to_dict(orient='records')},
        )

        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Baseline Policy Comparison Results}",
            "\\label{tab:baseline_comparison}",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "Policy & Mean Reward & Std Reward & Unsafe Rate & Constraint Sat. \\\\",
            "\\hline",
        ]
        for row in summary_reset.to_dict(orient='records'):
            lines.append(
                f"{row['policy']} & {row['mean_reward']:.4f} & {row['std_reward']:.4f} & "
                f"{row['unsafe_action_rate']:.4f} & {row['constraint_satisfaction_rate']:.4f} \\\\"
            )
        lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
        tex_path.write_text("\n".join(lines) + "\n")

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

    def _to_ope_trajectories(self, transitions):
        """Convert flat transitions into OffPolicyEvaluator Trajectory objects."""
        from evaluation.off_policy_eval import Trajectory
        episodes = self._transitions_to_episodes(transitions)
        out = []
        for ep in episodes:
            out.append(Trajectory(
                states=np.asarray(ep['states'], dtype=np.float32),
                actions=np.asarray(ep['actions'], dtype=np.float32).reshape(-1, 1),
                rewards=np.asarray(ep['rewards'], dtype=np.float32),
                next_states=np.asarray(ep['next_states'], dtype=np.float32),
                dones=np.zeros(len(ep['rewards']), dtype=np.float32),
            ))
        return out

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

    def _generate_master_results(self) -> None:
        if 'evaluation' not in self.results:
            return
        rollout_df = self.results['evaluation'].get('rollout_summary')
        if rollout_df is None or rollout_df.empty:
            return

        master = rollout_df.reset_index().rename(columns={'index': 'policy'})
        ope_df = self.results['evaluation'].get('ope_table')
        wis_map, dr_map, ess_map = {}, {}, {}
        if isinstance(ope_df, pd.DataFrame) and not ope_df.empty:
            for _, row in ope_df.iterrows():
                p = row['policy']
                est = str(row['estimator']).upper()
                if est == 'WIS':
                    wis_map[p] = float(row['value_estimate'])
                    ess_map[p] = float(row['ess'])
                if est == 'DR':
                    dr_map[p] = float(row['value_estimate'])
                    ess_map[p] = float(row['ess'])

        master['WIS'] = master['policy'].map(wis_map)
        master['DR'] = master['policy'].map(dr_map)
        master['ESS'] = master['policy'].map(ess_map)
        if len(master) > 1:
            master['subgroup_gap'] = float(master['mean_reward'].max() - master['mean_reward'].min())
        else:
            master['subgroup_gap'] = 0.0
        fidelity = self.results.get('policy_distillation', {}).get('fidelity')
        master['interpretability_fidelity'] = float(fidelity) if fidelity is not None else np.nan

        keep_cols = [
            'policy', 'mean_reward', 'std_reward', 'mean_episode_length',
            'unsafe_action_rate', 'constraint_satisfaction_rate',
            'WIS', 'DR', 'ESS', 'subgroup_gap', 'interpretability_fidelity',
        ]
        master = master[keep_cols]
        for c in ['data_source', 'seed', 'git_commit']:
            if c == 'data_source':
                master[c] = self.results.get('data', {}).get('source', 'unknown')
            elif c == 'seed':
                master[c] = int(getattr(self.config, 'seed', 42))
            else:
                master[c] = self._get_git_commit()
        master_path = self.output_dir / 'MASTER_RESULTS.csv'
        master.to_csv(master_path, index=False)

        md_lines = [
            "# MASTER RESULTS",
            "",
            f"- Generated: {datetime.now().isoformat()}",
            f"- Data source: {self.results.get('data', {}).get('source', 'unknown')}",
            "",
            master.to_markdown(index=False),
            "",
        ]
        (self.output_dir / 'MASTER_RESULTS.md').write_text("\n".join(md_lines))
        self.results['master_results'] = master

    def _write_limitations_file(self) -> None:
        lines = [
            "# LIMITATIONS",
            "",
            "## Synthetic vs Real Data",
            "- Synthetic trajectories simplify clinical dynamics and may not capture all real-world confounders.",
            "- MIMIC-derived trajectories are retrospective and reflect logged clinician behavior, not prospective interventions.",
            "",
            "## Support Mismatch",
            "- OPE estimates are sensitive to limited overlap between target policy actions and behavior policy support.",
            "- Low ESS or heavy clipping indicates unstable estimates and limited reliability.",
            "",
            "## Offline RL Constraints",
            "- No online interaction or prospective validation was performed.",
            "- Policy quality is bounded by logging policy quality and dataset coverage.",
            "",
            "## Non-Deployment Disclaimer",
            "- This artifact is for retrospective research and thesis defense only.",
            "- Outputs must not be used for live clinical decision support.",
            "",
        ]
        (self.output_dir / 'LIMITATIONS.md').write_text("\n".join(lines))

    def _run_artifact_validation(self) -> None:
        failures = []
        rules_path = self.output_dir / 'decision_rules.txt'
        if rules_path.exists() and rules_path.read_text().strip() == "|--- class: 0":
            failures.append("Placeholder decision rules detected in decision_rules.txt.")

        fi_path = self.output_dir / 'feature_importances.json'
        if fi_path.exists():
            try:
                fi = json.loads(fi_path.read_text())
                if fi and all(abs(float(v)) <= 1e-12 for v in fi.values()):
                    failures.append("All feature importances are zero.")
            except Exception:
                failures.append("Failed to parse feature_importances.json.")

        cf_path = self.output_dir / 'counterfactuals.json'
        if cf_path.exists():
            try:
                cfs = json.loads(cf_path.read_text())
                if not cfs:
                    failures.append("counterfactuals.json is empty.")
            except Exception:
                failures.append("Failed to parse counterfactuals.json.")

        baseline_json = self.output_dir / 'baseline_comparison_report.json'
        if baseline_json.exists():
            payload = json.loads(baseline_json.read_text())
            rows = payload.get('baseline_metrics', [])
            mean_rewards = [float(r['mean_reward']) for r in rows if 'mean_reward' in r]
            if len(mean_rewards) > 1 and len(set(mean_rewards)) == 1:
                failures.append("Identical mean rewards across distinct baseline policies.")

        safety_csv = self.output_dir / 'safety_summary.csv'
        if safety_csv.exists() and 'evaluation' in self.results:
            sdf = pd.read_csv(safety_csv)
            for _, row in sdf.iterrows():
                p = row['policy']
                eval_s = self.results['evaluation']['safety'].get(p, {})
                expected = float(eval_s.get('constraint_satisfaction_rate', np.nan))
                observed = float(row['constraint_satisfaction_rate'])
                if np.isfinite(expected) and abs(expected - observed) > 1e-9:
                    failures.append(f"Contradictory safety values for policy {p}.")

        source = self.results.get('data', {}).get('source', 'unknown')
        if source == 'synthetic' and 'mimic' in self.output_dir.name.lower():
            failures.append("Output directory label implies MIMIC while data_source is synthetic.")

        master_path = self.output_dir / 'MASTER_RESULTS.csv'
        if master_path.exists():
            master = pd.read_csv(master_path)
            expected_policies = set(self.results.get('evaluation', {}).get('rollout_summary', pd.DataFrame()).index.tolist())
            if not expected_policies.issubset(set(master['policy'].tolist())):
                failures.append("MASTER_RESULTS.csv missing required policy rows.")

        if failures:
            out = ["# VALIDATION FAILURES", ""]
            for idx, f in enumerate(failures, 1):
                out.append(f"{idx}. {f}")
            out.extend([
                "",
                "Likely root causes:",
                "- policy-specific rollout path not used consistently across exporters",
                "- placeholder interpretability files persisted from fallback code paths",
                "- metadata/data-source labels not propagated uniformly",
            ])
            (self.output_dir / 'VALIDATION_FAILURES.md').write_text("\n".join(out) + "\n")
            raise RuntimeError(f"Artifact validation failed with {len(failures)} issue(s).")

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
            safety_rate = row.get('constraint_satisfaction_rate', row.get('safety_rate', float('nan')))
            mean_len = row.get('mean_episode_length', row.get('total_steps', 0))
            lines.append(
                f"{policy_name} & {mean_reward:.4f} & "
                f"{safety_rate:.2%} & {int(mean_len)} \\\\"
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

    # ------------------------------------------------------------------
    # MIMIC-III trajectory helpers
    # ------------------------------------------------------------------

    def _extract_insulin_daily(
        self, prescriptions: pd.DataFrame, subject_ids: set
    ) -> pd.DataFrame:
        """Return a DataFrame of daily insulin usage: subject_id, _lab_date, insulin_flag, insulin_dose."""
        empty = pd.DataFrame(columns=['subject_id', '_lab_date', 'insulin_flag', 'insulin_dose'])
        if len(prescriptions) == 0:
            return empty

        mask    = prescriptions['drug'].str.contains('insulin', case=False, na=False)
        insulin = prescriptions[mask & prescriptions['subject_id'].isin(subject_ids)].copy()
        if len(insulin) == 0:
            return empty

        date_col = 'startdate' if 'startdate' in insulin.columns else None
        if date_col is None:
            return empty

        insulin['_lab_date'] = pd.to_datetime(insulin[date_col], errors='coerce').dt.date
        insulin = insulin.dropna(subset=['_lab_date'])
        if len(insulin) == 0:
            return empty

        daily = (
            insulin.groupby(['subject_id', '_lab_date'])
            .size().reset_index(name='_cnt')
        )
        daily['insulin_flag'] = 1.0
        daily['insulin_dose'] = 1.0  # binary (actual dosing info not reliably available)
        return daily[['subject_id', '_lab_date', 'insulin_flag', 'insulin_dose']]

    def _compute_med_history_features(
        self, prescriptions: pd.DataFrame, subject_ids: set
    ) -> pd.DataFrame:
        """Return per-patient adherence_rate_7d and medication_count."""
        result = pd.DataFrame({'subject_id': list(subject_ids)})
        if len(prescriptions) == 0:
            result['adherence_rate_7d'] = 0.5
            result['medication_count']  = 0.0
            return result

        presc = prescriptions[prescriptions['subject_id'].isin(subject_ids)]
        counts = (
            presc.groupby('subject_id')['drug']
            .nunique().reset_index(name='medication_count')
        )
        max_cnt = counts['medication_count'].max()
        if max_cnt > 0:
            counts['medication_count'] = counts['medication_count'] / max_cnt

        result = result.merge(counts, on='subject_id', how='left')
        result['medication_count']  = result['medication_count'].fillna(0.0)
        result['adherence_rate_7d'] = 0.5  # no direct adherence data in MIMIC
        return result[['subject_id', 'adherence_rate_7d', 'medication_count']]

    def _build_mimic_state_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add BASE_STATE_COLS columns from MIMIC-derived features (in-place copy)."""
        r = df.copy()

        # glucose_mean ← rolling_mean_glucose (from create_temporal_features) or glucose
        for src in ['rolling_mean_glucose', 'glucose']:
            if src in df.columns:
                r['glucose_mean'] = df[src].fillna(0.0)
                break
        else:
            r['glucose_mean'] = 0.0

        # glucose_std ← rolling_std_glucose
        r['glucose_std'] = (
            df['rolling_std_glucose'].fillna(0.0)
            if 'rolling_std_glucose' in df.columns else 0.0
        )

        # glucose_min / glucose_max ← approximate from mean ± std
        r['glucose_min'] = r['glucose_mean'] - r['glucose_std']
        r['glucose_max'] = r['glucose_mean'] + r['glucose_std']

        # insulin_mean ← insulin_dose (binary proxy)
        r['insulin_mean'] = (
            df['insulin_dose'].fillna(0.0) if 'insulin_dose' in df.columns else 0.0
        )

        # medication_taken ← insulin_flag
        r['medication_taken'] = (
            df['insulin_flag'].fillna(0.0) if 'insulin_flag' in df.columns else 0.0
        )

        # reminder_sent ← 0 (not in MIMIC)
        r['reminder_sent'] = 0.0

        # hypoglycemia / hyperglycemia ← computed from raw glucose before normalization
        for col in ['hypoglycemia', 'hyperglycemia']:
            if col not in r.columns:
                r[col] = 0.0
            else:
                r[col] = r[col].fillna(0.0)

        # day ← should already be normalized [0,1]; ensure it exists
        if 'day' not in r.columns:
            r['day'] = 0.0
        else:
            r['day'] = r['day'].fillna(0.0)

        return r

    def _build_mimic_trajectories(
        self, df: pd.DataFrame, state_cols: list
    ) -> list:
        """Build (state, action, reward, next_state, done) tuples from MIMIC data."""
        if df.empty:
            return []

        # Pre-ensure all state columns exist
        for col in state_cols:
            if col not in df.columns:
                df = df.copy()
                df[col] = 0.0

        df = df.sort_values(['subject_id', 'hadm_id', 'charttime']).reset_index(drop=True)
        trajectories = []

        for (_, _), adm_df in df.groupby(['subject_id', 'hadm_id']):
            adm_df  = adm_df.reset_index(drop=True)
            n_steps = len(adm_df)
            if n_steps < 2:
                continue

            states      = adm_df[state_cols].fillna(0.0).values.astype(np.float32)
            med_col     = 'medication_taken'
            actions_raw = (
                adm_df[med_col].fillna(0.0).values
                if med_col in adm_df.columns else np.zeros(n_steps)
            )
            hypo_raw  = adm_df['hypoglycemia'].fillna(0.0).values  if 'hypoglycemia'  in adm_df.columns else np.zeros(n_steps)
            hyper_raw = adm_df['hyperglycemia'].fillna(0.0).values if 'hyperglycemia' in adm_df.columns else np.zeros(n_steps)

            for i in range(n_steps - 1):
                action  = np.array([float(actions_raw[i])], dtype=np.float32)
                reward  = float(1.0 - 2.0 * hypo_raw[i] - 1.0 * hyper_raw[i])
                done    = (i == n_steps - 2)
                trajectories.append((
                    states[i], action, reward, states[i + 1], done
                ))

        return trajectories

    def _convert_to_trajectory_format(self, data):
        """Return trajectory tuples unchanged; MIMIC now produces tuples directly."""
        if isinstance(data, list):
            return data
        logger.warning(
            "_convert_to_trajectory_format: unexpected type %s — returning []", type(data)
        )
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
    parser.add_argument('--train-iql', action='store_true', help='Train discrete IQL offline RL baseline')
    parser.add_argument('--discrete-actions', type=int, default=2, help='Number of discrete action buckets for IQL')
    parser.add_argument('--iql-updates', type=int, default=400, help='Number of IQL updates')
    parser.add_argument('--demo', action='store_true', help='Run deterministic quick CPU demo')
    parser.add_argument('--defense-bundle', action='store_true', help='Generate full thesis defense artifact bundle')
    parser.add_argument('--run-id', type=str, default=None, help='Custom run id for outputs/<run_id>')
    parser.add_argument('--seed', type=int, default=42, help='Global deterministic seed')
    parser.add_argument('--run-distillation', action='store_true', default=True, help='Run policy distillation step')

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
    if args.defense_bundle:
        args.mode = 'synthetic'
        args.use_synthetic = True
        args.train_iql = True
        args.train_cql = False
        if not args.run_id:
            args.run_id = datetime.now().strftime('defense_%Y%m%d_%H%M%S')
    if args.demo:
        args.mode = 'synthetic'
        args.use_synthetic = True
        args.n_synthetic_patients = min(args.n_synthetic_patients, 120)
        args.trajectory_length = min(args.trajectory_length, 14)
        args.train_iql = True
        args.train_cql = False
        args.output_dir = args.output_dir or 'outputs/demo'
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

        if 'defense_bundle_dir' in results:
            logger.info(f"\nDefense bundle path: {results['defense_bundle_dir']}")
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
