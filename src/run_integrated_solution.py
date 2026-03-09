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
import shutil
import time
from typing import Dict, List, Tuple, Any, Optional, Sequence, Set

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

    CHECKPOINT_SCHEMA_VERSION = 2
    ROLLOUT_CACHE_SCHEMA_VERSION = 2

    PIPELINE_STAGE_ORDER = [
        'stage_1_data_preparation',
        'stage_2_environment_setup',
        'stage_2b_encoder_training',
        'stage_3_baseline_training',
        'stage_4_cql_training',
        'stage_4b_iql_training',
        'stage_5_evaluation',
        'stage_6b_transfer',
        'stage_7_interpretability',
        'stage_7b_policy_distillation',
        'stage_6_report_inputs',
        'stage_6_generate_reports',
    ]

    STAGE_CHECKPOINT_KEYS = {
        'stage_1_data_preparation': ['data'],
        'stage_2_environment_setup': ['environments'],
        'stage_2b_encoder_training': ['encoder_wrapper', 'encoded_data'],
        'stage_3_baseline_training': ['baselines', 'baseline_rollouts'],
        'stage_4_cql_training': ['cql'],
        'stage_4b_iql_training': ['iql'],
        'stage_5_evaluation': ['evaluation'],
        'stage_6b_transfer': ['transfer'],
        'stage_7_interpretability': ['interpretability'],
        'stage_7b_policy_distillation': ['policy_distillation'],
        'stage_6_report_inputs': ['report_inputs'],
        'stage_6_generate_reports': ['plot_artifacts', 'master_results'],
    }
    STAGE_CHECKPOINT_SCHEMA_VERSIONS = {
        'stage_1_data_preparation': 2,
        'stage_2_environment_setup': 1,
        'stage_2b_encoder_training': 2,
        'stage_3_baseline_training': 3,
        'stage_4_cql_training': 2,
        'stage_4b_iql_training': 2,
        'stage_5_evaluation': 3,
        'stage_6b_transfer': 1,
        'stage_7_interpretability': 1,
        'stage_7b_policy_distillation': 1,
        'stage_6_report_inputs': 1,
        'stage_6_generate_reports': 2,
    }

    STAGE_ALIAS = {
        'stage1': 'stage_1_data_preparation',
        'stage2': 'stage_2_environment_setup',
        'stage2b': 'stage_2b_encoder_training',
        'stage3': 'stage_3_baseline_training',
        'stage4': 'stage_4_cql_training',
        'stage4b': 'stage_4b_iql_training',
        'stage5': 'stage_5_evaluation',
        'stage6': 'stage_6_generate_reports',
        'stage6b': 'stage_6b_transfer',
        'stage7': 'stage_7_interpretability',
        'stage7b': 'stage_7b_policy_distillation',
        'report_inputs': 'stage_6_report_inputs',
    }
    STAGES_ALWAYS_REBUILD_ON_RESUME = {
        # Baseline policies include closures/rule functions; recreate from stage logic.
        'stage_3_baseline_training',
    }

    def __init__(self, config):
        self.config = config
        seed_all(getattr(config, "seed", 42))
        self.results = {}
        self.run_started_at = datetime.now()
        self.requested_data_source = self._expected_data_source()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_ckpt_dir = self.output_dir / 'checkpoints' / 'pipeline'
        self.pipeline_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_manifest_path = self.output_dir / 'pipeline_manifest.json'
        self.pipeline_manifest = self._load_pipeline_manifest()
        self._clear_previous_run_artifacts()

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

    def _expected_data_source(self) -> str:
        """Infer intended dataset source from CLI flags."""
        if bool(getattr(self.config, 'use_synthetic', False)) or str(getattr(self.config, 'mode', '')).lower() == 'synthetic':
            return 'synthetic'
        return 'mimic'

    def _is_strict_mimic_run(self) -> bool:
        return self.requested_data_source == 'mimic'

    def _path_within_output(self, path_like: Any) -> bool:
        try:
            p = Path(path_like).resolve()
            root = self.output_dir.resolve()
            return p == root or root in p.parents
        except Exception:
            return False

    def _register_plot_file(self, path: Path) -> None:
        p = path.resolve()
        if not self._path_within_output(p):
            raise RuntimeError(f"Plot artifact path escapes output directory: {p}")
        self.results.setdefault('plot_artifacts', [])
        self.results['plot_artifacts'].append(str(p))

    def _pipeline_checkpoint_path(self, step_name: str) -> Path:
        return self.pipeline_ckpt_dir / f"{step_name}.pkl"

    def _normalize_step_name(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        canonical = self.STAGE_ALIAS.get(raw.lower(), raw)
        if canonical not in self.PIPELINE_STAGE_ORDER:
            raise ValueError(
                f"Unknown stage/step '{value}'. Allowed: {self.PIPELINE_STAGE_ORDER}"
            )
        return canonical

    def _compute_config_hash(self) -> str:
        cfg = vars(self.config).copy()
        for volatile in [
            'resume', 'start_from', 'stop_after',
            'force_stage', 'invalidate_from',
        ]:
            cfg.pop(volatile, None)
        payload = json.dumps(cfg, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _compute_dataset_signature(self) -> str:
        payload = {
            'mode': getattr(self.config, 'mode', 'unknown'),
            'use_synthetic': bool(getattr(self.config, 'use_synthetic', False)),
            'mimic_dir': str(getattr(self.config, 'mimic_dir', '')),
            'n_synthetic_patients': int(getattr(self.config, 'n_synthetic_patients', 0)),
            'sample_size': int(getattr(self.config, 'sample_size', 0)),
            'use_sample': bool(getattr(self.config, 'use_sample', False)),
            'trajectory_length': int(getattr(self.config, 'trajectory_length', 0)),
            'batch_size': int(getattr(self.config, 'batch_size', 0)),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
        ).hexdigest()

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        try:
            with open(tmp_path, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        blob = json.dumps(payload, indent=2, default=str).encode('utf-8')
        self._atomic_write_bytes(path, blob)

    def _atomic_write_pickle(self, path: Path, obj: Any) -> None:
        import pickle
        blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self._atomic_write_bytes(path, blob)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        blob = text.encode('utf-8')
        self._atomic_write_bytes(path, blob)

    def _load_pipeline_manifest(self) -> Dict[str, Any]:
        if self.pipeline_manifest_path.exists():
            try:
                with open(self.pipeline_manifest_path, 'r') as f:
                    manifest = json.load(f)
                manifest.setdefault('stages', {})
                return manifest
            except Exception:
                logger.warning("Failed to parse pipeline manifest; starting fresh.")
        return {
            'version': 1,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'config_hash': self._compute_config_hash() if hasattr(self, 'config') else None,
            'dataset_signature': None,
            'git_commit': self._get_git_commit() if hasattr(self, 'config') else "unknown",
            'stages': {},
        }

    def _persist_pipeline_manifest(self) -> None:
        self.pipeline_manifest['updated_at'] = datetime.now().isoformat()
        self.pipeline_manifest['config_hash'] = self._compute_config_hash()
        self.pipeline_manifest['dataset_signature'] = self._compute_dataset_signature()
        self.pipeline_manifest['git_commit'] = self._get_git_commit()
        self._atomic_write_json(self.pipeline_manifest_path, self.pipeline_manifest)

    def save_checkpoint(self, step_name: str, obj: Any, metadata: Optional[Dict[str, Any]] = None) -> Path:
        ckpt_path = self._pipeline_checkpoint_path(step_name)
        meta = {
            'step_name': step_name,
            'saved_at': datetime.now().isoformat(),
            'config_hash': self._compute_config_hash(),
            'dataset_signature': self._compute_dataset_signature(),
            'git_commit': self._get_git_commit(),
            'checkpoint_schema_version': int(self.CHECKPOINT_SCHEMA_VERSION),
            'stage_schema_version': int(self.STAGE_CHECKPOINT_SCHEMA_VERSIONS.get(step_name, 1)),
        }
        if metadata:
            meta.update(metadata)
        payload = {'meta': meta, 'payload': obj}
        import pickle
        try:
            blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(
                "Non-serializable object detected in checkpoint payload for %s. "
                "Sanitizing payload before saving checkpoint. (%s)",
                step_name, e,
            )
            payload = {'meta': meta, 'payload': self._sanitize_checkpoint_payload(obj)}
            blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self._atomic_write_bytes(ckpt_path, blob)
        logger.info("Saving checkpoint for %s", step_name)
        stage_entry = self.pipeline_manifest.setdefault('stages', {}).setdefault(step_name, {})
        stage_entry.update({
            'status': 'completed',
            'checkpoint_path': str(ckpt_path),
            'last_completed_at': datetime.now().isoformat(),
            'meta': meta,
        })
        self._persist_pipeline_manifest()
        return ckpt_path

    def load_checkpoint(self, step_name: str) -> Optional[Any]:
        ckpt_path = self._pipeline_checkpoint_path(step_name)
        if not ckpt_path.exists():
            return None
        try:
            import pickle
            with open(ckpt_path, 'rb') as f:
                payload = pickle.load(f)
            logger.info("Loading checkpoint for %s", step_name)
            return payload.get('payload')
        except Exception as e:
            logger.warning("Checkpoint load failed for %s: %s", step_name, e)
            return None

    def is_checkpoint_valid(self, step_name: str, config_hash: str, dataset_signature: str) -> bool:
        ckpt_path = self._pipeline_checkpoint_path(step_name)
        if not ckpt_path.exists():
            return False
        try:
            import pickle
            with open(ckpt_path, 'rb') as f:
                payload = pickle.load(f)
            meta = payload.get('meta', {})
            saved_ckpt_schema = int(meta.get('checkpoint_schema_version', -1))
            if saved_ckpt_schema != int(self.CHECKPOINT_SCHEMA_VERSION):
                logger.info(
                    "Checkpoint invalid for %s: checkpoint schema mismatch (saved=%s current=%s)",
                    step_name, saved_ckpt_schema, self.CHECKPOINT_SCHEMA_VERSION,
                )
                return False
            saved_stage_schema = int(meta.get('stage_schema_version', -1))
            current_stage_schema = int(self.STAGE_CHECKPOINT_SCHEMA_VERSIONS.get(step_name, 1))
            if saved_stage_schema != current_stage_schema:
                logger.info(
                    "Checkpoint invalid for %s: schema version mismatch (saved=%s current=%s)",
                    step_name, saved_stage_schema, current_stage_schema,
                )
                return False
            if meta.get('config_hash') != config_hash:
                logger.info("Checkpoint invalid for %s: config hash mismatch", step_name)
                return False
            if meta.get('dataset_signature') != dataset_signature:
                logger.info("Checkpoint invalid for %s: dataset signature mismatch", step_name)
                return False
            logger.info("Checkpoint valid for %s", step_name)
            return True
        except Exception as e:
            logger.info("Checkpoint invalid for %s: failed to parse checkpoint (%s)", step_name, e)
            return False

    def _collect_step_artifact_paths(self, step_name: str) -> List[str]:
        paths: List[str] = []
        if step_name == 'stage_1_data_preparation':
            for name in ['synthetic_data.pkl', 'mimic_trajectories.pkl', 'mimic_dataset_summary.json']:
                p = self.output_dir / name
                if p.exists():
                    paths.append(str(p.resolve()))
        elif step_name == 'stage_3_baseline_training':
            for name in [
                'baseline_policy_summary.csv',
                'baseline_policy_raw_evaluation.csv',
                'baseline_policy_episode_summary.csv',
                'baseline_comparison_report.json',
            ]:
                p = self.output_dir / name
                if p.exists():
                    paths.append(str(p.resolve()))
        elif step_name == 'stage_4_cql_training':
            p = self.output_dir / 'cql'
            if p.exists():
                paths.append(str(p.resolve()))
        elif step_name == 'stage_5_evaluation':
            for name in ['evaluation_policy_summary.csv', 'evaluation_policy_raw_evaluation.csv', 'evaluation_policy_episode_summary.csv', 'ope_estimates.csv', 'safety_summary.csv']:
                p = self.output_dir / name
                if p.exists():
                    paths.append(str(p.resolve()))
        elif step_name == 'stage_6_generate_reports':
            for name in ['results_summary.json', 'MASTER_RESULTS.csv', 'RUN_PROVENANCE.json']:
                p = self.output_dir / name
                if p.exists():
                    paths.append(str(p.resolve()))
        return paths

    def _build_stage_checkpoint_payload(self, step_name: str) -> Dict[str, Any]:
        """Build stage payload while excluding runtime-only/non-serializable objects."""
        ckpt_keys = self.STAGE_CHECKPOINT_KEYS.get(step_name, [])
        payload = {k: self.results[k] for k in ckpt_keys if k in self.results}

        if step_name == 'stage_3_baseline_training':
            baselines = payload.get('baselines')
            if isinstance(baselines, dict):
                # Do not checkpoint policy objects; they may contain local closures.
                policies = baselines.get('policies', {})
                slim = {k: v for k, v in baselines.items() if k != 'policies'}
                slim['baseline_names'] = list(policies.keys()) if isinstance(policies, dict) else []
                payload['baselines'] = slim
            rollout = payload.get('baseline_rollouts')
            if isinstance(rollout, dict):
                slim_rollout = {k: v for k, v in rollout.items() if k not in ('raw_eval_df', 'episode_summary_df')}
                payload['baseline_rollouts'] = slim_rollout

        return payload

    def _sanitize_checkpoint_payload(self, obj: Any) -> Any:
        """Recursively sanitize non-pickleable objects in checkpoint payloads."""
        import pickle
        import types

        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k == 'policies' and isinstance(v, dict):
                    out['baseline_names'] = list(v.keys())
                    continue
                out[k] = self._sanitize_checkpoint_payload(v)
            return out
        if isinstance(obj, (list, tuple, set)):
            seq = [self._sanitize_checkpoint_payload(x) for x in obj]
            return seq if isinstance(obj, list) else tuple(seq)
        if isinstance(obj, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
            return f"<non-serializable:{type(obj).__module__}.{type(obj).__name__}>"
        if callable(obj):
            return f"<non-serializable-callable:{type(obj).__module__}.{type(obj).__name__}>"

        try:
            pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return obj
        except Exception:
            return f"<non-serializable:{type(obj).__module__}.{type(obj).__name__}>"

    def _invalidate_from(self, step_name: str) -> None:
        idx = self.PIPELINE_STAGE_ORDER.index(step_name)
        to_clear = self.PIPELINE_STAGE_ORDER[idx:]
        logger.info("Invalidating downstream checkpoints from %s", step_name)
        for step in to_clear:
            ckpt = self._pipeline_checkpoint_path(step)
            if ckpt.exists():
                ckpt.unlink()
            self.pipeline_manifest.setdefault('stages', {}).pop(step, None)
        self._persist_pipeline_manifest()

    def _execute_pipeline_step(self, step_name: str, fn: Any) -> None:
        config_hash = self._compute_config_hash()
        dataset_signature = self._compute_dataset_signature()
        force_stage_raw = str(getattr(self.config, 'force_stage', '') or '')
        forced = False
        if force_stage_raw:
            force_steps = [self._normalize_step_name(x.strip()) for x in force_stage_raw.split(',') if x.strip()]
            forced = step_name in force_steps

        can_resume = bool(getattr(self.config, 'resume', False)) and not forced
        if can_resume and step_name in self.STAGES_ALWAYS_REBUILD_ON_RESUME:
            logger.info(
                "Stage %s will be recomputed on resume to reconstruct runtime objects (e.g., policies).",
                step_name,
            )
            can_resume = False
        if can_resume:
            if self.is_checkpoint_valid(step_name, config_hash, dataset_signature):
                loaded = self.load_checkpoint(step_name)
                if loaded is not None:
                    if isinstance(loaded, dict):
                        self.results.update(loaded)
                    logger.info("Skipping %s; valid checkpoint found", step_name)
                    stage_entry = self.pipeline_manifest.setdefault('stages', {}).setdefault(step_name, {})
                    stage_entry['status'] = 'completed'
                    stage_entry['last_loaded_at'] = datetime.now().isoformat()
                    self._persist_pipeline_manifest()
                    return
            else:
                logger.info("Recomputing %s due to checkpoint validation failure.", step_name)

        stage_entry = self.pipeline_manifest.setdefault('stages', {}).setdefault(step_name, {})
        stage_entry['status'] = 'running'
        stage_entry['last_started_at'] = datetime.now().isoformat()
        self._persist_pipeline_manifest()

        try:
            fn()
            payload = self._build_stage_checkpoint_payload(step_name)
            meta = {
                'artifact_keys': list(payload.keys()),
                'artifact_paths': self._collect_step_artifact_paths(step_name),
            }
            self.save_checkpoint(step_name, payload, metadata=meta)
        except Exception as e:
            stage_entry = self.pipeline_manifest.setdefault('stages', {}).setdefault(step_name, {})
            stage_entry['status'] = 'failed'
            stage_entry['last_failed_at'] = datetime.now().isoformat()
            stage_entry['error'] = str(e)
            self._persist_pipeline_manifest()
            raise

    def _clear_previous_run_artifacts(self) -> None:
        """Prevent stale artifacts from previous runs in reused output dirs."""
        if bool(getattr(self.config, 'resume', False)):
            logger.info("Resume mode enabled; preserving existing output artifacts.")
            return
        stale_files = [
            'synthetic_data.pkl',
            'mimic_trajectories.pkl',
            'mimic_dataset_summary.json',
            'results_summary.json',
            'MASTER_RESULTS.csv',
            'MASTER_RESULTS.md',
            'safety_summary.csv',
            'ope_estimates.csv',
            'baseline_comparison_report.json',
            'baseline_comparison_report.md',
            'results_table.tex',
            'thesis_figures.pdf',
            'VALIDATION_FAILURES.md',
            'RUN_PROVENANCE.json',
            'ROOT_CAUSE_REPORT.md',
        ]
        stale_dirs = ['cql', 'encoder', 'interpretability']
        for name in stale_files:
            p = self.output_dir / name
            if p.exists():
                p.unlink()
        for name in stale_dirs:
            p = self.output_dir / name
            if p.exists():
                shutil.rmtree(p)
        for p in self.output_dir.glob('*.png'):
            p.unlink()

    def _ensure_data_prepared(self) -> None:
        if 'data' not in self.results:
            self.stage_1_data_preparation()
        source = self.results.get('data', {}).get('source', 'unknown')
        if self._is_strict_mimic_run() and source != 'mimic':
            raise RuntimeError(f"Strict provenance violation: expected MIMIC data, got '{source}'.")

    def _checkpoint_root(self) -> Path:
        explicit = getattr(self.config, 'checkpoint_dir', None)
        root = Path(explicit) if explicit else (self.output_dir / 'checkpoints' / 'mimic')
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _checkpoint_path(self, name: str, suffix: str) -> Path:
        return self._checkpoint_root() / f"{name}.{suffix.lstrip('.')}"

    def _checkpoint_exists(self, name: str, suffix: str) -> bool:
        return self._checkpoint_path(name, suffix).exists()

    def _should_load_checkpoint(self, name: str, suffix: str) -> bool:
        if getattr(self.config, 'ignore_checkpoints', False):
            return False
        if not getattr(self.config, 'resume', False):
            return False
        return self._checkpoint_exists(name, suffix)

    def _save_pickle_checkpoint(self, name: str, obj: Any) -> Path:
        path = self._checkpoint_path(name, 'pkl')
        self._atomic_write_pickle(path, obj)
        logger.info("Atomic checkpoint write complete: %s", path)
        return path

    def _load_pickle_checkpoint(self, name: str) -> Optional[Any]:
        path = self._checkpoint_path(name, 'pkl')
        import pickle
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            logger.info("Loaded checkpoint: %s", path)
            return obj
        except (pickle.UnpicklingError, EOFError, OSError, ValueError) as e:
            logger.warning(
                "Failed to load checkpoint %s: corrupt or incomplete file; recomputing (%s)",
                name, e,
            )
            return None

    def _save_parquet_checkpoint(self, name: str, df: pd.DataFrame) -> Path:
        path = self._checkpoint_path(name, 'parquet')
        tmp_path = path.with_name(path.name + '.tmp')
        out = df.copy()
        cat_cols = out.select_dtypes(include=['category']).columns.tolist()
        if cat_cols:
            logger.info(
                "Converting categorical columns to string before parquet checkpoint '%s': %s",
                name,
                cat_cols,
            )
            for c in cat_cols:
                out[c] = out[c].astype('string')
        try:
            out.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise
        logger.info("Atomic checkpoint write complete: %s (rows=%s)", path, len(df))
        return path

    def _load_parquet_checkpoint(self, name: str) -> Optional[pd.DataFrame]:
        path = self._checkpoint_path(name, 'parquet')
        try:
            df = pd.read_parquet(path)
            logger.info("Loaded checkpoint: %s (rows=%s)", path, len(df))
            return df
        except (OSError, ValueError, EOFError) as e:
            logger.warning(
                "Ignoring unreadable checkpoint at %s: corrupt/incomplete parquet (%s)",
                path, e,
            )
            return None

    def _write_missing_data_report(self, policy) -> None:
        """Persist compact missing-data policy report for debugging/reproducibility."""
        rows = []
        for col, strat in policy.strategies.items():
            rows.append({
                'column_name': col,
                'missing_rate_before': float(policy.missing_rates_before.get(col, np.nan)),
                'strategy': strat,
                'fill_value': policy.numeric_fill_values.get(col, None),
                'category_policy': (
                    f"vocab_size={len(policy.categorical_vocab.get(col, []))}"
                    if col in policy.categorical_vocab else ''
                ),
                'mask_added': int(f"{col}_missing" in policy.mask_columns),
            })
        if not rows:
            return
        report_df = pd.DataFrame(rows).sort_values('column_name')
        report_path = self.output_dir / 'missing_data_report.csv'
        report_df.to_csv(report_path, index=False)
        logger.info("Saved missing-data report: %s", report_path)

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
        source = data.get('source', 'unknown')
        inferred_state_dim = int(len(np.asarray(data['train'][0][0]).reshape(-1))) if data.get('train') else int(self._get_state_dim())
        return {
            'run_timestamp': datetime.now().isoformat(),
            'seed': int(getattr(self.config, 'seed', 42)),
            'git_commit': self._get_git_commit(),
            'data_source': source,
            'input_dataset_path': str(getattr(self.config, 'mimic_dir', '')) if source == 'mimic' else 'synthetic_generator',
            'n_patients': int(len(data.get('patients', data.get('cohort', [])))),
            'n_trajectories': int(n_train + n_val + n_test),
            'state_dim': inferred_state_dim,
            'action_dim': 1,
            'state_dimension': int(self._get_state_dim()),
            'action_dimension': 1,
            'reward_definition': 'in_range - 2*hypoglycemia - hyperglycemia + 0.5*medication_taken',
            'output_directory': str(self.output_dir),
        }

    def _write_json_with_provenance(self, path: Path, payload: Dict[str, Any]) -> None:
        out = dict(payload)
        out['provenance'] = self._provenance()
        # Use atomic write to prevent corrupted artifacts if run is interrupted
        self._atomic_write_json(path, out)

    def _write_run_provenance_manifest(self) -> None:
        """Persist canonical provenance metadata for validation and packaging."""
        manifest_path = self.output_dir / 'RUN_PROVENANCE.json'
        manifest = self._provenance()
        data = self.results.get('data', {})
        manifest['input_dataset_path'] = data.get(
            'input_dataset_path',
            str(getattr(self.config, 'mimic_dir', '')) if manifest['data_source'] == 'mimic' else 'synthetic_generator',
        )
        manifest['requested_data_source'] = self.requested_data_source
        manifest['mode'] = str(getattr(self.config, 'mode', 'unknown'))
        # Use atomic write to prevent corrupted artifacts if run is interrupted
        self._atomic_write_json(manifest_path, manifest)

    def _write_root_cause_report(self) -> None:
        report = [
            "# Root Cause Report: MIMIC Mode Producing Synthetic Artifacts",
            "",
            "## Why this happened",
            "- CLI scenario used `--mode train-eval --mimic-dir ... --output-dir outputs/mimic_full`.",
            "- `run_full_pipeline()` previously skipped Stage 1 for `train-eval`, so no MIMIC data was loaded.",
            "- Downstream stages (`stage_2b_encoder_training` / `stage_3_baseline_training`) auto-generated synthetic data when `self.results['data']` was missing.",
            "- Output folder and summary exporters then mixed labels/files, causing synthetic artifacts under a MIMIC-looking path.",
            "",
            "## Exact code path before fix",
            "- `run_full_pipeline(): if mode in ['full','data-only'] -> stage_1_data_preparation()`",
            "- `stage_2b_encoder_training(): if 'data' not in results -> prepare_synthetic_data()`",
            "- `stage_3_baseline_training(): if 'data' not in results -> prepare_synthetic_data()`",
            "- `stage_1_data_preparation(): except -> fallback to synthetic`",
            "",
            "## Preventive changes",
            "- Stage 1 now runs for `train-eval`/`eval-only` modes as well.",
            "- Removed silent fallback to synthetic from Stage 1 and later stages.",
            "- Added strict source checks (`_ensure_data_prepared`) so MIMIC-intended runs fail hard if source is not MIMIC.",
            "- Added `RUN_PROVENANCE.json` and stricter artifact validation before final packaging/reporting.",
            "- Added stale-artifact cleanup and run-local plot tracking to avoid cross-run contamination.",
        ]
        (self.output_dir / 'ROOT_CAUSE_REPORT.md').write_text("\n".join(report) + "\n")

    def _normalize_output_dir_by_data_source(self) -> None:
        """Ensure folder naming reflects the actual data source."""
        source = self.results.get('data', {}).get('source', '').lower()
        cur_name = self.output_dir.name
        cur_lower = cur_name.lower()
        new_name = None
        if source == 'synthetic':
            if 'mimic' in cur_lower:
                new_name = cur_name.lower().replace('mimic', 'synthetic')
            elif 'synthetic' not in cur_lower:
                new_name = f"{cur_name}_synthetic"
        elif source == 'mimic':
            if 'synthetic' in cur_lower:
                new_name = cur_name.lower().replace('synthetic', 'mimic')
        if new_name:
            new_dir = self.output_dir.parent / new_name
            new_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "Output directory name '%s' mismatches source '%s'. Using '%s' instead.",
                self.output_dir.name, source, new_dir,
            )
            self.output_dir = new_dir
            self._clear_previous_run_artifacts()

    def run_full_pipeline(self):
        """Run complete end-to-end pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 80)

        if getattr(self.config, "defense_bundle", False):
            return self.run_defense_bundle()

        start_from = self._normalize_step_name(getattr(self.config, 'start_from', None))
        stop_after = self._normalize_step_name(getattr(self.config, 'stop_after', None))
        invalidate_from = self._normalize_step_name(getattr(self.config, 'invalidate_from', None))
        if invalidate_from is not None:
            self._invalidate_from(invalidate_from)

        stage_plan: List[Tuple[str, bool, Any]] = [
            ('stage_1_data_preparation', self.config.mode in ['full', 'data-only', 'train-eval', 'eval-only', 'synthetic'], self.stage_1_data_preparation),
            ('stage_2_environment_setup', self.config.mode in ['full', 'train-eval', 'synthetic'], self.stage_2_environment_setup),
            ('stage_2b_encoder_training', True, self.stage_2b_encoder_training),
            ('stage_3_baseline_training', self.config.mode in ['full', 'train-eval', 'synthetic'], self.stage_3_baseline_training),
            ('stage_4_cql_training', self.config.mode in ['full', 'train-eval'], self.stage_4_cql_training),
            ('stage_4b_iql_training', True, self.stage_4b_iql_training),
            ('stage_5_evaluation', self.config.mode in ['full', 'train-eval', 'eval-only', 'synthetic'], self.stage_5_evaluation),
            ('stage_6b_transfer', True, self.stage_6b_transfer),
            ('stage_7_interpretability', True, self.stage_7_interpretability),
            ('stage_7b_policy_distillation', True, self.stage_7b_policy_distillation),
            ('stage_6_report_inputs', not getattr(self.config, 'no_report', False), self.stage_6_prepare_report_inputs),
            ('stage_6_generate_reports', not getattr(self.config, 'no_report', False), self.stage_6_generate_reports),
        ]

        started = start_from is None
        for step_name, enabled, fn in stage_plan:
            if not enabled:
                continue
            if not started:
                if step_name != start_from:
                    continue
                started = True
            self._execute_pipeline_step(step_name, fn)
            if stop_after is not None and step_name == stop_after:
                logger.info("Stopping pipeline after %s due to --stop-after", step_name)
                break

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
        if self.requested_data_source == 'synthetic':
            data = self.prepare_synthetic_data()
        else:
            data = self.prepare_mimic_data()
        self.results['data'] = data
        self._normalize_output_dir_by_data_source()
        logger.info("Data preparation complete")

    def prepare_synthetic_data(self):
        """Generate synthetic patient data."""
        logger.info("Generating synthetic diabetes patient data...")

        generator = SyntheticDataGenerator(random_seed=int(getattr(self.config, 'seed', 42)))

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
            seed=int(getattr(self.config, 'seed', 42)),
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
        # Use atomic write to prevent corrupted artifacts if run is interrupted
        self._atomic_write_pickle(output_path, data)

        logger.info(f"Saved synthetic data to {output_path}")
        return data

    def prepare_mimic_data(self):
        """Prepare MIMIC-III data with resumable cohort + batch checkpoints."""
        logger.info("Loading MIMIC-III data...")
        logger.info(
            "Checkpoint settings: resume=%s ignore_checkpoints=%s checkpoint_dir=%s",
            bool(getattr(self.config, 'resume', False)),
            bool(getattr(self.config, 'ignore_checkpoints', False)),
            str(self._checkpoint_root()),
        )

        if self._should_load_checkpoint('prepared_mimic_data', 'pkl'):
            payload = self._load_pickle_checkpoint('prepared_mimic_data')
            if payload is not None:
                logger.info("Using fully prepared MIMIC dataset from checkpoint.")
                return payload

        from data import (
            MIMICLoader,
            CohortBuilder,
            FeatureEngineer,
            MissingDataPolicyConfig,
            fit_missing_data_policy,
            transform_with_missing_data_policy,
        )

        mimic_dir = Path(self.config.mimic_dir)
        if not mimic_dir.exists():
            raise FileNotFoundError(f"MIMIC directory not found: {mimic_dir}")
        loader = MIMICLoader(data_dir=str(mimic_dir), use_cache=True)

        patients = loader.load_patients()
        admissions = loader.load_admissions()
        diagnoses = loader.load_diagnoses_icd()
        logger.info(f"Loaded {len(patients):,} patients")

        builder = CohortBuilder(patients, admissions, diagnoses)

        if self._should_load_checkpoint('01_diabetes_patients', 'pkl'):
            diabetes_pts = self._load_pickle_checkpoint('01_diabetes_patients')
            if diabetes_pts is None:
                diabetes_pts = builder.define_diabetes_cohort()
                self._save_pickle_checkpoint('01_diabetes_patients', diabetes_pts)
        else:
            diabetes_pts = builder.define_diabetes_cohort()
            self._save_pickle_checkpoint('01_diabetes_patients', diabetes_pts)

        if self._should_load_checkpoint('02_inclusion_filtered_patients', 'pkl'):
            filtered_pts = self._load_pickle_checkpoint('02_inclusion_filtered_patients')
            if filtered_pts is None:
                filtered_pts = builder.apply_inclusion_criteria(
                    diabetes_pts, min_age=18, max_age=80, min_admissions=2
                )
                self._save_pickle_checkpoint('02_inclusion_filtered_patients', filtered_pts)
        else:
            filtered_pts = builder.apply_inclusion_criteria(
                diabetes_pts, min_age=18, max_age=80, min_admissions=2
            )
            self._save_pickle_checkpoint('02_inclusion_filtered_patients', filtered_pts)

        if self._should_load_checkpoint('03_final_cohort_patients', 'pkl'):
            final_cohort = self._load_pickle_checkpoint('03_final_cohort_patients')
            if final_cohort is None:
                final_cohort = builder.apply_exclusion_criteria(
                    filtered_pts, exclude_pregnancy=True, exclude_pediatric=True
                )
                self._save_pickle_checkpoint('03_final_cohort_patients', final_cohort)
        else:
            final_cohort = builder.apply_exclusion_criteria(
                filtered_pts, exclude_pregnancy=True, exclude_pediatric=True
            )
            self._save_pickle_checkpoint('03_final_cohort_patients', final_cohort)

        final_cohort = sorted(set(int(x) for x in final_cohort))
        logger.info(f"Final cohort before sampling: {len(final_cohort):,} patients")

        if self.config.use_sample:
            import random
            random.seed(int(getattr(self.config, 'seed', 42)))
            final_cohort = sorted(random.sample(
                final_cohort, min(self.config.sample_size, len(final_cohort))
            ))
            logger.info(f"Using sample of {len(final_cohort)} patients")

        if not final_cohort:
            raise RuntimeError("Final MIMIC cohort is empty after inclusion/exclusion criteria.")

        if self._should_load_checkpoint('04_demographics', 'parquet'):
            demographics = self._load_parquet_checkpoint('04_demographics')
            if demographics is None:
                engineer = FeatureEngineer()
                demographics = engineer.extract_demographics(patients, admissions)
                demographics = demographics[demographics['subject_id'].isin(final_cohort)]
                self._save_parquet_checkpoint('04_demographics', demographics)
        else:
            engineer = FeatureEngineer()
            demographics = engineer.extract_demographics(patients, admissions)
            demographics = demographics[demographics['subject_id'].isin(final_cohort)]
            self._save_parquet_checkpoint('04_demographics', demographics)

        batch_size = int(getattr(self.config, 'batch_size', 512))
        if batch_size <= 0:
            raise ValueError("--batch-size must be > 0")

        n_batches = int(np.ceil(len(final_cohort) / batch_size))
        logger.info(
            "Preparing MIMIC feature batches: %s patients in %s batches (batch_size=%s).",
            len(final_cohort),
            n_batches,
            batch_size,
        )
        batch_names: List[str] = []
        non_empty_batch_names: List[str] = []
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            stop = min(len(final_cohort), (batch_idx + 1) * batch_size)
            batch_subject_ids = final_cohort[start:stop]
            batch_name = f"05_features_batch_{batch_idx:04d}"
            batch_names.append(batch_name)

            if self._should_load_checkpoint(batch_name, 'parquet'):
                batch_df = self._load_parquet_checkpoint(batch_name)
                if batch_df is not None:
                    logger.info(
                        "Batch %s/%s loaded from checkpoint (%s patients, %s rows).",
                        batch_idx + 1, n_batches, len(batch_subject_ids), len(batch_df),
                    )
                    if len(batch_df) > 0:
                        non_empty_batch_names.append(batch_name)
                    continue

            logger.info(
                "Processing batch %s/%s (%s patients)...",
                batch_idx + 1, n_batches, len(batch_subject_ids),
            )
            batch_df = self._prepare_mimic_feature_batch(
                loader=loader,
                subject_ids=batch_subject_ids,
            )
            self._save_parquet_checkpoint(batch_name, batch_df)
            if len(batch_df) > 0:
                non_empty_batch_names.append(batch_name)

        if not non_empty_batch_names:
            raise RuntimeError(
                "All MIMIC feature batches are empty. Verify mimic-dir data integrity and cohort filters."
            )

        logger.info("Assembling %s non-empty feature batches...", len(non_empty_batch_names))
        batch_frames = []
        for name in non_empty_batch_names:
            df_ckpt = self._load_parquet_checkpoint(name)
            if df_ckpt is None:
                raise RuntimeError(f"Checkpoint {name} became unreadable during assembly; rerun with --ignore-checkpoints.")
            batch_frames.append(df_ckpt)
        normalized_data = pd.concat(batch_frames, ignore_index=True)

        state_cols = self._get_state_cols()
        for col in state_cols:
            if col not in normalized_data.columns:
                logger.warning("State column '%s' missing from MIMIC data, filling 0.", col)
                normalized_data[col] = 0.0

        unique_pts = list(sorted(set(final_cohort)))
        rng = np.random.default_rng(int(getattr(self.config, 'seed', 42)))
        rng.shuffle(unique_pts)
        n_train = int(0.70 * len(unique_pts))
        n_val = int(0.15 * len(unique_pts))
        train_pts = set(unique_pts[:n_train])
        val_pts = set(unique_pts[n_train:n_train + n_val])
        test_pts = set(unique_pts[n_train + n_val:])

        train_df = normalized_data[normalized_data['subject_id'].isin(train_pts)].copy()
        val_df = normalized_data[normalized_data['subject_id'].isin(val_pts)].copy()
        test_df = normalized_data[normalized_data['subject_id'].isin(test_pts)].copy()

        md_cfg = MissingDataPolicyConfig(
            group_cols=('subject_id', 'hadm_id'),
            time_col='charttime',
            enable_missingness_masks=not bool(getattr(self.config, 'disable_missingness_masks', False)),
            enable_time_since_last_observed=bool(getattr(self.config, 'enable_time_since_last_observed', False)),
            numeric_fill_stat='median',
            categorical_unknown_token='UNKNOWN',
            lab_max_hold_steps=getattr(self.config, 'lab_max_hold_steps', None),
            vital_max_hold_steps=getattr(self.config, 'vital_max_hold_steps', None),
            drop_high_missingness_columns=bool(getattr(self.config, 'drop_high_missingness_columns', False)),
            high_missingness_threshold=float(getattr(self.config, 'high_missingness_threshold', 0.95)),
            columns_exempt_from_imputation=('day',),
            zero_fill_columns=('medication_taken', 'reminder_sent', 'hypoglycemia', 'hyperglycemia'),
            report_path=str(self.output_dir / 'missing_data_report.csv'),
        )
        missing_policy = fit_missing_data_policy(
            train_df,
            md_cfg,
            numeric_columns=state_cols,
            categorical_columns=[],
            time_varying_columns=[c for c in state_cols if c not in MED_HISTORY_COLS],
            static_numeric_columns=[c for c in state_cols if c in MED_HISTORY_COLS],
        )
        train_df = transform_with_missing_data_policy(train_df, missing_policy)
        val_df = transform_with_missing_data_policy(val_df, missing_policy)
        test_df = transform_with_missing_data_policy(test_df, missing_policy)

        mask_cols = [c for c in train_df.columns if c.endswith('_missing') and c[:-8] in state_cols]
        tsl_cols = [c for c in train_df.columns if c.startswith('time_since_last_') and c[len('time_since_last_'):] in state_cols]
        base_model_state_cols = [c for c in state_cols if c in train_df.columns]
        model_state_cols = base_model_state_cols + sorted(mask_cols) + sorted(tsl_cols)

        self._write_missing_data_report(missing_policy)
        logger.info(
            "Missing-data policy fit on train only. Added %s mask columns. Final state_dim=%s",
            len(mask_cols),
            len(model_state_cols),
        )

        train_data = self._build_mimic_trajectories(train_df, model_state_cols)
        val_data = self._build_mimic_trajectories(val_df, model_state_cols)
        test_data = self._build_mimic_trajectories(test_df, model_state_cols)

        data = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'demographics': demographics,
            'cohort': final_cohort,
            'source': 'mimic',
            'state_cols': model_state_cols,
            'input_dataset_path': str(mimic_dir.resolve()),
            'batch_size': batch_size,
            'n_feature_batches': n_batches,
        }
        if not train_data or not test_data:
            raise RuntimeError(
                f"MIMIC trajectory build produced insufficient data (train={len(train_data)}, test={len(test_data)})."
            )

        mimic_path = self.output_dir / 'mimic_trajectories.pkl'
        # Use atomic write to prevent corrupted artifacts if run is interrupted
        self._atomic_write_pickle(mimic_path, data)
        self._save_pickle_checkpoint('prepared_mimic_data', data)

        summary_path = self.output_dir / 'mimic_dataset_summary.json'
        # Use atomic write to prevent corrupted artifacts if run is interrupted
        self._atomic_write_json(summary_path, {
            'input_dataset_path': str(mimic_dir.resolve()),
            'n_patients': int(len(final_cohort)),
            'n_train_trajectories': int(len(train_data)),
            'n_val_trajectories': int(len(val_data)),
            'n_test_trajectories': int(len(test_data)),
            'state_dim': int(len(model_state_cols)),
            'action_dim': 1,
            'batch_size': batch_size,
            'n_feature_batches': n_batches,
            'resume_used': bool(getattr(self.config, 'resume', False)),
            'ignore_checkpoints': bool(getattr(self.config, 'ignore_checkpoints', False)),
            'missing_masks_added': int(len(mask_cols)),
        })
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

        self._ensure_data_prepared()
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

        self._ensure_data_prepared()
        data = self.results['data']
        if data['source'] == 'synthetic':
            test_data = data['test']
            train_data = data['train']
        else:
            test_data = self._convert_to_trajectory_format(data['test'])
            train_data = self._convert_to_trajectory_format(data['train'])

        # Derive state_dim from actual data to stay correct with --use-vitals / --use-med-history
        state_dim = len(train_data[0][0]) if train_data else self._get_state_dim()
        action_dim = 1

        train_states = np.array([d[0] for d in train_data])
        train_actions = np.array([d[1] for d in train_data])

        logger.info(f"Training data: {len(train_data)} transitions")
        logger.info(f"Test data: {len(test_data)} transitions")

        include_baselines = self._parse_name_list(getattr(self.config, 'include_baselines', None))
        exclude_baselines = self._parse_name_list(getattr(self.config, 'exclude_baselines', None))
        baseline_order = [
            'Rule-Based',
            'Random-Uniform',
            'Random-Safe',
            'Mean-Action',
            'Ridge-Regression',
            'KNN-5',
            'Behavior-Cloning',
        ]
        selected_names = self._filter_policy_dict(
            {n: object() for n in baseline_order},
            include_names=include_baselines,
            exclude_names=exclude_baselines,
            skip_slow=bool(getattr(self.config, 'skip_slow_baselines', False)),
            max_policies=getattr(self.config, 'max_rollout_policies', None),
        )
        selected_name_set = set(selected_names.keys())

        logger.info("Creating baseline policies...")
        baselines = {}

        if 'Rule-Based' in selected_name_set:
            logger.info("  Rule-based policy...")
            baselines['Rule-Based'] = create_diabetes_rule_policy(state_dim, action_dim)

        if 'Random-Uniform' in selected_name_set:
            logger.info("  Random policy...")
            baselines['Random-Uniform'] = create_random_policy(
                action_dim, state_dim, seed=42, distribution='uniform'
            )

        if 'Random-Safe' in selected_name_set:
            logger.info("  Safe random policy...")
            baselines['Random-Safe'] = create_safe_random_policy(
                action_dim, state_dim, seed=42, num_samples=10
            )

        if 'Mean-Action' in selected_name_set:
            logger.info("  Mean action policy...")
            mean_policy = create_mean_action_policy(action_dim, state_dim)
            mean_policy.fit(train_states, train_actions)
            baselines['Mean-Action'] = mean_policy

        if 'Ridge-Regression' in selected_name_set:
            logger.info("  Ridge regression policy...")
            ridge_policy = create_regression_policy(
                state_dim, action_dim, regression_type='ridge', alpha=1.0
            )
            ridge_policy.fit(train_states, train_actions)
            baselines['Ridge-Regression'] = ridge_policy

        if 'KNN-5' in selected_name_set:
            logger.info("  KNN policy...")
            knn_policy = create_knn_policy(state_dim, action_dim, k=5)
            knn_policy.fit(train_states, train_actions)
            baselines['KNN-5'] = knn_policy

        if 'Behavior-Cloning' in selected_name_set:
            logger.info("  Behavior cloning policy...")
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
        baselines = self._filter_policy_dict(
            baselines,
            include_names=include_baselines,
            exclude_names=exclude_baselines,
            skip_slow=bool(getattr(self.config, 'skip_slow_baselines', False)),
            max_policies=getattr(self.config, 'max_rollout_policies', None),
        )

        eval_data_for_rollout = test_data
        fast_eval_enabled = bool(getattr(self.config, 'fast_eval', False))
        if fast_eval_enabled:
            max_eval_samples = int(getattr(self.config, 'max_eval_samples', 2000) or 2000)
            eval_data_for_rollout = self._deterministic_subsample_transitions(
                test_data, max_eval_samples=max_eval_samples, label='stage_3_baselines',
            )
            logger.warning(
                "Fast eval mode enabled for baseline rollouts. Outputs are for debug/iteration, not final thesis metrics."
            )

        rollout_payload = self._run_policy_rollout_evaluation(
            policies=baselines,
            train_data=train_data,
            eval_data=eval_data_for_rollout,
            export_prefix='baseline',
            use_cache=True,
            force_recompute=bool(getattr(self.config, 'force_recompute_baselines', False)),
            cache_extra={
                'fast_eval': fast_eval_enabled,
                'max_eval_samples': int(getattr(self.config, 'max_eval_samples', 0) or 0),
                'include_baselines': include_baselines or [],
                'exclude_baselines': exclude_baselines or [],
                'skip_slow_baselines': bool(getattr(self.config, 'skip_slow_baselines', False)),
                'max_rollout_policies': int(getattr(self.config, 'max_rollout_policies', 0) or 0),
            },
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
            'test_data': eval_data_for_rollout,
            'raw_eval_path': rollout_payload['raw_eval_path'],
            'episode_summary_path': rollout_payload['episode_summary_path'],
            'fast_eval': fast_eval_enabled,
            'fast_eval_original_test_size': int(len(test_data)),
            'fast_eval_used_test_size': int(len(eval_data_for_rollout)),
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
        # Reuse Stage 3 rollout only when full evaluation signature matches.
        baseline_rollout = self.results.get('baseline_rollouts')
        eval_policy_names = sorted(list(eval_policies.keys()))
        current_eval_signature = self._build_rollout_signature(
            policies=eval_policies,
            train_data=train_data,
            eval_data=test_data,
            cache_extra={
                'fast_eval': bool(getattr(self.config, 'fast_eval', False)),
                'max_eval_samples': int(getattr(self.config, 'max_eval_samples', 0) or 0),
                'policy_set': eval_policy_names,
            },
        )
        can_reuse_stage3 = False
        if baseline_rollout is not None:
            prev_sig = baseline_rollout.get('evaluation_signature', {})
            can_reuse_stage3, reason = self._rollout_reuse_check(prev_sig, current_eval_signature)
            if can_reuse_stage3:
                logger.info("Stage 5 rollout reuse signature matched; reusing Stage 3 results")
            else:
                if reason == "dataset signature changed":
                    logger.info("Stage 5 rollout reuse rejected: dataset signature changed")
                elif reason == "policy artifact fingerprint changed":
                    logger.info("Stage 5 rollout reuse rejected: policy artifact fingerprint changed")
                else:
                    logger.info("Stage 5 rollout reuse rejected: evaluation signature changed")

        if can_reuse_stage3:
            rollout_eval = baseline_rollout
        else:
            rollout_eval = self._run_policy_rollout_evaluation(
                policies=eval_policies,
                train_data=train_data,
                eval_data=test_data,
                export_prefix='evaluation',
                use_cache=True,
                force_recompute=bool(getattr(self.config, 'force_recompute_baselines', False)),
                cache_extra={
                    'fast_eval': bool(getattr(self.config, 'fast_eval', False)),
                    'max_eval_samples': int(getattr(self.config, 'max_eval_samples', 0) or 0),
                    'policy_set': eval_policy_names,
                },
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
            'fast_eval': bool(getattr(self.config, 'fast_eval', False)),
            'max_eval_samples': int(getattr(self.config, 'max_eval_samples', 0) or 0),
            'skip_slow_baselines': bool(getattr(self.config, 'skip_slow_baselines', False)),
            'output_directory': str(self.output_dir),
            'dataset_identity': {
                'data_source': self.results.get('data', {}).get('source', 'unknown'),
                'input_dataset_path': self.results.get('data', {}).get(
                    'input_dataset_path',
                    str(getattr(self.config, 'mimic_dir', '')) if self.results.get('data', {}).get('source') == 'mimic' else 'synthetic_generator',
                ),
                'n_patients': len(self.results.get('data', {}).get('patients', self.results.get('data', {}).get('cohort', []))),
                'n_trajectories': len(self.results.get('data', {}).get('train', [])) + len(self.results.get('data', {}).get('val', [])) + len(self.results.get('data', {}).get('test', [])),
                'state_dim': self._get_state_dim(),
                'action_dim': 1,
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
        self._write_run_provenance_manifest()
        self._write_root_cause_report()

        logger.info(f"Results summary saved to {summary_path}")

        self._generate_latex_table()
        self._generate_master_results()
        self._write_limitations_file()
        if getattr(self.config, 'light_report', False):
            self.results['plot_artifacts'] = []
            logger.info("Light report enabled: skipping heavy visualization generation.")
        elif not getattr(self.config, 'demo', False) and not getattr(self.config, 'defense_bundle', False):
            self._generate_visualizations()
        else:
            self.results['plot_artifacts'] = []
            logger.info("Visualization generation skipped in demo/defense-bundle mode.")
        self._run_artifact_validation()

        logger.info("Report generation complete")

    def stage_6_prepare_report_inputs(self):
        """Capture report-input summary payload for resumable reporting."""
        payload = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.config.mode,
            'data_source': self.results.get('data', {}).get('source', 'unknown'),
            'n_baselines': len(self.results.get('baselines', {}).get('policies', {})),
            'has_evaluation': 'evaluation' in self.results,
            'output_directory': str(self.output_dir),
        }
        self.results['report_inputs'] = payload

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

    def _known_baseline_names(self) -> Set[str]:
        return {
            'Rule-Based',
            'Random-Uniform',
            'Random-Safe',
            'Mean-Action',
            'Ridge-Regression',
            'KNN-5',
            'Behavior-Cloning',
        }

    @staticmethod
    def _parse_name_list(raw: Optional[str]) -> Optional[List[str]]:
        if raw is None:
            return None
        names = [x.strip() for x in str(raw).split(',') if x.strip()]
        return names if names else None

    def _filter_policy_dict(
        self,
        policies: Dict[str, Any],
        include_names: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        skip_slow: bool = False,
        max_policies: Optional[int] = None,
        slow_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        selected = list(policies.keys())
        all_names = set(selected)

        if include_names:
            unknown = sorted(set(include_names) - all_names)
            if unknown:
                raise ValueError(f"Unknown baseline names in --include-baselines: {unknown}")
            include_set = set(include_names)
            selected = [n for n in selected if n in include_set]

        if exclude_names:
            unknown = sorted(set(exclude_names) - all_names)
            if unknown:
                raise ValueError(f"Unknown baseline names in --exclude-baselines: {unknown}")
            exclude_set = set(exclude_names)
            selected = [n for n in selected if n not in exclude_set]

        if skip_slow:
            slow = set(slow_names or ['KNN-5', 'Behavior-Cloning'])
            selected = [n for n in selected if n not in slow]

        if max_policies is not None and max_policies > 0:
            selected = selected[:max_policies]

        if not selected:
            raise ValueError("No baselines selected after include/exclude/skip filtering.")

        logger.info("Final baseline policy set (%s): %s", len(selected), selected)
        return {n: policies[n] for n in selected}

    def _deterministic_subsample_transitions(
        self,
        transitions: List[Tuple],
        max_samples: Optional[int] = None,
        label: str = "",
        max_eval_samples: Optional[int] = None,
    ) -> List[Tuple]:
        # Backward-compatible alias support: callers may pass max_eval_samples.
        if max_samples is None:
            max_samples = max_eval_samples
        if max_samples is None or max_samples <= 0 or len(transitions) <= max_samples:
            return transitions
        seed = int(getattr(self.config, 'seed', 42))
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(transitions), size=max_samples, replace=False))
        sampled = [transitions[int(i)] for i in idx]
        logger.info(
            "Fast eval sampling for %s: %s -> %s transitions (seed=%s).",
            label,
            len(transitions),
            len(sampled),
            seed,
        )
        return sampled

    def _transitions_signature(self, transitions: List[Tuple], limit: int = 128) -> str:
        h = hashlib.sha256()
        h.update(str(len(transitions)).encode('utf-8'))
        max_n = min(len(transitions), max(1, limit))
        for i in range(max_n):
            s, a, r, ns, d = transitions[i]
            s_arr = np.asarray(s, dtype=np.float32).reshape(-1)
            a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
            ns_arr = np.asarray(ns, dtype=np.float32).reshape(-1)
            h.update(s_arr[:16].round(5).tobytes())
            h.update(a_arr[:4].round(5).tobytes())
            h.update(np.asarray([float(r), float(d)], dtype=np.float32).tobytes())
            h.update(ns_arr[:16].round(5).tobytes())
        return h.hexdigest()

    def _policy_fingerprint(self, policy: Any) -> Dict[str, Any]:
        """Build a stable lightweight fingerprint for policy identity validation."""
        fp: Dict[str, Any] = {
            'module': type(policy).__module__,
            'class': type(policy).__name__,
            'name': getattr(policy, 'name', None),
        }
        for attr in ['seed', 'distribution', 'k', 'regression_type', 'alpha', 'normalize', 'state_dim']:
            if hasattr(policy, attr):
                try:
                    fp[attr] = getattr(policy, attr)
                except Exception:
                    pass
        # Include small scalar fields from __dict__ for stricter identity checks.
        try:
            for k, v in getattr(policy, '__dict__', {}).items():
                if k in fp or k.startswith('_'):
                    continue
                if isinstance(v, (str, int, float, bool, type(None))):
                    fp[f"attr_{k}"] = v
                elif isinstance(v, np.ndarray) and v.size <= 32:
                    fp[f"attr_{k}"] = np.asarray(v).reshape(-1).tolist()
        except Exception:
            pass
        if hasattr(policy, 'is_fitted'):
            fp['is_fitted'] = bool(getattr(policy, 'is_fitted'))

        def _array_sample_digest(arr: np.ndarray, max_elems: int = 2048) -> str:
            flat = np.asarray(arr).reshape(-1)
            if flat.size > max_elems:
                flat = flat[:max_elems]
            return hashlib.sha256(np.asarray(flat, dtype=np.float32).round(6).tobytes()).hexdigest()

        # sklearn linear models
        model = getattr(policy, 'model', None)
        if model is not None:
            if hasattr(model, 'coef_'):
                try:
                    coef = np.asarray(model.coef_)
                    fp['model_coef_shape'] = tuple(coef.shape)
                    fp['model_coef_digest'] = _array_sample_digest(coef)
                except Exception:
                    pass
            if hasattr(model, 'intercept_'):
                try:
                    intercept = np.asarray(model.intercept_)
                    fp['model_intercept_shape'] = tuple(intercept.shape)
                    fp['model_intercept_digest'] = _array_sample_digest(intercept)
                except Exception:
                    pass

        # KNN fitted data digest (sampled)
        knn_model = getattr(policy, 'knn_model', None)
        if knn_model is not None and hasattr(knn_model, '_fit_X'):
            try:
                xfit = np.asarray(knn_model._fit_X)
                fp['knn_fit_shape'] = tuple(xfit.shape)
                fp['knn_fit_digest'] = _array_sample_digest(xfit)
            except Exception:
                pass

        # Torch-network digest (sampled from state_dict)
        network = getattr(policy, 'network', None)
        if network is not None and hasattr(network, 'state_dict'):
            try:
                sd = network.state_dict()
                h = hashlib.sha256()
                total_params = 0
                for k in sorted(sd.keys()):
                    t = sd[k].detach().cpu().numpy().reshape(-1)
                    total_params += int(t.size)
                    if t.size > 2048:
                        t = t[:2048]
                    h.update(k.encode('utf-8'))
                    h.update(np.asarray(t, dtype=np.float32).round(6).tobytes())
                fp['network_param_count'] = total_params
                fp['network_digest'] = h.hexdigest()
            except Exception:
                pass

        return fp

    def _build_rollout_signature(
        self,
        policies: Dict[str, Any],
        train_data: List[Tuple],
        eval_data: List[Tuple],
        cache_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        safety_cfg = EvaluationConfig().safety
        signature = {
            'rollout_cache_schema_version': int(self.ROLLOUT_CACHE_SCHEMA_VERSION),
            'policy_names': sorted(list(policies.keys())),
            'policy_fingerprints': {
                k: self._policy_fingerprint(v) for k, v in sorted(policies.items(), key=lambda kv: kv[0])
            },
            'train_signature': self._transitions_signature(train_data),
            'eval_signature': self._transitions_signature(eval_data),
            'eval_size': int(len(eval_data)),
            'seed': int(getattr(self.config, 'seed', 42)),
            'safe_glucose_range': list(safety_cfg.safe_glucose_range),
            'rollout_model_version': 1,
        }
        if cache_extra:
            signature['cache_extra'] = dict(cache_extra)
        signature_json = json.dumps(signature, sort_keys=True, default=str)
        signature['signature_hash'] = hashlib.sha256(signature_json.encode('utf-8')).hexdigest()
        return signature

    def _rollout_reuse_check(
        self,
        previous_signature: Optional[Dict[str, Any]],
        current_signature: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if not previous_signature:
            return False, "no previous signature"
        if previous_signature.get('rollout_cache_schema_version') != current_signature.get('rollout_cache_schema_version'):
            return False, "evaluation signature changed"
        if previous_signature.get('eval_signature') != current_signature.get('eval_signature'):
            return False, "dataset signature changed"
        if previous_signature.get('policy_names') != current_signature.get('policy_names'):
            return False, "policy set changed"
        if previous_signature.get('cache_extra') != current_signature.get('cache_extra'):
            return False, "evaluation signature changed"
        if previous_signature.get('policy_fingerprints') != current_signature.get('policy_fingerprints'):
            return False, "policy artifact fingerprint changed"
        return True, "signature matched"

    def _rollout_cache_path(self, export_prefix: str) -> Path:
        return self.output_dir / f"{export_prefix}_rollout_cache.pkl"

    def _load_rollout_cache(self, cache_file: Path) -> Optional[Dict[str, Any]]:
        if not cache_file.exists():
            return None
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict):
                raise ValueError("cache payload must be dict")
            return payload
        except Exception as e:
            logger.warning(
                "Ignoring corrupt rollout cache / failed to load cache, recomputing: %s (%s)",
                cache_file, e,
            )
            return None

    def _save_rollout_cache(self, cache_file: Path, payload: Dict[str, Any]) -> None:
        import pickle
        logger.info("Saving rollout cache atomically to %s", cache_file)
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self._atomic_write_bytes(cache_file, blob)
        logger.info("Rollout cache write complete")

    def _run_policy_rollout_evaluation(
        self,
        policies: Dict[str, Any],
        train_data: List[Tuple],
        eval_data: List[Tuple],
        export_prefix: str,
        use_cache: bool = True,
        force_recompute: bool = False,
        cache_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate each policy with independent model-based rollouts and export raw traces."""
        cache_file = self._rollout_cache_path(export_prefix)
        eval_signature = self._build_rollout_signature(
            policies=policies,
            train_data=train_data,
            eval_data=eval_data,
            cache_extra=cache_extra,
        )

        if use_cache and cache_file.exists() and not force_recompute:
            payload = self._load_rollout_cache(cache_file)
            if payload is not None:
                saved_sig = payload.get('evaluation_signature', {})
                can_reuse, reason = self._rollout_reuse_check(saved_sig, eval_signature)
                if can_reuse:
                    logger.info("Found cached baseline evaluation results at %s", cache_file)
                    logger.info("Reusing cached results for %s baselines", len(policies))
                    summary_path = str(payload.get('summary_path', ''))
                    if summary_path and Path(summary_path).exists():
                        logger.info("Loading rollout summary from disk artifact: %s", summary_path)
                        summary_df = pd.read_csv(summary_path).set_index('policy')
                    else:
                        summary_df = pd.DataFrame(payload.get('summary_table', {}))
                        if not summary_df.empty and 'policy' in summary_df.columns:
                            summary_df = summary_df.set_index('policy')
                    raw_eval_path = str(payload.get('raw_eval_path', ''))
                    episode_summary_path = str(payload.get('episode_summary_path', ''))
                    loaded_payload = {
                        'summary_df': summary_df,
                        'raw_eval_df': None,
                        'episode_summary_df': None,
                        'summary_path': summary_path,
                        'raw_eval_path': raw_eval_path,
                        'episode_summary_path': episode_summary_path,
                        'evaluation_signature': eval_signature,
                        'evaluation_signature_hash': eval_signature.get('signature_hash'),
                        'cache_manifest': payload.get('cache_manifest', {}),
                    }
                    logger.info("Raw rollout trace not cached in memory; loading from artifact on demand.")
                    return loaded_payload
                logger.info("Cache miss: %s", reason)
        elif force_recompute:
            logger.info("Force recompute enabled; ignoring cache")

        sim_model = self._fit_linear_simulator(train_data)
        episodes = self._transitions_to_episodes(eval_data)
        safety_cfg = EvaluationConfig().safety
        g_low, g_high = safety_cfg.safe_glucose_range
        raw_rows = []
        ep_rows = []
        summary_rows = []
        action_signatures: Dict[str, str] = {}
        data_source = self.results.get('data', {}).get('source', 'unknown')
        seed = int(getattr(self.config, 'seed', 42))
        git_commit = self._get_git_commit()
        t0_all = time.perf_counter()
        logger.info(
            "Baseline rollout evaluation start: baselines=%s episodes=%s transitions=%s",
            len(policies),
            len(episodes),
            len(eval_data),
        )

        for policy_name, policy in policies.items():
            t0_policy = time.perf_counter()
            logger.info("Evaluating baseline: %s (samples=%s)", policy_name, len(eval_data))
            per_policy_actions = []
            episode_returns = []
            episode_lengths = []
            unsafe_count = 0
            total_steps = 0
            n_episodes = len(episodes)
            configured_interval = int(getattr(self.config, 'rollout_progress_every_episodes', 0) or 0)
            progress_every = configured_interval if configured_interval > 0 else max(1, min(100, n_episodes // 20 if n_episodes >= 20 else n_episodes))

            for ep_id, ep in enumerate(episodes):
                if not ep['states']:
                    continue
                state = np.asarray(ep['states'][0], dtype=np.float32).reshape(-1)
                ep_return = 0.0
                ep_unsafe_count = 0

                for step_id in range(len(ep['states'])):
                    raw_action, clipped_action = self._safe_policy_action(policy, state)
                    next_state, reward = self._simulate_step_with_model(state, clipped_action, sim_model)
                    glucose = float(next_state[0]) if len(next_state) > 0 else 0.0
                    unsafe = bool(glucose < g_low or glucose > g_high)
                    done = step_id == (len(ep['states']) - 1)
                    constraint_satisfied = not unsafe
                    unsafe_count += int(unsafe)
                    ep_unsafe_count += int(unsafe)
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
                        'data_source': data_source,
                        'seed': seed,
                        'git_commit': git_commit,
                    })
                    state = next_state

                episode_returns.append(ep_return)
                episode_lengths.append(len(ep['states']))
                ep_rows.append({
                    'policy_name': policy_name,
                    'episode_id': ep_id,
                    'episode_return': ep_return,
                    'episode_length': len(ep['states']),
                    'unsafe_steps': int(ep_unsafe_count),
                    'data_source': data_source,
                    'seed': seed,
                    'git_commit': git_commit,
                })
                ep_done = ep_id + 1
                if ep_done % progress_every == 0 or ep_done == n_episodes:
                    elapsed = time.perf_counter() - t0_policy
                    rate = ep_done / elapsed if elapsed > 0 else 0.0
                    eta = ((n_episodes - ep_done) / rate) if rate > 0 else float('inf')
                    logger.info(
                        "[policy=%s] rollout progress %s/%s episodes, elapsed=%.1fs, rate=%.2f eps/s, eta=%.1fs",
                        policy_name, ep_done, n_episodes, elapsed, rate, eta if np.isfinite(eta) else -1.0,
                    )

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
            logger.info("Completed baseline: %s in %.2fs", policy_name, time.perf_counter() - t0_policy)

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
        summary_path = self.output_dir / f'{export_prefix}_policy_summary.csv'
        raw_eval_path = self.output_dir / f'{export_prefix}_policy_raw_evaluation.csv'
        episode_summary_path = self.output_dir / f'{export_prefix}_policy_episode_summary.csv'
        summary_df.reset_index().to_csv(summary_path, index=False)
        raw_df.to_csv(raw_eval_path, index=False)
        ep_df.to_csv(episode_summary_path, index=False)
        logger.info("Completed all baseline evaluations in %.2fs", time.perf_counter() - t0_all)
        rollout_payload = {
            'summary_df': summary_df,
            'raw_eval_df': raw_df,
            'episode_summary_df': ep_df,
            'summary_path': str(summary_path),
            'raw_eval_path': str(raw_eval_path),
            'episode_summary_path': str(episode_summary_path),
            'evaluation_signature': eval_signature,
            'evaluation_signature_hash': eval_signature.get('signature_hash'),
        }
        if use_cache:
            try:
                logger.info("Storing lightweight rollout cache manifest")
                cache_payload = {
                    'rollout_cache_schema_version': int(self.ROLLOUT_CACHE_SCHEMA_VERSION),
                    'created_at': datetime.now().isoformat(),
                    'evaluation_signature': eval_signature,
                    'summary_table': summary_df.reset_index().to_dict(orient='list'),
                    'summary_path': str(summary_path),
                    'raw_eval_path': str(raw_eval_path),
                    'episode_summary_path': str(episode_summary_path),
                    'cache_manifest': {
                        'policy_names': sorted(list(policies.keys())),
                        'n_eval_transitions': int(len(eval_data)),
                        'n_eval_episodes': int(len(episodes)),
                    },
                }
                self._save_rollout_cache(cache_file, cache_payload)
            except Exception as e:
                logger.warning("Failed to save rollout cache to %s: %s", cache_file, e)
        return rollout_payload

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
        source = self.results.get('data', {}).get('source', 'unknown')
        out_name = self.output_dir.name.lower()
        manifest_path = self.output_dir / 'RUN_PROVENANCE.json'
        if not manifest_path.exists():
            failures.append("RUN_PROVENANCE.json missing.")
        else:
            try:
                manifest = json.loads(manifest_path.read_text())
                if manifest.get('data_source') != source:
                    failures.append("RUN_PROVENANCE data_source mismatches in-memory data source.")
            except Exception:
                failures.append("RUN_PROVENANCE.json is not valid JSON.")

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

        if source == 'synthetic' and 'mimic' in out_name:
            failures.append("Output directory label implies MIMIC while data_source is synthetic.")
        if source == 'mimic' and 'synthetic' in out_name:
            failures.append("Output directory label implies synthetic while data_source is MIMIC.")
        if self._is_strict_mimic_run() and source != 'mimic':
            failures.append("Strict MIMIC run ended with non-MIMIC data source.")
        if source == 'mimic' and (self.output_dir / 'synthetic_data.pkl').exists():
            failures.append("synthetic_data.pkl present in MIMIC run output.")
        if source == 'mimic':
            required = [
                self.output_dir / 'mimic_trajectories.pkl',
                self.output_dir / 'mimic_dataset_summary.json',
            ]
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                failures.append(f"Missing required MIMIC-derived trajectory artifacts: {missing}")
            expected_path = str(Path(getattr(self.config, 'mimic_dir', '')).resolve())
            got_path = str(self.results.get('data', {}).get('input_dataset_path', ''))
            if got_path != expected_path:
                failures.append(f"MIMIC input_dataset_path mismatch: expected '{expected_path}', got '{got_path}'.")

        for pstr in self.results.get('plot_artifacts', []):
            p = Path(pstr)
            if not self._path_within_output(p):
                failures.append(f"Plot artifact outside run directory: {p}")
                continue
            if p.exists() and datetime.fromtimestamp(p.stat().st_mtime) < self.run_started_at:
                failures.append(f"Stale plot artifact reused from earlier run: {p.name}")

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
            generated_plots: List[Path] = []

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
            generated_plots.append(path1)
            self._register_plot_file(path1)
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
            generated_plots.append(path2)
            self._register_plot_file(path2)
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
                if path_comp.exists():
                    generated_plots.append(path_comp)
                    self._register_plot_file(path_comp)
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
                    if path_hm.exists():
                        generated_plots.append(path_hm)
                        self._register_plot_file(path_hm)
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
                    generated_plots.append(path3)
                    self._register_plot_file(path3)
                    logger.info(f"Saved {path3}")

            # ----------------------------------------------------------
            # Fig 4: Feature importance (only if interpretability ran)
            # ----------------------------------------------------------
            if 'interpretability' in self.results:
                import json as _json
                fi_path = self.output_dir / 'feature_importances.json'
                if fi_path.exists():
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
                            generated_plots.append(path4)
                            self._register_plot_file(path4)
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
                    generated_plots.append(path5)
                    self._register_plot_file(path5)
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
                    generated_plots.append(path6)
                    self._register_plot_file(path6)
                    logger.info(f"Saved {path6}")

            # ----------------------------------------------------------
            # Fig 6: Thesis summary PDF (combines all plots)
            # ----------------------------------------------------------
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                pdf_path = self.output_dir / 'thesis_figures.pdf'
                pngs = [p for p in generated_plots if p.exists()]
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
                    self.results['plot_artifacts'] = [str(p.resolve()) for p in pngs]
                    logger.info(f"Saved thesis PDF: {pdf_path}")
            except Exception as e:
                logger.warning(f"Thesis PDF generation failed: {e}")

            if 'plot_artifacts' not in self.results:
                self.results['plot_artifacts'] = [str(p.resolve()) for p in generated_plots if p.exists()]

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

    def _prepare_mimic_feature_batch(
        self,
        loader,
        subject_ids: List[int],
    ) -> pd.DataFrame:
        """Build normalized feature rows for a patient batch."""
        from data import FeatureEngineer, DataPreprocessor

        engineer = FeatureEngineer()
        preprocessor = DataPreprocessor()
        subj_set = set(int(s) for s in subject_ids)

        labs = loader.load_lab_events(subject_ids=list(subj_set))
        prescriptions = loader.load_prescriptions(subject_ids=list(subj_set))
        if len(labs) == 0:
            logger.warning("No lab events in current batch (patients=%s).", len(subject_ids))
            return pd.DataFrame(columns=['subject_id', 'hadm_id', 'charttime'])

        lab_sequence = engineer.extract_lab_sequence(labs, list(subj_set))
        if len(lab_sequence) == 0:
            logger.warning("Lab sequence extraction returned no rows for batch.")
            return pd.DataFrame(columns=['subject_id', 'hadm_id', 'charttime'])

        lab_sequence = (
            lab_sequence
            .drop_duplicates(subset=['subject_id', 'hadm_id', 'charttime'])
            .sort_values(['subject_id', 'hadm_id', 'charttime'])
            .reset_index(drop=True)
        )

        clin_flags = lab_sequence[['subject_id', 'hadm_id', 'charttime']].copy()
        if 'glucose' in lab_sequence.columns:
            clin_flags['hypoglycemia'] = (lab_sequence['glucose'] < 70.0).astype(float)
            clin_flags['hyperglycemia'] = (lab_sequence['glucose'] > 180.0).astype(float)
        else:
            clin_flags['hypoglycemia'] = 0.0
            clin_flags['hyperglycemia'] = 0.0

        day_idx = lab_sequence[['subject_id', 'hadm_id', 'charttime']].copy()
        day_idx['day_raw'] = (
            lab_sequence.groupby(['subject_id', 'hadm_id']).cumcount().astype(float)
        )
        insulin_daily = self._extract_insulin_daily(prescriptions, subj_set)

        temporal_features = engineer.create_temporal_features(
            lab_sequence, time_column='charttime'
        )
        clean_data = preprocessor.clean_missing_values(temporal_features)
        clean_data = preprocessor.handle_outliers(clean_data)
        normalized_data = preprocessor.normalize_labs(clean_data)

        normalized_data = normalized_data.merge(
            clin_flags, on=['subject_id', 'hadm_id', 'charttime'], how='left'
        )
        normalized_data = normalized_data.merge(
            day_idx, on=['subject_id', 'hadm_id', 'charttime'], how='left'
        )
        max_days = (
            normalized_data.groupby(['subject_id', 'hadm_id'])['day_raw']
            .transform('max').replace(0, 1)
        )
        normalized_data['day'] = (normalized_data['day_raw'] / max_days).fillna(0.0)

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

        vitals_df = None
        if getattr(self.config, 'use_vitals', False):
            all_vital_ids = [
                211, 220045,
                51, 442, 455, 6701, 220050, 220179,
                618, 615, 220210, 224690,
                646, 220277,
            ]
            chartevents = loader.load_chartevents(
                subject_ids=list(subj_set), item_ids=all_vital_ids,
            )
            if len(chartevents) > 0:
                vitals_df = engineer.extract_vitals_sequence(
                    chartevents, subject_ids=list(subj_set)
                )
        if getattr(self.config, 'use_vitals', False) and vitals_df is not None and len(vitals_df) > 0:
            avail_vcols = [c for c in VITAL_COLS if c in vitals_df.columns]
            if avail_vcols:
                vdf = vitals_df.copy()
                vdf['_vd'] = pd.to_datetime(vdf['charttime'], errors='coerce').dt.date
                vagg = (
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

        if getattr(self.config, 'use_med_history', False):
            med_hist = self._compute_med_history_features(prescriptions, subj_set)
            normalized_data = normalized_data.merge(med_hist, on='subject_id', how='left')
        for col in MED_HISTORY_COLS:
            if col not in normalized_data.columns:
                normalized_data[col] = 0.0
            else:
                normalized_data[col] = normalized_data[col].fillna(0.0)

        normalized_data = normalized_data.drop(columns=['day_raw'], errors='ignore')
        normalized_data = self._build_mimic_state_columns(normalized_data)

        state_cols = self._get_state_cols()
        for col in state_cols:
            if col not in normalized_data.columns:
                normalized_data[col] = 0.0

        return normalized_data

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
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Patient batch size for MIMIC feature preparation')
    parser.add_argument('--resume', action='store_true',
                        help='Resume MIMIC preparation from existing checkpoints when available')
    parser.add_argument('--start-from', type=str, default=None,
                        help='Start pipeline from this stage/step (e.g., stage_3_baseline_training)')
    parser.add_argument('--stop-after', type=str, default=None,
                        help='Stop pipeline after this stage/step')
    parser.add_argument('--force-stage', type=str, default=None,
                        help='Comma-separated stage/step names to force recompute even if checkpoint exists')
    parser.add_argument('--invalidate-from', type=str, default=None,
                        help='Invalidate checkpoints from this stage/step onward before run')
    parser.add_argument('--ignore-checkpoints', action='store_true',
                        help='Ignore all existing checkpoints and recompute from scratch')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory for stage/batch checkpoints (default: <output-dir>/checkpoints/mimic)')
    parser.add_argument('--fast-eval', action='store_true',
                        help='Fast local evaluation mode (deterministic subsampling for quicker debug runs)')
    parser.add_argument('--max-eval-samples', type=int, default=2000,
                        help='Maximum number of evaluation transitions when --fast-eval is enabled')
    parser.add_argument('--max-rollout-policies', type=int, default=None,
                        help='Optional cap on number of baselines/policies evaluated in rollout mode')
    parser.add_argument('--include-baselines', type=str, default=None,
                        help='Comma-separated baseline names to include (e.g., Rule-Based,Mean-Action)')
    parser.add_argument('--exclude-baselines', type=str, default=None,
                        help='Comma-separated baseline names to exclude')
    parser.add_argument('--skip-slow-baselines', action='store_true',
                        help='Skip known slow baselines (KNN-5, Behavior-Cloning) for debug runs')
    parser.add_argument('--force-recompute-baselines', action='store_true',
                        help='Ignore baseline rollout cache and recompute baseline evaluations')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip final report generation stage (faster local iteration)')
    parser.add_argument('--light-report', action='store_true',
                        help='Generate summary files but skip heavy visualization generation')
    parser.add_argument('--disable-missingness-masks', action='store_true',
                        help='Disable explicit missingness mask features in MIMIC preprocessing')
    parser.add_argument('--enable-time-since-last-observed', action='store_true',
                        help='Enable optional time-since-last-observed features (if supported by policy)')
    parser.add_argument('--lab-max-hold-steps', type=int, default=None,
                        help='Max forward-fill hold steps for sparse lab features (None = unbounded)')
    parser.add_argument('--vital-max-hold-steps', type=int, default=None,
                        help='Max forward-fill hold steps for vital features (None = unbounded)')
    parser.add_argument('--drop-high-missingness-columns', action='store_true',
                        help='Drop columns above high missingness threshold in missing-data policy')
    parser.add_argument('--high-missingness-threshold', type=float, default=0.95,
                        help='Threshold used with --drop-high-missingness-columns')
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

    def _parse_csv(raw: Optional[str]) -> List[str]:
        if raw is None:
            return []
        return [x.strip() for x in str(raw).split(',') if x.strip()]

    known = {
        'Rule-Based', 'Random-Uniform', 'Random-Safe',
        'Mean-Action', 'Ridge-Regression', 'KNN-5', 'Behavior-Cloning',
    }
    requested = set(_parse_csv(args.include_baselines)) | set(_parse_csv(args.exclude_baselines))
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown baseline names requested: {unknown}. Known names: {sorted(known)}")

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
