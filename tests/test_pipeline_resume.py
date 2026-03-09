import argparse
import pickle
import os
from pathlib import Path
from types import MethodType

import pandas as pd
import pytest

from src.run_integrated_solution import IntegratedSolutionRunner


def _config(tmp_path: Path, **overrides):
    base = dict(
        mode='train-eval',
        output_dir=str(tmp_path / 'run'),
        seed=42,
        resume=False,
        start_from=None,
        stop_after=None,
        force_stage=None,
        invalidate_from=None,
        no_report=False,
        defense_bundle=False,
        use_synthetic=True,
        use_encoder=False,
        use_transfer=False,
        use_interpretability=False,
        run_distillation=False,
        train_cql=False,
        train_iql=False,
        batch_size=32,
        ignore_checkpoints=False,
        checkpoint_dir=None,
        mimic_dir='data/raw/mimic-iii',
        n_synthetic_patients=10,
        sample_size=10,
        fast_eval=False,
        max_eval_samples=100,
        skip_slow_baselines=False,
        max_rollout_policies=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _attach_fake_stage_methods(runner: IntegratedSolutionRunner, events):
    for step_name in runner.PIPELINE_STAGE_ORDER:
        def _fn(self, _step=step_name):
            events.append(_step)
            for key in self.STAGE_CHECKPOINT_KEYS.get(_step, []):
                self.results[key] = {'step': _step, 'key': key}
        setattr(runner, step_name, MethodType(_fn, runner))

    # aliases used in run_full_pipeline
    runner.stage_1_data_preparation = runner.stage_1_data_preparation
    runner.stage_2_environment_setup = runner.stage_2_environment_setup
    runner.stage_2b_encoder_training = runner.stage_2b_encoder_training
    runner.stage_3_baseline_training = runner.stage_3_baseline_training
    runner.stage_4_cql_training = runner.stage_4_cql_training
    runner.stage_4b_iql_training = runner.stage_4b_iql_training
    runner.stage_5_evaluation = runner.stage_5_evaluation
    runner.stage_6b_transfer = runner.stage_6b_transfer
    runner.stage_7_interpretability = runner.stage_7_interpretability
    runner.stage_7b_policy_distillation = runner.stage_7b_policy_distillation
    runner.stage_6_prepare_report_inputs = runner.stage_6_report_inputs
    runner.stage_6_generate_reports = runner.stage_6_generate_reports


def test_resume_skips_completed_checkpoints(tmp_path):
    out_dir = tmp_path / 'resume_run'

    cfg1 = _config(out_dir, output_dir=str(out_dir), stop_after='stage_3_baseline_training')
    runner1 = IntegratedSolutionRunner(cfg1)
    events1 = []
    _attach_fake_stage_methods(runner1, events1)
    runner1.run_full_pipeline()
    assert 'stage_1_data_preparation' in events1
    assert 'stage_3_baseline_training' in events1

    cfg2 = _config(out_dir, output_dir=str(out_dir), resume=True)
    runner2 = IntegratedSolutionRunner(cfg2)
    events2 = []
    _attach_fake_stage_methods(runner2, events2)
    runner2.run_full_pipeline()

    assert 'stage_1_data_preparation' not in events2
    assert 'stage_3_baseline_training' not in events2
    assert 'stage_5_evaluation' in events2


def test_config_change_invalidates_resume(tmp_path):
    out_dir = tmp_path / 'config_change'

    cfg1 = _config(out_dir, output_dir=str(out_dir), stop_after='stage_2_environment_setup', seed=7)
    runner1 = IntegratedSolutionRunner(cfg1)
    events1 = []
    _attach_fake_stage_methods(runner1, events1)
    runner1.run_full_pipeline()
    assert 'stage_1_data_preparation' in events1

    # change seed -> config hash changes -> stage should recompute
    cfg2 = _config(out_dir, output_dir=str(out_dir), resume=True, stop_after='stage_2_environment_setup', seed=99)
    runner2 = IntegratedSolutionRunner(cfg2)
    events2 = []
    _attach_fake_stage_methods(runner2, events2)
    runner2.run_full_pipeline()
    assert 'stage_1_data_preparation' in events2


def test_start_from_and_stop_after(tmp_path):
    cfg = _config(
        tmp_path,
        start_from='stage_3_baseline_training',
        stop_after='stage_4b_iql_training',
    )
    runner = IntegratedSolutionRunner(cfg)
    events = []
    _attach_fake_stage_methods(runner, events)
    runner.run_full_pipeline()

    assert events == [
        'stage_3_baseline_training',
        'stage_4_cql_training',
        'stage_4b_iql_training',
    ]


def test_corrupted_checkpoint_not_reused(tmp_path):
    out_dir = tmp_path / 'corrupted'
    cfg1 = _config(out_dir, output_dir=str(out_dir), stop_after='stage_1_data_preparation')
    runner1 = IntegratedSolutionRunner(cfg1)
    events1 = []
    _attach_fake_stage_methods(runner1, events1)
    runner1.run_full_pipeline()
    assert events1 == ['stage_1_data_preparation']

    ckpt = runner1._pipeline_checkpoint_path('stage_1_data_preparation')
    ckpt.write_bytes(b'not-a-valid-pickle')

    cfg2 = _config(out_dir, output_dir=str(out_dir), resume=True, stop_after='stage_1_data_preparation')
    runner2 = IntegratedSolutionRunner(cfg2)
    events2 = []
    _attach_fake_stage_methods(runner2, events2)
    runner2.run_full_pipeline()

    assert events2 == ['stage_1_data_preparation']


def test_checkpoint_schema_version_mismatch_invalidates(tmp_path):
    out_dir = tmp_path / 'schema_mismatch'
    cfg = _config(out_dir, output_dir=str(out_dir), stop_after='stage_1_data_preparation')
    runner = IntegratedSolutionRunner(cfg)
    events = []
    _attach_fake_stage_methods(runner, events)
    runner.run_full_pipeline()

    ckpt = runner._pipeline_checkpoint_path('stage_1_data_preparation')
    with open(ckpt, 'rb') as f:
        payload = pickle.load(f)
    payload['meta']['stage_schema_version'] = 0
    runner._atomic_write_bytes(ckpt, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    valid = runner.is_checkpoint_valid(
        'stage_1_data_preparation',
        runner._compute_config_hash(),
        runner._compute_dataset_signature(),
    )
    assert not valid


def test_pickle_checkpoint_uses_atomic_write_helper(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    runner = IntegratedSolutionRunner(cfg)
    called = {'ok': False}

    def _fake_atomic_write_pickle(path, obj):
        called['ok'] = True
        runner._atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    monkeypatch.setattr(runner, "_atomic_write_pickle", _fake_atomic_write_pickle)
    runner._save_pickle_checkpoint('unit_pickle_ckpt', {'x': 1})
    assert called['ok'] is True
    assert runner._checkpoint_path('unit_pickle_ckpt', 'pkl').exists()


def test_corrupt_pickle_checkpoint_load_returns_none(tmp_path):
    cfg = _config(tmp_path)
    runner = IntegratedSolutionRunner(cfg)
    path = runner._checkpoint_path('corrupt_pickle', 'pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'not-a-pickle')
    loaded = runner._load_pickle_checkpoint('corrupt_pickle')
    assert loaded is None


def test_manifest_not_marked_complete_if_checkpoint_write_fails(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    runner = IntegratedSolutionRunner(cfg)

    def _boom(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(runner, "_atomic_write_bytes", _boom)
    with pytest.raises(OSError):
        runner.save_checkpoint('stage_3_baseline_training', {'dummy': 1})

    stage_meta = runner.pipeline_manifest.get('stages', {}).get('stage_3_baseline_training', {})
    assert stage_meta.get('status') != 'completed'


def test_parquet_checkpoint_write_is_atomic(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    runner = IntegratedSolutionRunner(cfg)
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    called = {'replace': False}
    real_replace = os.replace

    def _replace(src, dst):
        called['replace'] = True
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _replace)
    out_path = runner._save_parquet_checkpoint('unit_parquet_ckpt', df)
    assert called['replace'] is True
    assert out_path.exists()
