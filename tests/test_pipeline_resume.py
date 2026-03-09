import argparse
from pathlib import Path
from types import MethodType

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
