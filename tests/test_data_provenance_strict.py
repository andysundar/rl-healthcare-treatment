from argparse import Namespace
from pathlib import Path

import pytest

from src.run_integrated_solution import IntegratedSolutionRunner


def _cfg(tmp_path: Path, **overrides):
    base = dict(
        output_dir=str(tmp_path / "out"),
        mode="train-eval",
        seed=42,
        use_synthetic=False,
        mimic_dir=str(tmp_path / "mimic"),
        use_vitals=False,
        use_med_history=False,
        defense_bundle=False,
        demo=False,
        train_cql=False,
        train_iql=False,
        use_encoder=False,
        use_transfer=False,
        use_interpretability=False,
        run_distillation=False,
        batch_size=32,
        resume=False,
        ignore_checkpoints=False,
        checkpoint_dir=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_train_eval_runs_stage1_data_preparation(tmp_path: Path):
    cfg = _cfg(tmp_path)
    runner = IntegratedSolutionRunner(cfg)

    calls = []
    runner.stage_1_data_preparation = lambda: calls.append("stage1")
    runner.stage_2_environment_setup = lambda: calls.append("stage2")
    runner.stage_2b_encoder_training = lambda: calls.append("stage2b")
    runner.stage_3_baseline_training = lambda: calls.append("stage3")
    runner.stage_4_cql_training = lambda: calls.append("stage4")
    runner.stage_4b_iql_training = lambda: calls.append("stage4b")
    runner.stage_5_evaluation = lambda: calls.append("stage5")
    runner.stage_6b_transfer = lambda: calls.append("stage6b")
    runner.stage_7_interpretability = lambda: calls.append("stage7")
    runner.stage_7b_policy_distillation = lambda: calls.append("stage7b")
    runner.stage_6_generate_reports = lambda: calls.append("stage6")

    runner.run_full_pipeline()
    assert calls[0] == "stage1"
    assert "stage1" in calls


def test_mimic_missing_dir_fails_without_synthetic_fallback(tmp_path: Path):
    cfg = _cfg(tmp_path, mimic_dir=str(tmp_path / "missing_mimic"))
    runner = IntegratedSolutionRunner(cfg)

    with pytest.raises(FileNotFoundError):
        runner.stage_1_data_preparation()
    assert "data" not in runner.results
    assert not (Path(cfg.output_dir) / "synthetic_data.pkl").exists()


def test_validation_fails_if_synthetic_artifact_exists_in_mimic_run(tmp_path: Path):
    mimic_dir = tmp_path / "mimic"
    mimic_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(tmp_path, mimic_dir=str(mimic_dir))
    runner = IntegratedSolutionRunner(cfg)

    runner.results["data"] = {
        "source": "mimic",
        "input_dataset_path": str(mimic_dir.resolve()),
        "train": [([0.0], [0.0], 0.0, [0.0], 1.0)],
        "val": [],
        "test": [([0.0], [0.0], 0.0, [0.0], 1.0)],
        "cohort": [1],
    }
    (runner.output_dir / "mimic_trajectories.pkl").write_bytes(b"x")
    (runner.output_dir / "mimic_dataset_summary.json").write_text("{}")
    (runner.output_dir / "synthetic_data.pkl").write_bytes(b"x")
    runner._write_run_provenance_manifest()

    with pytest.raises(RuntimeError, match="Artifact validation failed"):
        runner._run_artifact_validation()


def test_prepare_mimic_data_uses_final_checkpoint_when_resume_enabled(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        resume=True,
        ignore_checkpoints=False,
        checkpoint_dir=str(tmp_path / "ckpts"),
        mimic_dir=str(tmp_path / "missing_dir_is_ok_when_checkpoint_exists"),
    )
    runner = IntegratedSolutionRunner(cfg)
    payload = {"source": "mimic", "train": [], "val": [], "test": [], "cohort": []}
    runner._save_pickle_checkpoint("prepared_mimic_data", payload)

    loaded = runner.prepare_mimic_data()
    assert loaded == payload


def test_prepare_mimic_data_missing_checkpoint_does_not_crash_resume_logic(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        resume=True,
        ignore_checkpoints=False,
        checkpoint_dir=str(tmp_path / "ckpts"),
        mimic_dir=str(tmp_path / "missing_mimic_dir"),
    )
    runner = IntegratedSolutionRunner(cfg)
    assert runner._should_load_checkpoint("prepared_mimic_data", "pkl") is False
    with pytest.raises(FileNotFoundError):
        runner.prepare_mimic_data()
