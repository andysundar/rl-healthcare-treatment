import subprocess
from pathlib import Path


def test_defense_bundle_smoke(tmp_path: Path):
    run_id = 'defense_smoke_test'
    cmd = [
        'python', 'src/run_integrated_solution.py',
        '--defense-bundle',
        '--run-id', run_id,
        '--n-synthetic-patients', '20',
        '--trajectory-length', '8',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-1200:]

    root = Path('outputs') / run_id
    required = [
        root / '00_metadata' / 'run_config.yaml',
        root / '01_data_summary' / 'leakage_checks.md',
        root / '03_evaluation' / 'ope_summary.csv',
        root / '03_evaluation' / 'ope_returns_ci.png',
        root / '04_interpretability' / 'distilled_tree.txt',
        root / '06_final_reports' / 'DEFENSE_REPORT.md',
    ]
    for p in required:
        assert p.exists(), f'missing artifact: {p}'
