import subprocess
from pathlib import Path


def test_demo_pipeline_runs(tmp_path: Path):
    out = tmp_path / 'demo_out'
    cmd = [
        'python', 'src/run_integrated_solution.py', '--demo',
        '--n-synthetic-patients', '30', '--trajectory-length', '10',
        '--output-dir', str(out)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-1000:]
    assert (out / 'results_summary.json').exists()
