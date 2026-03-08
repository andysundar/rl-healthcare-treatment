import subprocess
from pathlib import Path
import sys


def test_demo_pipeline_runs(tmp_path: Path):
    out = tmp_path / 'demo_out'
    cmd = [
        sys.executable, 'src/run_integrated_solution.py', '--demo',
        '--n-synthetic-patients', '30', '--trajectory-length', '10',
        '--output-dir', str(out)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-1000:]
    resolved_out = out if (out / 'results_summary.json').exists() else out.with_name(f"{out.name}_synthetic")
    assert (resolved_out / 'results_summary.json').exists()
    assert (resolved_out / 'MASTER_RESULTS.csv').exists()
    assert (resolved_out / 'safety_summary.csv').exists()
    assert (resolved_out / 'ope_estimates.csv').exists()
