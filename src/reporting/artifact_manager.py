from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


class ArtifactManager:
    def __init__(self, base_output: Path, run_id: str):
        self.run_root = base_output / run_id
        self.dirs = {
            'meta': self.run_root / '00_metadata',
            'data': self.run_root / '01_data_summary',
            'train': self.run_root / '02_training',
            'eval': self.run_root / '03_evaluation',
            'interp': self.run_root / '04_interpretability',
            'demo': self.run_root / '05_demo_assets',
            'final': self.run_root / '06_final_reports',
        }
        for p in self.dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        (self.dirs['train'] / 'training_curves').mkdir(exist_ok=True)
        (self.dirs['train'] / 'checkpoints').mkdir(exist_ok=True)

    def write_metadata(self, config: Any, argv: str, seed: int):
        meta = self.dirs['meta']
        cfg_dict = asdict(config) if is_dataclass(config) else vars(config)
        (meta / 'run_config.yaml').write_text(yaml.safe_dump(cfg_dict, sort_keys=True))
        (meta / 'command_invocation.txt').write_text(argv + '\n')
        (meta / 'seed.txt').write_text(str(seed) + '\n')
        (meta / 'environment.txt').write_text('\n'.join([f"platform={platform.platform()}", f"cwd={os.getcwd()}"]))

        try:
            versions = subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True)
        except Exception:
            versions = 'pip freeze unavailable\n'
        (meta / 'versions.txt').write_text(versions)

        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()
            status = subprocess.check_output(['git', 'status', '--short'], text=True)
            (meta / 'git_info.txt').write_text(f"commit={commit}\nbranch={branch}\n\n{status}")
        except Exception:
            (meta / 'git_info.txt').write_text('git unavailable\n')

    def write_data_summary(self, raw_df: pd.DataFrame, split_info: Dict[str, int], state_cols: list[str], leakage_text: str):
        d = self.dirs['data']
        cohort = pd.DataFrame([split_info])
        cohort.to_csv(d / 'cohort_stats.csv', index=False)
        miss = raw_df[state_cols].isna().mean().reset_index()
        miss.columns = ['feature', 'missing_rate']
        miss.to_csv(d / 'missingness_report.csv', index=False)
        pd.DataFrame({'feature': state_cols}).to_csv(d / 'feature_list.csv', index=False)
        profile = [
            '# Dataset Profile',
            f"- rows: {len(raw_df)}",
            f"- unique_subjects: {raw_df['subject_id'].nunique() if 'subject_id' in raw_df else 'n/a'}",
            f"- generated_at: {datetime.now().isoformat()}",
        ]
        (d / 'dataset_profile.md').write_text('\n'.join(profile) + '\n')
        (d / 'leakage_checks.md').write_text(leakage_text)

    def write_eval_tables(self, metrics_df: pd.DataFrame, ope_rows: list[dict], safety_rows: list[dict], subgroup_rows: list[dict], warnings_md: str):
        e = self.dirs['eval']
        metrics_df.to_csv(e / 'metrics_summary.csv', index=False)
        metrics_df.to_json(e / 'metrics_summary.json', orient='records', indent=2)

        ope_df = pd.DataFrame(ope_rows)
        ope_df.to_csv(e / 'ope_summary.csv', index=False)
        ope_df.to_json(e / 'ope_summary.json', orient='records', indent=2)

        ci_df = ope_df[[c for c in ope_df.columns if c in ['policy', 'method', 'ci_low', 'ci_high']]] if len(ope_df) else pd.DataFrame(columns=['policy','method','ci_low','ci_high'])
        ci_df.to_csv(e / 'ope_bootstrap_ci.csv', index=False)
        ess_df = ope_df[[c for c in ope_df.columns if c in ['policy', 'method', 'ess']]] if len(ope_df) else pd.DataFrame(columns=['policy','method','ess'])
        ess_df.to_csv(e / 'ess_report.csv', index=False)

        pd.DataFrame(safety_rows).to_csv(e / 'safety_summary.csv', index=False)
        pd.DataFrame(subgroup_rows).to_csv(e / 'subgroup_analysis.csv', index=False)
        (e / 'support_mismatch_warnings.md').write_text(warnings_md)

    def write_demo_assets(self):
        d = self.dirs['demo']
        (d / 'demo_script.md').write_text("""# 5-minute Demo Script\n1. Show metadata + deterministic seed.\n2. Show OPE comparison (offline RL vs baselines).\n3. Show safety-on vs safety-off ablation.\n4. Show distilled rules and fidelity.\n""")
        (d / 'demo_slides_outline.md').write_text("""# Slides Outline\n- Problem and constraints\n- Methods\n- Results C1-C4\n- Limitations and future work\n""")
        (d / 'what_to_show_live.md').write_text("""# What to Show Live\n- 00_metadata/run_config.yaml\n- 03_evaluation/ope_summary.csv\n- 03_evaluation/safety_summary.csv\n- 04_interpretability/distilled_tree.txt\n""")

    def write_final_reports(self, defense_md: str, one_page_md: str, figures_index_md: str):
        f = self.dirs['final']
        (f / 'DEFENSE_REPORT.md').write_text(defense_md)
        (f / 'ONE_PAGE_EXEC_SUMMARY.md').write_text(one_page_md)
        (f / 'FIGURES_INDEX.md').write_text(figures_index_md)
