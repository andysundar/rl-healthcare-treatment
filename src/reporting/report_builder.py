from __future__ import annotations

from pathlib import Path


def build_defense_report(run_root: Path, transfer_future_work: bool = True) -> str:
    c5 = (
        "Transfer adaptation is future work in this bundle; we provide shift-analysis placeholders and clearly avoid adaptation claims."
        if transfer_future_work else
        "Transfer adaptation evidence is included in subgroup/domain metrics."
    )
    return f"""# DEFENSE REPORT

## 1) Project Overview
Offline RL for daily treatment recommendation with explicit safety constraints, retrospective-only evaluation.

## 2) Experimental Setup
- MDP/action/reward/safety config: `00_metadata/run_config.yaml`
- Data/leakage profile: `01_data_summary/dataset_profile.md`, `01_data_summary/leakage_checks.md`

## 3) Models
- Baselines and RL outputs in `03_evaluation/metrics_summary.csv`

## 4) Evaluation Methods
- OPE (WIS/DR, ESS, CI): `03_evaluation/ope_summary.csv`, `03_evaluation/ess_report.csv`, `03_evaluation/ope_bootstrap_ci.csv`
- Safety metrics: `03_evaluation/safety_summary.csv`

## 5) Results and Evidence by Claim
### C1 Evidence: Offline RL > baseline under OPE
- Table: `03_evaluation/ope_summary.csv`
- CI table: `03_evaluation/ope_bootstrap_ci.csv`
- Plot: `03_evaluation/ope_returns_ci.png`
- Interpretation: Compare IQL/CQL rows against BC/rule-based rows with CI overlap.

### C2 Evidence: Safety constraints reduce unsafe actions
- Table: `03_evaluation/safety_summary.csv`
- Plot: `03_evaluation/safety_vs_performance.png`
- Warnings audit: `03_evaluation/support_mismatch_warnings.md`
- Interpretation: Safety-on should lower unsafe_action_rate with bounded value drop.

### C3 Evidence: Interpretability via distillation
- Rules text: `04_interpretability/distilled_tree.txt`
- Rules figure: `04_interpretability/distilled_tree.png`
- Fidelity: `04_interpretability/distillation_fidelity.csv`
- Examples: `04_interpretability/example_explanations.md`

### C4 Evidence: Reproducible + no leakage
- Metadata snapshot: `00_metadata/*`
- Leakage checks: `01_data_summary/leakage_checks.md`
- Seed/config: `00_metadata/seed.txt`, `00_metadata/run_config.yaml`

### C5 Evidence: Transfer adaptation or shift-only
{c5}

## 6) Demo Script
See `05_demo_assets/demo_script.md`, `05_demo_assets/what_to_show_live.md`.

## 7) Appendix
- Full metrics: `03_evaluation/metrics_summary.csv`
- Figures index: `06_final_reports/FIGURES_INDEX.md`
- Limitations: retrospective/OPE support limits, non-deployment scope.
"""


def build_one_page_summary(run_root: Path) -> str:
    return """# One-Page Executive Summary
- Goal: safe offline treatment recommendation from retrospective data.
- Main evidence: OPE + safety + interpretability artifacts.
- Reproducibility: full metadata + deterministic seed + command snapshot.
"""


def build_figures_index() -> str:
    return """# Figures Index
- `03_evaluation/ope_returns_ci.png`: OPE value with CI.
- `03_evaluation/safety_vs_performance.png`: safety-performance tradeoff.
- `04_interpretability/distilled_tree.png`: distilled rule visualization.
"""
