from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_ope_returns_ci(ope_df: pd.DataFrame, out_path: Path):
    if ope_df.empty:
        return
    wis = ope_df[ope_df['method'] == 'wis'].copy()
    if wis.empty:
        return
    x = np.arange(len(wis))
    y = wis['value_estimate'].values
    yerr = np.vstack([
        y - wis['ci_low'].values,
        wis['ci_high'].values - y,
    ])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, y, color='steelblue')
    ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(wis['policy'].tolist(), rotation=45, ha='right')
    ax.set_ylabel('OPE value (WIS)')
    ax.set_title('OPE Returns with 95% CI')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_safety_vs_performance(safety_df: pd.DataFrame, ope_df: pd.DataFrame, out_path: Path):
    if safety_df.empty or ope_df.empty:
        return
    wis = ope_df[ope_df['method'] == 'wis'][['policy', 'value_estimate']].copy()
    merged = safety_df.merge(wis, on='policy', how='left')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged['unsafe_action_rate'], merged['value_estimate'], color='darkred')
    for _, r in merged.iterrows():
        ax.annotate(str(r['policy']), (r['unsafe_action_rate'], r['value_estimate']), fontsize=7)
    ax.set_xlabel('Unsafe action rate')
    ax.set_ylabel('OPE value (WIS)')
    ax.set_title('Safety vs Performance')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_distilled_tree_placeholder(out_path: Path, text_rules: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.text(0.01, 0.99, text_rules[:1500], va='top', ha='left', family='monospace', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
