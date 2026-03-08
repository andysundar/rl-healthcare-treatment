"""Leakage-safe missing-data policy for MIMIC-style trajectory preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


DEFAULT_UNKNOWN_TOKEN = "UNKNOWN"
NULL_LIKE_STRINGS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown/not specified",
    "not specified",
    "not recorded",
}


@dataclass
class MissingDataPolicyConfig:
    """Configuration for missing-data handling."""

    group_cols: Sequence[str] = ("subject_id", "hadm_id")
    time_col: str = "charttime"
    enable_missingness_masks: bool = True
    enable_time_since_last_observed: bool = False
    numeric_fill_stat: str = "median"
    categorical_unknown_token: str = DEFAULT_UNKNOWN_TOKEN
    lab_max_hold_steps: Optional[int] = None
    vital_max_hold_steps: Optional[int] = None
    drop_high_missingness_columns: bool = False
    high_missingness_threshold: float = 0.95
    columns_exempt_from_imputation: Sequence[str] = ()
    zero_fill_columns: Sequence[str] = ()
    report_path: Optional[str] = None


@dataclass
class FittedMissingDataPolicy:
    """Fitted policy parameters learned from the training split only."""

    config: MissingDataPolicyConfig
    numeric_fill_values: Dict[str, float] = field(default_factory=dict)
    categorical_vocab: Dict[str, List[str]] = field(default_factory=dict)
    strategies: Dict[str, str] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    mask_columns: List[str] = field(default_factory=list)
    time_since_columns: List[str] = field(default_factory=list)
    missing_rates_before: Dict[str, float] = field(default_factory=dict)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    time_varying_columns: List[str] = field(default_factory=list)
    static_numeric_columns: List[str] = field(default_factory=list)
    feature_order: List[str] = field(default_factory=list)
    unknown_token: str = DEFAULT_UNKNOWN_TOKEN


def _normalize_categorical_strings(series: pd.Series, unknown_token: str) -> pd.Series:
    s = series.copy()
    if isinstance(s.dtype, pd.CategoricalDtype):
        if unknown_token not in s.cat.categories:
            s = s.cat.add_categories([unknown_token])
        s = s.astype("string")
    else:
        s = s.astype("string")
    s = s.str.strip()
    lower = s.str.lower()
    mask = s.isna() | lower.isin(NULL_LIKE_STRINGS)
    s = s.mask(mask, unknown_token)
    return s.fillna(unknown_token)


def _ffill_with_max_hold_steps(
    df: pd.DataFrame,
    col: str,
    group_cols: Sequence[str],
    max_hold_steps: Optional[int],
) -> pd.Series:
    """Forward-fill within group, optionally capping hold length."""
    if max_hold_steps is None:
        return df.groupby(list(group_cols))[col].ffill()

    tmp = df[list(group_cols)].copy()
    tmp[col] = df[col]
    notna = tmp[col].notna()
    run_id = notna.groupby([tmp[g] for g in group_cols]).cumsum()
    since_obs = (~notna).groupby([tmp[g] for g in group_cols] + [run_id]).cumsum()
    ffilled = tmp.groupby(list(group_cols))[col].ffill()
    ffilled = ffilled.mask(since_obs > int(max_hold_steps), np.nan)
    return ffilled


def fit_missing_data_policy(
    train_df: pd.DataFrame,
    config: MissingDataPolicyConfig,
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str] = (),
    time_varying_columns: Sequence[str] = (),
    static_numeric_columns: Sequence[str] = (),
) -> FittedMissingDataPolicy:
    """Fit leakage-safe missing-data policy on the training split."""
    unknown_token = config.categorical_unknown_token
    policy = FittedMissingDataPolicy(
        config=config,
        unknown_token=unknown_token,
    )

    cols_present = lambda cols: [c for c in cols if c in train_df.columns]
    policy.numeric_columns = cols_present(numeric_columns)
    policy.categorical_columns = cols_present(categorical_columns)
    policy.time_varying_columns = [c for c in cols_present(time_varying_columns) if c in policy.numeric_columns]
    policy.static_numeric_columns = [c for c in cols_present(static_numeric_columns) if c in policy.numeric_columns]

    for col in policy.numeric_columns + policy.categorical_columns:
        policy.missing_rates_before[col] = float(train_df[col].isna().mean())
    if policy.missing_rates_before:
        logger.info("Missing-data policy fit: per-column missing rates (train-only) computed for %s columns.", len(policy.missing_rates_before))

    if config.drop_high_missingness_columns:
        for col in list(policy.numeric_columns):
            if col in config.columns_exempt_from_imputation:
                continue
            if policy.missing_rates_before.get(col, 0.0) >= config.high_missingness_threshold:
                policy.dropped_columns.append(col)
                policy.numeric_columns.remove(col)
                if col in policy.time_varying_columns:
                    policy.time_varying_columns.remove(col)
                if col in policy.static_numeric_columns:
                    policy.static_numeric_columns.remove(col)
                policy.strategies[col] = "dropped_high_missingness"
                logger.warning("Dropping column '%s' due to high missingness rate %.3f", col, policy.missing_rates_before.get(col, np.nan))

    train_work = train_df.copy()
    if config.time_col in train_work.columns:
        train_work[config.time_col] = pd.to_datetime(train_work[config.time_col], errors="coerce")
        train_work = train_work.sort_values(list(config.group_cols) + [config.time_col])
    else:
        train_work = train_work.sort_values(list(config.group_cols))

    for col in policy.time_varying_columns:
        if col in config.columns_exempt_from_imputation:
            continue
        if col in config.zero_fill_columns:
            policy.strategies[col] = "zero_fill"
            policy.numeric_fill_values[col] = 0.0
            continue
        max_hold = (
            config.vital_max_hold_steps
            if "heart_rate" in col or "sbp" in col or "respiratory" in col or "spo2" in col
            else config.lab_max_hold_steps
        )
        held = _ffill_with_max_hold_steps(train_work, col, config.group_cols, max_hold)
        train_work[col] = held
        if config.numeric_fill_stat == "median":
            fill = float(pd.to_numeric(train_work[col], errors="coerce").median())
        else:
            fill = float(pd.to_numeric(train_work[col], errors="coerce").mean())
        if not np.isfinite(fill):
            fill = 0.0
        policy.numeric_fill_values[col] = fill
        if max_hold is None:
            policy.strategies[col] = "ffill_then_train_median"
        else:
            policy.strategies[col] = f"ffill_max_hold_{max_hold}_then_train_median"

    for col in policy.numeric_columns:
        if col in policy.time_varying_columns:
            continue
        if col in config.columns_exempt_from_imputation:
            policy.strategies[col] = "exempt"
            continue
        if col in config.zero_fill_columns:
            policy.numeric_fill_values[col] = 0.0
            policy.strategies[col] = "zero_fill"
            continue
        s = pd.to_numeric(train_work[col], errors="coerce")
        fill = float(s.median()) if config.numeric_fill_stat == "median" else float(s.mean())
        if not np.isfinite(fill):
            fill = 0.0
        policy.numeric_fill_values[col] = fill
        policy.strategies[col] = "train_median_fill"

    for col in policy.categorical_columns:
        s = _normalize_categorical_strings(train_work[col], unknown_token)
        vocab = sorted(set(s.tolist() + [unknown_token]))
        policy.categorical_vocab[col] = vocab
        policy.strategies[col] = "normalize_unknown_token"

    if config.enable_missingness_masks:
        policy.mask_columns = [f"{c}_missing" for c in policy.numeric_columns + policy.categorical_columns]
    if config.enable_time_since_last_observed:
        policy.time_since_columns = [f"time_since_last_{c}" for c in policy.time_varying_columns]

    base_order = [c for c in train_df.columns if c not in policy.dropped_columns]
    policy.feature_order = base_order + policy.mask_columns + policy.time_since_columns
    logger.info(
        "Missing-data policy fit complete: numeric=%s categorical=%s masks=%s tsl=%s",
        len(policy.numeric_columns),
        len(policy.categorical_columns),
        len(policy.mask_columns),
        len(policy.time_since_columns),
    )
    if config.report_path:
        rows = []
        for col, strat in policy.strategies.items():
            rows.append({
                "column_name": col,
                "missing_rate_before": float(policy.missing_rates_before.get(col, np.nan)),
                "strategy": strat,
                "fill_value": policy.numeric_fill_values.get(col, None),
                "category_policy": (
                    f"vocab_size={len(policy.categorical_vocab.get(col, []))}"
                    if col in policy.categorical_vocab else ""
                ),
                "mask_added": int(f"{col}_missing" in policy.mask_columns),
            })
        if rows:
            pd.DataFrame(rows).sort_values("column_name").to_csv(config.report_path, index=False)
            logger.info("Saved missing-data policy report: %s", config.report_path)
    return policy


def transform_with_missing_data_policy(
    df: pd.DataFrame,
    policy: FittedMissingDataPolicy,
) -> pd.DataFrame:
    """Apply fitted missing-data policy to any split (train/val/test)."""
    cfg = policy.config
    out = df.copy()
    for col in policy.dropped_columns:
        if col in out.columns:
            out = out.drop(columns=[col])

    if cfg.time_col in out.columns:
        out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="coerce")
        out = out.sort_values(list(cfg.group_cols) + [cfg.time_col])
    else:
        out = out.sort_values(list(cfg.group_cols))

    if cfg.enable_missingness_masks:
        for col in policy.numeric_columns:
            if col in out.columns:
                out[f"{col}_missing"] = out[col].isna().astype(np.float32)
        for col in policy.categorical_columns:
            if col in out.columns:
                out[f"{col}_missing"] = out[col].isna().astype(np.float32)

    if cfg.enable_time_since_last_observed:
        for col in policy.time_varying_columns:
            if col not in out.columns:
                continue
            notna = out[col].notna()
            run_id = notna.groupby([out[g] for g in cfg.group_cols]).cumsum()
            since_obs = (~notna).groupby([out[g] for g in cfg.group_cols] + [run_id]).cumsum()
            tcol = f"time_since_last_{col}"
            out[tcol] = since_obs.astype(np.float32)
            if tcol not in policy.time_since_columns:
                policy.time_since_columns.append(tcol)

    for col in policy.time_varying_columns:
        if col not in out.columns:
            continue
        if col in cfg.columns_exempt_from_imputation:
            continue
        if col in cfg.zero_fill_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            continue
        max_hold = (
            cfg.vital_max_hold_steps
            if "heart_rate" in col or "sbp" in col or "respiratory" in col or "spo2" in col
            else cfg.lab_max_hold_steps
        )
        out[col] = _ffill_with_max_hold_steps(out, col, cfg.group_cols, max_hold)
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(policy.numeric_fill_values.get(col, 0.0))

    for col in policy.numeric_columns:
        if col not in out.columns:
            continue
        if col in policy.time_varying_columns:
            continue
        if col in cfg.columns_exempt_from_imputation:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            continue
        if col in cfg.zero_fill_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(policy.numeric_fill_values.get(col, 0.0))

    for col in policy.categorical_columns:
        if col not in out.columns:
            out[col] = policy.unknown_token
        s = _normalize_categorical_strings(out[col], policy.unknown_token)
        vocab = set(policy.categorical_vocab.get(col, [policy.unknown_token]))
        out[col] = s.where(s.isin(vocab), policy.unknown_token)

    # enforce stable order; keep extra columns at end deterministically
    ordered = [c for c in policy.feature_order if c in out.columns]
    ordered += [c for c in policy.time_since_columns if c in out.columns and c not in ordered]
    extras = [c for c in out.columns if c not in ordered]
    return out[ordered + sorted(extras)]
