"""Deterministic trajectory builder for offline RL from tabular EHR-like data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd


@dataclass
class TrajectoryBuildConfig:
    patient_keys: Sequence[str] = ("subject_id", "hadm_id")
    time_col: str = "day"
    state_cols: Sequence[str] = ()
    action_col: str = "action"
    reward_col: str = "reward"
    timestep: str = "1D"
    seed: int = 42


class DeterministicTrajectoryBuilder:
    """Build deterministic (s,a,r,s',done) transitions with leakage-safe split."""

    def __init__(self, config: TrajectoryBuildConfig):
        if not config.state_cols:
            raise ValueError("state_cols must be non-empty")
        self.config = config

    def build(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        c = self.config
        required = set(c.patient_keys) | {c.time_col, c.action_col, c.reward_col} | set(c.state_cols)
        missing = sorted([col for col in required if col not in df.columns])
        if missing:
            raise ValueError(f"Missing required columns for trajectory build: {missing}")

        ordered = df.sort_values(list(c.patient_keys) + [c.time_col]).copy()

        # forward-fill within patient episode and add missing indicators
        for col in c.state_cols:
            miss_col = f"{col}_is_missing"
            ordered[miss_col] = ordered[col].isna().astype(np.float32)
            ordered[col] = ordered.groupby(list(c.patient_keys))[col].ffill()
            ordered[col] = ordered[col].fillna(ordered[col].median())

        model_state_cols = list(c.state_cols) + [f"{x}_is_missing" for x in c.state_cols]

        transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        for _, g in ordered.groupby(list(c.patient_keys), sort=False):
            if len(g) < 2:
                continue
            g = g.reset_index(drop=True)
            for i in range(len(g) - 1):
                s = g.loc[i, model_state_cols].to_numpy(dtype=np.float32)
                sp = g.loc[i + 1, model_state_cols].to_numpy(dtype=np.float32)
                a = np.asarray([g.loc[i, c.action_col]], dtype=np.float32)
                r = float(g.loc[i, c.reward_col])
                done = (i == len(g) - 2)
                transitions.append((s, a, r, sp, done))
        return transitions

    def patient_split(self, df: pd.DataFrame, ratios=(0.7, 0.15, 0.15)):
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")
        c = self.config
        key_df = df[list(c.patient_keys)].drop_duplicates().copy()
        keys = key_df.to_records(index=False)

        rng = np.random.default_rng(c.seed)
        perm = rng.permutation(len(keys))
        keys = keys[perm]

        n = len(keys)
        n_train = int(ratios[0] * n)
        n_val = int(ratios[1] * n)

        train_keys = set(map(tuple, keys[:n_train]))
        val_keys = set(map(tuple, keys[n_train:n_train + n_val]))
        test_keys = set(map(tuple, keys[n_train + n_val:]))

        to_key = lambda row: tuple(row[k] for k in c.patient_keys)
        train = df[df.apply(lambda r: to_key(r) in train_keys, axis=1)]
        val = df[df.apply(lambda r: to_key(r) in val_keys, axis=1)]
        test = df[df.apply(lambda r: to_key(r) in test_keys, axis=1)]

        return train.copy(), val.copy(), test.copy()
