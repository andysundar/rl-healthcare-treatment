import numpy as np
import pandas as pd

from src.data.trajectory_builder import DeterministicTrajectoryBuilder, TrajectoryBuildConfig


def test_trajectory_builder_deterministic_and_split_no_leakage():
    rows = []
    for pid in ['p1', 'p2', 'p3', 'p4']:
        for day in range(3):
            rows.append({
                'subject_id': pid,
                'hadm_id': pid,
                'day': day,
                'x': np.nan if day == 0 else day,
                'action': float(day % 2),
                'reward': float(day),
            })
    df = pd.DataFrame(rows)
    cfg = TrajectoryBuildConfig(
        patient_keys=('subject_id', 'hadm_id'),
        time_col='day',
        state_cols=('x',),
        action_col='action',
        reward_col='reward',
        seed=7,
    )
    b = DeterministicTrajectoryBuilder(cfg)
    tr, va, te = b.patient_split(df)
    tr_keys = set(zip(tr.subject_id, tr.hadm_id))
    va_keys = set(zip(va.subject_id, va.hadm_id))
    te_keys = set(zip(te.subject_id, te.hadm_id))
    assert tr_keys.isdisjoint(va_keys)
    assert tr_keys.isdisjoint(te_keys)
    assert va_keys.isdisjoint(te_keys)

    t1 = b.build(tr)
    t2 = b.build(tr)
    assert len(t1) == len(t2)
    assert np.allclose(t1[0][0], t2[0][0])
    assert t1[0][0].shape[0] == 2  # x plus missingness indicator
