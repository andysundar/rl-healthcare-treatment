import pandas as pd

from src.data.feature_engineering import FeatureEngineer


def test_create_temporal_features_excludes_timedelta_from_rolling():
    df = pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 2],
        "charttime": pd.to_datetime([
            "2020-01-01 00:00:00",
            "2020-01-01 01:00:00",
            "2020-01-01 02:00:00",
            "2020-01-01 00:00:00",
            "2020-01-01 01:00:00",
        ]),
        "glucose": [100.0, 105.0, 102.0, 120.0, 118.0],
    })
    eng = FeatureEngineer()
    out = eng.create_temporal_features(df, time_column="charttime", subject_column="subject_id")

    assert "time_since_last_hours" in out.columns
    assert "rolling_mean_glucose" in out.columns
    assert "rolling_std_glucose" in out.columns
    assert "trend_glucose" in out.columns
    assert "rolling_mean_time_since_last" not in out.columns
    assert "rolling_std_time_since_last" not in out.columns
    assert "trend_time_since_last" not in out.columns


def test_create_temporal_features_skips_non_numeric_columns_and_handles_bad_time():
    df = pd.DataFrame({
        "subject_id": [1, 1, 1],
        "charttime": ["2020-01-01 00:00:00", "bad-time", "2020-01-01 02:00:00"],
        "glucose": [100.0, None, 102.0],
        "note": ["a", "b", "c"],
    })
    eng = FeatureEngineer()
    out = eng.create_temporal_features(df, time_column="charttime", subject_column="subject_id")

    assert len(out) == 2  # invalid timestamp row dropped
    assert "rolling_mean_glucose" in out.columns
    assert "rolling_mean_note" not in out.columns
    assert "trend_note" not in out.columns


def test_create_temporal_features_regression_for_mimic_lab_sequence_shape():
    # Mimics extract_lab_sequence output shape used in runner path.
    df = pd.DataFrame({
        "subject_id": [10, 10, 10, 11, 11],
        "hadm_id": [100, 100, 100, 110, 110],
        "charttime": pd.to_datetime([
            "2020-01-01 00:00:00",
            "2020-01-01 03:00:00",
            "2020-01-01 06:00:00",
            "2020-01-01 00:00:00",
            "2020-01-01 02:00:00",
        ]),
        "glucose": [100.0, 105.0, None, 99.0, 101.0],
        "sodium": [140.0, None, 141.0, 138.0, None],
    })
    eng = FeatureEngineer()
    out = eng.create_temporal_features(df, time_column="charttime", subject_column="subject_id")

    assert "rolling_mean_glucose" in out.columns
    assert "rolling_mean_sodium" in out.columns
    assert "trend_glucose" in out.columns
    assert "trend_sodium" in out.columns
