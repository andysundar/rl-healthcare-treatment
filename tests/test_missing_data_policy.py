import numpy as np
import pandas as pd

from src.data.missing_data_policy import (
    MissingDataPolicyConfig,
    fit_missing_data_policy,
    transform_with_missing_data_policy,
)


def _base_train_df():
    return pd.DataFrame({
        "subject_id": [1, 1, 2, 2],
        "hadm_id": [10, 10, 20, 20],
        "charttime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
        "heart_rate": [80.0, np.nan, np.nan, 90.0],
        "glucose_mean": [100.0, np.nan, 110.0, np.nan],
        "adherence_rate_7d": [np.nan, 0.5, 0.7, np.nan],
        "ethnicity": pd.Series(["WHITE", None, "ASIAN", " "], dtype="category"),
    })


def test_forward_fill_does_not_cross_patient_boundary():
    train = _base_train_df()
    cfg = MissingDataPolicyConfig(vital_max_hold_steps=None, lab_max_hold_steps=None)
    policy = fit_missing_data_policy(
        train,
        cfg,
        numeric_columns=["heart_rate", "glucose_mean", "adherence_rate_7d"],
        categorical_columns=["ethnicity"],
        time_varying_columns=["heart_rate", "glucose_mean"],
        static_numeric_columns=["adherence_rate_7d"],
    )
    out = transform_with_missing_data_policy(train, policy)
    # subject 2 first row should not inherit subject 1 values
    row = out[(out["subject_id"] == 2) & (out["charttime"] == pd.Timestamp("2020-01-01"))].iloc[0]
    assert np.isfinite(float(row["heart_rate"]))
    assert float(row["heart_rate"]) == policy.numeric_fill_values["heart_rate"]


def test_val_uses_train_only_fill_statistics():
    train = _base_train_df()
    val = pd.DataFrame({
        "subject_id": [3],
        "hadm_id": [30],
        "charttime": pd.to_datetime(["2020-01-01"]),
        "heart_rate": [np.nan],
        "glucose_mean": [np.nan],
        "adherence_rate_7d": [np.nan],
        "ethnicity": [None],
    })
    cfg = MissingDataPolicyConfig()
    policy = fit_missing_data_policy(
        train,
        cfg,
        numeric_columns=["heart_rate", "glucose_mean", "adherence_rate_7d"],
        categorical_columns=["ethnicity"],
        time_varying_columns=["heart_rate", "glucose_mean"],
        static_numeric_columns=["adherence_rate_7d"],
    )
    out = transform_with_missing_data_policy(val, policy)
    assert float(out.iloc[0]["heart_rate"]) == policy.numeric_fill_values["heart_rate"]
    assert float(out.iloc[0]["glucose_mean"]) == policy.numeric_fill_values["glucose_mean"]
    assert str(out.iloc[0]["ethnicity"]) == "UNKNOWN"


def test_categorical_unknown_handles_pandas_categorical_dtype():
    train = _base_train_df()
    cfg = MissingDataPolicyConfig()
    policy = fit_missing_data_policy(
        train,
        cfg,
        numeric_columns=["heart_rate"],
        categorical_columns=["ethnicity"],
        time_varying_columns=["heart_rate"],
    )
    out = transform_with_missing_data_policy(train, policy)
    assert "UNKNOWN" in set(out["ethnicity"].astype(str).tolist())


def test_missingness_masks_and_feature_shape_stable():
    train = _base_train_df()
    cfg = MissingDataPolicyConfig(enable_missingness_masks=True)
    policy = fit_missing_data_policy(
        train,
        cfg,
        numeric_columns=["heart_rate", "glucose_mean", "adherence_rate_7d"],
        categorical_columns=["ethnicity"],
        time_varying_columns=["heart_rate", "glucose_mean"],
        static_numeric_columns=["adherence_rate_7d"],
    )
    out1 = transform_with_missing_data_policy(train.iloc[:2].copy(), policy)
    out2 = transform_with_missing_data_policy(train.iloc[2:].copy(), policy)
    expected_masks = {"heart_rate_missing", "glucose_mean_missing", "adherence_rate_7d_missing", "ethnicity_missing"}
    assert expected_masks.issubset(set(out1.columns))
    assert expected_masks.issubset(set(out2.columns))
    assert out1.shape[1] == out2.shape[1]
    for c in ["heart_rate", "glucose_mean", "adherence_rate_7d"] + sorted(expected_masks):
        assert pd.api.types.is_numeric_dtype(out1[c])


def test_max_hold_steps_for_labs():
    train = pd.DataFrame({
        "subject_id": [1, 1, 1],
        "hadm_id": [10, 10, 10],
        "charttime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "glucose_mean": [100.0, np.nan, np.nan],
    })
    cfg = MissingDataPolicyConfig(lab_max_hold_steps=1)
    policy = fit_missing_data_policy(
        train,
        cfg,
        numeric_columns=["glucose_mean"],
        time_varying_columns=["glucose_mean"],
    )
    out = transform_with_missing_data_policy(train, policy)
    # step2 can hold, step3 should revert to median fill because hold exceeded
    assert float(out.iloc[1]["glucose_mean"]) == 100.0
    assert float(out.iloc[2]["glucose_mean"]) == policy.numeric_fill_values["glucose_mean"]
