import pandas as pd

from src.data.feature_engineering import FeatureEngineer, DEFAULT_MISSING_TOKEN


def test_extract_demographics_handles_categorical_fillna_without_crash():
    patients = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "gender": pd.Series(["M", None, " "], dtype="category"),
        "dob": ["1970-01-01", "1980-01-01", "1990-01-01"],
        "dod": [None, None, None],
    })
    admissions = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "admittime": ["2020-01-01", "2020-01-01", "2020-01-01"],
        "ethnicity": pd.Series(["WHITE", None, "UNKNOWN/NOT SPECIFIED"], dtype="category"),
        "insurance": pd.Series(["Medicare", " ", None], dtype="category"),
        "marital_status": [None, "MARRIED", "nan"],
        "admission_type": ["EMERGENCY", "", None],
    })

    fe = FeatureEngineer()
    out = fe.extract_demographics(patients, admissions)

    assert "ethnicity_encoded" in out.columns
    assert "insurance_encoded" in out.columns
    assert "gender_encoded" in out.columns
    assert DEFAULT_MISSING_TOKEN in set(out["ethnicity"].astype(str).tolist())
    assert DEFAULT_MISSING_TOKEN in set(out["insurance"].astype(str).tolist())
    assert DEFAULT_MISSING_TOKEN in set(out["gender"].astype(str).tolist())


def test_normalize_string_missing_maps_placeholders_to_unknown():
    s = pd.Series(["", "  ", "nan", "None", "NULL", "ok"], dtype="string")
    norm = FeatureEngineer.normalize_string_missing(s, fill_value=DEFAULT_MISSING_TOKEN)
    assert norm.tolist() == [
        DEFAULT_MISSING_TOKEN,
        DEFAULT_MISSING_TOKEN,
        DEFAULT_MISSING_TOKEN,
        DEFAULT_MISSING_TOKEN,
        DEFAULT_MISSING_TOKEN,
        "ok",
    ]
