import pandas as pd

from src.data.cohort_builder import CohortBuilder


def _mk_builder():
    patients = pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "gender": ["M", "F", "M", "F"],
        "dob": ["1800-01-01", "2263-01-01", "2100-01-01", "1970-06-15"],
        "dod": [None, None, None, None],
    })
    admissions = pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "hadm_id": [11, 22, 33, 44],
        "admittime": ["2101-01-01", "2140-01-01", "2099-01-01", "2020-01-01"],
        "dischtime": ["2101-01-03", "2140-01-03", "2099-01-03", "2020-01-03"],
        "admission_type": ["EMERGENCY", "EMERGENCY", "EMERGENCY", "EMERGENCY"],
    })
    diagnoses = pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "hadm_id": [11, 22, 33, 44],
        "icd9_code": ["25000", "25000", "25000", "25000"],
        "seq_num": [1, 1, 1, 1],
    })
    return CohortBuilder(
        patients=patients,
        admissions=admissions,
        diagnoses=diagnoses,
    )


def test_filter_by_age_handles_invalid_dates_without_overflow():
    builder = _mk_builder()
    subject_ids = {1, 2, 3, 4}
    filtered = builder._filter_by_age(subject_ids, min_age=18, max_age=80)
    assert filtered == {4}


def test_get_cohort_statistics_age_safe():
    builder = _mk_builder()
    stats = builder.get_cohort_statistics([1, 2, 3, 4])
    assert "age" in stats
    assert stats["age"]["mean"] >= 0
