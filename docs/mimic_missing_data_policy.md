# MIMIC Missing-Data Policy

This project uses a leakage-safe missing-data policy for MIMIC trajectory preparation.

## Why

MIMIC ICU measurements are irregular and missingness is often informative (not MCAR).
To keep preprocessing clinically sensible and model-safe, we use conservative imputation:

1. Within-patient sample-and-hold (forward fill) for time-varying measurements.
2. Training-split-only statistical fill (median by default) for any remaining gaps.
3. Explicit missingness indicators (`*_missing`) to preserve missingness signal.
4. Robust categorical normalization with an explicit `UNKNOWN` token.

## Leakage Rule

- The policy is fit **only on the train split**.
- Validation and test are transformed with train-derived parameters only.
- No medians/vocabularies are learned from val/test.

## Column Behavior

- Time-varying vitals/labs: per-episode forward fill, optional max-hold limit, then train median.
- Static numeric features: train median fill.
- Categorical fields: normalize null-like strings and unknown/unseen values to `UNKNOWN`.
- Optional `time_since_last_*` features can be enabled for time-varying columns.

## Robustness Notes

- Pandas categorical columns are handled safely by adding `UNKNOWN` before assignment when needed.
- Batch checkpoints are saved with categorical columns converted to stable string dtype to avoid category-mismatch failures during concat/reload.
- A `missing_data_report.csv` artifact is emitted for traceability (strategy, missing rate, fill value, mask usage).
