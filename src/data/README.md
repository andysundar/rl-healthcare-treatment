# Healthcare RL Data Pipeline

Comprehensive data processing pipeline for reinforcement learning in healthcare, designed for the M.Tech thesis project on "RL for Healthcare Treatment Recommendations."

## Author
**Anindya Bandopadhyay** (m23csa508@iitj.ac.in)  
IIT Jodhpur, January 2026

## Overview

This data pipeline provides production-ready modules for:
- Loading MIMIC-III clinical data efficiently
- Preprocessing with missing value handling, normalization, and outlier detection
- Feature engineering with categorical age bucketing and temporal features
- Cohort building with flexible inclusion/exclusion criteria
- Synthetic data generation using physiological models
- Comprehensive data quality validation

## Installation

```bash
# Required packages
pip install pandas numpy scikit-learn scipy

# Optional for advanced features
pip install pyarrow  # For parquet files
```

## Module Structure

```
src/data/
├── __init__.py                  # Package initialization
├── mimic_loader.py             # MIMIC-III data loading
├── preprocessor.py             # Data preprocessing
├── feature_engineering.py      # Feature extraction
├── cohort_builder.py          # Cohort definition
├── synthetic_generator.py     # Synthetic data generation
├── data_validator.py          # Data quality validation
└── utils.py                   # Utility functions
```

## Quick Start

### 1. Loading MIMIC-III Data

```python
from src.data import MIMICLoader

# Initialize loader
loader = MIMICLoader(data_dir='data/raw/mimic-iii')

# Load patients table
patients = loader.load_patients()
print(f"Loaded {len(patients):,} patients")

# Load admissions
admissions = loader.load_admissions()

# Load lab events for specific patients
patient_ids = patients['subject_id'].head(100).tolist()
labs = loader.load_lab_events(subject_ids=patient_ids)

# Load complete patient record
record = loader.get_patient_complete_record(subject_id=10)
```

### 2. Data Preprocessing

```python
from src.data import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Handle missing values
clean_data = preprocessor.clean_missing_values(
    df=labs,
    strategy='median'  # or 'mean', 'forward_fill', 'knn'
)

# Normalize vitals
normalized = preprocessor.normalize_vitals(
    df=clean_data,
    method='robust'  # or 'standard', 'minmax'
)

# Handle outliers
no_outliers = preprocessor.handle_outliers(
    df=normalized,
    method='clip',  # or 'remove', 'replace_nan'
    threshold=3.0
)

# Validate physiological ranges
validated, violations = preprocessor.validate_physiological_ranges(no_outliers)
```

### 3. Feature Engineering

```python
from src.data import FeatureEngineer

# Initialize engineer
engineer = FeatureEngineer()

# Extract demographics with categorical age bucketing
demographics = engineer.extract_demographics(
    patients=patients,
    admissions=admissions
)

# Age buckets: neonate, infant, child, adolescent, 
#              young_adult, adult, elderly, very_elderly

# Extract vitals sequence
vitals = engineer.extract_vitals_sequence(
    chartevents=chartevents,
    subject_ids=patient_ids
)

# Extract lab sequence
labs_seq = engineer.extract_lab_sequence(
    labevents=labs,
    subject_ids=patient_ids
)

# Create temporal features
temporal = engineer.create_temporal_features(
    df=vitals,
    time_column='charttime'
)

# Encode comorbidities from ICD-9 codes
comorbidities = engineer.encode_comorbidities(
    diagnoses=diagnoses
)
```

### 4. Cohort Building

```python
from src.data import CohortBuilder

# Initialize builder
builder = CohortBuilder(
    patients=patients,
    admissions=admissions,
    diagnoses=diagnoses
)

# Define diabetes cohort
diabetes_patients = builder.define_diabetes_cohort()

# Define multimorbidity cohort
multimorbid = builder.define_multimorbidity_cohort(
    conditions=['diabetes', 'hypertension', 'ckd'],
    min_conditions=2
)

# Apply inclusion criteria
filtered = builder.apply_inclusion_criteria(
    subject_ids=diabetes_patients,
    min_age=18,
    max_age=80,
    min_admissions=2
)

# Apply exclusion criteria
final_cohort = builder.apply_exclusion_criteria(
    subject_ids=filtered,
    exclude_pregnancy=True,
    exclude_pediatric=True,
    exclude_died_in_hospital=True
)

# Get statistics
stats = builder.get_cohort_statistics(final_cohort)
print(f"Final cohort: {len(final_cohort)} patients")

# Export cohort definition
builder.export_cohort_definition(
    subject_ids=final_cohort,
    filepath='cohort_definition.txt'
)
```

### 5. Synthetic Data Generation

```python
from src.data import SyntheticDataGenerator, PopulationParameters

# Initialize generator
generator = SyntheticDataGenerator(random_seed=42)

# Generate population
population_params = PopulationParameters(
    n_patients=1000,
    age_mean=55.0,
    age_std=15.0
)

patients = generator.generate_diabetes_population(
    n_patients=1000,
    population_params=population_params
)

# Simulate patient trajectory
trajectory = generator.simulate_patient_trajectory(
    patient=patients[0],
    time_horizon_days=365,
    treatment_policy={'insulin_dose': 0.1, 'reminder_frequency': 3}
)

# Generate complete dataset
patients, trajectories = generator.generate_dataset(
    n_patients=100,
    time_horizon_days=365
)
```

### 6. Data Quality Validation

```python
from src.data import DataQualityValidator

# Initialize validator
validator = DataQualityValidator()

# Validate patient record
is_valid, issues = validator.validate_patient_record(patient_data)
if not is_valid:
    print(f"Found issues: {issues}")

# Generate quality report
report = validator.generate_quality_report(
    dataset=labs,
    dataset_name="Lab Events"
)

# Print report summary
validator.print_report_summary(report)

# Save report
validator.save_report(report, 'quality_report.json')
```

### 7. Utility Functions

```python
from src.data import (
    calculate_age,
    create_sliding_windows,
    split_train_val_test,
    save_processed_data,
    load_processed_data,
    lookup_icd9_description
)

# Calculate age
age = calculate_age(
    dob=pd.Timestamp('1970-01-01'),
    reference_date=pd.Timestamp('2020-01-01')
)

# Create sliding windows
windows = create_sliding_windows(
    sequence=vitals,
    window_size=24,
    stride=12
)

# Split data
train, val, test = split_train_val_test(
    data=features,
    ratios=(0.7, 0.15, 0.15),
    random_state=42
)

# Save processed data
save_processed_data(train, 'data/processed/train.parquet')

# Load processed data
train_loaded = load_processed_data('data/processed/train.parquet')

# ICD-9 lookup
description = lookup_icd9_description('250')
print(description)  # "Diabetes mellitus"
```

## Complete Pipeline Example

```python
from src.data import *

# 1. Load MIMIC-III data
loader = MIMICLoader('data/raw/mimic-iii')
patients = loader.load_patients()
admissions = loader.load_admissions()
diagnoses = loader.load_diagnoses_icd()

# 2. Build cohort
builder = CohortBuilder(patients, admissions, diagnoses)
diabetes_cohort = builder.define_diabetes_cohort()
filtered_cohort = builder.apply_inclusion_criteria(
    diabetes_cohort,
    min_age=18,
    min_admissions=2
)

# 3. Load data for cohort
labs = loader.load_lab_events(subject_ids=filtered_cohort)

# 4. Preprocess
preprocessor = DataPreprocessor()
clean_labs = preprocessor.clean_missing_values(labs, strategy='median')
normalized_labs = preprocessor.normalize_labs(clean_labs, method='robust')

# 5. Feature engineering
engineer = FeatureEngineer()
demographics = engineer.extract_demographics(patients, admissions)
lab_sequence = engineer.extract_lab_sequence(normalized_labs)
temporal_features = engineer.create_temporal_features(lab_sequence)

# 6. Validate quality
validator = DataQualityValidator()
report = validator.generate_quality_report(temporal_features)
validator.print_report_summary(report)

# 7. Split and save
train, val, test = split_train_val_test(temporal_features)
save_processed_data(train, 'data/processed/train.parquet')
save_processed_data(val, 'data/processed/val.parquet')
save_processed_data(test, 'data/processed/test.parquet')
```

## Key Features

### AgeBucketing Class
Single source of truth for age categorization, avoiding datetime overflow errors:
- **Neonate**: 0-28 days
- **Infant**: 29 days - 1 year
- **Child**: 1-12 years
- **Adolescent**: 13-18 years
- **Young Adult**: 19-35 years
- **Adult**: 36-60 years
- **Elderly**: 61-80 years
- **Very Elderly**: 80+ years

Handles MIMIC-III privacy-shifted birthdates automatically.

### Data Quality Thresholds
Default thresholds (configurable):
- Missing values: ≤30%
- Outliers: ≤5%
- Temporal consistency: ≥95%

### Physiological Ranges
Built-in validation for:
- **Vitals**: HR (40-200), SBP (70-250), DBP (30-150), Temp (35-42°C)
- **Labs**: Glucose (20-800 mg/dL), Sodium (120-160 mmol/L), etc.

## Testing

```python
# Run module examples
python src/data/mimic_loader.py
python src/data/preprocessor.py
python src/data/feature_engineering.py
python src/data/cohort_builder.py
python src/data/synthetic_generator.py
python src/data/data_validator.py
python src/data/utils.py
```

## Best Practices

1. **Memory Management**: Use chunked reading for large files (CHARTEVENTS, LABEVENTS)
2. **Missing Values**: Use 'median' or 'robust' methods for clinical data
3. **Normalization**: Use 'robust' scaling due to outliers in clinical data
4. **Age Handling**: Always use AgeBucketing class for MIMIC-III data
5. **Validation**: Run quality validation before training RL models
6. **Documentation**: Use cohort builder's export feature to document selection

## Performance Tips

```python
# 1. Downsample for prototyping
from src.data import downsample_data
sample_data = downsample_data(full_data, fraction=0.1)

# 2. Use caching in loader
loader = MIMICLoader('data/raw/mimic-iii', use_cache=True)

# 3. Filter early
labs = loader.load_lab_events(subject_ids=cohort)  # Filter at load time

# 4. Save intermediate results
save_processed_data(preprocessed, 'cache/preprocessed.parquet')
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check MIMIC-III data directory path
2. **Memory Error**: Use chunked reading or downsample
3. **Age Overflow**: Ensure using AgeBucketing class
4. **Missing Values**: Check validation report for data quality

## Citation

If you use this pipeline in your research, please cite:

```
@mastersthesis{bandopadhyay2026rl,
  author = {Anindya Bandopadhyay},
  title = {Reinforcement Learning for Healthcare Treatment Recommendations},
  school = {Indian Institute of Technology Jodhpur},
  year = {2026},
  type = {M.Tech Thesis}
}
```

## License

This code is part of an academic thesis project. For reuse, please contact the author.

## Contact

**Anindya Bandopadhyay**  
M.Tech Student, Department of Computer Science and Engineering  
IIT Jodhpur  
Email: m23csa508@iitj.ac.in

---

**Last Updated**: January 27, 2026
