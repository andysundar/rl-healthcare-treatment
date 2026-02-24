"""
Complete Pipeline Example

This script demonstrates the complete data processing pipeline from
raw MIMIC-III data to processed features ready for RL training.

Steps:
1. Load MIMIC-III data
2. Build patient cohort
3. Extract and preprocess features
4. Validate data quality
5. Split into train/val/test sets
6. Save processed data

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is on sys.path when run directly
_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from data import (
    MIMICLoader,
    DataPreprocessor,
    FeatureEngineer,
    CohortBuilder,
    DataQualityValidator,
    split_train_val_test,
    save_processed_data,
    print_data_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_pipeline(
    mimic_data_dir: str = 'data/raw/mimic-iii',
    output_dir: str = 'data/processed',
    use_sample: bool = False,
    sample_size: int = 100
):
    """
    Run complete data processing pipeline.
    
    Args:
        mimic_data_dir: Path to MIMIC-III data directory
        output_dir: Path to save processed data
        use_sample: If True, use sample of patients for quick testing
        sample_size: Number of patients in sample
    """
    
    print("=" * 80)
    print("HEALTHCARE RL DATA PROCESSING PIPELINE")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load MIMIC-III Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Loading MIMIC-III Data")
    print("=" * 80)
    
    try:
        loader = MIMICLoader(data_dir=mimic_data_dir, use_cache=True)
        
        # Load core tables
        print("\nLoading PATIENTS table...")
        patients = loader.load_patients()
        print(f"✓ Loaded {len(patients):,} patients")
        
        print("\nLoading ADMISSIONS table...")
        admissions = loader.load_admissions()
        print(f"✓ Loaded {len(admissions):,} admissions")
        
        print("\nLoading DIAGNOSES_ICD table...")
        diagnoses = loader.load_diagnoses_icd()
        print(f"✓ Loaded {len(diagnoses):,} diagnoses")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure MIMIC-III data is available at the specified path.")
        print("For testing, you can use synthetic data generation instead.")
        return
    
    # ========================================================================
    # STEP 2: Build Patient Cohort
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Building Patient Cohort")
    print("=" * 80)
    
    builder = CohortBuilder(
        patients=patients,
        admissions=admissions,
        diagnoses=diagnoses
    )
    
    # Define diabetes cohort
    print("\nDefining diabetes cohort...")
    diabetes_patients = builder.define_diabetes_cohort()
    print(f"✓ Found {len(diabetes_patients):,} diabetes patients")
    
    # Apply inclusion criteria
    print("\nApplying inclusion criteria...")
    filtered_patients = builder.apply_inclusion_criteria(
        subject_ids=diabetes_patients,
        min_age=18,
        max_age=80,
        min_admissions=2
    )
    print(f"✓ {len(filtered_patients):,} patients after inclusion criteria")
    
    # Apply exclusion criteria
    print("\nApplying exclusion criteria...")
    final_cohort = builder.apply_exclusion_criteria(
        subject_ids=filtered_patients,
        exclude_pregnancy=True,
        exclude_pediatric=True,
        exclude_died_in_hospital=False
    )
    print(f"✓ Final cohort: {len(final_cohort):,} patients")
    
    # Get cohort statistics
    print("\nCohort statistics:")
    stats = builder.get_cohort_statistics(final_cohort)
    print(f"  Mean age: {stats['age']['mean']:.1f} years")
    print(f"  Gender: {stats['demographics']['n_male']} M, {stats['demographics']['n_female']} F")
    print(f"  Mean admissions: {stats['admissions']['mean_per_patient']:.1f}")
    
    # Export cohort definition
    cohort_def_path = output_path / 'cohort_definition.txt'
    builder.export_cohort_definition(final_cohort, cohort_def_path)
    print(f"\n✓ Cohort definition saved to {cohort_def_path}")
    
    # Use sample for quick testing
    if use_sample:
        import random
        random.seed(42)
        final_cohort = random.sample(final_cohort, min(sample_size, len(final_cohort)))
        print(f"\n⚠ Using sample of {len(final_cohort)} patients for testing")
    
    # ========================================================================
    # STEP 3: Load Patient Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Loading Patient Data")
    print("=" * 80)
    
    print("\nLoading lab events for cohort...")
    labs = loader.load_lab_events(subject_ids=final_cohort)
    print(f"✓ Loaded {len(labs):,} lab events")
    
    print("\nLoading prescriptions for cohort...")
    prescriptions = loader.load_prescriptions(subject_ids=final_cohort)
    print(f"✓ Loaded {len(prescriptions):,} prescriptions")
    
    # ========================================================================
    # STEP 4: Feature Engineering
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Feature Engineering")
    print("=" * 80)
    
    engineer = FeatureEngineer()
    
    # Extract demographics
    print("\nExtracting demographics...")
    demographics = engineer.extract_demographics(
        patients=patients,
        admissions=admissions
    )
    demographics = demographics[demographics['subject_id'].isin(final_cohort)]
    print(f"✓ Extracted demographics for {len(demographics):,} patients")
    print(f"  Features: {', '.join(demographics.columns.tolist()[:10])}...")
    
    # Extract lab sequence
    print("\nExtracting lab sequences...")
    lab_sequence = engineer.extract_lab_sequence(
        labevents=labs,
        subject_ids=final_cohort
    )
    print(f"✓ Extracted {len(lab_sequence):,} lab measurements")
    
    # Create temporal features
    if len(lab_sequence) > 0:
        print("\nCreating temporal features...")
        temporal_features = engineer.create_temporal_features(
            df=lab_sequence,
            time_column='charttime'
        )
        print(f"✓ Created temporal features")
        print(f"  Total features: {len(temporal_features.columns)}")
    else:
        print("\n⚠ No lab data available for temporal features")
        temporal_features = lab_sequence
    
    # Extract medication history
    print("\nExtracting medication history...")
    medication_features = engineer.extract_medication_history(
        prescriptions=prescriptions,
        encoding='binary'
    )
    print(f"✓ Encoded {len(medication_features.columns)-1} unique medications")
    
    # Encode comorbidities
    print("\nEncoding comorbidities...")
    comorbidities = engineer.encode_comorbidities(
        diagnoses=diagnoses[diagnoses['subject_id'].isin(final_cohort)]
    )
    print(f"✓ Encoded comorbidities for {len(comorbidities):,} patients")
    
    # ========================================================================
    # STEP 5: Data Preprocessing
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Data Preprocessing")
    print("=" * 80)
    
    preprocessor = DataPreprocessor()
    
    if len(temporal_features) > 0:
        # Handle missing values
        print("\nHandling missing values...")
        clean_data = preprocessor.clean_missing_values(
            df=temporal_features,
            strategy='median'
        )
        
        # Handle outliers
        print("\nHandling outliers...")
        clean_data = preprocessor.handle_outliers(
            df=clean_data,
            method='clip',
            threshold=3.0
        )
        
        # Validate physiological ranges
        print("\nValidating physiological ranges...")
        clean_data, violations = preprocessor.validate_physiological_ranges(clean_data)
        if violations:
            print(f"  Fixed {sum(violations.values())} range violations")
        
        # Normalize features
        print("\nNormalizing features...")
        normalized_data = preprocessor.normalize_labs(
            df=clean_data,
            method='robust'
        )
        
        print(f"✓ Preprocessing complete")
        processed_features = normalized_data
    else:
        processed_features = temporal_features
    
    # ========================================================================
    # STEP 6: Data Quality Validation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Data Quality Validation")
    print("=" * 80)
    
    validator = DataQualityValidator()
    
    if len(processed_features) > 0:
        print("\nGenerating quality report...")
        quality_report = validator.generate_quality_report(
            dataset=processed_features,
            dataset_name="Processed Features"
        )
        
        print(f"\n✓ Quality Score: {quality_report['overall_quality_score']:.2%}")
        print(f"  Quality Level: {quality_report['quality_level'].upper()}")
        print(f"  Completeness: {quality_report['scores']['completeness']:.2%}")
        
        # Save quality report
        report_path = output_path / 'quality_report.json'
        validator.save_report(quality_report, report_path)
        print(f"\n✓ Quality report saved to {report_path}")
    else:
        print("\n⚠ No data to validate")
    
    # ========================================================================
    # STEP 7: Split and Save Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Splitting and Saving Data")
    print("=" * 80)
    
    if len(processed_features) > 0:
        # Split into train/val/test
        print("\nSplitting data into train/val/test...")
        train, val, test = split_train_val_test(
            data=processed_features,
            ratios=(0.7, 0.15, 0.15),
            random_state=42
        )
        print(f"✓ Train: {len(train):,} samples")
        print(f"  Val:   {len(val):,} samples")
        print(f"  Test:  {len(test):,} samples")
        
        # Save splits
        print("\nSaving processed data...")
        save_processed_data(train, output_path / 'train_features.parquet')
        save_processed_data(val, output_path / 'val_features.parquet')
        save_processed_data(test, output_path / 'test_features.parquet')
        print(f"✓ Saved to {output_path}/")
        
        # Save auxiliary data
        print("\nSaving auxiliary data...")
        save_processed_data(demographics, output_path / 'demographics.parquet')
        save_processed_data(medication_features, output_path / 'medications.parquet')
        save_processed_data(comorbidities, output_path / 'comorbidities.parquet')
        print(f"✓ Auxiliary data saved")
    else:
        print("\n⚠ No data to save")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    
    print("\nOutput Files:")
    print(f"  📁 {output_path}/")
    print(f"    ├── train_features.parquet ({len(train):,} samples)")
    print(f"    ├── val_features.parquet ({len(val):,} samples)")
    print(f"    ├── test_features.parquet ({len(test):,} samples)")
    print(f"    ├── demographics.parquet ({len(demographics):,} patients)")
    print(f"    ├── medications.parquet ({len(medication_features):,} patients)")
    print(f"    ├── comorbidities.parquet ({len(comorbidities):,} patients)")
    print(f"    ├── cohort_definition.txt")
    print(f"    └── quality_report.json")
    
    print("\nNext Steps:")
    print("  1. Load processed data for RL training")
    print("  2. Review quality report for data issues")
    print("  3. Check cohort definition for selection criteria")
    print("  4. Train RL policy on processed features")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Healthcare RL Data Processing Pipeline'
    )
    parser.add_argument(
        '--mimic-dir',
        type=str,
        default='data/raw/mimic-iii',
        help='Path to MIMIC-III data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Path to save processed data'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample of patients for quick testing'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of patients in sample (if --sample is used)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    run_complete_pipeline(
        mimic_data_dir=args.mimic_dir,
        output_dir=args.output_dir,
        use_sample=args.sample,
        sample_size=args.sample_size
    )
