"""
Data Quality Validation Module for Healthcare RL

This module provides comprehensive data quality checks for clinical datasets:
- Missing value analysis and reporting
- Temporal consistency validation
- Physiological range validation
- Outlier detection and flagging
- Data completeness scoring
- Quality report generation

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation check."""
    is_valid: bool
    issues: Dict[str, any]
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    
    def __repr__(self):
        return f"ValidationResult(valid={self.is_valid}, severity={self.severity}, issues={len(self.issues)})"


class DataQualityValidator:
    """
    Comprehensive data quality validation for clinical datasets.
    
    Performs multi-level validation:
    - Level 1: Structural (columns, dtypes, nulls)
    - Level 2: Semantic (value ranges, consistency)
    - Level 3: Statistical (distributions, outliers)
    - Level 4: Clinical (physiological plausibility)
    
    Attributes:
        thresholds: Quality thresholds for validation
        validation_results: History of validation checks
        
    Example:
        >>> validator = DataQualityValidator()
        >>> is_valid, issues = validator.validate_patient_record(patient_data)
        >>> report = validator.generate_quality_report(dataset)
    """
    
    # Physiological ranges (same as in preprocessor)
    VITAL_RANGES = {
        'heart_rate': (40, 200),
        'sbp': (70, 250),
        'dbp': (30, 150),
        'temperature': (35.0, 42.0),
        'respiratory_rate': (8, 40),
        'spo2': (70, 100),
    }
    
    LAB_RANGES = {
        'glucose': (20, 800),
        'sodium': (120, 160),
        'potassium': (2.0, 8.0),
        'chloride': (80, 120),
        'bicarbonate': (10, 45),
        'bun': (5, 150),
        'creatinine': (0.3, 15.0),
        'hematocrit': (15, 60),
        'wbc': (0.1, 50),
        'hemoglobin': (5, 20),
        'platelet': (10, 1000),
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data quality validator.
        
        Args:
            config: Configuration with quality thresholds
                   Example: {
                       'missing_threshold': 0.3,
                       'outlier_threshold': 0.05,
                       'temporal_consistency': 0.95
                   }
        """
        default_config = {
            'missing_threshold': 0.3,  # Max 30% missing values
            'outlier_threshold': 0.05,  # Max 5% outliers
            'temporal_consistency': 0.95,  # Min 95% temporal consistency
            'min_records_per_patient': 1,
            'max_age': 120,
            'min_age': 0
        }
        
        self.config = {**default_config, **(config or {})}
        self.validation_results: List[ValidationResult] = []
        
        logger.info("Initialized DataQualityValidator")
    
    def validate_patient_record(
        self,
        patient_data: pd.DataFrame
    ) -> Tuple[bool, Dict]:
        """
        Validate individual patient record for quality issues.
        
        Args:
            patient_data: DataFrame containing patient data
            
        Returns:
            Tuple of (is_valid, issues_dict)
            
        Example:
            >>> validator = DataQualityValidator()
            >>> is_valid, issues = validator.validate_patient_record(df)
            >>> if not is_valid:
            ...     print(f"Found {len(issues)} issues")
        """
        issues = {}
        
        # Check 1: Completeness
        missing_check = self._check_completeness(patient_data)
        if not missing_check.is_valid:
            issues['completeness'] = missing_check.issues
        
        # Check 2: Temporal consistency
        if 'charttime' in patient_data.columns or 'admittime' in patient_data.columns:
            temporal_check = self._check_temporal_consistency(patient_data)
            if not temporal_check.is_valid:
                issues['temporal'] = temporal_check.issues
        
        # Check 3: Value ranges
        range_check = self._check_value_ranges(patient_data)
        if not range_check.is_valid:
            issues['value_ranges'] = range_check.issues
        
        # Check 4: Duplicates
        duplicate_check = self._check_duplicates(patient_data)
        if not duplicate_check.is_valid:
            issues['duplicates'] = duplicate_check.issues
        
        # Overall validity
        is_valid = len(issues) == 0
        
        # Log result
        result = ValidationResult(
            is_valid=is_valid,
            issues=issues,
            severity='error' if not is_valid else 'info',
            message=f"Validation {'passed' if is_valid else 'failed'} with {len(issues)} issue categories"
        )
        self.validation_results.append(result)
        
        return is_valid, issues
    
    def generate_quality_report(
        self,
        dataset: pd.DataFrame,
        dataset_name: str = "dataset"
    ) -> Dict:
        """
        Generate comprehensive quality report for entire dataset.
        
        Args:
            dataset: Complete dataset to validate
            dataset_name: Name for the dataset in report
            
        Returns:
            Dictionary containing quality metrics and issues
        """
        logger.info(f"Generating quality report for {dataset_name}")
        
        report = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'n_records': len(dataset),
            'n_columns': len(dataset.columns),
            'issues': {},
            'scores': {}
        }
        
        # Overall missing value analysis
        missing_analysis = self._analyze_missing_values(dataset)
        report['missing_values'] = missing_analysis
        report['issues']['missing_values'] = [
            col for col, rate in missing_analysis['missing_rates'].items()
            if rate > self.config['missing_threshold']
        ]
        
        # Completeness score
        completeness = 1 - dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))
        report['scores']['completeness'] = completeness
        
        # Outlier analysis
        outlier_analysis = self._analyze_outliers(dataset)
        report['outliers'] = outlier_analysis
        report['scores']['outlier_rate'] = outlier_analysis['overall_outlier_rate']
        
        # Temporal consistency
        if 'charttime' in dataset.columns or 'admittime' in dataset.columns:
            temporal_analysis = self._analyze_temporal_consistency(dataset)
            report['temporal_consistency'] = temporal_analysis
            report['scores']['temporal_consistency'] = temporal_analysis['consistency_rate']
        
        # Value range violations
        range_analysis = self._analyze_value_ranges(dataset)
        report['value_ranges'] = range_analysis
        report['scores']['range_compliance'] = range_analysis['compliance_rate']
        
        # Duplicate analysis
        duplicate_analysis = self._analyze_duplicates(dataset)
        report['duplicates'] = duplicate_analysis
        
        # Overall quality score (weighted average)
        weights = {
            'completeness': 0.3,
            'temporal_consistency': 0.2,
            'range_compliance': 0.3,
            'outlier_rate': 0.2
        }
        
        quality_score = 0
        for metric, weight in weights.items():
            if metric in report['scores']:
                if metric == 'outlier_rate':
                    # Lower outlier rate is better
                    score_component = 1 - min(1, report['scores'][metric] / 0.1)
                else:
                    score_component = report['scores'][metric]
                quality_score += weight * score_component
        
        report['overall_quality_score'] = quality_score
        
        # Severity classification
        if quality_score >= 0.9:
            report['quality_level'] = 'excellent'
        elif quality_score >= 0.75:
            report['quality_level'] = 'good'
        elif quality_score >= 0.6:
            report['quality_level'] = 'acceptable'
        elif quality_score >= 0.4:
            report['quality_level'] = 'poor'
        else:
            report['quality_level'] = 'critical'
        
        logger.info(f"Quality report generated: {report['quality_level']} "
                   f"(score: {quality_score:.2f})")
        
        return report
    
    def _check_completeness(self, df: pd.DataFrame) -> ValidationResult:
        """Check data completeness (missing values)."""
        missing_rates = df.isnull().sum() / len(df)
        problematic_cols = missing_rates[
            missing_rates > self.config['missing_threshold']
        ].to_dict()
        
        is_valid = len(problematic_cols) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=problematic_cols,
            severity='warning' if not is_valid else 'info',
            message=f"Found {len(problematic_cols)} columns with excessive missing values"
        )
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Check temporal consistency of timestamps."""
        issues = {}
        
        # Find time columns
        time_cols = [col for col in df.columns 
                    if 'time' in col.lower() or 'date' in col.lower()]
        
        for col in time_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    issues[col] = "Cannot parse as datetime"
                    continue
            
            # Check for future dates
            future_dates = df[col] > pd.Timestamp.now()
            if future_dates.sum() > 0:
                issues[f"{col}_future"] = f"{future_dates.sum()} future dates"
            
            # Check for very old dates (likely errors)
            very_old = df[col] < pd.Timestamp('1900-01-01')
            if very_old.sum() > 0:
                issues[f"{col}_ancient"] = f"{very_old.sum()} dates before 1900"
            
            # Check chronological order (if patient/admission grouping available)
            if 'subject_id' in df.columns:
                non_chronological = 0
                for subject_id in df['subject_id'].unique():
                    subject_data = df[df['subject_id'] == subject_id][col].dropna()
                    if len(subject_data) > 1:
                        if not subject_data.is_monotonic_increasing:
                            non_chronological += 1
                
                if non_chronological > 0:
                    issues[f"{col}_order"] = f"{non_chronological} patients with non-chronological {col}"
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            severity='warning' if not is_valid else 'info',
            message=f"Temporal consistency check: {len(issues)} issues"
        )
    
    def _check_value_ranges(self, df: pd.DataFrame) -> ValidationResult:
        """Check if values are within physiological ranges."""
        issues = {}
        
        # Check vitals
        for vital, (min_val, max_val) in self.VITAL_RANGES.items():
            matching_cols = [col for col in df.columns if vital in col.lower()]
            
            for col in matching_cols:
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                n_violations = out_of_range.sum()
                
                if n_violations > 0:
                    violation_rate = n_violations / len(df)
                    if violation_rate > 0.01:  # More than 1% violations
                        issues[col] = {
                            'n_violations': n_violations,
                            'rate': violation_rate,
                            'expected_range': (min_val, max_val),
                            'actual_range': (df[col].min(), df[col].max())
                        }
        
        # Check labs
        for lab, (min_val, max_val) in self.LAB_RANGES.items():
            matching_cols = [col for col in df.columns if lab in col.lower()]
            
            for col in matching_cols:
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                n_violations = out_of_range.sum()
                
                if n_violations > 0:
                    violation_rate = n_violations / len(df)
                    if violation_rate > 0.01:
                        issues[col] = {
                            'n_violations': n_violations,
                            'rate': violation_rate,
                            'expected_range': (min_val, max_val),
                            'actual_range': (df[col].min(), df[col].max())
                        }
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            severity='warning' if not is_valid else 'info',
            message=f"Value range check: {len(issues)} columns with violations"
        )
    
    def _check_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        """Check for duplicate records."""
        issues = {}
        
        # Check for exact duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            issues['exact_duplicates'] = n_duplicates
        
        # Check for duplicates in key columns (if available)
        key_columns = ['subject_id', 'hadm_id', 'charttime']
        available_keys = [col for col in key_columns if col in df.columns]
        
        if available_keys:
            n_key_duplicates = df.duplicated(subset=available_keys).sum()
            if n_key_duplicates > 0:
                issues['key_duplicates'] = {
                    'columns': available_keys,
                    'count': n_key_duplicates
                }
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            severity='warning' if not is_valid else 'info',
            message=f"Duplicate check: {len(issues)} types of duplicates found"
        )
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Detailed missing value analysis."""
        missing_counts = df.isnull().sum()
        missing_rates = missing_counts / len(df)
        
        return {
            'total_missing': missing_counts.sum(),
            'missing_rate_overall': missing_counts.sum() / (len(df) * len(df.columns)),
            'missing_rates': missing_rates.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'completely_missing_columns': missing_rates[missing_rates == 1.0].index.tolist()
        }
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict:
        """Detailed outlier analysis using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        total_outliers = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                outlier_info[col] = {
                    'count': n_outliers,
                    'rate': n_outliers / len(df),
                    'bounds': (lower_bound, upper_bound),
                    'outlier_values': df[outliers][col].describe().to_dict()
                }
                total_outliers += n_outliers
        
        return {
            'columns_with_outliers': outlier_info,
            'total_outliers': total_outliers,
            'overall_outlier_rate': total_outliers / (len(df) * len(numeric_cols)) if len(numeric_cols) > 0 else 0
        }
    
    def _analyze_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        """Detailed temporal consistency analysis."""
        time_cols = [col for col in df.columns 
                    if 'time' in col.lower() or 'date' in col.lower()]
        
        consistency_info = {}
        total_inconsistencies = 0
        
        for col in time_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            
            # Check chronological order per patient
            if 'subject_id' in df.columns:
                inconsistent_patients = 0
                total_patients = df['subject_id'].nunique()
                
                for subject_id in df['subject_id'].unique():
                    subject_data = df[df['subject_id'] == subject_id][col].dropna()
                    if len(subject_data) > 1:
                        if not subject_data.is_monotonic_increasing:
                            inconsistent_patients += 1
                
                consistency_info[col] = {
                    'inconsistent_patients': inconsistent_patients,
                    'total_patients': total_patients,
                    'consistency_rate': 1 - (inconsistent_patients / total_patients)
                }
                total_inconsistencies += inconsistent_patients
        
        overall_consistency = 1 - (total_inconsistencies / (df['subject_id'].nunique() * len(time_cols))) if 'subject_id' in df.columns and len(time_cols) > 0 else 1.0
        
        return {
            'columns': consistency_info,
            'consistency_rate': overall_consistency
        }
    
    def _analyze_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Detailed value range analysis."""
        all_ranges = {**self.VITAL_RANGES, **self.LAB_RANGES}
        
        range_info = {}
        total_violations = 0
        total_values = 0
        
        for measurement, (min_val, max_val) in all_ranges.items():
            matching_cols = [col for col in df.columns if measurement in col.lower()]
            
            for col in matching_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                n_violations = out_of_range.sum()
                n_values = df[col].notna().sum()
                
                if n_values > 0:
                    range_info[col] = {
                        'violations': n_violations,
                        'total_values': n_values,
                        'violation_rate': n_violations / n_values,
                        'expected_range': (min_val, max_val),
                        'actual_range': (df[col].min(), df[col].max())
                    }
                    total_violations += n_violations
                    total_values += n_values
        
        compliance_rate = 1 - (total_violations / total_values) if total_values > 0 else 1.0
        
        return {
            'columns': range_info,
            'total_violations': total_violations,
            'total_values': total_values,
            'compliance_rate': compliance_rate
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict:
        """Detailed duplicate analysis."""
        return {
            'exact_duplicates': df.duplicated().sum(),
            'exact_duplicate_rate': df.duplicated().sum() / len(df) if len(df) > 0 else 0,
            'unique_rows': len(df) - df.duplicated().sum()
        }
    
    def save_report(self, report: Dict, filepath: str) -> None:
        """Save quality report to file."""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Quality report saved to {filepath}")
    
    def print_report_summary(self, report: Dict) -> None:
        """Print human-readable report summary."""
        print("=" * 80)
        print(f"DATA QUALITY REPORT: {report['dataset_name']}")
        print("=" * 80)
        print(f"\nTimestamp: {report['timestamp']}")
        print(f"Records: {report['n_records']:,}")
        print(f"Columns: {report['n_columns']}")
        print(f"\nOverall Quality Score: {report['overall_quality_score']:.2%}")
        print(f"Quality Level: {report['quality_level'].upper()}")
        
        print("\n" + "-" * 80)
        print("SCORES")
        print("-" * 80)
        for metric, score in report['scores'].items():
            print(f"{metric:.<40} {score:.2%}")
        
        if report['issues']:
            print("\n" + "-" * 80)
            print("ISSUES")
            print("-" * 80)
            for category, issues in report['issues'].items():
                if issues:
                    print(f"\n{category}:")
                    print(f"  {issues}")


if __name__ == '__main__':
    # Example usage
    
    print("=== Data Quality Validation Example ===\n")
    
    # Create sample data with quality issues
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'subject_id': np.repeat(range(100), 10),
        'charttime': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        'heart_rate': np.random.normal(75, 15, n_samples),
        'sbp': np.random.normal(120, 20, n_samples),
        'glucose': np.random.normal(100, 30, n_samples),
        'sodium': np.random.normal(140, 3, n_samples),
    })
    
    # Introduce quality issues
    # Missing values
    sample_data.loc[np.random.choice(n_samples, 100), 'heart_rate'] = np.nan
    
    # Outliers
    sample_data.loc[np.random.choice(n_samples, 20), 'heart_rate'] = 300
    sample_data.loc[np.random.choice(n_samples, 10), 'glucose'] = 1000
    
    # Out of range values
    sample_data.loc[np.random.choice(n_samples, 15), 'sodium'] = 180
    
    # Duplicates
    sample_data = pd.concat([sample_data, sample_data.head(50)], ignore_index=True)
    
    # Initialize validator
    validator = DataQualityValidator()
    
    # Validate single patient
    print("Validating single patient record...")
    patient_data = sample_data[sample_data['subject_id'] == 0]
    is_valid, issues = validator.validate_patient_record(patient_data)
    print(f"Valid: {is_valid}")
    if not is_valid:
        print(f"Issues: {issues}")
    
    # Generate full quality report
    print("\n\nGenerating quality report for full dataset...")
    report = validator.generate_quality_report(sample_data, dataset_name="Sample Clinical Data")
    
    # Print summary
    validator.print_report_summary(report)
    
    # Save report
    print("\n\nSaving report to file...")
    validator.save_report(report, 'quality_report.json')
    print("Report saved!")
