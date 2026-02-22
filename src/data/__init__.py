"""
Data Pipeline Module for Healthcare RL

This module provides comprehensive data loading, preprocessing, and feature engineering
capabilities for clinical data, specifically designed for reinforcement learning applications.

Main Components:
- MIMICLoader: Load MIMIC-III clinical database tables
- DataPreprocessor: Clean, normalize, and preprocess clinical data
- FeatureEngineer: Extract and engineer features from raw data
- CohortBuilder: Define patient cohorts with inclusion/exclusion criteria
- SyntheticDataGenerator: Generate synthetic patient trajectories
- DataQualityValidator: Validate data quality and generate reports
- Utils: Utility functions for common data operations

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

from .mimic_loader import MIMICLoader
from .preprocessor import DataPreprocessor
from .feature_engineering import (
    FeatureEngineer,
    AgeBucketing,
    AgeBucket
)
from .cohort_builder import CohortBuilder
from .synthetic_generator import (
    SyntheticDataGenerator,
    PatientParameters,
    PopulationParameters,
    BergmanMinimalModel,
    AdherenceModel
)
from .data_validator import (
    DataQualityValidator,
    ValidationResult
)
from .utils import (
    load_icd9_mappings,
    lookup_icd9_description,
    calculate_age,
    calculate_age_at_events,
    create_sliding_windows,
    split_train_val_test,
    save_processed_data,
    load_processed_data,
    merge_patient_tables,
    downsample_data,
    compute_statistics,
    print_data_summary
)

__version__ = '1.0.0'

__all__ = [
    # Loaders
    'MIMICLoader',
    
    # Preprocessing
    'DataPreprocessor',
    
    # Feature Engineering
    'FeatureEngineer',
    'AgeBucketing',
    'AgeBucket',
    
    # Cohort Building
    'CohortBuilder',
    
    # Synthetic Data
    'SyntheticDataGenerator',
    'PatientParameters',
    'PopulationParameters',
    'BergmanMinimalModel',
    'AdherenceModel',
    
    # Validation
    'DataQualityValidator',
    'ValidationResult',
    
    # Utilities
    'load_icd9_mappings',
    'lookup_icd9_description',
    'calculate_age',
    'calculate_age_at_events',
    'create_sliding_windows',
    'split_train_val_test',
    'save_processed_data',
    'load_processed_data',
    'merge_patient_tables',
    'downsample_data',
    'compute_statistics',
    'print_data_summary',
]
