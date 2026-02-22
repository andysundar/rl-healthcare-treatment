"""
Utility Functions for Healthcare RL Data Pipeline

This module provides common utility functions for:
- ICD-9 code mappings and lookups
- Age calculations with privacy shift handling
- Sliding window generation for time-series
- Train/validation/test splitting
- Data persistence (save/load processed data)
- Helper functions for data manipulation

Author: Anindya Bandopadhyay
Date: January 2026
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ICD-9 Code Mappings
# Simplified mapping - in practice, load from comprehensive database
ICD9_DESCRIPTIONS = {
    # Diabetes
    '250': 'Diabetes mellitus',
    '25000': 'Diabetes mellitus without complication, type II or unspecified',
    '25001': 'Diabetes mellitus without complication, type I',
    '25002': 'Diabetes mellitus without complication, uncontrolled',
    
    # Hypertension
    '401': 'Essential hypertension',
    '40100': 'Essential hypertension, malignant',
    '40101': 'Essential hypertension, benign',
    '40190': 'Essential hypertension, unspecified',
    
    # Heart disease
    '410': 'Acute myocardial infarction',
    '41000': 'AMI anterolateral wall, episode of care unspecified',
    '41001': 'AMI anterolateral wall, initial episode of care',
    '428': 'Heart failure',
    '42800': 'Congestive heart failure, unspecified',
    
    # Kidney disease
    '585': 'Chronic kidney disease',
    '58581': 'Chronic kidney disease, stage I',
    '58582': 'Chronic kidney disease, stage II',
    '58583': 'Chronic kidney disease, stage III',
    '58584': 'Chronic kidney disease, stage IV',
    '58585': 'Chronic kidney disease, stage V',
    '58589': 'Chronic kidney disease, unspecified stage',
    
    # COPD
    '491': 'Chronic bronchitis',
    '492': 'Emphysema',
    '496': 'Chronic airway obstruction, not elsewhere classified',
    
    # Stroke
    '430': 'Subarachnoid hemorrhage',
    '431': 'Intracerebral hemorrhage',
    '434': 'Occlusion of cerebral arteries',
    '436': 'Acute, but ill-defined, cerebrovascular disease',
}


def load_icd9_mappings(
    filepath: Optional[Union[str, Path]] = None
) -> Dict[str, str]:
    """
    Load ICD-9 code to description mappings.
    
    Args:
        filepath: Path to ICD-9 mapping file (JSON or CSV)
                 If None, uses built-in simplified mappings
                 
    Returns:
        Dictionary mapping ICD-9 codes to descriptions
        
    Example:
        >>> mappings = load_icd9_mappings()
        >>> print(mappings['250'])
        'Diabetes mellitus'
    """
    if filepath is None:
        logger.info("Using built-in ICD-9 mappings")
        return ICD9_DESCRIPTIONS.copy()
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"ICD-9 mapping file not found: {filepath}")
        return ICD9_DESCRIPTIONS.copy()
    
    try:
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                mappings = json.load(f)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
            mappings = dict(zip(df['code'], df['description']))
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded {len(mappings)} ICD-9 mappings from {filepath}")
        return mappings
        
    except Exception as e:
        logger.error(f"Error loading ICD-9 mappings: {e}")
        return ICD9_DESCRIPTIONS.copy()


def lookup_icd9_description(
    icd9_code: str,
    mappings: Optional[Dict[str, str]] = None
) -> str:
    """
    Look up description for ICD-9 code.
    
    Args:
        icd9_code: ICD-9 code to look up
        mappings: ICD-9 mappings dictionary (None = use default)
        
    Returns:
        Description string, or 'Unknown' if not found
        
    Example:
        >>> desc = lookup_icd9_description('250')
        >>> print(desc)
        'Diabetes mellitus'
    """
    if mappings is None:
        mappings = ICD9_DESCRIPTIONS
    
    # Try exact match first
    if icd9_code in mappings:
        return mappings[icd9_code]
    
    # Try prefix matching (e.g., '25000' -> '250')
    for length in range(len(icd9_code), 0, -1):
        prefix = icd9_code[:length]
        if prefix in mappings:
            return mappings[prefix]
    
    return 'Unknown'


def calculate_age(
    dob: Union[pd.Timestamp, datetime],
    reference_date: Union[pd.Timestamp, datetime],
    handle_privacy_shift: bool = True
) -> float:
    """
    Calculate age from date of birth to reference date.
    
    Args:
        dob: Date of birth
        reference_date: Reference date (e.g., admission date, current date)
        handle_privacy_shift: If True, handles MIMIC-III privacy shifts
        
    Returns:
        Age in years (float)
        
    Note:
        MIMIC-III shifts birthdates by ~300 years for patients >89.
        This function detects and corrects for this shift.
        
    Example:
        >>> age = calculate_age(
        ...     dob=pd.Timestamp('1970-01-01'),
        ...     reference_date=pd.Timestamp('2020-01-01')
        ... )
        >>> print(f"Age: {age:.1f} years")
    """
    if pd.isna(dob) or pd.isna(reference_date):
        return np.nan
    
    # Convert to pandas Timestamp if needed
    if not isinstance(dob, pd.Timestamp):
        dob = pd.Timestamp(dob)
    if not isinstance(reference_date, pd.Timestamp):
        reference_date = pd.Timestamp(reference_date)
    
    # Calculate age
    age_timedelta = reference_date - dob
    age_years = age_timedelta.days / 365.25
    
    # Handle MIMIC-III privacy shift
    if handle_privacy_shift and age_years > 200:
        # Privacy-shifted - patient is >89
        # Assign median age for very elderly
        age_years = 91.5
        logger.debug("Privacy-shifted age detected")
    
    # Handle negative ages (data error)
    if age_years < 0:
        logger.warning(f"Negative age calculated: {age_years}, returning NaN")
        return np.nan
    
    return age_years


def calculate_age_at_events(
    dob_series: pd.Series,
    event_times: pd.Series,
    handle_privacy_shift: bool = True
) -> pd.Series:
    """
    Calculate ages at multiple events efficiently.
    
    Args:
        dob_series: Series of dates of birth
        event_times: Series of event timestamps
        handle_privacy_shift: Handle MIMIC-III privacy shifts
        
    Returns:
        Series of ages in years
    """
    ages = (event_times - dob_series).dt.days / 365.25
    
    if handle_privacy_shift:
        # Replace privacy-shifted ages (>200) with 91.5
        ages = ages.where(ages <= 200, 91.5)
    
    # Replace negative ages with NaN
    ages = ages.where(ages >= 0, np.nan)
    
    return ages


def create_sliding_windows(
    sequence: Union[pd.DataFrame, np.ndarray],
    window_size: int,
    stride: int = 1,
    time_column: Optional[str] = None
) -> List[Union[pd.DataFrame, np.ndarray]]:
    """
    Create sliding windows from time-series sequence.
    
    Args:
        sequence: Input sequence (DataFrame or array)
        window_size: Number of time steps per window
        stride: Step size between windows
        time_column: Column to sort by (for DataFrames)
        
    Returns:
        List of windows (DataFrames or arrays)
        
    Example:
        >>> data = pd.DataFrame({'time': range(100), 'value': range(100)})
        >>> windows = create_sliding_windows(data, window_size=10, stride=5)
        >>> print(f"Created {len(windows)} windows")
    """
    if isinstance(sequence, pd.DataFrame):
        # Sort by time if specified
        if time_column is not None and time_column in sequence.columns:
            sequence = sequence.sort_values(time_column)
        
        windows = []
        for start_idx in range(0, len(sequence) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = sequence.iloc[start_idx:end_idx].copy()
            windows.append(window)
    
    elif isinstance(sequence, np.ndarray):
        windows = []
        for start_idx in range(0, len(sequence) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = sequence[start_idx:end_idx].copy()
            windows.append(window)
    
    else:
        raise ValueError(f"Unsupported sequence type: {type(sequence)}")
    
    logger.info(f"Created {len(windows)} windows (size={window_size}, stride={stride})")
    return windows


def split_train_val_test(
    data: Union[pd.DataFrame, np.ndarray, List],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int = 42,
    shuffle: bool = True,
    stratify_by: Optional[pd.Series] = None
) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Input data (DataFrame, array, or list)
        ratios: Tuple of (train, val, test) ratios (must sum to 1.0)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting
        stratify_by: Series for stratified splitting (e.g., labels)
        
    Returns:
        Tuple of (train, val, test) data
        
    Example:
        >>> data = pd.DataFrame({'x': range(1000), 'y': range(1000)})
        >>> train, val, test = split_train_val_test(data)
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    
    train_ratio, val_ratio, test_ratio = ratios
    n_total = len(data)
    
    # Set random seed
    np.random.seed(random_state)
    
    # Get indices
    indices = np.arange(n_total)
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Calculate split points
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Split data based on type
    if isinstance(data, pd.DataFrame):
        train = data.iloc[train_indices].reset_index(drop=True)
        val = data.iloc[val_indices].reset_index(drop=True)
        test = data.iloc[test_indices].reset_index(drop=True)
    
    elif isinstance(data, np.ndarray):
        train = data[train_indices]
        val = data[val_indices]
        test = data[test_indices]
    
    elif isinstance(data, list):
        train = [data[i] for i in train_indices]
        val = [data[i] for i in val_indices]
        test = [data[i] for i in test_indices]
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    logger.info(f"Split data: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def save_processed_data(
    data: Union[pd.DataFrame, Dict, List],
    filepath: Union[str, Path],
    format: str = 'auto'
) -> None:
    """
    Save processed data to disk.
    
    Args:
        data: Data to save (DataFrame, dict, list, etc.)
        filepath: Output file path
        format: File format - 'auto', 'csv', 'parquet', 'pickle', 'json'
                'auto' detects from file extension
                
    Example:
        >>> data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> save_processed_data(data, 'processed_data.parquet')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format from extension
    if format == 'auto':
        format = filepath.suffix[1:]  # Remove leading dot
    
    try:
        if format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                raise ValueError("CSV format requires DataFrame")
        
        elif format == 'parquet':
            if isinstance(data, pd.DataFrame):
                data.to_parquet(filepath, index=False)
            else:
                raise ValueError("Parquet format requires DataFrame")
        
        elif format in ['pickle', 'pkl']:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif format == 'json':
            with open(filepath, 'w') as f:
                if isinstance(data, pd.DataFrame):
                    data.to_json(f, orient='records', indent=2)
                else:
                    json.dump(data, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved data to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def load_processed_data(
    filepath: Union[str, Path],
    format: str = 'auto'
) -> Union[pd.DataFrame, Dict, List]:
    """
    Load processed data from disk.
    
    Args:
        filepath: Input file path
        format: File format - 'auto', 'csv', 'parquet', 'pickle', 'json'
                'auto' detects from file extension
                
    Returns:
        Loaded data
        
    Example:
        >>> data = load_processed_data('processed_data.parquet')
        >>> print(type(data), len(data))
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect format from extension
    if format == 'auto':
        format = filepath.suffix[1:]  # Remove leading dot
    
    try:
        if format == 'csv':
            data = pd.read_csv(filepath)
        
        elif format == 'parquet':
            data = pd.read_parquet(filepath)
        
        elif format in ['pickle', 'pkl']:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        elif format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Loaded data from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def merge_patient_tables(
    tables: List[pd.DataFrame],
    on: Union[str, List[str]] = 'subject_id',
    how: str = 'inner'
) -> pd.DataFrame:
    """
    Merge multiple patient tables on common key(s).
    
    Args:
        tables: List of DataFrames to merge
        on: Column(s) to merge on
        how: Join type - 'inner', 'outer', 'left', 'right'
        
    Returns:
        Merged DataFrame
    """
    if len(tables) == 0:
        raise ValueError("No tables provided")
    
    if len(tables) == 1:
        return tables[0]
    
    result = tables[0]
    for i, table in enumerate(tables[1:], start=1):
        result = result.merge(table, on=on, how=how, suffixes=('', f'_t{i}'))
        logger.info(f"Merged table {i}: {len(result)} rows")
    
    return result


def downsample_data(
    df: pd.DataFrame,
    fraction: float = 0.1,
    random_state: int = 42,
    stratify_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Downsample data for faster prototyping.
    
    Args:
        df: Input DataFrame
        fraction: Fraction of data to keep
        random_state: Random seed
        stratify_by: Column for stratified sampling
        
    Returns:
        Downsampled DataFrame
    """
    if stratify_by is not None and stratify_by in df.columns:
        sampled = df.groupby(stratify_by, group_keys=False).apply(
            lambda x: x.sample(frac=fraction, random_state=random_state)
        )
    else:
        sampled = df.sample(frac=fraction, random_state=random_state)
    
    sampled = sampled.reset_index(drop=True)
    
    logger.info(f"Downsampled data: {len(df)} -> {len(sampled)} rows ({fraction:.1%})")
    return sampled


def compute_statistics(
    df: pd.DataFrame,
    numeric_only: bool = True
) -> Dict[str, Dict]:
    """
    Compute comprehensive statistics for DataFrame columns.
    
    Args:
        df: Input DataFrame
        numeric_only: Only compute for numeric columns
        
    Returns:
        Dictionary of statistics per column
    """
    stats = {}
    
    cols = df.select_dtypes(include=[np.number]).columns if numeric_only else df.columns
    
    for col in cols:
        if df[col].dtype in [np.number, float, int]:
            stats[col] = {
                'count': df[col].notna().sum(),
                'missing': df[col].isna().sum(),
                'missing_rate': df[col].isna().sum() / len(df),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'q25': df[col].quantile(0.25),
                'median': df[col].median(),
                'q75': df[col].quantile(0.75),
                'max': df[col].max(),
                'n_unique': df[col].nunique()
            }
        else:
            stats[col] = {
                'count': df[col].notna().sum(),
                'missing': df[col].isna().sum(),
                'missing_rate': df[col].isna().sum() / len(df),
                'n_unique': df[col].nunique(),
                'top': df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else None,
                'freq': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            }
    
    return stats


def print_data_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print readable summary of DataFrame."""
    print("=" * 80)
    print(f"DATA SUMMARY: {name}")
    print("=" * 80)
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn Information:")
    print("-" * 80)
    for col in df.columns:
        dtype = df[col].dtype
        n_missing = df[col].isna().sum()
        missing_pct = n_missing / len(df) * 100
        n_unique = df[col].nunique()
        print(f"{col:.<40} {str(dtype):.<15} Missing: {missing_pct:>5.1f}%  Unique: {n_unique:>6}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Example usage
    
    print("=== Utility Functions Example ===\n")
    
    # ICD-9 lookup
    print("1. ICD-9 Code Lookup")
    print("-" * 40)
    icd_codes = ['250', '401', '410', '428', '585']
    for code in icd_codes:
        desc = lookup_icd9_description(code)
        print(f"{code}: {desc}")
    
    # Age calculation
    print("\n2. Age Calculation")
    print("-" * 40)
    dob = pd.Timestamp('1970-01-15')
    ref_date = pd.Timestamp('2020-01-01')
    age = calculate_age(dob, ref_date)
    print(f"DOB: {dob.date()}")
    print(f"Reference: {ref_date.date()}")
    print(f"Age: {age:.1f} years")
    
    # Sliding windows
    print("\n3. Sliding Windows")
    print("-" * 40)
    data = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=100, freq='H'),
        'value': np.random.randn(100)
    })
    windows = create_sliding_windows(data, window_size=24, stride=12)
    print(f"Created {len(windows)} windows from {len(data)} time points")
    print(f"Window shape: {windows[0].shape}")
    
    # Train/val/test split
    print("\n4. Data Splitting")
    print("-" * 40)
    sample_data = pd.DataFrame({
        'x': range(1000),
        'y': np.random.randn(1000)
    })
    train, val, test = split_train_val_test(sample_data)
    print(f"Train: {len(train)} samples")
    print(f"Val: {len(val)} samples")
    print(f"Test: {len(test)} samples")
    
    # Save and load
    print("\n5. Save/Load Data")
    print("-" * 40)
    save_path = Path('/tmp/test_data.parquet')
    save_processed_data(train, save_path)
    print(f"Saved to: {save_path}")
    
    loaded = load_processed_data(save_path)
    print(f"Loaded: {len(loaded)} samples")
    
    # Statistics
    print("\n6. Data Statistics")
    print("-" * 40)
    stats = compute_statistics(sample_data)
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            if isinstance(stat_value, float):
                print(f"  {stat_name}: {stat_value:.3f}")
            else:
                print(f"  {stat_name}: {stat_value}")
