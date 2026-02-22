"""
Data Preprocessing Module for Healthcare RL

This module provides comprehensive preprocessing functionality for clinical data:
- Missing value imputation with multiple strategies
- Feature normalization and scaling
- Outlier detection and handling
- Timestamp alignment across multiple data sources
- Data quality validation

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for clinical datasets.
    
    Provides methods for:
    - Missing value imputation (mean, median, forward fill, KNN)
    - Feature normalization (standard, robust, minmax)
    - Outlier detection and handling (IQR, z-score, clipping)
    - Timestamp alignment across multiple dataframes
    - Data quality checks and reporting
    
    Attributes:
        config (Dict): Configuration parameters for preprocessing
        scalers (Dict): Fitted scaler objects for each feature type
        imputers (Dict): Fitted imputer objects for each strategy
        
    Example:
        >>> preprocessor = DataPreprocessor()
        >>> clean_data = preprocessor.clean_missing_values(df, strategy='median')
        >>> normalized = preprocessor.normalize_vitals(clean_data, method='robust')
    """
    
    # Physiological ranges for clinical validation
    VITAL_RANGES = {
        'heart_rate': (40, 200),
        'sbp': (70, 250),  # systolic blood pressure
        'dbp': (30, 150),  # diastolic blood pressure
        'temperature': (35.0, 42.0),  # Celsius
        'respiratory_rate': (8, 40),
        'spo2': (70, 100),  # oxygen saturation
    }
    
    LAB_RANGES = {
        'glucose': (20, 800),  # mg/dL
        'sodium': (120, 160),  # mmol/L
        'potassium': (2.0, 8.0),  # mmol/L
        'chloride': (80, 120),  # mmol/L
        'bicarbonate': (10, 45),  # mmol/L
        'bun': (5, 150),  # blood urea nitrogen, mg/dL
        'creatinine': (0.3, 15.0),  # mg/dL
        'hematocrit': (15, 60),  # %
        'wbc': (0.1, 50),  # white blood cell count, K/uL
        'hemoglobin': (5, 20),  # g/dL
        'platelet': (10, 1000),  # K/uL
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
                   Example: {'missing_threshold': 0.3, 'outlier_method': 'iqr'}
        """
        self.config = config or {}
        self.scalers: Dict[str, Union[StandardScaler, RobustScaler, MinMaxScaler]] = {}
        self.imputers: Dict[str, Union[SimpleImputer, KNNImputer]] = {}
        
        # Set default configuration values
        self.missing_threshold = self.config.get('missing_threshold', 0.3)
        self.outlier_method = self.config.get('outlier_method', 'iqr')
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        
        logger.info("Initialized DataPreprocessor")
    
    def clean_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'median',
        columns: Optional[List[str]] = None,
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values with various imputation strategies.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy - one of:
                     'mean': Replace with column mean
                     'median': Replace with column median (default)
                     'forward_fill': Forward fill missing values
                     'backward_fill': Backward fill missing values
                     'constant': Fill with constant value
                     'knn': KNN-based imputation (slow but accurate)
                     'drop': Drop rows with missing values
            columns: Specific columns to impute (None = all numeric columns)
            fill_value: Constant value for 'constant' strategy
            
        Returns:
            DataFrame with imputed missing values
            
        Raises:
            ValueError: If invalid strategy specified
        """
        if strategy not in ['mean', 'median', 'forward_fill', 'backward_fill', 
                           'constant', 'knn', 'drop']:
            raise ValueError(f"Invalid strategy: {strategy}")
        
        df_clean = df.copy()
        
        # Determine columns to process
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        missing_counts = df_clean[columns].isnull().sum()
        logger.info(f"Missing values before imputation:\n{missing_counts[missing_counts > 0]}")
        
        if strategy == 'drop':
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=columns)
            dropped = initial_rows - len(df_clean)
            logger.info(f"Dropped {dropped} rows with missing values")
            
        elif strategy in ['mean', 'median', 'constant']:
            imputer_key = f"{strategy}_{','.join(columns)}"
            
            if imputer_key not in self.imputers:
                if strategy == 'constant':
                    if fill_value is None:
                        raise ValueError("fill_value required for constant strategy")
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                else:
                    imputer = SimpleImputer(strategy=strategy)
                
                self.imputers[imputer_key] = imputer
                df_clean[columns] = imputer.fit_transform(df_clean[columns])
                logger.info(f"Fitted and applied {strategy} imputer")
            else:
                df_clean[columns] = self.imputers[imputer_key].transform(df_clean[columns])
                logger.info(f"Applied existing {strategy} imputer")
        
        elif strategy == 'forward_fill':
            df_clean[columns] = df_clean[columns].fillna(method='ffill')
            logger.info("Applied forward fill")
        
        elif strategy == 'backward_fill':
            df_clean[columns] = df_clean[columns].fillna(method='bfill')
            logger.info("Applied backward fill")
        
        elif strategy == 'knn':
            imputer_key = f"knn_{','.join(columns)}"
            
            if imputer_key not in self.imputers:
                imputer = KNNImputer(n_neighbors=5)
                self.imputers[imputer_key] = imputer
                df_clean[columns] = imputer.fit_transform(df_clean[columns])
                logger.info("Fitted and applied KNN imputer")
            else:
                df_clean[columns] = self.imputers[imputer_key].transform(df_clean[columns])
                logger.info("Applied existing KNN imputer")
        
        # Check remaining missing values
        remaining_missing = df_clean[columns].isnull().sum().sum()
        logger.info(f"Remaining missing values: {remaining_missing}")
        
        return df_clean
    
    def normalize_vitals(
        self,
        df: pd.DataFrame,
        method: str = 'robust',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize vital signs using specified scaling method.
        
        Args:
            df: Input DataFrame containing vital signs
            method: Normalization method - one of:
                   'standard': Standardization (zero mean, unit variance)
                   'robust': Robust scaling (median and IQR) - DEFAULT
                   'minmax': Min-max scaling to [0, 1]
            columns: Specific vital columns to normalize
                    (None = auto-detect based on column names)
            
        Returns:
            DataFrame with normalized vitals
            
        Note:
            Robust scaling is recommended for clinical data due to outliers
        """
        df_norm = df.copy()
        
        # Auto-detect vital columns if not specified
        if columns is None:
            vital_keywords = ['heart_rate', 'hr', 'sbp', 'dbp', 'bp', 'temperature', 
                            'temp', 'respiratory', 'rr', 'spo2', 'oxygen']
            columns = [col for col in df.columns 
                      if any(kw in col.lower() for kw in vital_keywords)]
        
        if not columns:
            logger.warning("No vital columns found for normalization")
            return df_norm
        
        logger.info(f"Normalizing {len(columns)} vital columns using {method} method")
        
        scaler_key = f"vitals_{method}"
        
        # Select appropriate scaler
        if scaler_key not in self.scalers:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Invalid method: {method}")
            
            self.scalers[scaler_key] = scaler
            df_norm[columns] = scaler.fit_transform(df_norm[columns])
            logger.info(f"Fitted and applied {method} scaler to vitals")
        else:
            df_norm[columns] = self.scalers[scaler_key].transform(df_norm[columns])
            logger.info(f"Applied existing {method} scaler to vitals")
        
        return df_norm
    
    def normalize_labs(
        self,
        df: pd.DataFrame,
        method: str = 'robust',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize laboratory values using specified scaling method.
        
        Args:
            df: Input DataFrame containing lab values
            method: Normalization method - 'standard', 'robust' (default), or 'minmax'
            columns: Specific lab columns to normalize
                    (None = auto-detect based on column names)
            
        Returns:
            DataFrame with normalized labs
            
        Note:
            Lab values often have outliers, so robust scaling is recommended
        """
        df_norm = df.copy()
        
        # Auto-detect lab columns if not specified
        if columns is None:
            lab_keywords = ['glucose', 'sodium', 'potassium', 'chloride', 'bicarbonate',
                          'bun', 'creatinine', 'hematocrit', 'wbc', 'hemoglobin', 
                          'platelet', 'ast', 'alt', 'albumin', 'bilirubin']
            columns = [col for col in df.columns 
                      if any(kw in col.lower() for kw in lab_keywords)]
        
        if not columns:
            logger.warning("No lab columns found for normalization")
            return df_norm
        
        logger.info(f"Normalizing {len(columns)} lab columns using {method} method")
        
        scaler_key = f"labs_{method}"
        
        # Select appropriate scaler
        if scaler_key not in self.scalers:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Invalid method: {method}")
            
            self.scalers[scaler_key] = scaler
            df_norm[columns] = scaler.fit_transform(df_norm[columns])
            logger.info(f"Fitted and applied {method} scaler to labs")
        else:
            df_norm[columns] = self.scalers[scaler_key].transform(df_norm[columns])
            logger.info(f"Applied existing {method} scaler to labs")
        
        return df_norm
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'clip',
        threshold: float = 3.0,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detect and handle outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            method: Outlier handling method - one of:
                   'clip': Clip values to threshold (default)
                   'remove': Remove rows containing outliers
                   'replace_nan': Replace outliers with NaN
                   'winsorize': Winsorize extreme values
            threshold: Threshold for outlier detection
                      For 'iqr': Multiplier for IQR (default: 3.0)
                      For 'zscore': Z-score threshold (default: 3.0)
            columns: Specific columns to check (None = all numeric columns)
            
        Returns:
            DataFrame with handled outliers
            
        Note:
            Default uses IQR method for outlier detection
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_counts = {}
        
        for col in columns:
            if col not in df_clean.columns:
                continue
            
            # IQR-based outlier detection
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Identify outliers
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                outlier_counts[col] = n_outliers
                
                if method == 'clip':
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                
                elif method == 'remove':
                    df_clean = df_clean[~outliers]
                
                elif method == 'replace_nan':
                    df_clean.loc[outliers, col] = np.nan
                
                elif method == 'winsorize':
                    # Cap at 5th and 95th percentiles
                    lower_cap = df_clean[col].quantile(0.05)
                    upper_cap = df_clean[col].quantile(0.95)
                    df_clean[col] = df_clean[col].clip(lower=lower_cap, upper=upper_cap)
                
                else:
                    raise ValueError(f"Invalid method: {method}")
        
        if outlier_counts:
            logger.info(f"Handled outliers using {method} method:")
            for col, count in outlier_counts.items():
                logger.info(f"  {col}: {count} outliers")
        else:
            logger.info("No outliers detected")
        
        return df_clean
    
    def validate_physiological_ranges(
        self,
        df: pd.DataFrame,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Validate that vital signs and lab values are within physiological ranges.
        
        Args:
            df: Input DataFrame
            column_mapping: Mapping from DataFrame columns to standard names
                          Example: {'hr': 'heart_rate', 'temp_c': 'temperature'}
            
        Returns:
            Tuple of (cleaned DataFrame, violation counts dictionary)
        """
        df_clean = df.copy()
        violations = {}
        
        # Use provided mapping or try to auto-detect
        if column_mapping is None:
            column_mapping = self._auto_detect_column_mapping(df)
        
        # Check vitals
        for col, standard_name in column_mapping.items():
            if col not in df_clean.columns:
                continue
            
            if standard_name in self.VITAL_RANGES:
                min_val, max_val = self.VITAL_RANGES[standard_name]
                invalid = (df_clean[col] < min_val) | (df_clean[col] > max_val)
                n_invalid = invalid.sum()
                
                if n_invalid > 0:
                    violations[col] = n_invalid
                    # Clip to valid range
                    df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
            
            elif standard_name in self.LAB_RANGES:
                min_val, max_val = self.LAB_RANGES[standard_name]
                invalid = (df_clean[col] < min_val) | (df_clean[col] > max_val)
                n_invalid = invalid.sum()
                
                if n_invalid > 0:
                    violations[col] = n_invalid
                    # Clip to valid range
                    df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
        
        if violations:
            logger.warning(f"Physiological range violations corrected:")
            for col, count in violations.items():
                logger.warning(f"  {col}: {count} violations")
        
        return df_clean, violations
    
    def align_timestamps(
        self,
        dfs_list: List[pd.DataFrame],
        time_column: str = 'charttime',
        join_method: str = 'outer',
        tolerance: Optional[pd.Timedelta] = None
    ) -> pd.DataFrame:
        """
        Align multiple DataFrames on timestamp column.
        
        Args:
            dfs_list: List of DataFrames to align
            time_column: Name of timestamp column
            join_method: Join method - 'inner', 'outer' (default), 'left'
            tolerance: Time tolerance for matching timestamps
                      Example: pd.Timedelta('1 hour')
            
        Returns:
            Aligned DataFrame with merged data
            
        Note:
            Uses pandas merge_asof for time-based joining with tolerance
        """
        if len(dfs_list) < 2:
            raise ValueError("Need at least 2 DataFrames to align")
        
        # Ensure all dataframes have the time column
        for i, df in enumerate(dfs_list):
            if time_column not in df.columns:
                raise ValueError(f"DataFrame {i} missing {time_column} column")
            
            # Ensure timestamp is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                dfs_list[i] = df.copy()
                dfs_list[i][time_column] = pd.to_datetime(df[time_column])
        
        logger.info(f"Aligning {len(dfs_list)} DataFrames on {time_column}")
        
        # Start with first dataframe
        result = dfs_list[0].copy()
        
        # Merge each subsequent dataframe
        for i, df in enumerate(dfs_list[1:], start=1):
            # Sort by time
            df_sorted = df.sort_values(time_column)
            result_sorted = result.sort_values(time_column)
            
            if tolerance is not None:
                # Use merge_asof for time-based joining with tolerance
                result = pd.merge_asof(
                    result_sorted,
                    df_sorted,
                    on=time_column,
                    direction='nearest',
                    tolerance=tolerance,
                    suffixes=('', f'_df{i}')
                )
            else:
                # Regular merge
                result = pd.merge(
                    result_sorted,
                    df_sorted,
                    on=time_column,
                    how=join_method,
                    suffixes=('', f'_df{i}')
                )
            
            logger.info(f"Merged DataFrame {i}: {len(result)} rows")
        
        logger.info(f"Final aligned DataFrame: {len(result)} rows")
        return result
    
    def create_sliding_windows(
        self,
        df: pd.DataFrame,
        window_size: int,
        stride: int = 1,
        time_column: Optional[str] = None
    ) -> List[pd.DataFrame]:
        """
        Create sliding windows from time-series data.
        
        Args:
            df: Input DataFrame (should be sorted by time)
            window_size: Number of rows per window
            stride: Number of rows to slide (default: 1)
            time_column: Column to sort by (if not already sorted)
            
        Returns:
            List of DataFrames, each representing a window
        """
        if time_column is not None:
            df = df.sort_values(time_column)
        
        windows = []
        
        for start_idx in range(0, len(df) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = df.iloc[start_idx:end_idx].copy()
            windows.append(window)
        
        logger.info(f"Created {len(windows)} windows (size={window_size}, stride={stride})")
        return windows
    
    def _auto_detect_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect mapping from DataFrame columns to standard names."""
        mapping = {}
        
        # Define patterns for different measurements
        patterns = {
            'heart_rate': ['heart_rate', 'hr', 'heartrate', 'pulse'],
            'sbp': ['sbp', 'systolic', 'sys_bp'],
            'dbp': ['dbp', 'diastolic', 'dia_bp'],
            'temperature': ['temperature', 'temp', 'temp_c', 'temp_f'],
            'respiratory_rate': ['respiratory_rate', 'rr', 'resprate'],
            'spo2': ['spo2', 'oxygen', 'o2sat', 'sao2'],
            'glucose': ['glucose', 'bg', 'bloodglucose'],
            'sodium': ['sodium', 'na'],
            'potassium': ['potassium', 'k'],
            'creatinine': ['creatinine', 'cr'],
        }
        
        for col in df.columns:
            col_lower = col.lower()
            for standard_name, keywords in patterns.items():
                if any(kw in col_lower for kw in keywords):
                    mapping[col] = standard_name
                    break
        
        return mapping
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing operations performed.
        
        Returns:
            Dictionary containing preprocessing statistics
        """
        summary = {
            'scalers_fitted': list(self.scalers.keys()),
            'imputers_fitted': list(self.imputers.keys()),
            'config': self.config
        }
        
        return summary
    
    def reset(self) -> None:
        """Reset all fitted transformers."""
        self.scalers.clear()
        self.imputers.clear()
        logger.info("Reset all preprocessing transformers")


if __name__ == '__main__':
    # Example usage
    
    # Create sample clinical data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'subject_id': range(n_samples),
        'heart_rate': np.random.normal(75, 15, n_samples),
        'sbp': np.random.normal(120, 20, n_samples),
        'dbp': np.random.normal(80, 10, n_samples),
        'temperature': np.random.normal(37, 0.5, n_samples),
        'glucose': np.random.normal(100, 30, n_samples),
        'sodium': np.random.normal(140, 3, n_samples),
        'charttime': pd.date_range('2020-01-01', periods=n_samples, freq='H')
    })
    
    # Add some missing values
    sample_data.loc[np.random.choice(n_samples, 50), 'heart_rate'] = np.nan
    sample_data.loc[np.random.choice(n_samples, 30), 'glucose'] = np.nan
    
    # Add some outliers
    sample_data.loc[np.random.choice(n_samples, 10), 'heart_rate'] = 250
    sample_data.loc[np.random.choice(n_samples, 10), 'glucose'] = 500
    
    print("Original data:")
    print(sample_data.describe())
    print(f"\nMissing values:\n{sample_data.isnull().sum()}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Clean missing values
    print("\n=== Cleaning Missing Values ===")
    clean_data = preprocessor.clean_missing_values(sample_data, strategy='median')
    print(f"Missing values after cleaning:\n{clean_data.isnull().sum()}")
    
    # Handle outliers
    print("\n=== Handling Outliers ===")
    clean_data = preprocessor.handle_outliers(clean_data, method='clip', threshold=3.0)
    
    # Validate physiological ranges
    print("\n=== Validating Physiological Ranges ===")
    clean_data, violations = preprocessor.validate_physiological_ranges(clean_data)
    print(f"Violations: {violations}")
    
    # Normalize vitals
    print("\n=== Normalizing Vitals ===")
    normalized_data = preprocessor.normalize_vitals(clean_data, method='robust')
    print(normalized_data[['heart_rate', 'sbp', 'dbp', 'temperature']].describe())
    
    # Normalize labs
    print("\n=== Normalizing Labs ===")
    normalized_data = preprocessor.normalize_labs(normalized_data, method='robust')
    print(normalized_data[['glucose', 'sodium']].describe())
    
    # Get summary
    print("\n=== Preprocessing Summary ===")
    summary = preprocessor.get_preprocessing_summary()
    print(f"Scalers fitted: {summary['scalers_fitted']}")
    print(f"Imputers fitted: {summary['imputers_fitted']}")
