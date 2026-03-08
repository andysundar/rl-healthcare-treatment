"""
Feature Engineering Module for Healthcare RL

This module extracts and engineers features from raw clinical data:
- Demographics with categorical age bucketing
- Vital signs sequences
- Laboratory values sequences
- Medication history encoding
- Temporal features (time gaps, trends, seasonality)
- Comorbidity encoding from ICD-9 codes

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_MISSING_TOKEN = "UNKNOWN"
MISSING_STRING_TOKENS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "unknown/not specified",
    "not specified",
    "not recorded",
    "not available",
}


class AgeBucket(Enum):
    """Age categories for patient stratification."""
    NEONATE = "neonate"  # 0-28 days
    INFANT = "infant"  # 29 days - 1 year
    CHILD = "child"  # 1-12 years
    ADOLESCENT = "adolescent"  # 13-18 years
    YOUNG_ADULT = "young_adult"  # 19-35 years
    ADULT = "adult"  # 36-60 years
    ELDERLY = "elderly"  # 61-80 years
    VERY_ELDERLY = "very_elderly"  # 80+ years


class AgeBucketing:
    """
    Single source of truth for age bucketing in MIMIC-III data.
    
    Handles privacy-shifted birth dates and provides categorical age groups
    that are clinically meaningful and avoid datetime overflow errors.
    """
    
    @staticmethod
    def get_age_bucket(age_years: float) -> AgeBucket:
        """
        Convert age in years to categorical age bucket.
        
        Args:
            age_years: Age in years (can be fractional for infants)
            
        Returns:
            AgeBucket enum value
        """
        if age_years < 0:
            logger.warning(f"Negative age: {age_years}, treating as 0")
            age_years = 0
        
        if age_years < 0.077:  # ~28 days
            return AgeBucket.NEONATE
        elif age_years < 1:
            return AgeBucket.INFANT
        elif age_years < 12:
            return AgeBucket.CHILD
        elif age_years < 18:
            return AgeBucket.ADOLESCENT
        elif age_years < 35:
            return AgeBucket.YOUNG_ADULT
        elif age_years < 60:
            return AgeBucket.ADULT
        elif age_years < 80:
            return AgeBucket.ELDERLY
        else:
            return AgeBucket.VERY_ELDERLY
    
    @staticmethod
    def calculate_age_at_admission(
        dob: pd.Timestamp,
        admittime: pd.Timestamp,
        handle_privacy_shift: bool = True
    ) -> float:
        """
        Calculate age at admission handling MIMIC-III privacy shifts.
        
        Args:
            dob: Date of birth (privacy-shifted in MIMIC-III)
            admittime: Admission timestamp
            handle_privacy_shift: If True, handles MIMIC-III privacy shift (~300 years)
            
        Returns:
            Age in years
            
        Note:
            MIMIC-III shifts DOB by ~300 years for patients >89 to protect privacy.
            This function detects and corrects for this shift.
        """
        try:
            dob = pd.to_datetime(dob, errors='coerce')
            admittime = pd.to_datetime(admittime, errors='coerce')
            if pd.isna(dob) or pd.isna(admittime):
                return np.nan
            if dob > admittime:
                return np.nan

            age_years = float(
                admittime.year - dob.year
                - int((admittime.month, admittime.day) < (dob.month, dob.day))
            )
            if handle_privacy_shift and age_years > 200:
                age_years = 91.5
            if age_years < 0 or age_years > 130:
                return np.nan
            return age_years
        except Exception as e:
            logger.error(f"Error calculating age: {e}")
            return np.nan
    
    @staticmethod
    def bucket_to_numeric(bucket: AgeBucket) -> int:
        """Convert age bucket to numeric value for modeling."""
        bucket_order = {
            AgeBucket.NEONATE: 0,
            AgeBucket.INFANT: 1,
            AgeBucket.CHILD: 2,
            AgeBucket.ADOLESCENT: 3,
            AgeBucket.YOUNG_ADULT: 4,
            AgeBucket.ADULT: 5,
            AgeBucket.ELDERLY: 6,
            AgeBucket.VERY_ELDERLY: 7
        }
        return bucket_order[bucket]
    
    @staticmethod
    def bucket_to_onehot(bucket: AgeBucket) -> np.ndarray:
        """Convert age bucket to one-hot encoding."""
        buckets = list(AgeBucket)
        onehot = np.zeros(len(buckets))
        onehot[buckets.index(bucket)] = 1
        return onehot


class FeatureEngineer:
    """
    Extract and engineer features from raw clinical data.
    
    Provides methods for:
    - Demographics extraction with categorical age bucketing
    - Vital signs sequence extraction
    - Laboratory values sequence extraction
    - Medication history encoding
    - Temporal feature engineering
    - Comorbidity encoding from ICD-9 codes
    
    Example:
        >>> engineer = FeatureEngineer()
        >>> demographics = engineer.extract_demographics(patients, admissions)
        >>> vitals_seq = engineer.extract_vitals_sequence(admissions, chartevents)
    """
    
    # Common ICD-9 codes for chronic diseases
    CHRONIC_CONDITIONS = {
        'diabetes': ['250'],  # Diabetes mellitus
        'hypertension': ['401', '402', '403', '404', '405'],
        'heart_disease': ['410', '411', '412', '413', '414'],
        'copd': ['491', '492', '496'],
        'ckd': ['585', '586'],  # Chronic kidney disease
        'heart_failure': ['428'],
        'stroke': ['430', '431', '432', '433', '434', '435', '436'],
        'cancer': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
                  '150', '151', '152', '153', '154', '155', '156', '157', '158', '159'],
    }
    
    def __init__(self):
        """Initialize feature engineer."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.age_bucketing = AgeBucketing()
        logger.info("Initialized FeatureEngineer")

    @staticmethod
    def normalize_string_missing(
        series: pd.Series,
        fill_value: str = DEFAULT_MISSING_TOKEN,
    ) -> pd.Series:
        """Normalize text-like missing placeholders to a stable sentinel."""
        s = series.copy()
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype('string')
        elif not pd.api.types.is_string_dtype(s):
            s = s.astype('string')
        else:
            s = s.astype('string')
        s = s.str.strip()
        lower = s.str.lower()
        missing_mask = s.isna() | lower.isin(MISSING_STRING_TOKENS)
        s = s.mask(missing_mask, fill_value)
        return s.fillna(fill_value)

    @classmethod
    def safe_fill_categorical(
        cls,
        series: pd.Series,
        fill_value: str = DEFAULT_MISSING_TOKEN,
    ) -> pd.Series:
        """Safely fill missing values for categorical/object-like columns."""
        s = series.copy()
        if isinstance(s.dtype, pd.CategoricalDtype):
            if fill_value not in s.cat.categories:
                s = s.cat.add_categories([fill_value])
            s = s.fillna(fill_value)
            s = cls.normalize_string_missing(s.astype('string'), fill_value=fill_value)
            return s
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            return cls.normalize_string_missing(s, fill_value=fill_value)
        return s

    @staticmethod
    def safe_fill_numeric(
        series: pd.Series,
        default: Optional[float] = None,
    ) -> pd.Series:
        """Safely coerce numeric series and fill missing values."""
        s = pd.to_numeric(series, errors='coerce')
        if default is None:
            default = float(s.median()) if s.notna().any() else 0.0
        return s.fillna(default)

    def _safe_label_encode(
        self,
        name: str,
        series: pd.Series,
        unknown_token: str = DEFAULT_MISSING_TOKEN,
    ) -> np.ndarray:
        """Label encode robustly, mapping unseen labels to UNKNOWN."""
        s = self.normalize_string_missing(series, fill_value=unknown_token)
        if name not in self.label_encoders:
            enc = LabelEncoder()
            enc.fit(sorted(set(s.tolist() + [unknown_token])))
            self.label_encoders[name] = enc
        enc = self.label_encoders[name]
        known_classes = set(enc.classes_)
        if unknown_token not in known_classes:
            enc.fit(sorted(set(list(enc.classes_) + [unknown_token])))
            known_classes = set(enc.classes_)
        s = s.where(s.isin(known_classes), unknown_token)
        return enc.transform(s)

    def _normalize_categorical_feature_column(
        self,
        df: pd.DataFrame,
        column: str,
        fill_value: str = DEFAULT_MISSING_TOKEN,
    ) -> pd.DataFrame:
        if column not in df.columns:
            logger.warning("Demographics column '%s' missing; creating with fallback '%s'.", column, fill_value)
            df[column] = fill_value
        df[column] = self.safe_fill_categorical(df[column], fill_value=fill_value)
        return df
    
    def extract_demographics(
        self,
        patients: pd.DataFrame,
        admissions: Optional[pd.DataFrame] = None,
        reference_time: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Extract demographic features from patient data.
        
        Args:
            patients: PATIENTS table DataFrame
            admissions: ADMISSIONS table DataFrame (optional, for age at admission)
            reference_time: Reference time for age calculation (default: now)
            
        Returns:
            DataFrame with demographic features:
            - subject_id: Patient identifier
            - gender: Gender (M/F)
            - gender_encoded: Numeric encoding (0/1)
            - age_years: Age in years
            - age_bucket: Categorical age group
            - age_bucket_numeric: Numeric age bucket (0-7)
            - age_bucket_onehot: One-hot encoded age (8 columns)
            - ethnicity: Ethnicity (if available from admissions)
            - ethnicity_encoded: Numeric encoding
            - is_deceased: Boolean flag
            
        Note:
            Uses AgeBucketing class as single source of truth for age categories
        """
        logger.info(f"Extracting demographics for {len(patients)} patients")
        
        if 'subject_id' not in patients.columns:
            raise ValueError("patients DataFrame must include 'subject_id'")

        demo_features = pd.DataFrame()
        demo_features['subject_id'] = patients['subject_id']

        # Gender (robust normalization + encoding)
        raw_gender = patients['gender'] if 'gender' in patients.columns else pd.Series([pd.NA] * len(patients))
        demo_features['gender'] = self.safe_fill_categorical(raw_gender, fill_value=DEFAULT_MISSING_TOKEN)
        demo_features['gender_encoded'] = self._safe_label_encode('gender', demo_features['gender'])
        
        # Age calculation
        if reference_time is None:
            reference_time = pd.Timestamp.now()
        
        admissions_usable = (
            admissions is not None
            and 'subject_id' in admissions.columns
            and 'admittime' in admissions.columns
        )
        if admissions_usable:
            admissions = admissions.copy()
            admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
            first_admissions = admissions.groupby('subject_id')['admittime'].min().reset_index()
            demo_features = demo_features.merge(first_admissions, on='subject_id', how='left')
            demo_features = demo_features.merge(
                patients[['subject_id', 'dob']] if 'dob' in patients.columns else pd.DataFrame({'subject_id': patients['subject_id'], 'dob': pd.NaT}),
                on='subject_id',
                how='left'
            )
            demo_features['dob'] = pd.to_datetime(demo_features['dob'], errors='coerce')
            demo_features['age_years'] = demo_features.apply(
                lambda row: self.age_bucketing.calculate_age_at_admission(
                    row['dob'], row['admittime']
                ) if pd.notna(row['admittime']) else np.nan,
                axis=1
            )
        else:
            if admissions is not None:
                logger.warning("Admissions missing subject_id/admittime; using reference_time age fallback.")
            # Calculate current age
            demo_features = demo_features.merge(
                patients[['subject_id', 'dob']] if 'dob' in patients.columns else pd.DataFrame({'subject_id': patients['subject_id'], 'dob': pd.NaT}),
                on='subject_id',
                how='left'
            )
            demo_features['dob'] = pd.to_datetime(demo_features['dob'], errors='coerce')
            demo_features['age_years'] = demo_features['dob'].apply(
                lambda dob: self.age_bucketing.calculate_age_at_admission(
                    dob, reference_time
                ) if pd.notna(dob) else np.nan
            )

        # Numeric stabilization for age
        demo_features['age_years'] = self.safe_fill_numeric(demo_features['age_years'], default=np.nan)
        
        # Age bucketing
        demo_features['age_bucket'] = demo_features['age_years'].apply(
            lambda age: self.age_bucketing.get_age_bucket(age).value 
            if pd.notna(age) else AgeBucket.ADULT.value
        )
        
        demo_features['age_bucket_numeric'] = demo_features['age_years'].apply(
            lambda age: self.age_bucketing.bucket_to_numeric(
                self.age_bucketing.get_age_bucket(age)
            ) if pd.notna(age) else 5  # Default to ADULT
        )
        
        # One-hot encode age buckets
        age_buckets = [bucket.value for bucket in AgeBucket]
        for i, bucket in enumerate(age_buckets):
            demo_features[f'age_bucket_{bucket}'] = (
                demo_features['age_bucket'] == bucket
            ).astype(int)
        
        # Mortality
        dod_src = patients[['subject_id', 'dod']].copy() if 'dod' in patients.columns else pd.DataFrame({'subject_id': patients['subject_id'], 'dod': pd.NaT})
        dod_src['dod'] = pd.to_datetime(dod_src['dod'], errors='coerce')
        demo_features = demo_features.merge(
            dod_src,
            on='subject_id',
            how='left',
            suffixes=('', '_patient')
        )
        demo_features['is_deceased'] = demo_features['dod'].notna().astype(int)

        # Demographic categorical features from admissions (first known value per subject)
        admission_demo_cols = [
            'ethnicity',
            'insurance',
            'marital_status',
            'admission_type',
            'admission_location',
            'discharge_location',
        ]
        admissions_for_demo = admissions if (admissions is not None and 'subject_id' in admissions.columns) else None
        if admissions_for_demo is not None:
            for col in admission_demo_cols:
                if col in admissions_for_demo.columns:
                    col_df = admissions_for_demo.groupby('subject_id')[col].first().reset_index()
                    demo_features = demo_features.merge(col_df, on='subject_id', how='left')
                else:
                    logger.warning("Admissions missing expected column '%s'.", col)
        elif admissions is not None:
            logger.warning("Admissions missing 'subject_id'; admission demographics fallback to UNKNOWN.")

        demo_features = self._normalize_categorical_feature_column(demo_features, 'gender')
        for col in admission_demo_cols:
            demo_features = self._normalize_categorical_feature_column(demo_features, col)
            demo_features[f'{col}_encoded'] = self._safe_label_encode(col, demo_features[col])

        # Missing-data summary logs
        for col in ['gender', 'ethnicity', 'insurance', 'marital_status', 'admission_type']:
            if col in demo_features.columns:
                n_unknown = int((demo_features[col] == DEFAULT_MISSING_TOKEN).sum())
                logger.info(
                    "Demographics normalization: column='%s' unknown_count=%s/%s",
                    col, n_unknown, len(demo_features),
                )
        
        # Clean up temporary columns
        cols_to_drop = ['dob', 'admittime', 'dod']
        demo_features = demo_features.drop(
            columns=[c for c in cols_to_drop if c in demo_features.columns]
        )
        
        # Ensure stable string dtype for categorical-like fields for batch concat robustness.
        for col in ['gender'] + admission_demo_cols + ['age_bucket']:
            if col in demo_features.columns:
                demo_features[col] = demo_features[col].astype('string')

        logger.info(f"Extracted {len(demo_features.columns)} demographic features")
        return demo_features
    
    def extract_vitals_sequence(
        self,
        chartevents: pd.DataFrame,
        subject_ids: Optional[List[int]] = None,
        vital_itemids: Optional[Dict[str, List[int]]] = None
    ) -> pd.DataFrame:
        """
        Extract vital signs sequences from CHARTEVENTS.
        
        Args:
            chartevents: CHARTEVENTS DataFrame
            subject_ids: Filter for specific patients
            vital_itemids: Mapping of vital names to MIMIC item IDs
                         Example: {'heart_rate': [211, 220045], 'sbp': [51, 220050]}
            
        Returns:
            DataFrame with vital signs sequences
            
        Note:
            Default vital_itemids for common vitals are used if not specified
        """
        logger.info("Extracting vital signs sequences")
        
        # Default MIMIC-III item IDs for common vitals
        if vital_itemids is None:
            vital_itemids = {
                'heart_rate': [211, 220045],
                'sbp': [51, 442, 455, 6701, 220050, 220179],
                'dbp': [8368, 8440, 8441, 8555, 220051, 220180],
                'temperature': [223761, 678],
                'respiratory_rate': [618, 615, 220210, 224690],
                'spo2': [646, 220277],
            }
        
        # Filter by subject_ids if provided
        if subject_ids is not None:
            chartevents = chartevents[chartevents['subject_id'].isin(subject_ids)]
        
        vitals_list = []
        
        for vital_name, itemids in vital_itemids.items():
            # Extract vital
            vital_data = chartevents[chartevents['itemid'].isin(itemids)].copy()
            
            if len(vital_data) == 0:
                continue
            
            # Use numeric value
            vital_data = vital_data[['subject_id', 'hadm_id', 'charttime', 'valuenum']].copy()
            vital_data = vital_data.dropna(subset=['valuenum'])
            vital_data = vital_data.rename(columns={'valuenum': vital_name})
            
            vitals_list.append(vital_data)
            logger.info(f"Extracted {len(vital_data)} {vital_name} measurements")
        
        if not vitals_list:
            logger.warning("No vital signs found")
            return pd.DataFrame()
        
        # Merge all vitals on timestamp
        vitals = vitals_list[0]
        for vital_df in vitals_list[1:]:
            vitals = vitals.merge(
                vital_df,
                on=['subject_id', 'hadm_id', 'charttime'],
                how='outer'
            )
        
        # Sort by time
        vitals = vitals.sort_values(['subject_id', 'hadm_id', 'charttime'])
        
        logger.info(f"Created vitals sequence with {len(vitals)} timepoints")
        return vitals
    
    def extract_lab_sequence(
        self,
        labevents: pd.DataFrame,
        subject_ids: Optional[List[int]] = None,
        lab_itemids: Optional[Dict[str, List[int]]] = None
    ) -> pd.DataFrame:
        """
        Extract laboratory values sequences from LABEVENTS.
        
        Args:
            labevents: LABEVENTS DataFrame
            subject_ids: Filter for specific patients
            lab_itemids: Mapping of lab names to MIMIC item IDs
            
        Returns:
            DataFrame with lab values sequences
        """
        logger.info("Extracting laboratory sequences")
        
        # Default MIMIC-III item IDs for common labs
        if lab_itemids is None:
            lab_itemids = {
                'glucose': [50809, 50931],
                'sodium': [50824, 50983],
                'potassium': [50822, 50971],
                'chloride': [50806, 50902],
                'bicarbonate': [50803, 50882],
                'bun': [51006],
                'creatinine': [50912],
                'hematocrit': [51221, 50810],
                'wbc': [51300, 51301],
                'hemoglobin': [51222, 50811],
                'platelet': [51265],
            }
        
        # Filter by subject_ids if provided
        if subject_ids is not None:
            labevents = labevents[labevents['subject_id'].isin(subject_ids)]
        
        labs_list = []
        
        for lab_name, itemids in lab_itemids.items():
            # Extract lab
            lab_data = labevents[labevents['itemid'].isin(itemids)].copy()
            
            if len(lab_data) == 0:
                continue
            
            # Use numeric value
            lab_data = lab_data[['subject_id', 'hadm_id', 'charttime', 'valuenum']].copy()
            lab_data = lab_data.dropna(subset=['valuenum'])
            lab_data = lab_data.rename(columns={'valuenum': lab_name})
            
            labs_list.append(lab_data)
            logger.info(f"Extracted {len(lab_data)} {lab_name} measurements")
        
        if not labs_list:
            logger.warning("No lab values found")
            return pd.DataFrame()
        
        # Merge all labs on timestamp
        labs = labs_list[0]
        for lab_df in labs_list[1:]:
            labs = labs.merge(
                lab_df,
                on=['subject_id', 'hadm_id', 'charttime'],
                how='outer'
            )
        
        # Sort by time
        labs = labs.sort_values(['subject_id', 'hadm_id', 'charttime'])
        
        logger.info(f"Created labs sequence with {len(labs)} timepoints")
        return labs
    
    def extract_medication_history(
        self,
        prescriptions: pd.DataFrame,
        encoding: str = 'binary'
    ) -> pd.DataFrame:
        """
        Extract medication history and encode.
        
        Args:
            prescriptions: PRESCRIPTIONS DataFrame
            encoding: Encoding method - 'binary', 'count', or 'frequency'
            
        Returns:
            DataFrame with encoded medication history
        """
        logger.info(f"Extracting medication history with {encoding} encoding")
        
        # Get unique medications per patient
        med_by_patient = prescriptions.groupby(['subject_id', 'drug']).size().reset_index(name='count')
        
        if encoding == 'binary':
            # Binary: whether patient has taken the medication
            med_pivot = med_by_patient.pivot_table(
                index='subject_id',
                columns='drug',
                values='count',
                fill_value=0
            )
            med_pivot = (med_pivot > 0).astype(int)
        
        elif encoding == 'count':
            # Count: number of times medication was prescribed
            med_pivot = med_by_patient.pivot_table(
                index='subject_id',
                columns='drug',
                values='count',
                fill_value=0
            )
        
        elif encoding == 'frequency':
            # Frequency: proportion of admissions with medication
            admissions_per_patient = prescriptions.groupby('subject_id')['hadm_id'].nunique()
            med_pivot = med_by_patient.pivot_table(
                index='subject_id',
                columns='drug',
                values='count',
                fill_value=0
            )
            med_pivot = med_pivot.div(admissions_per_patient, axis=0)
        
        else:
            raise ValueError(f"Invalid encoding: {encoding}")
        
        # Reset index
        med_pivot = med_pivot.reset_index()
        
        # Rename columns for clarity
        med_pivot.columns = ['subject_id'] + [f'med_{col}' for col in med_pivot.columns[1:]]
        
        logger.info(f"Encoded {len(med_pivot.columns)-1} unique medications")
        return med_pivot
    
    def create_temporal_features(
        self,
        df: pd.DataFrame,
        time_column: str = 'charttime',
        subject_column: str = 'subject_id'
    ) -> pd.DataFrame:
        """
        Create temporal features from time-series data.
        
        Args:
            df: Input DataFrame with timestamps
            time_column: Name of timestamp column
            subject_column: Name of patient ID column
            
        Returns:
            DataFrame with additional temporal features:
            - time_since_last: Time since previous measurement (hours)
            - time_of_day: Hour of day (0-23)
            - day_of_week: Day of week (0-6)
            - is_weekend: Weekend indicator
            - is_night: Night time indicator (22:00-06:00)
            - rolling_mean_*: Rolling mean of numeric features
            - rolling_std_*: Rolling standard deviation
            - trend_*: Recent trend (difference from previous value)
        """
        logger.info("Creating temporal features")
        
        df_temporal = df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df_temporal[time_column]):
            df_temporal[time_column] = pd.to_datetime(df_temporal[time_column], errors='coerce')
        n_bad_time = int(df_temporal[time_column].isna().sum())
        if n_bad_time > 0:
            logger.warning(
                "Temporal features: dropping %s rows with invalid '%s' timestamps.",
                n_bad_time,
                time_column,
            )
            df_temporal = df_temporal.dropna(subset=[time_column])
        
        # Sort by subject and time
        df_temporal = df_temporal.sort_values([subject_column, time_column])
        
        # Time since last measurement
        df_temporal['time_since_last'] = df_temporal.groupby(subject_column)[time_column].diff()
        df_temporal['time_since_last_hours'] = df_temporal['time_since_last'].dt.total_seconds() / 3600
        
        # Time of day features
        df_temporal['time_of_day'] = df_temporal[time_column].dt.hour
        df_temporal['day_of_week'] = df_temporal[time_column].dt.dayofweek
        df_temporal['is_weekend'] = (df_temporal['day_of_week'] >= 5).astype(int)
        df_temporal['is_night'] = ((df_temporal['time_of_day'] >= 22) | 
                                    (df_temporal['time_of_day'] <= 6)).astype(int)
        
        # Rolling statistics for numeric columns
        excluded_cols = {
            subject_column,
            time_column,
            'time_since_last',           # timedelta helper column
            'time_since_last_hours',     # engineered gap feature (do not roll recursively)
            'time_of_day',
            'day_of_week',
            'is_weekend',
            'is_night',
        }
        numeric_cols = []
        for col in df_temporal.columns:
            if col in excluded_cols:
                continue
            s = df_temporal[col]
            if pd.api.types.is_timedelta64_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                logger.warning(
                    "Skipping temporal rolling features for '%s' (dtype=%s: non-scalar time dtype).",
                    col,
                    s.dtype,
                )
                continue
            if not pd.api.types.is_numeric_dtype(s):
                logger.warning(
                    "Skipping temporal rolling features for '%s' (dtype=%s: non-numeric).",
                    col,
                    s.dtype,
                )
                continue
            coerced = pd.to_numeric(s, errors='coerce')
            if coerced.notna().sum() == 0:
                logger.warning(
                    "Skipping temporal rolling features for '%s' (all values non-numeric after coercion).",
                    col,
                )
                continue
            df_temporal[col] = coerced
            numeric_cols.append(col)
        
        for col in numeric_cols:
            # 3-point rolling mean
            df_temporal[f'rolling_mean_{col}'] = df_temporal.groupby(subject_column)[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            # 3-point rolling std
            df_temporal[f'rolling_std_{col}'] = df_temporal.groupby(subject_column)[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).std()
            )
            
            # Trend (difference from previous)
            df_temporal[f'trend_{col}'] = df_temporal.groupby(subject_column)[col].diff()
        
        # Drop intermediate columns
        df_temporal = df_temporal.drop(columns=['time_since_last'])
        
        logger.info(f"Created {len(df_temporal.columns) - len(df.columns)} temporal features")
        return df_temporal
    
    def encode_comorbidities(
        self,
        diagnoses: pd.DataFrame,
        conditions: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Encode comorbidities from ICD-9 diagnosis codes.
        
        Args:
            diagnoses: DIAGNOSES_ICD DataFrame
            conditions: Dict mapping condition names to ICD-9 code prefixes
                       (None = use default chronic conditions)
            
        Returns:
            DataFrame with binary indicators for each condition
        """
        logger.info("Encoding comorbidities from ICD-9 codes")
        
        if conditions is None:
            conditions = self.CHRONIC_CONDITIONS
        
        # Initialize result
        comorbidity_features = pd.DataFrame()
        comorbidity_features['subject_id'] = diagnoses['subject_id'].unique()
        
        # Check each condition
        for condition_name, icd_prefixes in conditions.items():
            # Find patients with this condition
            has_condition = diagnoses[
                diagnoses['icd9_code'].str.startswith(tuple(icd_prefixes), na=False)
            ]['subject_id'].unique()
            
            # Create binary indicator
            comorbidity_features[f'has_{condition_name}'] = comorbidity_features['subject_id'].isin(
                has_condition
            ).astype(int)
            
            logger.info(f"{condition_name}: {len(has_condition)} patients")
        
        # Count total comorbidities
        condition_cols = [c for c in comorbidity_features.columns if c.startswith('has_')]
        comorbidity_features['total_comorbidities'] = comorbidity_features[condition_cols].sum(axis=1)
        
        logger.info(f"Encoded {len(condition_cols)} comorbidities")
        return comorbidity_features


if __name__ == '__main__':
    # Example usage
    
    # Create sample patient data
    sample_patients = pd.DataFrame({
        'subject_id': [1, 2, 3, 4, 5],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'dob': pd.to_datetime(['1950-01-01', '1985-05-15', '2010-12-01', 
                               '1960-03-20', '1945-11-10']),
        'dod': [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.Timestamp('2020-01-01')]
    })
    
    sample_admissions = pd.DataFrame({
        'subject_id': [1, 1, 2, 3, 4, 5],
        'hadm_id': [100, 101, 102, 103, 104, 105],
        'admittime': pd.to_datetime(['2020-01-01', '2020-06-01', '2020-02-01',
                                     '2020-03-01', '2020-04-01', '2019-12-01']),
        'ethnicity': ['WHITE', 'WHITE', 'ASIAN', 'HISPANIC', 'BLACK', 'WHITE']
    })
    
    # Initialize engineer
    engineer = FeatureEngineer()
    
    print("=== Demographics Extraction ===")
    demographics = engineer.extract_demographics(sample_patients, sample_admissions)
    print(demographics)
    
    print("\n=== Age Bucketing Examples ===")
    for age in [0.01, 0.5, 5, 15, 25, 45, 70, 85]:
        bucket = engineer.age_bucketing.get_age_bucket(age)
        print(f"Age {age}: {bucket.value}")
    
    # Create sample temporal data
    sample_vitals = pd.DataFrame({
        'subject_id': [1, 1, 1, 2, 2, 2],
        'hadm_id': [100, 100, 100, 102, 102, 102],
        'charttime': pd.date_range('2020-01-01', periods=6, freq='H')[:6],
        'heart_rate': [75, 80, 78, 85, 90, 88],
        'sbp': [120, 125, 122, 130, 135, 132]
    })
    
    print("\n=== Temporal Features ===")
    temporal_features = engineer.create_temporal_features(sample_vitals)
    print(temporal_features[['subject_id', 'charttime', 'time_since_last_hours', 
                             'time_of_day', 'is_night']].head())
    
    # Sample diagnoses
    sample_diagnoses = pd.DataFrame({
        'subject_id': [1, 1, 2, 3, 4],
        'hadm_id': [100, 100, 102, 103, 104],
        'icd9_code': ['25000', '40100', '25010', '41000', '4280']
    })
    
    print("\n=== Comorbidity Encoding ===")
    comorbidities = engineer.encode_comorbidities(sample_diagnoses)
    print(comorbidities)
