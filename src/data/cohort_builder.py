"""
Cohort Building Module for Healthcare RL

This module provides functionality for defining and building patient cohorts
with specific inclusion and exclusion criteria for research studies.

Supports:
- Disease-specific cohorts (diabetes, hypertension, etc.)
- Age-based cohorts
- Admission-based filtering
- Lab value requirements
- Exclusion criteria (pregnancy, pediatric, data quality)
- Cohort statistics and documentation

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CohortBuilder:
    """
    Build patient cohorts with specific inclusion/exclusion criteria.
    
    This class provides methods to:
    - Define disease-specific cohorts using ICD-9 codes
    - Apply age, admission, and lab requirements
    - Apply exclusion criteria
    - Generate cohort statistics and documentation
    - Track cohort selection process
    
    Attributes:
        patients: PATIENTS table DataFrame
        admissions: ADMISSIONS table DataFrame
        diagnoses: DIAGNOSES_ICD table DataFrame
        labevents: LABEVENTS table DataFrame (optional)
        prescriptions: PRESCRIPTIONS table DataFrame (optional)
        
    Example:
        >>> builder = CohortBuilder(patients, admissions, diagnoses)
        >>> diabetes_patients = builder.define_diabetes_cohort()
        >>> filtered = builder.apply_inclusion_criteria(
        ...     subject_ids=diabetes_patients,
        ...     min_age=18,
        ...     min_admissions=2
        ... )
    """
    
    # ICD-9 code patterns for common diseases
    DISEASE_PATTERNS = {
        'diabetes': {
            'codes': ['250'],  # All diabetes codes (250.xx)
            'description': 'Diabetes mellitus'
        },
        'diabetes_type1': {
            'codes': ['25001', '25003', '25011', '25013', '25021', '25023',
                     '25031', '25033', '25041', '25043', '25051', '25053',
                     '25061', '25063', '25071', '25073', '25081', '25083',
                     '25091', '25093'],
            'description': 'Type 1 diabetes (insulin-dependent)'
        },
        'diabetes_type2': {
            'codes': ['25000', '25002', '25010', '25012', '25020', '25022',
                     '25030', '25032', '25040', '25042', '25050', '25052',
                     '25060', '25062', '25070', '25072', '25080', '25082',
                     '25090', '25092'],
            'description': 'Type 2 diabetes (non-insulin-dependent)'
        },
        'hypertension': {
            'codes': ['401', '402', '403', '404', '405'],
            'description': 'Hypertensive diseases'
        },
        'heart_failure': {
            'codes': ['428'],
            'description': 'Heart failure'
        },
        'ckd': {
            'codes': ['585', '586'],
            'description': 'Chronic kidney disease'
        },
        'copd': {
            'codes': ['491', '492', '496'],
            'description': 'Chronic obstructive pulmonary disease'
        },
        'stroke': {
            'codes': ['430', '431', '432', '433', '434', '435', '436'],
            'description': 'Cerebrovascular disease'
        },
        'ami': {
            'codes': ['410'],
            'description': 'Acute myocardial infarction'
        },
        'sepsis': {
            'codes': ['038', '995.91', '995.92'],
            'description': 'Sepsis and septic shock'
        }
    }
    
    def __init__(
        self,
        patients: pd.DataFrame,
        admissions: pd.DataFrame,
        diagnoses: pd.DataFrame,
        labevents: Optional[pd.DataFrame] = None,
        prescriptions: Optional[pd.DataFrame] = None
    ):
        """
        Initialize cohort builder with clinical data tables.
        
        Args:
            patients: PATIENTS table
            admissions: ADMISSIONS table
            diagnoses: DIAGNOSES_ICD table
            labevents: LABEVENTS table (optional, for lab-based criteria)
            prescriptions: PRESCRIPTIONS table (optional, for medication-based criteria)
        """
        self.patients = patients
        self.admissions = admissions
        self.diagnoses = diagnoses
        self.labevents = labevents
        self.prescriptions = prescriptions
        
        # Track cohort selection process
        self.selection_history: List[Dict] = []
        
        logger.info("Initialized CohortBuilder")
        logger.info(f"Total patients: {len(patients):,}")
        logger.info(f"Total admissions: {len(admissions):,}")
        logger.info(f"Total diagnoses: {len(diagnoses):,}")
    
    def define_cohort_by_icd(
        self,
        icd_codes: List[str],
        cohort_name: str = "custom",
        require_primary: bool = False
    ) -> List[int]:
        """
        Define cohort based on ICD-9 diagnosis codes.
        
        Args:
            icd_codes: List of ICD-9 code prefixes
                      Example: ['250'] matches all 250.xx codes
            cohort_name: Name for the cohort
            require_primary: If True, only match primary diagnosis (seq_num=1)
            
        Returns:
            List of subject_ids matching the criteria
        """
        logger.info(f"Defining {cohort_name} cohort with ICD codes: {icd_codes}")
        
        # Filter diagnoses
        diagnoses_subset = self.diagnoses.copy()
        
        if require_primary:
            diagnoses_subset = diagnoses_subset[diagnoses_subset['seq_num'] == 1]
        
        # Match ICD codes (supports prefix matching)
        matched_subjects = set()
        
        for code in icd_codes:
            # Convert to regex pattern for prefix matching
            pattern = f"^{code}"
            matches = diagnoses_subset[
                diagnoses_subset['icd9_code'].str.match(pattern, na=False)
            ]
            matched_subjects.update(matches['subject_id'].unique())
        
        subject_ids = list(matched_subjects)
        
        # Record selection
        self._record_selection(
            step=f"ICD-9 codes: {icd_codes}",
            n_subjects=len(subject_ids),
            cohort_name=cohort_name
        )
        
        logger.info(f"Found {len(subject_ids):,} patients with {cohort_name}")
        return subject_ids
    
    def define_diabetes_cohort(
        self,
        diabetes_type: str = 'all'
    ) -> List[int]:
        """
        Define diabetes cohort.
        
        Args:
            diabetes_type: 'all', 'type1', or 'type2'
            
        Returns:
            List of subject_ids with diabetes
        """
        if diabetes_type == 'all':
            pattern_key = 'diabetes'
        elif diabetes_type == 'type1':
            pattern_key = 'diabetes_type1'
        elif diabetes_type == 'type2':
            pattern_key = 'diabetes_type2'
        else:
            raise ValueError(f"Invalid diabetes_type: {diabetes_type}")
        
        disease_info = self.DISEASE_PATTERNS[pattern_key]
        
        return self.define_cohort_by_icd(
            icd_codes=disease_info['codes'],
            cohort_name=disease_info['description']
        )
    
    def define_hypertension_cohort(self) -> List[int]:
        """Define hypertension cohort."""
        disease_info = self.DISEASE_PATTERNS['hypertension']
        return self.define_cohort_by_icd(
            icd_codes=disease_info['codes'],
            cohort_name=disease_info['description']
        )
    
    def define_heart_failure_cohort(self) -> List[int]:
        """Define heart failure cohort."""
        disease_info = self.DISEASE_PATTERNS['heart_failure']
        return self.define_cohort_by_icd(
            icd_codes=disease_info['codes'],
            cohort_name=disease_info['description']
        )
    
    def define_ckd_cohort(self) -> List[int]:
        """Define chronic kidney disease cohort."""
        disease_info = self.DISEASE_PATTERNS['ckd']
        return self.define_cohort_by_icd(
            icd_codes=disease_info['codes'],
            cohort_name=disease_info['description']
        )
    
    def define_copd_cohort(self) -> List[int]:
        """Define COPD cohort."""
        disease_info = self.DISEASE_PATTERNS['copd']
        return self.define_cohort_by_icd(
            icd_codes=disease_info['codes'],
            cohort_name=disease_info['description']
        )
    
    def define_multimorbidity_cohort(
        self,
        conditions: List[str],
        min_conditions: int = 2
    ) -> List[int]:
        """
        Define cohort with multiple chronic conditions.
        
        Args:
            conditions: List of condition names from DISEASE_PATTERNS
            min_conditions: Minimum number of conditions required
            
        Returns:
            List of subject_ids with >= min_conditions
        """
        logger.info(f"Defining multimorbidity cohort: {conditions}, min={min_conditions}")
        
        # Get subjects for each condition
        condition_subjects = {}
        for condition in conditions:
            if condition not in self.DISEASE_PATTERNS:
                logger.warning(f"Unknown condition: {condition}")
                continue
            
            disease_info = self.DISEASE_PATTERNS[condition]
            subjects = self.define_cohort_by_icd(
                icd_codes=disease_info['codes'],
                cohort_name=condition
            )
            condition_subjects[condition] = set(subjects)
        
        # Count conditions per patient
        all_subjects = set()
        for subjects in condition_subjects.values():
            all_subjects.update(subjects)
        
        multimorbid_subjects = []
        for subject_id in all_subjects:
            n_conditions = sum(
                subject_id in subjects
                for subjects in condition_subjects.values()
            )
            if n_conditions >= min_conditions:
                multimorbid_subjects.append(subject_id)
        
        self._record_selection(
            step=f"Multimorbidity: {conditions} (>= {min_conditions})",
            n_subjects=len(multimorbid_subjects),
            cohort_name="multimorbidity"
        )
        
        logger.info(f"Found {len(multimorbid_subjects):,} patients with multimorbidity")
        return multimorbid_subjects
    
    def apply_inclusion_criteria(
        self,
        subject_ids: List[int],
        min_age: Optional[float] = None,
        max_age: Optional[float] = None,
        min_admissions: Optional[int] = None,
        max_admissions: Optional[int] = None,
        lab_requirements: Optional[Dict[str, Tuple[float, float]]] = None,
        admission_types: Optional[List[str]] = None
    ) -> List[int]:
        """
        Apply inclusion criteria to filter cohort.
        
        Args:
            subject_ids: Initial list of subject IDs
            min_age: Minimum age at first admission
            max_age: Maximum age at first admission
            min_admissions: Minimum number of admissions
            max_admissions: Maximum number of admissions
            lab_requirements: Dict of lab tests with (min, max) ranges
                            Example: {'glucose': (70, 200)}
            admission_types: List of allowed admission types
                           Example: ['EMERGENCY', 'ELECTIVE']
            
        Returns:
            Filtered list of subject_ids meeting all criteria
        """
        logger.info(f"Applying inclusion criteria to {len(subject_ids):,} subjects")
        
        filtered_ids = set(subject_ids)
        
        # Age criteria
        if min_age is not None or max_age is not None:
            filtered_ids = self._filter_by_age(filtered_ids, min_age, max_age)
        
        # Admission count criteria
        if min_admissions is not None or max_admissions is not None:
            filtered_ids = self._filter_by_admission_count(
                filtered_ids, min_admissions, max_admissions
            )
        
        # Admission type criteria
        if admission_types is not None:
            filtered_ids = self._filter_by_admission_type(filtered_ids, admission_types)
        
        # Lab value criteria
        if lab_requirements is not None and self.labevents is not None:
            filtered_ids = self._filter_by_lab_values(filtered_ids, lab_requirements)
        
        self._record_selection(
            step=f"Inclusion criteria applied",
            n_subjects=len(filtered_ids),
            cohort_name="filtered"
        )
        
        logger.info(f"After inclusion criteria: {len(filtered_ids):,} subjects")
        return list(filtered_ids)
    
    def apply_exclusion_criteria(
        self,
        subject_ids: List[int],
        exclude_pregnancy: bool = False,
        exclude_pediatric: bool = False,
        exclude_age_over: Optional[float] = None,
        missing_data_threshold: float = 0.5,
        exclude_short_los: Optional[float] = None,
        exclude_died_in_hospital: bool = False
    ) -> List[int]:
        """
        Apply exclusion criteria to filter cohort.
        
        Args:
            subject_ids: Initial list of subject IDs
            exclude_pregnancy: Exclude pregnancy-related admissions
            exclude_pediatric: Exclude patients under 18
            exclude_age_over: Exclude patients over specified age
            missing_data_threshold: Exclude patients with missing data rate above threshold
            exclude_short_los: Exclude admissions shorter than X days
            exclude_died_in_hospital: Exclude patients who died during admission
            
        Returns:
            Filtered list of subject_ids after exclusions
        """
        logger.info(f"Applying exclusion criteria to {len(subject_ids):,} subjects")
        
        filtered_ids = set(subject_ids)
        
        # Pregnancy exclusion
        if exclude_pregnancy:
            filtered_ids = self._exclude_pregnancy(filtered_ids)
        
        # Pediatric exclusion
        if exclude_pediatric:
            filtered_ids = self._exclude_pediatric(filtered_ids)
        
        # Age ceiling
        if exclude_age_over is not None:
            filtered_ids = self._exclude_age_over(filtered_ids, exclude_age_over)
        
        # Data quality exclusion
        if missing_data_threshold < 1.0:
            filtered_ids = self._exclude_poor_data_quality(
                filtered_ids, missing_data_threshold
            )
        
        # Length of stay exclusion
        if exclude_short_los is not None:
            filtered_ids = self._exclude_short_los(filtered_ids, exclude_short_los)
        
        # In-hospital mortality exclusion
        if exclude_died_in_hospital:
            filtered_ids = self._exclude_died_in_hospital(filtered_ids)
        
        self._record_selection(
            step=f"Exclusion criteria applied",
            n_subjects=len(filtered_ids),
            cohort_name="excluded"
        )
        
        logger.info(f"After exclusion criteria: {len(filtered_ids):,} subjects")
        return list(filtered_ids)
    
    def get_cohort_statistics(
        self,
        subject_ids: List[int]
    ) -> Dict:
        """
        Generate comprehensive statistics for a cohort.
        
        Args:
            subject_ids: List of subject IDs in cohort
            
        Returns:
            Dictionary containing cohort statistics
        """
        logger.info(f"Generating statistics for cohort of {len(subject_ids):,} patients")
        
        stats = {}
        
        # Basic counts
        stats['n_patients'] = len(subject_ids)
        
        # Demographics
        cohort_patients = self.patients[self.patients['subject_id'].isin(subject_ids)]
        
        stats['demographics'] = {
            'n_male': (cohort_patients['gender'] == 'M').sum(),
            'n_female': (cohort_patients['gender'] == 'F').sum(),
            'n_deceased': cohort_patients['dod'].notna().sum()
        }
        
        # Age statistics
        cohort_admissions = self.admissions[
            self.admissions['subject_id'].isin(subject_ids)
        ].merge(
            cohort_patients[['subject_id', 'dob']],
            on='subject_id'
        )
        cohort_admissions['admittime'] = pd.to_datetime(
            cohort_admissions['admittime'], errors='coerce'
        )
        cohort_admissions['dischtime'] = pd.to_datetime(
            cohort_admissions['dischtime'], errors='coerce'
        )
        cohort_admissions['dob'] = pd.to_datetime(
            cohort_admissions['dob'], errors='coerce'
        )
        cohort_admissions['age_years'] = cohort_admissions.apply(
            lambda row: self._safe_age_at_admission(row['dob'], row['admittime']),
            axis=1,
        )
        n_invalid_age = int(cohort_admissions['age_years'].isna().sum())
        if n_invalid_age > 0:
            logger.warning(
                "Cohort statistics: %s/%s admission rows have invalid age and were excluded from age stats.",
                n_invalid_age,
                len(cohort_admissions),
            )
        
        stats['age'] = {
            'mean': cohort_admissions['age_years'].mean(),
            'std': cohort_admissions['age_years'].std(),
            'median': cohort_admissions['age_years'].median(),
            'min': cohort_admissions['age_years'].min(),
            'max': cohort_admissions['age_years'].max()
        }
        
        # Admission statistics
        admissions_per_patient = cohort_admissions.groupby('subject_id').size()
        
        stats['admissions'] = {
            'total': len(cohort_admissions),
            'mean_per_patient': admissions_per_patient.mean(),
            'median_per_patient': admissions_per_patient.median(),
            'max_per_patient': admissions_per_patient.max()
        }
        
        # Admission types
        if 'admission_type' in cohort_admissions.columns:
            stats['admission_types'] = cohort_admissions['admission_type'].value_counts().to_dict()
        
        # Length of stay
        cohort_admissions['los_days'] = (
            cohort_admissions['dischtime'] - cohort_admissions['admittime']
        ).dt.total_seconds() / (24 * 3600)
        
        stats['length_of_stay'] = {
            'mean_days': cohort_admissions['los_days'].mean(),
            'median_days': cohort_admissions['los_days'].median(),
            'min_days': cohort_admissions['los_days'].min(),
            'max_days': cohort_admissions['los_days'].max()
        }
        
        # Diagnosis statistics
        cohort_diagnoses = self.diagnoses[
            self.diagnoses['subject_id'].isin(subject_ids)
        ]
        
        diagnoses_per_patient = cohort_diagnoses.groupby('subject_id').size()
        
        stats['diagnoses'] = {
            'total': len(cohort_diagnoses),
            'mean_per_patient': diagnoses_per_patient.mean(),
            'median_per_patient': diagnoses_per_patient.median(),
            'unique_codes': cohort_diagnoses['icd9_code'].nunique()
        }
        
        # Top diagnoses
        top_diagnoses = cohort_diagnoses['icd9_code'].value_counts().head(10)
        stats['top_diagnoses'] = top_diagnoses.to_dict()
        
        return stats
    
    def export_cohort_definition(
        self,
        subject_ids: List[int],
        filepath: str,
        include_statistics: bool = True
    ) -> None:
        """
        Export cohort definition and statistics to file.
        
        Args:
            subject_ids: List of subject IDs in cohort
            filepath: Path to save cohort definition
            include_statistics: Whether to include detailed statistics
        """
        logger.info(f"Exporting cohort definition to {filepath}")
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COHORT DEFINITION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total patients: {len(subject_ids):,}\n\n")
            
            # Selection history
            f.write("SELECTION PROCESS:\n")
            f.write("-" * 80 + "\n")
            for i, step in enumerate(self.selection_history, 1):
                f.write(f"{i}. {step['step']}\n")
                f.write(f"   Cohort: {step['cohort_name']}\n")
                f.write(f"   N subjects: {step['n_subjects']:,}\n")
                f.write(f"   Timestamp: {step['timestamp']}\n\n")
            
            # Statistics
            if include_statistics:
                f.write("\nCOHORT STATISTICS:\n")
                f.write("-" * 80 + "\n")
                
                stats = self.get_cohort_statistics(subject_ids)
                
                for section, data in stats.items():
                    f.write(f"\n{section.upper()}:\n")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {data}\n")
        
        logger.info(f"Cohort definition exported to {filepath}")
    
    # Private helper methods

    @staticmethod
    def _safe_age_at_admission(
        dob: pd.Timestamp,
        admittime: pd.Timestamp,
        privacy_shift_threshold: float = 200.0,
        privacy_shift_age: float = 91.5,
        max_reasonable_age: float = 130.0,
    ) -> float:
        """
        Safely compute age in years without timedelta subtraction overflow.

        Uses calendar arithmetic on year/month/day to avoid int64 datetime
        subtraction overflow for malformed or shifted timestamps.
        """
        if pd.isna(dob) or pd.isna(admittime):
            return np.nan

        try:
            dob = pd.to_datetime(dob, errors='coerce')
            admittime = pd.to_datetime(admittime, errors='coerce')
        except Exception:
            return np.nan

        if pd.isna(dob) or pd.isna(admittime):
            return np.nan

        if dob > admittime:
            return np.nan

        age = float(
            admittime.year - dob.year
            - int((admittime.month, admittime.day) < (dob.month, dob.day))
        )

        if age < 0:
            return np.nan
        if age > privacy_shift_threshold:
            return float(privacy_shift_age)
        if age > max_reasonable_age:
            return np.nan
        return age

    def _build_age_frame_for_first_admission(self, subject_ids: Set[int]) -> pd.DataFrame:
        """Return per-subject safe age at first admission with quality flags."""
        cohort_admissions = self.admissions[
            self.admissions['subject_id'].isin(subject_ids)
        ][['subject_id', 'admittime']].copy()
        cohort_patients = self.patients[
            self.patients['subject_id'].isin(subject_ids)
        ][['subject_id', 'dob']].copy()

        cohort_admissions['admittime'] = pd.to_datetime(
            cohort_admissions['admittime'], errors='coerce'
        )
        cohort_patients['dob'] = pd.to_datetime(
            cohort_patients['dob'], errors='coerce'
        )

        first_adm = (
            cohort_admissions
            .dropna(subset=['admittime'])
            .groupby('subject_id', as_index=False)['admittime']
            .min()
        )
        merged = first_adm.merge(cohort_patients, on='subject_id', how='left')
        merged['age_years'] = merged.apply(
            lambda row: self._safe_age_at_admission(row['dob'], row['admittime']),
            axis=1,
        )
        return merged

    def _filter_by_age(
        self,
        subject_ids: Set[int],
        min_age: Optional[float],
        max_age: Optional[float]
    ) -> Set[int]:
        """Filter by age at first admission."""
        age_df = self._build_age_frame_for_first_admission(subject_ids)
        n_missing_dates = int(age_df['dob'].isna().sum() + age_df['admittime'].isna().sum())
        invalid_age_mask = age_df['age_years'].isna()
        n_invalid_age = int(invalid_age_mask.sum())
        if n_missing_dates > 0 or n_invalid_age > 0:
            logger.warning(
                "Age filter dropped %s/%s subjects due to datetime issues "
                "(missing_dob_or_admittime=%s, invalid_or_impossible_age=%s).",
                n_invalid_age,
                len(age_df),
                n_missing_dates,
                n_invalid_age,
            )

        ages = age_df[['subject_id', 'age_years']].dropna(subset=['age_years'])
        if min_age is not None:
            ages = ages[ages['age_years'] >= min_age]
        if max_age is not None:
            ages = ages[ages['age_years'] <= max_age]

        logger.info(f"Age filter: {len(ages)} subjects remain")
        return set(ages['subject_id'].astype(int).tolist())
    
    def _filter_by_admission_count(
        self,
        subject_ids: Set[int],
        min_admissions: Optional[int],
        max_admissions: Optional[int]
    ) -> Set[int]:
        """Filter by number of admissions."""
        admission_counts = self.admissions[
            self.admissions['subject_id'].isin(subject_ids)
        ].groupby('subject_id').size()
        
        if min_admissions is not None:
            admission_counts = admission_counts[admission_counts >= min_admissions]
        if max_admissions is not None:
            admission_counts = admission_counts[admission_counts <= max_admissions]
        
        logger.info(f"Admission count filter: {len(admission_counts)} subjects remain")
        return set(admission_counts.index)
    
    def _filter_by_admission_type(
        self,
        subject_ids: Set[int],
        admission_types: List[str]
    ) -> Set[int]:
        """Filter by admission type."""
        valid_admissions = self.admissions[
            (self.admissions['subject_id'].isin(subject_ids)) &
            (self.admissions['admission_type'].isin(admission_types))
        ]
        
        filtered = set(valid_admissions['subject_id'].unique())
        logger.info(f"Admission type filter: {len(filtered)} subjects remain")
        return filtered
    
    def _filter_by_lab_values(
        self,
        subject_ids: Set[int],
        lab_requirements: Dict[str, Tuple[float, float]]
    ) -> Set[int]:
        """Filter by lab value requirements."""
        # This is a simplified version - in practice, you'd need to map
        # lab names to MIMIC item IDs
        logger.info("Lab value filtering not fully implemented")
        return subject_ids
    
    def _exclude_pregnancy(self, subject_ids: Set[int]) -> Set[int]:
        """Exclude pregnancy-related admissions."""
        pregnancy_codes = ['V22', 'V23', 'V24', '630', '631', '632', '633', 
                          '634', '635', '636', '637', '638', '639', '640', 
                          '641', '642', '643', '644', '645', '646', '647', 
                          '648', '649', '650', '651', '652', '653', '654', 
                          '655', '656', '657', '658', '659']
        
        pregnancy_patients = set()
        for code in pregnancy_codes:
            matches = self.diagnoses[
                self.diagnoses['icd9_code'].str.startswith(code, na=False)
            ]
            pregnancy_patients.update(matches['subject_id'].unique())
        
        filtered = subject_ids - pregnancy_patients
        logger.info(f"Pregnancy exclusion: {len(filtered)} subjects remain")
        return filtered
    
    def _exclude_pediatric(self, subject_ids: Set[int]) -> Set[int]:
        """Exclude pediatric patients (< 18 years)."""
        return self._filter_by_age(subject_ids, min_age=18, max_age=None)
    
    def _exclude_age_over(self, subject_ids: Set[int], max_age: float) -> Set[int]:
        """Exclude patients over specified age."""
        return self._filter_by_age(subject_ids, min_age=None, max_age=max_age)
    
    def _exclude_poor_data_quality(
        self,
        subject_ids: Set[int],
        threshold: float
    ) -> Set[int]:
        """Exclude patients with high missing data rate."""
        # Simplified implementation
        logger.info("Data quality exclusion not fully implemented")
        return subject_ids
    
    def _exclude_short_los(self, subject_ids: Set[int], min_days: float) -> Set[int]:
        """Exclude admissions with short length of stay."""
        cohort_admissions = self.admissions[
            self.admissions['subject_id'].isin(subject_ids)
        ].copy()
        
        cohort_admissions['los_days'] = (
            cohort_admissions['dischtime'] - cohort_admissions['admittime']
        ).dt.total_seconds() / (24 * 3600)
        
        valid_subjects = cohort_admissions[
            cohort_admissions['los_days'] >= min_days
        ]['subject_id'].unique()
        
        filtered = subject_ids & set(valid_subjects)
        logger.info(f"LOS exclusion: {len(filtered)} subjects remain")
        return filtered
    
    def _exclude_died_in_hospital(self, subject_ids: Set[int]) -> Set[int]:
        """Exclude patients who died during hospitalization."""
        died_in_hospital = self.admissions[
            (self.admissions['subject_id'].isin(subject_ids)) &
            (self.admissions['hospital_expire_flag'] == 1)
        ]['subject_id'].unique()
        
        filtered = subject_ids - set(died_in_hospital)
        logger.info(f"In-hospital mortality exclusion: {len(filtered)} subjects remain")
        return filtered
    
    def _record_selection(
        self,
        step: str,
        n_subjects: int,
        cohort_name: str
    ) -> None:
        """Record a cohort selection step."""
        self.selection_history.append({
            'step': step,
            'n_subjects': n_subjects,
            'cohort_name': cohort_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })


if __name__ == '__main__':
    # Example usage
    print("=== CohortBuilder Example ===\n")
    
    # Note: This example requires actual MIMIC-III data
    # Here we show the API usage pattern
    
    print("Usage example:")
    print("""
    from src.data.mimic_loader import MIMICLoader
    from src.data.cohort_builder import CohortBuilder
    
    # Load data
    loader = MIMICLoader('data/raw/mimic-iii')
    patients = loader.load_patients()
    admissions = loader.load_admissions()
    diagnoses = loader.load_diagnoses_icd()
    
    # Build cohort
    builder = CohortBuilder(patients, admissions, diagnoses)
    
    # Define diabetes cohort
    diabetes_patients = builder.define_diabetes_cohort()
    
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
        exclude_died_in_hospital=True
    )
    
    # Get statistics
    stats = builder.get_cohort_statistics(final_cohort)
    print(f"Final cohort: {len(final_cohort)} patients")
    print(f"Mean age: {stats['age']['mean']:.1f} years")
    
    # Export definition
    builder.export_cohort_definition(final_cohort, 'cohort_definition.txt')
    """)
