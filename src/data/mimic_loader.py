"""
MIMIC-III Data Loader Module

This module provides functionality for loading MIMIC-III clinical database tables
with efficient memory management, validation, and comprehensive logging.

MIMIC-III Tables Supported:
- PATIENTS: Patient demographics and mortality
- ADMISSIONS: Hospital admission details
- LABEVENTS: Laboratory test results
- PRESCRIPTIONS: Medication prescriptions
- CHARTEVENTS: Vital signs and clinical observations (use with caution - large file)
- DIAGNOSES_ICD: ICD-9 diagnosis codes
- ICUSTAYS: ICU stay information
- PROCEDURES_ICD: ICD-9 procedure codes

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MIMICLoader:
    """
    Load and manage MIMIC-III clinical database tables.
    
    This class provides methods to load various MIMIC-III tables with:
    - Efficient chunked reading for large files
    - Data validation and quality checks
    - Progress logging
    - Memory-efficient data types
    - Automatic column name normalization (UPPERCASE -> lowercase)
    
    Note: MIMIC-III CSV files use UPPERCASE column names (e.g., SUBJECT_ID),
    but this loader automatically normalizes them to lowercase (e.g., subject_id)
    for consistency with pandas conventions.
    
    Attributes:
        data_dir (Path): Directory containing MIMIC-III CSV files
        chunk_size (int): Number of rows to read per chunk for large files
        cache (Dict): In-memory cache of loaded tables
    
    Example:
        >>> loader = MIMICLoader(data_dir='data/raw/mimic-iii')
        >>> patients = loader.load_patients()
        >>> admissions = loader.load_admissions()
        >>> labs = loader.load_lab_events(subject_ids=[1, 2, 3])
    """
    
    # Table schemas for validation
    # Note: These are lowercase - MIMIC-III CSV headers are UPPERCASE but we normalize on load
    EXPECTED_COLUMNS = {
        'PATIENTS': ['subject_id', 'gender', 'dob', 'dod'],
        'ADMISSIONS': ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type'],
        'LABEVENTS': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum'],
        'PRESCRIPTIONS': ['subject_id', 'hadm_id', 'drug', 'dose_val_rx', 'dose_unit_rx'],
        'CHARTEVENTS': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum'],
        'DIAGNOSES_ICD': ['subject_id', 'hadm_id', 'icd9_code', 'seq_num'],
        'ICUSTAYS': ['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime'],
        'PROCEDURES_ICD': ['subject_id', 'hadm_id', 'icd9_code', 'seq_num']
    }
    
    # Large files that should use chunked reading (> 1GB typically)
    LARGE_FILES = ['CHARTEVENTS', 'LABEVENTS']
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        chunk_size: int = 100000,
        use_cache: bool = True
    ):
        """
        Initialize MIMIC-III data loader.
        
        Args:
            data_dir: Path to directory containing MIMIC-III CSV files
            chunk_size: Number of rows per chunk for large files
            use_cache: Whether to cache loaded tables in memory
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"Initialized MIMICLoader with data_dir: {self.data_dir}")
    
    def load_table(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        use_chunked: bool = False
    ) -> pd.DataFrame:
        """
        Load a MIMIC-III table from CSV.
        
        Args:
            table_name: Name of the table (e.g., 'PATIENTS', 'ADMISSIONS')
            columns: Specific columns to load (None = all columns)
            nrows: Number of rows to load (None = all rows)
            use_chunked: Force chunked reading for large files
            
        Returns:
            DataFrame containing the table data
            
        Raises:
            FileNotFoundError: If table CSV file doesn't exist
            ValueError: If table has invalid format
        """
        # Check cache
        cache_key = f"{table_name}_{columns}_{nrows}"
        if self.use_cache and cache_key in self.cache:
            logger.info(f"Loading {table_name} from cache")
            return self.cache[cache_key].copy()
        
        # Find CSV file (case-insensitive)
        csv_files = list(self.data_dir.glob(f"{table_name}.csv"))
        if not csv_files:
            csv_files = list(self.data_dir.glob(f"{table_name.lower()}.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"Table {table_name} not found in {self.data_dir}"
            )
        
        csv_path = csv_files[0]
        logger.info(f"Loading {table_name} from {csv_path}")
        
        # Determine if chunked reading needed
        use_chunked = use_chunked or (table_name.upper() in self.LARGE_FILES)
        
        try:
            if use_chunked and nrows is None:
                df = self._load_chunked(csv_path, columns)
            else:
                df = pd.read_csv(
                    csv_path,
                    usecols=columns,
                    nrows=nrows,
                    low_memory=False
                )
            
            # CRITICAL: Normalize column names to lowercase (MIMIC-III uses UPPERCASE)
            df.columns = df.columns.str.lower()
            
            # Validate columns
            if table_name.upper() in self.EXPECTED_COLUMNS:
                self._validate_columns(df, table_name)
            
            # Optimize dtypes
            df = self._optimize_dtypes(df)
            
            # Cache if enabled
            if self.use_cache:
                self.cache[cache_key] = df.copy()
            
            logger.info(f"Loaded {table_name}: {len(df):,} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {table_name}: {str(e)}")
            raise
    
    def load_patients(self) -> pd.DataFrame:
        """
        Load PATIENTS table with demographics and mortality.
        
        Returns:
            DataFrame with columns: subject_id, gender, dob, dod, dod_hosp, dod_ssn, expire_flag
            
        Note:
            - dob (date of birth) is privacy-shifted by ~300 years
            - Use utility functions to calculate actual ages
            - dod = date of death (if applicable)
        """
        logger.info("Loading PATIENTS table")
        df = self.load_table('PATIENTS')
        
        # Parse dates
        date_columns = ['dob', 'dod', 'dod_hosp', 'dod_ssn']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def load_admissions(self) -> pd.DataFrame:
        """
        Load ADMISSIONS table with hospital admission details.
        
        Returns:
            DataFrame with admission information including:
            - subject_id, hadm_id: Patient and admission identifiers
            - admittime, dischtime: Admission and discharge timestamps
            - admission_type, admission_location: Type and location of admission
            - discharge_location: Where patient was discharged to
            - insurance, language, religion, marital_status: Demographics
            - ethnicity: Patient ethnicity
            - diagnosis: Admission diagnosis
        """
        logger.info("Loading ADMISSIONS table")
        df = self.load_table('ADMISSIONS')
        
        # Parse timestamps
        time_columns = ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def load_lab_events(
        self,
        subject_ids: Optional[List[int]] = None,
        hadm_ids: Optional[List[int]] = None,
        item_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Load LABEVENTS table with laboratory test results.
        
        Args:
            subject_ids: Filter by specific patient IDs
            hadm_ids: Filter by specific admission IDs
            item_ids: Filter by specific lab test item IDs
            
        Returns:
            DataFrame with lab results including:
            - subject_id, hadm_id: Patient and admission identifiers
            - itemid: Lab test identifier (use D_LABITEMS for mapping)
            - charttime: Time of measurement
            - value: Result value (string)
            - valuenum: Numeric result value
            - valueuom: Unit of measurement
            - flag: Abnormal flag
            
        Note:
            This is a LARGE file. Filtering by IDs is recommended.
        """
        logger.info("Loading LABEVENTS table")
        
        # Load in chunks due to file size
        chunks = []
        csv_path = self.data_dir / 'LABEVENTS.csv'
        
        if not csv_path.exists():
            csv_path = self.data_dir / 'labevents.csv'
        
        if not csv_path.exists():
            raise FileNotFoundError(f"LABEVENTS not found in {self.data_dir}")
        
        logger.info(f"Reading LABEVENTS in chunks (chunk_size={self.chunk_size})")
        
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=self.chunk_size, low_memory=False)):
            # Normalize column names to lowercase
            chunk.columns = chunk.columns.str.lower()
            
            # Apply filters
            if subject_ids is not None:
                chunk = chunk[chunk['subject_id'].isin(subject_ids)]
            if hadm_ids is not None:
                chunk = chunk[chunk['hadm_id'].isin(hadm_ids)]
            if item_ids is not None:
                chunk = chunk[chunk['itemid'].isin(item_ids)]
            
            if len(chunk) > 0:
                chunks.append(chunk)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {(i+1) * self.chunk_size:,} rows...")
        
        if not chunks:
            logger.warning("No lab events found matching filters")
            return pd.DataFrame()
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Parse charttime
        df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
        
        logger.info(f"Loaded {len(df):,} lab events")
        return df
    
    def load_prescriptions(
        self,
        subject_ids: Optional[List[int]] = None,
        hadm_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Load PRESCRIPTIONS table with medication orders.
        
        Args:
            subject_ids: Filter by specific patient IDs
            hadm_ids: Filter by specific admission IDs
            
        Returns:
            DataFrame with prescription information including:
            - subject_id, hadm_id: Patient and admission identifiers
            - startdate, enddate: Start and end dates of prescription
            - drug_type: Type of medication (MAIN, BASE, ADDITIVE)
            - drug: Medication name
            - drug_name_poe, drug_name_generic: Generic drug names
            - formulary_drug_cd: Formulary code
            - gsn: Generic Sequence Number
            - ndc: National Drug Code
            - prod_strength: Product strength
            - dose_val_rx, dose_unit_rx: Prescribed dose and unit
            - form_val_disp, form_unit_disp: Dispensed form and unit
            - route: Administration route
        """
        logger.info("Loading PRESCRIPTIONS table")
        
        df = self.load_table('PRESCRIPTIONS')
        
        # Apply filters
        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)]
        if hadm_ids is not None:
            df = df[df['hadm_id'].isin(hadm_ids)]
        
        # Parse dates
        date_columns = ['startdate', 'enddate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        logger.info(f"Loaded {len(df):,} prescriptions")
        return df
    
    def load_diagnoses_icd(
        self,
        subject_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Load DIAGNOSES_ICD table with ICD-9 diagnosis codes.
        
        Args:
            subject_ids: Filter by specific patient IDs
            
        Returns:
            DataFrame with diagnosis codes including:
            - subject_id, hadm_id: Patient and admission identifiers
            - seq_num: Diagnosis sequence number (1 = primary diagnosis)
            - icd9_code: ICD-9 diagnosis code
        """
        logger.info("Loading DIAGNOSES_ICD table")
        
        df = self.load_table('DIAGNOSES_ICD')
        
        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)]
        
        logger.info(f"Loaded {len(df):,} diagnoses")
        return df
    
    def load_icustays(
        self,
        subject_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Load ICUSTAYS table with ICU stay information.
        
        Args:
            subject_ids: Filter by specific patient IDs
            
        Returns:
            DataFrame with ICU stay details including:
            - subject_id, hadm_id, icustay_id: Identifiers
            - dbsource: Database source
            - first_careunit, last_careunit: ICU care units
            - first_wardid, last_wardid: Ward identifiers
            - intime, outtime: ICU in and out times
            - los: Length of stay (days)
        """
        logger.info("Loading ICUSTAYS table")
        
        df = self.load_table('ICUSTAYS')
        
        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)]
        
        # Parse timestamps
        time_columns = ['intime', 'outtime']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        logger.info(f"Loaded {len(df):,} ICU stays")
        return df
    
    def load_procedures_icd(
        self,
        subject_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Load PROCEDURES_ICD table with ICD-9 procedure codes.
        
        Args:
            subject_ids: Filter by specific patient IDs
            
        Returns:
            DataFrame with procedure codes including:
            - subject_id, hadm_id: Patient and admission identifiers
            - seq_num: Procedure sequence number
            - icd9_code: ICD-9 procedure code
        """
        logger.info("Loading PROCEDURES_ICD table")
        
        df = self.load_table('PROCEDURES_ICD')
        
        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)]
        
        logger.info(f"Loaded {len(df):,} procedures")
        return df
    
    def get_patient_complete_record(
        self,
        subject_id: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Load complete medical record for a single patient.
        
        Args:
            subject_id: Patient identifier
            
        Returns:
            Dictionary containing all available data tables for the patient:
            - 'demographics': Patient demographics
            - 'admissions': All hospital admissions
            - 'lab_events': All lab results
            - 'prescriptions': All medication prescriptions
            - 'diagnoses': All diagnoses
            - 'icu_stays': All ICU stays
            - 'procedures': All procedures
        """
        logger.info(f"Loading complete record for patient {subject_id}")
        
        record = {}
        
        try:
            # Demographics
            patients = self.load_patients()
            record['demographics'] = patients[patients['subject_id'] == subject_id]
            
            # Admissions
            admissions = self.load_admissions()
            patient_admissions = admissions[admissions['subject_id'] == subject_id]
            record['admissions'] = patient_admissions
            
            if len(patient_admissions) > 0:
                hadm_ids = patient_admissions['hadm_id'].unique().tolist()
                
                # Lab events
                record['lab_events'] = self.load_lab_events(subject_ids=[subject_id])
                
                # Prescriptions
                record['prescriptions'] = self.load_prescriptions(subject_ids=[subject_id])
                
                # Diagnoses
                record['diagnoses'] = self.load_diagnoses_icd(subject_ids=[subject_id])
                
                # ICU stays
                record['icu_stays'] = self.load_icustays(subject_ids=[subject_id])
                
                # Procedures
                record['procedures'] = self.load_procedures_icd(subject_ids=[subject_id])
            
            logger.info(f"Loaded complete record for patient {subject_id}")
            return record
            
        except Exception as e:
            logger.error(f"Error loading patient {subject_id} record: {str(e)}")
            raise
    
    def _load_chunked(
        self,
        csv_path: Path,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load large CSV file in chunks."""
        chunks = []
        total_rows = 0
        
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=self.chunk_size, usecols=columns, low_memory=False)):
            # Normalize column names to lowercase
            chunk.columns = chunk.columns.str.lower()
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Loaded {total_rows:,} rows...")
        
        return pd.concat(chunks, ignore_index=True)
    
    def _validate_columns(self, df: pd.DataFrame, table_name: str) -> None:
        """Validate that DataFrame has expected columns."""
        expected = set(self.EXPECTED_COLUMNS[table_name.upper()])
        actual = set(df.columns)
        
        missing = expected - actual
        if missing:
            logger.warning(f"{table_name} missing expected columns: {missing}")
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage."""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if low cardinality
                num_unique = df[col].nunique()
                if num_unique / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
            
            elif col_type == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            elif col_type == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def clear_cache(self) -> None:
        """Clear the in-memory table cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached tables."""
        return {
            table: len(df)
            for table, df in self.cache.items()
        }


if __name__ == '__main__':
    # Example usage
    import sys
    
    # Example: Load MIMIC-III data
    try:
        # Initialize loader
        loader = MIMICLoader(data_dir='/Users/andy/Documents/Coding/Learning/mtp/healthcare_rl/data/mimic3')
        
        # Load patients table
        print("\n=== Loading Patients ===")
        patients = loader.load_patients()
        print(f"Loaded {len(patients):,} patients")
        print(patients.head())
        
        # Load admissions
        print("\n=== Loading Admissions ===")
        admissions = loader.load_admissions()
        print(f"Loaded {len(admissions):,} admissions")
        print(admissions.head())
        
        # Load lab events for first 100 patients
        print("\n=== Loading Lab Events (sample) ===")
        sample_patients = patients['subject_id'].head(100).tolist()
        labs = loader.load_lab_events(subject_ids=sample_patients)
        print(f"Loaded {len(labs):,} lab events")
        
        # Load complete record for one patient
        print("\n=== Loading Complete Patient Record ===")
        sample_subject_id = patients['subject_id'].iloc[0]
        record = loader.get_patient_complete_record(sample_subject_id)
        
        for table_name, data in record.items():
            print(f"{table_name}: {len(data)} records")
        
        # Cache info
        print("\n=== Cache Info ===")
        print(loader.get_cache_info())
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please specify the correct MIMIC-III data directory")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)