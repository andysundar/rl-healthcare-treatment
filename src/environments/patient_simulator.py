"""
Patient Simulator
=================
Generate diverse patient populations with heterogeneous characteristics.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from .disease_models import (
    BergmanModelParams, 
    BergmanMinimalModel,
    AdherenceModelParams,
    AdherenceDynamicsModel
)


class DiseaseSeverity(Enum):
    """Disease severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class PatientDemographics:
    """Patient demographic information."""
    patient_id: int
    age: int
    gender: str  # 'M', 'F', 'O'
    bmi: float
    ethnicity: str
    socioeconomic_status: str  # 'low', 'medium', 'high'


@dataclass
class PatientClinical:
    """Patient clinical characteristics."""
    disease_type: str
    disease_severity: DiseaseSeverity
    comorbidities: List[str] = field(default_factory=list)
    baseline_adherence: float = 0.7
    medication_sensitivity: float = 1.0  # Response multiplier


@dataclass
class DiabetesPatient:
    """Complete diabetes patient profile."""
    demographics: PatientDemographics
    clinical: PatientClinical
    bergman_params: BergmanModelParams
    adherence_params: AdherenceModelParams
    initial_glucose: float
    initial_insulin: float
    initial_X: float = 0.0


@dataclass
class AdherencePatient:
    """Complete adherence-focused patient profile."""
    demographics: PatientDemographics
    clinical: PatientClinical
    adherence_params: AdherenceModelParams
    initial_adherence: float
    initial_satisfaction: float
    initial_side_effects: float


class PatientSimulator:
    """
    Generate diverse patient populations for healthcare RL experiments.
    
    Creates realistic patient cohorts with:
    - Demographic diversity (age, gender, BMI, ethnicity)
    - Disease heterogeneity (severity, comorbidities)
    - Physiological variability (response to treatment)
    - Behavioral differences (adherence patterns)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize patient simulator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.next_patient_id = 0
    
    def generate_diabetes_cohort(self,
                                 n_patients: int,
                                 severity_distribution: Optional[Dict[DiseaseSeverity, float]] = None,
                                 demographic_distributions: Optional[Dict[str, Any]] = None
                                 ) -> List[DiabetesPatient]:
        """
        Generate a cohort of diabetes patients.
        
        Args:
            n_patients: Number of patients to generate
            severity_distribution: Distribution of disease severity
            demographic_distributions: Demographic parameter distributions
            
        Returns:
            cohort: List of DiabetesPatient objects
        """
        # Default severity distribution
        if severity_distribution is None:
            severity_distribution = {
                DiseaseSeverity.MILD: 0.3,
                DiseaseSeverity.MODERATE: 0.5,
                DiseaseSeverity.SEVERE: 0.2
            }
        
        # Default demographic distributions
        if demographic_distributions is None:
            demographic_distributions = {
                'age': {'mean': 55, 'std': 15, 'min': 18, 'max': 90},
                'bmi': {'mean': 28, 'std': 5, 'min': 18, 'max': 50},
                'gender': {'M': 0.48, 'F': 0.50, 'O': 0.02},
                'ethnicity': {
                    'Caucasian': 0.4,
                    'African American': 0.2,
                    'Hispanic': 0.2,
                    'Asian': 0.15,
                    'Other': 0.05
                },
                'ses': {'low': 0.3, 'medium': 0.5, 'high': 0.2}
            }
        
        cohort = []
        
        for _ in range(n_patients):
            # Sample severity
            severity = self._sample_categorical(severity_distribution)
            
            # Generate patient
            patient = self._generate_diabetes_patient(
                severity, 
                demographic_distributions
            )
            cohort.append(patient)
        
        return cohort
    
    def generate_adherence_cohort(self,
                                  n_patients: int,
                                  baseline_adherence_distribution: Optional[Dict[str, float]] = None,
                                  demographic_distributions: Optional[Dict[str, Any]] = None
                                  ) -> List[AdherencePatient]:
        """
        Generate a cohort focused on medication adherence.
        
        Args:
            n_patients: Number of patients to generate
            baseline_adherence_distribution: Distribution of baseline adherence
            demographic_distributions: Demographic parameter distributions
            
        Returns:
            cohort: List of AdherencePatient objects
        """
        # Default adherence distribution
        if baseline_adherence_distribution is None:
            baseline_adherence_distribution = {
                'mean': 0.7,
                'std': 0.15,
                'min': 0.2,
                'max': 0.95
            }
        
        if demographic_distributions is None:
            demographic_distributions = {
                'age': {'mean': 55, 'std': 15, 'min': 18, 'max': 90},
                'bmi': {'mean': 28, 'std': 5, 'min': 18, 'max': 50},
                'gender': {'M': 0.48, 'F': 0.50, 'O': 0.02},
                'ethnicity': {
                    'Caucasian': 0.4,
                    'African American': 0.2,
                    'Hispanic': 0.2,
                    'Asian': 0.15,
                    'Other': 0.05
                },
                'ses': {'low': 0.3, 'medium': 0.5, 'high': 0.2}
            }
        
        cohort = []
        
        for _ in range(n_patients):
            patient = self._generate_adherence_patient(
                baseline_adherence_distribution,
                demographic_distributions
            )
            cohort.append(patient)
        
        return cohort
    
    def _generate_diabetes_patient(self,
                                   severity: DiseaseSeverity,
                                   demo_dist: Dict[str, Any]) -> DiabetesPatient:
        """Generate a single diabetes patient."""
        # Demographics
        demographics = self._generate_demographics(demo_dist)
        
        # Clinical characteristics
        clinical = PatientClinical(
            disease_type='diabetes_type2',
            disease_severity=severity,
            comorbidities=self._sample_comorbidities(demographics.age, demographics.bmi),
            baseline_adherence=self._sample_baseline_adherence(demographics),
            medication_sensitivity=self._sample_medication_sensitivity(severity)
        )
        
        # Bergman model parameters (disease-specific)
        bergman_params = self._generate_bergman_params(severity, demographics)
        
        # Adherence parameters
        adherence_params = self._generate_adherence_params(clinical.baseline_adherence)
        
        # Initial state
        initial_glucose = self._sample_initial_glucose(severity)
        initial_insulin = self._sample_initial_insulin(severity)
        initial_X = self.rng.normal(0, 2)
        
        return DiabetesPatient(
            demographics=demographics,
            clinical=clinical,
            bergman_params=bergman_params,
            adherence_params=adherence_params,
            initial_glucose=initial_glucose,
            initial_insulin=initial_insulin,
            initial_X=initial_X
        )
    
    def _generate_adherence_patient(self,
                                   adh_dist: Dict[str, float],
                                   demo_dist: Dict[str, Any]) -> AdherencePatient:
        """Generate a single adherence-focused patient."""
        # Demographics
        demographics = self._generate_demographics(demo_dist)
        
        # Sample baseline adherence
        baseline = self._sample_truncated_normal(
            adh_dist['mean'],
            adh_dist['std'],
            adh_dist['min'],
            adh_dist['max']
        )
        
        # Clinical characteristics
        clinical = PatientClinical(
            disease_type='chronic_condition',
            disease_severity=DiseaseSeverity.MODERATE,
            baseline_adherence=baseline
        )
        
        # Adherence parameters
        adherence_params = self._generate_adherence_params(baseline)
        
        # Initial state
        initial_adherence = baseline + self.rng.normal(0, 0.1)
        initial_adherence = np.clip(initial_adherence, 0, 1)
        
        initial_satisfaction = self.rng.beta(7, 3)  # Skewed toward higher satisfaction
        initial_side_effects = self.rng.beta(2, 8)  # Skewed toward lower side effects
        
        return AdherencePatient(
            demographics=demographics,
            clinical=clinical,
            adherence_params=adherence_params,
            initial_adherence=initial_adherence,
            initial_satisfaction=initial_satisfaction,
            initial_side_effects=initial_side_effects
        )
    
    def _generate_demographics(self, demo_dist: Dict[str, Any]) -> PatientDemographics:
        """Generate patient demographics."""
        patient_id = self.next_patient_id
        self.next_patient_id += 1
        
        # Age
        age = int(self._sample_truncated_normal(
            demo_dist['age']['mean'],
            demo_dist['age']['std'],
            demo_dist['age']['min'],
            demo_dist['age']['max']
        ))
        
        # Gender
        gender = self._sample_categorical(demo_dist['gender'])
        
        # BMI
        bmi = self._sample_truncated_normal(
            demo_dist['bmi']['mean'],
            demo_dist['bmi']['std'],
            demo_dist['bmi']['min'],
            demo_dist['bmi']['max']
        )
        
        # Ethnicity
        ethnicity = self._sample_categorical(demo_dist['ethnicity'])
        
        # Socioeconomic status
        ses = self._sample_categorical(demo_dist['ses'])
        
        return PatientDemographics(
            patient_id=patient_id,
            age=age,
            gender=gender,
            bmi=bmi,
            ethnicity=ethnicity,
            socioeconomic_status=ses
        )
    
    def _generate_bergman_params(self,
                                 severity: DiseaseSeverity,
                                 demographics: PatientDemographics) -> BergmanModelParams:
        """Generate Bergman model parameters based on severity and demographics."""
        # Base parameters
        base = BergmanModelParams()
        
        # Adjust for severity
        severity_multipliers = {
            DiseaseSeverity.MILD: {'p1': 1.0, 'p3': 1.2, 'Gb': 0.9},
            DiseaseSeverity.MODERATE: {'p1': 1.0, 'p3': 1.0, 'Gb': 1.0},
            DiseaseSeverity.SEVERE: {'p1': 0.8, 'p3': 0.7, 'Gb': 1.2}
        }
        
        mult = severity_multipliers[severity]
        
        # Adjust for age (insulin sensitivity decreases with age)
        age_factor = 1.0 - (demographics.age - 50) * 0.005
        age_factor = np.clip(age_factor, 0.7, 1.2)
        
        # Adjust for BMI (higher BMI → lower insulin sensitivity)
        bmi_factor = 1.0 - (demographics.bmi - 25) * 0.02
        bmi_factor = np.clip(bmi_factor, 0.6, 1.2)
        
        # Sample with variability
        variability = 0.15
        
        return BergmanModelParams(
            p1=base.p1 * mult['p1'] * self.rng.lognormal(0, variability),
            p2=base.p2 * self.rng.lognormal(0, variability),
            p3=base.p3 * mult['p3'] * age_factor * bmi_factor * self.rng.lognormal(0, variability),
            n=base.n * self.rng.lognormal(0, variability),
            Gb=base.Gb * mult['Gb'],
            Ib=base.Ib * self.rng.lognormal(0, variability * 0.5),
            Vg=base.Vg * self.rng.lognormal(0, variability * 0.3),
            basal_insulin=base.basal_insulin * self.rng.lognormal(0, variability)
        )
    
    def _generate_adherence_params(self, baseline_adherence: float) -> AdherenceModelParams:
        """Generate adherence model parameters."""
        base = AdherenceModelParams(baseline=baseline_adherence)
        
        # Sample with moderate variability
        variability = 0.2
        
        return AdherenceModelParams(
            alpha=np.clip(self.rng.normal(base.alpha, base.alpha * variability), 0, 1),
            beta=max(0, self.rng.normal(base.beta, base.beta * variability)),
            gamma=max(0, self.rng.normal(base.gamma, base.gamma * variability)),
            delta=max(0, self.rng.normal(base.delta, base.delta * variability)),
            noise_std=max(0.01, self.rng.normal(base.noise_std, base.noise_std * variability)),
            baseline=baseline_adherence
        )
    
    def _sample_comorbidities(self, age: int, bmi: float) -> List[str]:
        """Sample comorbidities based on age and BMI."""
        comorbidities = []
        
        # Probabilities increase with age and BMI
        age_factor = (age - 40) / 50  # Normalized age risk
        bmi_factor = (bmi - 25) / 15   # Normalized BMI risk
        
        # Common comorbidities with their base probabilities
        conditions = {
            'hypertension': 0.3 + 0.2 * age_factor + 0.15 * bmi_factor,
            'hyperlipidemia': 0.25 + 0.15 * age_factor + 0.1 * bmi_factor,
            'cardiovascular_disease': 0.1 + 0.2 * age_factor,
            'chronic_kidney_disease': 0.05 + 0.1 * age_factor,
            'neuropathy': 0.15 + 0.1 * age_factor,
            'retinopathy': 0.1 + 0.08 * age_factor
        }
        
        for condition, prob in conditions.items():
            if self.rng.random() < np.clip(prob, 0, 0.8):
                comorbidities.append(condition)
        
        return comorbidities
    
    def _sample_baseline_adherence(self, demographics: PatientDemographics) -> float:
        """Sample baseline adherence based on demographics."""
        # Base adherence
        base = 0.7
        
        # Adjust for socioeconomic status
        ses_adjustment = {
            'low': -0.1,
            'medium': 0.0,
            'high': 0.1
        }
        
        # Adjust for age (middle-aged tend to have better adherence)
        age_adjustment = -0.05 * abs(demographics.age - 55) / 20
        
        adherence = base + ses_adjustment[demographics.socioeconomic_status] + age_adjustment
        adherence += self.rng.normal(0, 0.1)
        
        return np.clip(adherence, 0.2, 0.95)
    
    def _sample_medication_sensitivity(self, severity: DiseaseSeverity) -> float:
        """Sample medication sensitivity based on disease severity."""
        sensitivity_map = {
            DiseaseSeverity.MILD: self.rng.normal(1.2, 0.15),
            DiseaseSeverity.MODERATE: self.rng.normal(1.0, 0.1),
            DiseaseSeverity.SEVERE: self.rng.normal(0.8, 0.15)
        }
        
        return max(0.5, sensitivity_map[severity])
    
    def _sample_initial_glucose(self, severity: DiseaseSeverity) -> float:
        """Sample initial glucose level."""
        glucose_map = {
            DiseaseSeverity.MILD: self.rng.normal(120, 15),
            DiseaseSeverity.MODERATE: self.rng.normal(150, 20),
            DiseaseSeverity.SEVERE: self.rng.normal(200, 30)
        }
        
        return np.clip(glucose_map[severity], 80, 300)
    
    def _sample_initial_insulin(self, severity: DiseaseSeverity) -> float:
        """Sample initial insulin level."""
        insulin_map = {
            DiseaseSeverity.MILD: self.rng.normal(15, 3),
            DiseaseSeverity.MODERATE: self.rng.normal(20, 5),
            DiseaseSeverity.SEVERE: self.rng.normal(25, 7)
        }
        
        return max(5, insulin_map[severity])
    
    def _sample_categorical(self, distribution: Dict) -> Any:
        """Sample from categorical distribution."""
        items = list(distribution.keys())
        probs = list(distribution.values())
        probs = np.array(probs) / sum(probs)  # Normalize
        
        return self.rng.choice(items, p=probs)
    
    def _sample_truncated_normal(self, mean: float, std: float, 
                                 min_val: float, max_val: float) -> float:
        """Sample from truncated normal distribution."""
        while True:
            sample = self.rng.normal(mean, std)
            if min_val <= sample <= max_val:
                return sample
    
    def sample_patient_trajectory(self,
                                  patient: DiabetesPatient,
                                  policy: Any,
                                  horizon: int = 90,
                                  env_class: Any = None) -> Dict[str, Any]:
        """
        Sample a trajectory for a patient under a given policy.
        
        Args:
            patient: Patient object
            policy: Policy object with get_action(state) method
            horizon: Episode length
            env_class: Environment class to use (must accept patient parameters)
            
        Returns:
            trajectory: Dictionary with states, actions, rewards, etc.
        """
        if env_class is None:
            raise ValueError("env_class must be provided")
        
        # Create environment with patient parameters
        # (This would need custom environment initialization)
        env = env_class(patient_id=patient.demographics.patient_id)
        
        states = []
        actions = []
        rewards = []
        
        state, _ = env.reset()
        states.append(state)
        
        for _ in range(horizon):
            action = policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
            
            if terminated or truncated:
                break
            
            state = next_state
        
        return {
            'patient_id': patient.demographics.patient_id,
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'total_reward': sum(rewards),
            'episode_length': len(rewards)
        }
    
    def get_cohort_statistics(self, cohort: List) -> Dict[str, Any]:
        """
        Compute statistics for a patient cohort.
        
        Args:
            cohort: List of patient objects
            
        Returns:
            stats: Dictionary of cohort statistics
        """
        if not cohort:
            return {}
        
        # Extract demographics
        ages = [p.demographics.age for p in cohort]
        bmis = [p.demographics.bmi for p in cohort]
        genders = [p.demographics.gender for p in cohort]
        
        # Extract clinical
        severities = [p.clinical.disease_severity.value for p in cohort 
                     if hasattr(p.clinical, 'disease_severity')]
        baseline_adherences = [p.clinical.baseline_adherence for p in cohort]
        
        stats = {
            'n_patients': len(cohort),
            'age_mean': np.mean(ages),
            'age_std': np.std(ages),
            'bmi_mean': np.mean(bmis),
            'bmi_std': np.std(bmis),
            'gender_distribution': {
                g: genders.count(g) / len(genders) 
                for g in set(genders)
            },
            'adherence_mean': np.mean(baseline_adherences),
            'adherence_std': np.std(baseline_adherences)
        }
        
        if severities:
            stats['severity_distribution'] = {
                s: severities.count(s) / len(severities)
                for s in set(severities)
            }
        
        return stats
