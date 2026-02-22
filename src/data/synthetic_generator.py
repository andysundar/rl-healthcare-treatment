"""
Synthetic Patient Data Generator for Healthcare RL

This module generates synthetic patient trajectories for testing and development:
- Bergman minimal model for glucose-insulin dynamics
- Stochastic medication adherence modeling
- Patient heterogeneity simulation
- Realistic treatment response patterns
- Configurable population parameters

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PatientParameters:
    """Parameters for a synthetic patient."""
    patient_id: str
    age: float
    gender: str
    weight: float  # kg
    height: float  # cm
    
    # Bergman model parameters
    p1: float = 0.028  # Glucose effectiveness (1/min)
    p2: float = 0.025  # Insulin action decay (1/min)
    p3: float = 0.000013  # Insulin sensitivity (1/(min*μU/mL))
    Gb: float = 95.0  # Basal glucose (mg/dL)
    Ib: float = 10.0  # Basal insulin (μU/mL)
    
    # Disease severity
    insulin_resistance: float = 1.0  # 1.0 = normal, >1 = resistant
    beta_cell_function: float = 1.0  # 1.0 = normal, <1 = impaired
    
    # Adherence profile
    adherence_baseline: float = 0.7  # Base adherence probability
    adherence_variability: float = 0.2  # Day-to-day variability
    response_to_reminders: float = 0.15  # Boost from reminders
    
    # Treatment response heterogeneity
    medication_sensitivity: float = 1.0  # Response to medications
    lifestyle_impact: float = 1.0  # Impact of diet/exercise
    
    # Random seed for reproducibility
    random_seed: int = 42


@dataclass
class PopulationParameters:
    """Parameters for generating a population of patients."""
    n_patients: int = 1000
    
    # Age distribution
    age_mean: float = 55.0
    age_std: float = 15.0
    age_min: float = 18.0
    age_max: float = 90.0
    
    # Gender distribution
    fraction_male: float = 0.5
    
    # Weight distribution (kg)
    weight_mean: float = 80.0
    weight_std: float = 15.0
    
    # Height distribution (cm)
    height_mean: float = 170.0
    height_std: float = 10.0
    
    # Disease severity distribution
    insulin_resistance_mean: float = 1.5
    insulin_resistance_std: float = 0.5
    beta_cell_function_mean: float = 0.7
    beta_cell_function_std: float = 0.2
    
    # Adherence distribution
    adherence_baseline_mean: float = 0.7
    adherence_baseline_std: float = 0.2
    
    random_seed: int = 42


class BergmanMinimalModel:
    """
    Bergman minimal model for glucose-insulin dynamics.
    
    Equations:
        dG/dt = -p1*G - X*(G - Gb) + D(t)
        dX/dt = -p2*X + p3*(I - Ib)
        dI/dt = -n*I + gamma*(G - h)^+ + U(t)
    
    Where:
        G = glucose concentration (mg/dL)
        X = remote insulin effect
        I = insulin concentration (μU/mL)
        D(t) = glucose input (meals)
        U(t) = insulin input (medication/injection)
    """
    
    def __init__(self, patient_params: PatientParameters):
        """Initialize Bergman model with patient parameters."""
        self.params = patient_params
    
    def glucose_insulin_dynamics(
        self,
        state: np.ndarray,
        t: float,
        D: float,
        U: float
    ) -> List[float]:
        """
        Differential equations for glucose-insulin system.
        
        Args:
            state: [G, X, I] - current state
            t: Current time
            D: Dietary glucose input
            U: Insulin input
            
        Returns:
            [dG/dt, dX/dt, dI/dt]
        """
        G, X, I = state
        
        p1 = self.params.p1
        p2 = self.params.p2
        p3 = self.params.p3 / self.params.insulin_resistance
        Gb = self.params.Gb
        Ib = self.params.Ib
        
        # Glucose dynamics
        dG_dt = -p1 * G - X * (G - Gb) + D
        
        # Remote insulin effect
        dX_dt = -p2 * X + p3 * (I - Ib)
        
        # Insulin dynamics (simplified)
        n = 0.2  # Insulin clearance rate
        gamma = 0.01 * self.params.beta_cell_function  # Pancreatic response
        h = 100  # Glucose threshold for insulin secretion
        
        dI_dt = -n * I + gamma * max(G - h, 0) + U
        
        return [dG_dt, dX_dt, dI_dt]
    
    def simulate(
        self,
        initial_state: np.ndarray,
        time_points: np.ndarray,
        meal_schedule: np.ndarray,
        insulin_schedule: np.ndarray
    ) -> np.ndarray:
        """
        Simulate glucose-insulin trajectory.
        
        Args:
            initial_state: [G0, X0, I0] initial conditions
            time_points: Array of time points (minutes)
            meal_schedule: Glucose input at each time point
            insulin_schedule: Insulin input at each time point
            
        Returns:
            Array of shape (len(time_points), 3) with [G, X, I] at each time
        """
        trajectory = np.zeros((len(time_points), 3))
        trajectory[0] = initial_state
        
        for i in range(len(time_points) - 1):
            t_span = [time_points[i], time_points[i + 1]]
            D = meal_schedule[i]
            U = insulin_schedule[i]
            
            # Solve ODE for this time step
            sol = odeint(
                self.glucose_insulin_dynamics,
                trajectory[i],
                t_span,
                args=(D, U)
            )
            
            trajectory[i + 1] = sol[-1]
        
        return trajectory


class AdherenceModel:
    """
    Stochastic model for medication adherence behavior.
    
    Adherence at time t depends on:
    - Baseline adherence probability
    - Day-to-day variability
    - Impact of reminders
    - Recent adherence history (habit formation)
    """
    
    def __init__(self, patient_params: PatientParameters):
        """Initialize adherence model."""
        self.params = patient_params
        self.rng = np.random.RandomState(patient_params.random_seed)
    
    def simulate_adherence(
        self,
        n_days: int,
        reminder_schedule: np.ndarray
    ) -> np.ndarray:
        """
        Simulate adherence trajectory.
        
        Args:
            n_days: Number of days to simulate
            reminder_schedule: Binary array indicating reminder days
            
        Returns:
            Binary array of adherence (1 = adherent, 0 = non-adherent)
        """
        adherence = np.zeros(n_days, dtype=int)
        
        alpha = 0.7  # Weight for history
        beta = self.params.response_to_reminders
        
        for day in range(n_days):
            # Base probability
            p_adhere = self.params.adherence_baseline
            
            # Add variability
            p_adhere += self.rng.normal(0, self.params.adherence_variability)
            
            # Reminder effect
            if day < len(reminder_schedule) and reminder_schedule[day] == 1:
                p_adhere += beta
            
            # History effect (habit formation)
            if day > 0:
                recent_adherence = np.mean(adherence[max(0, day-7):day])
                p_adhere = alpha * p_adhere + (1 - alpha) * recent_adherence
            
            # Clip to [0, 1]
            p_adhere = np.clip(p_adhere, 0, 1)
            
            # Sample adherence
            adherence[day] = self.rng.binomial(1, p_adhere)
        
        return adherence


class SyntheticDataGenerator:
    """
    Generate synthetic patient trajectories for RL experiments.
    
    Combines:
    - Physiological modeling (Bergman model)
    - Behavioral modeling (adherence)
    - Population heterogeneity
    - Realistic noise and variability
    
    Example:
        >>> generator = SyntheticDataGenerator()
        >>> population = generator.generate_diabetes_population(n_patients=1000)
        >>> trajectory = generator.simulate_patient_trajectory(
        ...     patient=population[0],
        ...     time_horizon_days=365
        ... )
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize synthetic data generator."""
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        logger.info("Initialized SyntheticDataGenerator")
    
    def generate_diabetes_population(
        self,
        n_patients: int = 1000,
        population_params: Optional[PopulationParameters] = None
    ) -> List[PatientParameters]:
        """
        Generate a population of synthetic diabetes patients.
        
        Args:
            n_patients: Number of patients to generate
            population_params: Population-level parameters
            
        Returns:
            List of PatientParameters objects
        """
        if population_params is None:
            population_params = PopulationParameters(n_patients=n_patients)
        
        logger.info(f"Generating {n_patients} synthetic patients")
        
        patients = []
        
        for i in range(n_patients):
            # Demographics
            age = self.rng.normal(
                population_params.age_mean,
                population_params.age_std
            )
            age = np.clip(age, population_params.age_min, population_params.age_max)
            
            gender = 'M' if self.rng.random() < population_params.fraction_male else 'F'
            
            weight = self.rng.normal(
                population_params.weight_mean,
                population_params.weight_std
            )
            weight = max(40, weight)
            
            height = self.rng.normal(
                population_params.height_mean,
                population_params.height_std
            )
            height = max(140, height)
            
            # Disease parameters
            insulin_resistance = self.rng.normal(
                population_params.insulin_resistance_mean,
                population_params.insulin_resistance_std
            )
            insulin_resistance = max(0.5, insulin_resistance)
            
            beta_cell_function = self.rng.normal(
                population_params.beta_cell_function_mean,
                population_params.beta_cell_function_std
            )
            beta_cell_function = np.clip(beta_cell_function, 0.1, 1.0)
            
            # Adherence profile
            adherence_baseline = self.rng.normal(
                population_params.adherence_baseline_mean,
                population_params.adherence_baseline_std
            )
            adherence_baseline = np.clip(adherence_baseline, 0.1, 0.95)
            
            adherence_variability = self.rng.uniform(0.05, 0.3)
            response_to_reminders = self.rng.uniform(0.05, 0.25)
            
            # Treatment response
            medication_sensitivity = self.rng.lognormal(0, 0.3)
            lifestyle_impact = self.rng.lognormal(0, 0.2)
            
            patient = PatientParameters(
                patient_id=f'synth_{i:05d}',
                age=age,
                gender=gender,
                weight=weight,
                height=height,
                insulin_resistance=insulin_resistance,
                beta_cell_function=beta_cell_function,
                adherence_baseline=adherence_baseline,
                adherence_variability=adherence_variability,
                response_to_reminders=response_to_reminders,
                medication_sensitivity=medication_sensitivity,
                lifestyle_impact=lifestyle_impact,
                random_seed=self.random_seed + i
            )
            
            patients.append(patient)
        
        logger.info(f"Generated population of {len(patients)} patients")
        return patients
    
    def simulate_patient_trajectory(
        self,
        patient: PatientParameters,
        time_horizon_days: int = 365,
        treatment_policy: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Simulate complete patient trajectory over time.
        
        Args:
            patient: Patient parameters
            time_horizon_days: Length of simulation (days)
            treatment_policy: Treatment policy (medication doses, reminder schedule)
                            If None, uses default conservative policy
            
        Returns:
            DataFrame with columns:
            - day: Day number
            - glucose: Blood glucose (mg/dL)
            - insulin: Insulin concentration (μU/mL)
            - medication_taken: Adherence indicator
            - reminder_sent: Reminder indicator
            - meal_glucose: Glucose from meals
        """
        logger.info(f"Simulating trajectory for patient {patient.patient_id}")
        
        # Initialize models
        bergman = BergmanMinimalModel(patient)
        adherence_model = AdherenceModel(patient)
        
        # Default treatment policy
        if treatment_policy is None:
            treatment_policy = {
                'insulin_dose': 0.1,  # Units per dose
                'reminder_frequency': 3  # Reminders per week
            }
        
        # Create reminder schedule
        reminder_schedule = np.zeros(time_horizon_days)
        reminder_frequency = treatment_policy.get('reminder_frequency', 3)
        reminder_days = self.rng.choice(
            time_horizon_days,
            size=int(time_horizon_days * reminder_frequency / 7),
            replace=False
        )
        reminder_schedule[reminder_days] = 1
        
        # Simulate adherence
        adherence = adherence_model.simulate_adherence(
            n_days=time_horizon_days,
            reminder_schedule=reminder_schedule
        )
        
        # Simulate physiological trajectory
        minutes_per_day = 1440
        time_points = np.arange(0, time_horizon_days * minutes_per_day, 60)  # Hourly
        
        # Initial state
        initial_state = np.array([patient.Gb, 0.0, patient.Ib])
        
        # Meal schedule (3 meals per day)
        meal_times = [8*60, 13*60, 19*60]  # 8am, 1pm, 7pm in minutes
        meal_schedule = np.zeros(len(time_points))
        for day in range(time_horizon_days):
            for meal_time in meal_times:
                meal_idx = day * 24 + meal_time // 60
                if meal_idx < len(meal_schedule):
                    # Glucose spike from meal
                    meal_schedule[meal_idx] = self.rng.normal(50, 10)
        
        # Insulin schedule (based on adherence)
        insulin_dose = treatment_policy.get('insulin_dose', 0.1)
        insulin_schedule = np.zeros(len(time_points))
        for day in range(time_horizon_days):
            if adherence[day] == 1:
                # Morning dose
                dose_idx = day * 24 + 8  # 8am
                if dose_idx < len(insulin_schedule):
                    insulin_schedule[dose_idx] = insulin_dose * patient.medication_sensitivity
        
        # Run simulation
        trajectory = bergman.simulate(
            initial_state=initial_state,
            time_points=time_points,
            meal_schedule=meal_schedule,
            insulin_schedule=insulin_schedule
        )
        
        # Add measurement noise
        noise_glucose = self.rng.normal(0, 5, len(trajectory))
        noise_insulin = self.rng.normal(0, 0.5, len(trajectory))
        
        trajectory[:, 0] += noise_glucose
        trajectory[:, 2] += noise_insulin
        
        # Ensure non-negative
        trajectory = np.maximum(trajectory, 0)
        
        # Create daily summary
        daily_data = []
        for day in range(time_horizon_days):
            day_start = day * 24
            day_end = (day + 1) * 24
            
            if day_end <= len(trajectory):
                day_glucose = trajectory[day_start:day_end, 0]
                day_insulin = trajectory[day_start:day_end, 2]
                
                daily_data.append({
                    'day': day,
                    'glucose_mean': np.mean(day_glucose),
                    'glucose_std': np.std(day_glucose),
                    'glucose_min': np.min(day_glucose),
                    'glucose_max': np.max(day_glucose),
                    'insulin_mean': np.mean(day_insulin),
                    'medication_taken': adherence[day],
                    'reminder_sent': reminder_schedule[day]
                })
        
        df = pd.DataFrame(daily_data)
        
        # Add adverse events
        df['hypoglycemia'] = (df['glucose_min'] < 70).astype(int)
        df['hyperglycemia'] = (df['glucose_max'] > 180).astype(int)
        
        logger.info(f"Simulated {len(df)} days for patient {patient.patient_id}")
        return df
    
    def add_noise(
        self,
        trajectory: pd.DataFrame,
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """
        Add realistic measurement noise to trajectory.
        
        Args:
            trajectory: Clean trajectory DataFrame
            noise_level: Standard deviation as fraction of mean
            
        Returns:
            Noisy trajectory
        """
        noisy = trajectory.copy()
        
        numeric_cols = trajectory.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'day':
                mean_val = trajectory[col].mean()
                noise = self.rng.normal(0, noise_level * mean_val, len(trajectory))
                noisy[col] = trajectory[col] + noise
                
                # Ensure non-negative for physical quantities
                if col in ['glucose_mean', 'glucose_min', 'glucose_max', 'insulin_mean']:
                    noisy[col] = np.maximum(noisy[col], 0)
        
        return noisy
    
    def generate_dataset(
        self,
        n_patients: int = 100,
        time_horizon_days: int = 365,
        population_params: Optional[PopulationParameters] = None
    ) -> Tuple[List[PatientParameters], List[pd.DataFrame]]:
        """
        Generate complete synthetic dataset.
        
        Args:
            n_patients: Number of patients
            time_horizon_days: Length of each trajectory
            population_params: Population parameters
            
        Returns:
            Tuple of (patient_list, trajectory_list)
        """
        logger.info(f"Generating dataset: {n_patients} patients, {time_horizon_days} days")
        
        # Generate population
        patients = self.generate_diabetes_population(n_patients, population_params)
        
        # Simulate trajectories
        trajectories = []
        for patient in patients:
            trajectory = self.simulate_patient_trajectory(
                patient=patient,
                time_horizon_days=time_horizon_days
            )
            trajectory['patient_id'] = patient.patient_id
            trajectories.append(trajectory)
        
        logger.info(f"Generated dataset with {len(trajectories)} trajectories")
        return patients, trajectories


if __name__ == '__main__':
    # Example usage
    
    print("=== Synthetic Data Generation Example ===\n")
    
    # Initialize generator
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate small population
    print("Generating population...")
    patients = generator.generate_diabetes_population(n_patients=10)
    
    print(f"\nGenerated {len(patients)} patients")
    print(f"\nSample patient:")
    sample_patient = patients[0]
    print(f"  ID: {sample_patient.patient_id}")
    print(f"  Age: {sample_patient.age:.1f} years")
    print(f"  Gender: {sample_patient.gender}")
    print(f"  Weight: {sample_patient.weight:.1f} kg")
    print(f"  Insulin resistance: {sample_patient.insulin_resistance:.2f}")
    print(f"  Beta cell function: {sample_patient.beta_cell_function:.2f}")
    print(f"  Adherence baseline: {sample_patient.adherence_baseline:.2f}")
    
    # Simulate trajectory
    print("\nSimulating patient trajectory...")
    trajectory = generator.simulate_patient_trajectory(
        patient=sample_patient,
        time_horizon_days=90
    )
    
    print(f"\nTrajectory summary ({len(trajectory)} days):")
    print(trajectory.describe())
    
    print(f"\nAdherence rate: {trajectory['medication_taken'].mean():.2%}")
    print(f"Hypoglycemia events: {trajectory['hypoglycemia'].sum()}")
    print(f"Hyperglycemia events: {trajectory['hyperglycemia'].sum()}")
    
    # Generate full dataset
    print("\n\nGenerating full dataset...")
    patients, trajectories = generator.generate_dataset(
        n_patients=5,
        time_horizon_days=30
    )
    
    print(f"\nDataset generated:")
    print(f"  Patients: {len(patients)}")
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  Total datapoints: {sum(len(t) for t in trajectories)}")
