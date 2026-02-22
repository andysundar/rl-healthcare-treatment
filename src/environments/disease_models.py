"""
Disease Models
==============
Physiological models for simulating patient dynamics.
"""

from typing import Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class BergmanModelParams:
    """Parameters for Bergman minimal model of glucose-insulin dynamics."""
    p1: float = 0.028735  # Glucose effectiveness (min^-1)
    p2: float = 0.028344  # Insulin clearance rate (min^-1)
    p3: float = 5.035e-5  # Insulin sensitivity (mU^-1 L min^-1)
    n: float = 0.09       # Insulin degradation rate (min^-1)
    Gb: float = 100.0     # Basal glucose (mg/dL)
    Ib: float = 15.0      # Basal insulin (mU/L)
    Vg: float = 12.0      # Glucose distribution volume (L)
    basal_insulin: float = 10.0  # Basal insulin infusion (mU/min)


class BergmanMinimalModel:
    """
    Bergman minimal model for glucose-insulin dynamics.
    
    The model consists of three ODEs:
    - dG/dt = -p1*G - X*(G - Gb) + D(t)/Vg
    - dX/dt = -p2*X + p3*(I - Ib)
    - dI/dt = -n*I + u(t) + basal_insulin
    
    where:
        G: glucose concentration (mg/dL)
        I: insulin concentration (mU/L)
        X: remote insulin effect
        D(t): dietary glucose input (mg/min)
        u(t): exogenous insulin input (mU/min)
    """
    
    def __init__(self, params: Optional[BergmanModelParams] = None):
        """
        Initialize Bergman minimal model.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params or BergmanModelParams()
        
    def step(self, 
             G: float, 
             I: float, 
             X: float,
             insulin_dose: float,
             meal_carbs: float,
             dt: float = 1.0) -> Tuple[float, float, float]:
        """
        Simulate one timestep of glucose-insulin dynamics.
        
        Args:
            G: Current glucose level (mg/dL)
            I: Current insulin level (mU/L)
            X: Current remote insulin effect
            insulin_dose: Exogenous insulin dose (units)
            meal_carbs: Dietary carbohydrate intake (grams)
            dt: Timestep size (minutes, default 1440 for 1 day)
            
        Returns:
            G_next: Next glucose level
            I_next: Next insulin level
            X_next: Next remote insulin effect
        """
        # Convert meal carbs to glucose input (1g carb ≈ 5mg glucose)
        D = meal_carbs * 5.0 / dt if meal_carbs > 0 else 0.0
        
        # Convert insulin dose from units to mU/min
        # 1 unit = 1000 mU, spread over dt minutes
        u = (insulin_dose * 1000.0) / dt if insulin_dose > 0 else 0.0
        
        # Compute derivatives using Euler method
        dG = (-self.params.p1 * G - 
              X * (G - self.params.Gb) + 
              D / self.params.Vg) * dt
        
        dX = (-self.params.p2 * X + 
              self.params.p3 * (I - self.params.Ib)) * dt
        
        dI = (-self.params.n * I + 
              u + 
              self.params.basal_insulin) * dt
        
        # Update state
        G_next = max(0, G + dG)  # Glucose can't be negative
        I_next = max(0, I + dI)  # Insulin can't be negative
        X_next = X + dX
        
        return G_next, I_next, X_next
    
    def simulate_trajectory(self,
                           initial_state: Tuple[float, float, float],
                           actions: np.ndarray,
                           meals: np.ndarray,
                           dt: float = 1.0) -> np.ndarray:
        """
        Simulate a full trajectory.
        
        Args:
            initial_state: (G0, I0, X0) initial conditions
            actions: Array of insulin doses over time
            meals: Array of meal carbohydrates over time
            dt: Timestep size (minutes)
            
        Returns:
            trajectory: (T, 3) array of [G, I, X] over time
        """
        T = len(actions)
        trajectory = np.zeros((T + 1, 3))
        trajectory[0] = initial_state
        
        for t in range(T):
            G, I, X = trajectory[t]
            G_next, I_next, X_next = self.step(
                G, I, X, 
                actions[t], 
                meals[t],
                dt
            )
            trajectory[t + 1] = [G_next, I_next, X_next]
        
        return trajectory
    
    @staticmethod
    def sample_patient_parameters(
        base_params: BergmanModelParams,
        variability: float = 0.2,
        rng: Optional[np.random.Generator] = None
    ) -> BergmanModelParams:
        """
        Sample patient-specific parameters with variability.
        
        Args:
            base_params: Base parameter values
            variability: Coefficient of variation for parameter sampling
            rng: Random number generator
            
        Returns:
            patient_params: Sampled parameters
        """
        rng = rng or np.random.default_rng()
        
        # Sample with log-normal distribution to ensure positive values
        def sample_param(mean, cv):
            sigma = np.sqrt(np.log(1 + cv**2))
            mu = np.log(mean) - 0.5 * sigma**2
            return rng.lognormal(mu, sigma)
        
        return BergmanModelParams(
            p1=sample_param(base_params.p1, variability),
            p2=sample_param(base_params.p2, variability),
            p3=sample_param(base_params.p3, variability),
            n=sample_param(base_params.n, variability),
            Gb=sample_param(base_params.Gb, variability * 0.5),  # Less variability in baseline
            Ib=sample_param(base_params.Ib, variability * 0.5),
            Vg=sample_param(base_params.Vg, variability * 0.3),
            basal_insulin=sample_param(base_params.basal_insulin, variability)
        )


@dataclass
class AdherenceModelParams:
    """Parameters for medication adherence dynamics model."""
    alpha: float = 0.8      # Adherence persistence
    beta: float = 0.15      # Reminder effectiveness
    gamma: float = 0.1      # Satisfaction effect
    delta: float = 0.05     # Side effect penalty
    noise_std: float = 0.05 # Stochastic noise
    baseline: float = 0.7   # Baseline adherence


class AdherenceDynamicsModel:
    """
    Model for medication adherence behavior.
    
    The adherence evolves according to:
    A(t+1) = α*A(t) + β*R(t) + γ*S(t) - δ*E(t) + ε
    
    where:
        A: adherence level [0, 1]
        R: reminder action (binary or continuous)
        S: treatment satisfaction [0, 1]
        E: side effect severity [0, 1]
        ε: random noise ~ N(0, σ²)
    """
    
    def __init__(self, params: Optional[AdherenceModelParams] = None):
        """
        Initialize adherence dynamics model.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params or AdherenceModelParams()
        
    def step(self,
             current_adherence: float,
             reminder_action: float,
             satisfaction: float = 0.7,
             side_effects: float = 0.1,
             rng: Optional[np.random.Generator] = None) -> float:
        """
        Simulate one timestep of adherence dynamics.
        
        Args:
            current_adherence: Current adherence level [0, 1]
            reminder_action: Reminder intensity [0, 1]
            satisfaction: Treatment satisfaction [0, 1]
            side_effects: Side effect severity [0, 1]
            rng: Random number generator
            
        Returns:
            next_adherence: Next adherence level [0, 1]
        """
        rng = rng or np.random.default_rng()
        
        # Compute deterministic component
        next_adherence = (
            self.params.alpha * current_adherence +
            self.params.beta * reminder_action +
            self.params.gamma * satisfaction -
            self.params.delta * side_effects
        )
        
        # Add stochastic noise
        noise = rng.normal(0, self.params.noise_std)
        next_adherence += noise
        
        # Clip to valid range
        next_adherence = np.clip(next_adherence, 0, 1)
        
        return next_adherence
    
    def simulate_trajectory(self,
                           initial_adherence: float,
                           reminder_actions: np.ndarray,
                           satisfaction_levels: Optional[np.ndarray] = None,
                           side_effect_levels: Optional[np.ndarray] = None,
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Simulate a full adherence trajectory.
        
        Args:
            initial_adherence: Starting adherence level
            reminder_actions: Array of reminder intensities over time
            satisfaction_levels: Array of satisfaction levels (uses default if None)
            side_effect_levels: Array of side effect levels (uses default if None)
            rng: Random number generator
            
        Returns:
            trajectory: Array of adherence levels over time
        """
        T = len(reminder_actions)
        trajectory = np.zeros(T + 1)
        trajectory[0] = initial_adherence
        
        # Use default values if not provided
        if satisfaction_levels is None:
            satisfaction_levels = np.full(T, 0.7)
        if side_effect_levels is None:
            side_effect_levels = np.full(T, 0.1)
        
        for t in range(T):
            trajectory[t + 1] = self.step(
                trajectory[t],
                reminder_actions[t],
                satisfaction_levels[t],
                side_effect_levels[t],
                rng
            )
        
        return trajectory
    
    @staticmethod
    def sample_patient_parameters(
        base_params: AdherenceModelParams,
        variability: float = 0.3,
        rng: Optional[np.random.Generator] = None
    ) -> AdherenceModelParams:
        """
        Sample patient-specific adherence parameters.
        
        Args:
            base_params: Base parameter values
            variability: Coefficient of variation
            rng: Random number generator
            
        Returns:
            patient_params: Sampled parameters
        """
        rng = rng or np.random.default_rng()
        
        def sample_param(mean, cv):
            return max(0, rng.normal(mean, mean * cv))
        
        return AdherenceModelParams(
            alpha=np.clip(sample_param(base_params.alpha, variability), 0, 1),
            beta=sample_param(base_params.beta, variability),
            gamma=sample_param(base_params.gamma, variability),
            delta=sample_param(base_params.delta, variability),
            noise_std=sample_param(base_params.noise_std, variability),
            baseline=np.clip(sample_param(base_params.baseline, variability), 0, 1)
        )


class BloodPressureModel:
    """
    Simple blood pressure dynamics model for hypertension management.
    
    Models systolic BP as:
    BP(t+1) = BP(t) + medication_effect - lifestyle_effect + noise
    """
    
    def __init__(self, 
                 baseline_systolic: float = 140.0,
                 baseline_diastolic: float = 90.0,
                 medication_effectiveness: float = -5.0,
                 lifestyle_effectiveness: float = -3.0,
                 noise_std: float = 5.0):
        """
        Initialize blood pressure model.
        
        Args:
            baseline_systolic: Baseline systolic BP (mmHg)
            baseline_diastolic: Baseline diastolic BP (mmHg)
            medication_effectiveness: Effect of medication per unit dose
            lifestyle_effectiveness: Effect of lifestyle changes
            noise_std: Standard deviation of BP fluctuations
        """
        self.baseline_systolic = baseline_systolic
        self.baseline_diastolic = baseline_diastolic
        self.med_effect = medication_effectiveness
        self.lifestyle_effect = lifestyle_effectiveness
        self.noise_std = noise_std
    
    def step(self,
             current_systolic: float,
             current_diastolic: float,
             medication_dose: float,
             lifestyle_score: float,
             rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """
        Simulate one timestep of blood pressure dynamics.
        
        Args:
            current_systolic: Current systolic BP
            current_diastolic: Current diastolic BP
            medication_dose: Medication dose [0, 1]
            lifestyle_score: Lifestyle adherence [0, 1]
            rng: Random number generator
            
        Returns:
            next_systolic: Next systolic BP
            next_diastolic: Next diastolic BP
        """
        rng = rng or np.random.default_rng()
        
        # Systolic BP update
        systolic_change = (
            self.med_effect * medication_dose +
            self.lifestyle_effect * lifestyle_score +
            rng.normal(0, self.noise_std)
        )
        next_systolic = max(90, current_systolic + systolic_change)
        
        # Diastolic BP update (correlated with systolic)
        diastolic_change = systolic_change * 0.6  # Typically 60% of systolic change
        next_diastolic = max(60, current_diastolic + diastolic_change)
        
        return next_systolic, next_diastolic
