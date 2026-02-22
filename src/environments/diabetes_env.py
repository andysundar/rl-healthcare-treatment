"""
Diabetes Management Environment
================================
Gymnasium environment for diabetes treatment optimization using Bergman minimal model.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass

from .base_env import BaseHealthcareEnv
from .disease_models import BergmanMinimalModel, BergmanModelParams


@dataclass
class DiabetesEnvConfig:
    """Configuration for diabetes management environment."""
    # Episode settings
    max_steps: int = 90  # 90 days
    dt: float = 1440.0   # 1 day in minutes
    
    # Bergman model parameters (mean values)
    p1_mean: float = 0.028735
    p2_mean: float = 0.028344
    p3_mean: float = 5.035e-5
    n_mean: float = 0.09
    Gb_mean: float = 100.0
    Ib_mean: float = 15.0
    Vg_mean: float = 12.0
    basal_insulin_mean: float = 10.0
    
    # Heterogeneity (standard deviations as fraction of mean)
    patient_variability: float = 0.2
    
    # Initial state ranges
    initial_glucose_range: Tuple[float, float] = (80, 180)
    initial_insulin_range: Tuple[float, float] = (10, 30)
    initial_X_range: Tuple[float, float] = (-5, 5)
    initial_adherence_range: Tuple[float, float] = (0.5, 0.9)
    
    # Action space bounds
    max_insulin_dose: float = 100.0  # units per day
    
    # Reward parameters
    target_glucose_range: Tuple[float, float] = (80, 130)
    mild_hypo_threshold: float = 60.0
    severe_hypo_threshold: float = 40.0
    mild_hyper_threshold: float = 180.0
    severe_hyper_threshold: float = 300.0
    
    # Reward weights
    reward_in_range: float = 1.0
    penalty_mild_hypo: float = -2.0
    penalty_severe_hypo: float = -10.0
    penalty_mild_hyper: float = -1.0
    penalty_severe_hyper: float = -5.0
    reward_adherence_bonus: float = 0.5
    
    # Meal simulation
    meals_per_day: int = 3
    meal_carb_range: Tuple[float, float] = (30, 80)  # grams
    meal_times: Tuple[int, int, int] = (8, 13, 19)  # hours
    
    # Safety thresholds
    critical_glucose_min: float = 40.0
    critical_glucose_max: float = 400.0


class DiabetesManagementEnv(BaseHealthcareEnv):
    """
    Environment for diabetes management using Bergman minimal model.
    
    State Space:
        - Glucose level (mg/dL): [40, 400]
        - Insulin level (mU/L): [0, 100]
        - Remote insulin effect X: [-20, 20]
        - Time of day: [0, 23]
        - Meal indicator: [0, 1]
        - Current adherence: [0, 1]
    
    Action Space:
        - Insulin dosage: [0, max_insulin_dose] units
        - Meal recommendation adjustment: [-1, 1] (scaling factor)
        - Reminder frequency: [0, 3] reminders per week (discrete)
    
    Reward:
        Composite reward balancing glucose control, safety, and adherence
    """
    
    def __init__(self, config: Optional[DiabetesEnvConfig] = None, 
                 render_mode: Optional[str] = None,
                 patient_id: Optional[int] = None):
        """
        Initialize diabetes management environment.
        
        Args:
            config: Environment configuration
            render_mode: Rendering mode
            patient_id: Specific patient ID for reproducibility
        """
        self.config = config or DiabetesEnvConfig()
        
        # Create patient-specific Bergman model
        base_params = BergmanModelParams(
            p1=self.config.p1_mean,
            p2=self.config.p2_mean,
            p3=self.config.p3_mean,
            n=self.config.n_mean,
            Gb=self.config.Gb_mean,
            Ib=self.config.Ib_mean,
            Vg=self.config.Vg_mean,
            basal_insulin=self.config.basal_insulin_mean
        )
        
        # Sample patient-specific parameters
        if patient_id is not None:
            rng = np.random.default_rng(patient_id)
        else:
            rng = np.random.default_rng()
        
        patient_params = BergmanMinimalModel.sample_patient_parameters(
            base_params, 
            self.config.patient_variability,
            rng
        )
        
        self.bergman_model = BergmanMinimalModel(patient_params)
        self.patient_id = patient_id
        
        # Meal schedule
        self.current_hour = 0
        self.meal_carbs_today = []
        
        # Initialize base environment
        super().__init__(config, render_mode)
    
    def _get_observation_space(self) -> spaces.Space:
        """Define observation space."""
        return spaces.Box(
            low=np.array([
                self.config.critical_glucose_min,  # glucose
                0.0,                                # insulin
                -20.0,                              # X
                0.0,                                # hour of day
                0.0,                                # meal indicator
                0.0                                 # adherence
            ]),
            high=np.array([
                self.config.critical_glucose_max,  # glucose
                100.0,                             # insulin
                20.0,                              # X
                23.0,                              # hour of day
                1.0,                               # meal indicator
                1.0                                # adherence
            ]),
            dtype=np.float32
        )
    
    def _get_action_space(self) -> spaces.Space:
        """Define action space."""
        return spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([self.config.max_insulin_dose, 1.0, 3.0]),
            dtype=np.float32
        )
    
    def _reset_state(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset to initial state."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        # Sample initial glucose and insulin
        G0 = rng.uniform(*self.config.initial_glucose_range)
        I0 = rng.uniform(*self.config.initial_insulin_range)
        X0 = rng.uniform(*self.config.initial_X_range)
        adherence0 = rng.uniform(*self.config.initial_adherence_range)
        
        # Initialize time
        self.current_hour = rng.integers(0, 24)
        
        # Generate meal schedule for the day
        self._generate_daily_meals(rng)
        meal_indicator = 1.0 if self._is_meal_time() else 0.0
        
        # State: [G, I, X, hour, meal_indicator, adherence]
        return np.array([G0, I0, X0, self.current_hour, meal_indicator, adherence0], 
                       dtype=np.float32)
    
    def _step_dynamics(self, action: np.ndarray) -> np.ndarray:
        """Update environment dynamics."""
        # Extract current state
        G, I, X, hour, _, adherence = self.state
        
        # Extract action components
        insulin_dose = action[0]
        meal_adjustment = action[1]
        reminder_freq = int(action[2])
        
        # Simulate adherence to prescribed dose
        # Lower adherence = higher chance of missing dose
        if np.random.random() > adherence:
            insulin_dose = 0.0  # Missed dose
        
        # Get meal carbs for current time
        meal_carbs = self._get_meal_carbs()
        
        # Apply meal adjustment (patient may adjust meal based on recommendation)
        if meal_carbs > 0:
            meal_carbs *= (1 + 0.2 * meal_adjustment)  # ±20% adjustment
            meal_carbs = np.clip(meal_carbs, 0, 150)
        
        # Simulate glucose-insulin dynamics
        G_next, I_next, X_next = self.bergman_model.step(
            G, I, X,
            insulin_dose,
            meal_carbs,
            dt=self.config.dt
        )
        
        # Update adherence based on reminder frequency
        # More reminders → better adherence (with diminishing returns)
        adherence_delta = 0.05 * np.sqrt(reminder_freq) * np.random.uniform(0.5, 1.0)
        adherence_next = np.clip(adherence + adherence_delta, 0, 1)
        
        # Update time
        self.current_hour = (self.current_hour + 24) % 24  # Next day
        if self.current_hour == 0:
            self._generate_daily_meals(np.random.default_rng())
        
        meal_indicator_next = 1.0 if self._is_meal_time() else 0.0
        
        # New state
        return np.array([G_next, I_next, X_next, self.current_hour, 
                        meal_indicator_next, adherence_next], dtype=np.float32)
    
    def _compute_reward(self, state: np.ndarray, action: np.ndarray, 
                       next_state: np.ndarray) -> float:
        """Compute composite reward."""
        G_next = next_state[0]
        adherence = next_state[5]
        
        reward = 0.0
        
        # Glucose control reward
        if self.config.target_glucose_range[0] <= G_next <= self.config.target_glucose_range[1]:
            reward += self.config.reward_in_range
        elif G_next < self.config.severe_hypo_threshold:
            reward += self.config.penalty_severe_hypo
        elif G_next < self.config.mild_hypo_threshold:
            reward += self.config.penalty_mild_hypo
        elif G_next > self.config.severe_hyper_threshold:
            reward += self.config.penalty_severe_hyper
        elif G_next > self.config.mild_hyper_threshold:
            reward += self.config.penalty_mild_hyper
        
        # Adherence bonus
        reward += self.config.reward_adherence_bonus * adherence
        
        # Small penalty for excessive insulin (to encourage minimal effective dose)
        insulin_dose = action[0]
        reward -= 0.001 * (insulin_dose / self.config.max_insulin_dose)
        
        return reward
    
    def _check_termination(self, state: np.ndarray) -> Tuple[bool, bool]:
        """Check for episode termination."""
        G = state[0]
        
        # Terminate if critical glucose levels reached
        if G < self.config.critical_glucose_min or G > self.config.critical_glucose_max:
            return True, False  # Natural termination (severe adverse event)
        
        return False, False  # Continue
    
    def _is_unsafe_state(self, state: np.ndarray) -> bool:
        """Check if state is unsafe."""
        G = state[0]
        return (G < self.config.severe_hypo_threshold or 
                G > self.config.severe_hyper_threshold)
    
    def _generate_daily_meals(self, rng: np.random.Generator):
        """Generate meal schedule for the day."""
        self.meal_carbs_today = []
        for _ in range(self.config.meals_per_day):
            carbs = rng.uniform(*self.config.meal_carb_range)
            self.meal_carbs_today.append(carbs)
    
    def _is_meal_time(self) -> bool:
        """Check if current hour is a meal time."""
        return self.current_hour in self.config.meal_times
    
    def _get_meal_carbs(self) -> float:
        """Get carbohydrates for current meal."""
        if not self._is_meal_time():
            return 0.0
        
        meal_idx = self.config.meal_times.index(self.current_hour)
        if meal_idx < len(self.meal_carbs_today):
            return self.meal_carbs_today[meal_idx]
        return 0.0
    
    def render(self):
        """Render current state."""
        if self.render_mode == 'human':
            G, I, X, hour, meal, adherence = self.state
            print(f"\n{'='*60}")
            print(f"Step {self.current_step} | Day {self.current_step + 1}")
            print(f"{'='*60}")
            print(f"Glucose: {G:.1f} mg/dL")
            print(f"Insulin: {I:.1f} mU/L")
            print(f"Remote Effect: {X:.2f}")
            print(f"Time: {int(hour)}:00")
            print(f"Meal Time: {'Yes' if meal > 0 else 'No'}")
            print(f"Adherence: {adherence:.2%}")
            
            # Glucose status
            if G < self.config.severe_hypo_threshold:
                status = "SEVERE HYPOGLYCEMIA ⚠️"
            elif G < self.config.mild_hypo_threshold:
                status = "Mild Hypoglycemia"
            elif G <= self.config.target_glucose_range[1]:
                status = "Target Range ✓"
            elif G < self.config.severe_hyper_threshold:
                status = "Mild Hyperglycemia"
            else:
                status = "SEVERE HYPERGLYCEMIA ⚠️"
            print(f"Status: {status}")
            print(f"{'='*60}\n")
    
    def get_physiological_state(self) -> Dict[str, float]:
        """Get detailed physiological state."""
        G, I, X, hour, meal, adherence = self.state
        
        return {
            'glucose_mgdl': G,
            'insulin_muL': I,
            'remote_insulin_effect': X,
            'hour_of_day': hour,
            'is_meal_time': bool(meal),
            'adherence_level': adherence,
            'in_target_range': self.config.target_glucose_range[0] <= G <= self.config.target_glucose_range[1],
            'is_hypoglycemic': G < self.config.mild_hypo_threshold,
            'is_hyperglycemic': G > self.config.mild_hyper_threshold,
            'is_critical': self._is_unsafe_state(self.state)
        }
