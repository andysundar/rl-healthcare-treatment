"""
Medication Adherence Environment
=================================
Gymnasium environment for optimizing medication adherence through reminders and interventions.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass

from .base_env import BaseHealthcareEnv
from .disease_models import AdherenceDynamicsModel, AdherenceModelParams


@dataclass
class AdherenceEnvConfig:
    """Configuration for medication adherence environment."""
    # Episode settings
    max_steps: int = 90  # 90 days
    
    # Adherence model parameters
    alpha_mean: float = 0.8      # Adherence persistence
    beta_mean: float = 0.15      # Reminder effectiveness
    gamma_mean: float = 0.1      # Satisfaction effect
    delta_mean: float = 0.05     # Side effect penalty
    noise_std: float = 0.05      # Stochastic noise
    baseline_mean: float = 0.7   # Baseline adherence
    
    # Patient heterogeneity
    patient_variability: float = 0.3
    
    # Initial state ranges
    initial_adherence_range: Tuple[float, float] = (0.4, 0.8)
    initial_satisfaction_range: Tuple[float, float] = (0.5, 0.9)
    initial_side_effects_range: Tuple[float, float] = (0.0, 0.3)
    initial_days_since_reminder_range: Tuple[int, int] = (0, 7)
    
    # Reminder types and effectiveness
    reminder_types: Dict[str, float] = None
    
    # Reward parameters
    high_adherence_threshold: float = 0.8
    medium_adherence_threshold: float = 0.5
    reward_high_adherence: float = 10.0
    reward_medium_adherence: float = 5.0
    penalty_low_adherence: float = -5.0
    penalty_reminder_cost: float = -0.1
    
    # Satisfaction dynamics
    satisfaction_decay: float = 0.02  # Per day without positive intervention
    side_effect_variability: float = 0.1
    
    def __post_init__(self):
        if self.reminder_types is None:
            self.reminder_types = {
                'none': 0.0,
                'sms': 0.3,
                'call': 0.5,
                'app_notification': 0.4,
                'personal_visit': 0.7
            }


class MedicationAdherenceEnv(BaseHealthcareEnv):
    """
    Environment for medication adherence optimization.
    
    State Space:
        - Current adherence level: [0, 1]
        - Days since last reminder: [0, 30]
        - Treatment satisfaction: [0, 1]
        - Side effect severity: [0, 1]
        - Consecutive days of good adherence: [0, max_steps]
    
    Action Space:
        - Reminder type: discrete {0: none, 1: sms, 2: call, 3: app, 4: visit}
        - Educational content: discrete {0: none, 1: benefits, 2: side_effects, 3: support}
    
    Reward:
        Balances adherence improvement against intervention costs
    """
    
    def __init__(self, config: Optional[AdherenceEnvConfig] = None,
                 render_mode: Optional[str] = None,
                 patient_id: Optional[int] = None):
        """
        Initialize medication adherence environment.
        
        Args:
            config: Environment configuration
            render_mode: Rendering mode
            patient_id: Specific patient ID for reproducibility
        """
        self.config = config or AdherenceEnvConfig()
        
        # Create patient-specific adherence model
        base_params = AdherenceModelParams(
            alpha=self.config.alpha_mean,
            beta=self.config.beta_mean,
            gamma=self.config.gamma_mean,
            delta=self.config.delta_mean,
            noise_std=self.config.noise_std,
            baseline=self.config.baseline_mean
        )
        
        # Sample patient-specific parameters
        if patient_id is not None:
            rng = np.random.default_rng(patient_id)
        else:
            rng = np.random.default_rng()
        
        patient_params = AdherenceDynamicsModel.sample_patient_parameters(
            base_params,
            self.config.patient_variability,
            rng
        )
        
        self.adherence_model = AdherenceDynamicsModel(patient_params)
        self.patient_id = patient_id
        
        # Reminder tracking
        self.days_since_last_reminder = 0
        self.consecutive_good_days = 0
        self.total_reminders_sent = 0
        
        # Map reminder type indices to effectiveness
        self.reminder_effectiveness = list(self.config.reminder_types.values())
        
        # Initialize base environment
        super().__init__(self.config, render_mode)
    
    def _get_observation_space(self) -> spaces.Space:
        """Define observation space."""
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 30.0, 1.0, 1.0, float(self.config.max_steps)]),
            dtype=np.float32
        )
    
    def _get_action_space(self) -> spaces.Space:
        """Define action space."""
        # Discrete action space: (reminder_type, educational_content)
        # We'll use MultiDiscrete for this
        from gymnasium.spaces import MultiDiscrete
        return MultiDiscrete([
            len(self.config.reminder_types),  # reminder type
            4  # educational content (none, benefits, side_effects, support)
        ])
    
    def _reset_state(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset to initial state."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        # Sample initial conditions
        adherence = rng.uniform(*self.config.initial_adherence_range)
        satisfaction = rng.uniform(*self.config.initial_satisfaction_range)
        side_effects = rng.uniform(*self.config.initial_side_effects_range)
        days_since_reminder = float(rng.integers(*self.config.initial_days_since_reminder_range))
        
        # Reset tracking
        self.days_since_last_reminder = int(days_since_reminder)
        self.consecutive_good_days = 0
        self.total_reminders_sent = 0
        
        # State: [adherence, days_since_reminder, satisfaction, side_effects, consecutive_good_days]
        return np.array([adherence, days_since_reminder, satisfaction, 
                        side_effects, 0.0], dtype=np.float32)
    
    def _step_dynamics(self, action: np.ndarray) -> np.ndarray:
        """Update environment dynamics."""
        # Extract current state
        adherence, days_since_reminder, satisfaction, side_effects, consecutive_good = self.state
        
        # Extract action components
        reminder_type = int(action[0])
        education_type = int(action[1])
        
        # Get reminder effectiveness
        reminder_intensity = self.reminder_effectiveness[reminder_type]
        
        # Update days since reminder
        if reminder_intensity > 0:
            self.days_since_last_reminder = 0
            self.total_reminders_sent += 1
        else:
            self.days_since_last_reminder += 1
        
        # Educational content effect on satisfaction
        education_boost = {
            0: 0.0,    # none
            1: 0.05,   # benefits
            2: 0.03,   # side_effects (smaller boost, addresses concerns)
            3: 0.04    # support
        }
        satisfaction_change = education_boost[education_type]
        
        # Satisfaction dynamics
        # Decays slowly without intervention, boosted by education
        satisfaction_next = satisfaction - self.config.satisfaction_decay + satisfaction_change
        satisfaction_next = np.clip(satisfaction_next, 0, 1)
        
        # Side effects vary stochastically (could be improved with treatment adjustments)
        if education_type == 2:  # Side effect education helps manage them
            side_effects_next = max(0, side_effects - 0.05 + 
                                  np.random.normal(0, self.config.side_effect_variability))
        else:
            side_effects_next = side_effects + np.random.normal(0, self.config.side_effect_variability)
        side_effects_next = np.clip(side_effects_next, 0, 1)
        
        # Update adherence using dynamics model
        adherence_next = self.adherence_model.step(
            adherence,
            reminder_intensity,
            satisfaction_next,
            side_effects_next
        )
        
        # Track consecutive good adherence days
        if adherence_next >= self.config.high_adherence_threshold:
            consecutive_good_next = consecutive_good + 1
        else:
            consecutive_good_next = 0
        
        # New state
        return np.array([
            adherence_next,
            float(self.days_since_last_reminder),
            satisfaction_next,
            side_effects_next,
            consecutive_good_next
        ], dtype=np.float32)
    
    def _compute_reward(self, state: np.ndarray, action: np.ndarray,
                       next_state: np.ndarray) -> float:
        """Compute reward."""
        adherence_next = next_state[0]
        reminder_type = int(action[0])
        
        reward = 0.0
        
        # Adherence-based reward
        if adherence_next >= self.config.high_adherence_threshold:
            reward += self.config.reward_high_adherence
        elif adherence_next >= self.config.medium_adherence_threshold:
            reward += self.config.reward_medium_adherence
        else:
            reward += self.config.penalty_low_adherence
        
        # Bonus for sustained adherence
        consecutive_good = next_state[4]
        if consecutive_good >= 7:  # Week of good adherence
            reward += 5.0
        if consecutive_good >= 30:  # Month of good adherence
            reward += 10.0
        
        # Penalty for reminder cost (scaled by reminder type)
        reminder_cost_multipliers = [0, 1, 3, 2, 5]  # none, sms, call, app, visit
        reminder_cost = reminder_cost_multipliers[reminder_type] * self.config.penalty_reminder_cost
        reward += reminder_cost
        
        # Satisfaction bonus/penalty
        satisfaction = next_state[2]
        reward += (satisfaction - 0.5) * 2.0  # Range: [-1, 1]
        
        return reward
    
    def _check_termination(self, state: np.ndarray) -> Tuple[bool, bool]:
        """Check for episode termination."""
        # No early termination for adherence environment
        # Episode continues until max_steps
        return False, False
    
    def _is_unsafe_state(self, state: np.ndarray) -> bool:
        """Check if state is unsafe."""
        adherence = state[0]
        # Very low adherence is concerning but not immediately "unsafe"
        # We consider it unsafe if adherence drops below 0.2 for tracking
        return adherence < 0.2
    
    def render(self):
        """Render current state."""
        if self.render_mode == 'human':
            adherence, days_since, satisfaction, side_effects, consecutive = self.state
            
            print(f"\n{'='*60}")
            print(f"Day {self.current_step + 1}")
            print(f"{'='*60}")
            print(f"Adherence: {adherence:.1%}")
            print(f"Days since reminder: {int(days_since)}")
            print(f"Satisfaction: {satisfaction:.1%}")
            print(f"Side effects: {side_effects:.1%}")
            print(f"Consecutive good days: {int(consecutive)}")
            print(f"Total reminders sent: {self.total_reminders_sent}")
            
            # Adherence status
            if adherence >= self.config.high_adherence_threshold:
                status = "Excellent Adherence ✓"
            elif adherence >= self.config.medium_adherence_threshold:
                status = "Moderate Adherence"
            else:
                status = "Poor Adherence ⚠️"
            print(f"Status: {status}")
            print(f"{'='*60}\n")
    
    def get_adherence_metrics(self) -> Dict[str, Any]:
        """Get detailed adherence metrics."""
        adherence, days_since, satisfaction, side_effects, consecutive = self.state
        
        # Calculate metrics from episode history
        if self.episode_history:
            adherence_levels = [step['next_state'][0] for step in self.episode_history]
            avg_adherence = np.mean(adherence_levels)
            adherence_trend = np.polyfit(range(len(adherence_levels)), adherence_levels, 1)[0]
        else:
            avg_adherence = adherence
            adherence_trend = 0.0
        
        return {
            'current_adherence': adherence,
            'average_adherence': avg_adherence,
            'adherence_trend': adherence_trend,
            'days_since_reminder': int(days_since),
            'satisfaction_level': satisfaction,
            'side_effect_severity': side_effects,
            'consecutive_good_days': int(consecutive),
            'total_reminders': self.total_reminders_sent,
            'reminder_efficiency': avg_adherence / max(self.total_reminders_sent, 1),
            'is_high_adherence': adherence >= self.config.high_adherence_threshold,
            'is_low_adherence': adherence < self.config.medium_adherence_threshold
        }
