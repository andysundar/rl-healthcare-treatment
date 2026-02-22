"""
Reward Function Configuration
Centralized configuration for all reward components.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Set


@dataclass
class RewardConfig:
    """
    Configuration for reward functions.
    
    This centralizes all reward-related parameters for easy tuning
    and experimentation.
    """
    
    # ========== Composite Reward Weights ==========
    w_adherence: float = 1.0
    w_health: float = 2.0
    w_safety: float = 5.0  # Safety most important
    w_cost: float = 0.1
    
    # ========== Adherence Reward Parameters ==========
    adherence_improvement_multiplier: float = 2.0
    high_adherence_threshold: float = 0.8
    high_adherence_bonus: float = 1.0
    sustained_adherence_days: int = 30
    sustained_adherence_bonus: float = 2.0
    
    # ========== Health Outcome Parameters ==========
    # Target ranges for health metrics
    glucose_target: Tuple[float, float] = (80.0, 130.0)
    systolic_bp_target: Tuple[float, float] = (90.0, 120.0)
    diastolic_bp_target: Tuple[float, float] = (60.0, 80.0)
    hba1c_target: Tuple[float, float] = (4.0, 6.5)
    weight_target: Tuple[float, float] = None  # Personalized per patient
    cholesterol_target: Tuple[float, float] = (0.0, 200.0)
    
    # Metric importance weights
    glucose_weight: float = 1.0
    systolic_bp_weight: float = 0.8
    diastolic_bp_weight: float = 0.6
    hba1c_weight: float = 1.2
    weight_weight: float = 0.5
    cholesterol_weight: float = 0.7
    
    health_improvement_multiplier: float = 1.5
    
    # ========== Safety Penalty Parameters ==========
    # Glucose safety thresholds
    severe_hypoglycemia_threshold: float = 60.0
    moderate_hypoglycemia_threshold: float = 70.0
    severe_hyperglycemia_threshold: float = 300.0
    moderate_hyperglycemia_threshold: float = 200.0
    
    # Blood pressure safety thresholds
    severe_hypertension_threshold: float = 180.0
    moderate_hypertension_threshold: float = 140.0
    severe_hypotension_threshold: float = 70.0
    
    # Safety penalty magnitudes (negative values)
    severe_hypoglycemia_penalty: float = -10.0
    moderate_hypoglycemia_penalty: float = -3.0
    severe_hyperglycemia_penalty: float = -5.0
    moderate_hyperglycemia_penalty: float = -2.0
    severe_hypertension_penalty: float = -8.0
    moderate_hypertension_penalty: float = -3.0
    severe_hypotension_penalty: float = -7.0
    
    # Medication safety penalties
    drug_interaction_penalty: float = -7.0
    overdose_penalty: float = -8.0
    contraindication_penalty: float = -10.0
    
    # Adverse event penalties
    emergency_visit_penalty: float = -20.0
    hospitalization_penalty: float = -15.0
    icu_admission_penalty: float = -25.0
    
    # Maximum safe medication dosages
    max_safe_dosages: Dict[str, float] = field(default_factory=lambda: {
        'insulin': 100.0,  # units
        'metformin': 2000.0,  # mg
        'glipizide': 20.0  # mg
    })
    
    # Dangerous drug interactions (set of tuples)
    dangerous_interactions: Set[Tuple[str, str]] = field(default_factory=set)
    
    # ========== Cost Parameters ==========
    # Medication costs (per unit/dose)
    medication_costs: Dict[str, float] = field(default_factory=lambda: {
        'insulin': 0.50,
        'metformin': 0.10,
        'glipizide': 0.15,
        'statin': 0.20,
        'ace_inhibitor': 0.25
    })
    
    # Service costs
    appointment_cost: float = 100.0
    lab_test_cost: float = 50.0
    emergency_visit_cost: float = 5000.0
    hospitalization_cost_per_day: float = 2000.0
    icu_cost_per_day: float = 10000.0
    
    # Communication costs
    reminder_cost: float = 0.50
    phone_call_cost: float = 5.0
    
    # Cost normalization (to scale costs to similar range as other rewards)
    cost_normalization_factor: float = 1000.0
    
    # ========== Reward Shaping Parameters ==========
    use_potential_shaping: bool = False
    gamma: float = 0.99  # Discount factor for potential shaping
    normalize_components: bool = True  # Normalize reward components
    
    # ========== Advanced Options ==========
    # Curriculum learning
    use_curriculum: bool = False
    curriculum_type: str = 'linear'  # 'linear', 'exponential', 'step'
    
    # Curiosity-driven exploration
    use_curiosity_bonus: bool = False
    curiosity_bonus_scale: float = 0.1
    
    # Reward normalization
    use_online_normalization: bool = False
    normalization_clip_range: Tuple[float, float] = (-10.0, 10.0)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Ensure weights are non-negative
        assert self.w_adherence >= 0, "Adherence weight must be non-negative"
        assert self.w_health >= 0, "Health weight must be non-negative"
        assert self.w_safety >= 0, "Safety weight must be non-negative"
        assert self.w_cost >= 0, "Cost weight must be non-negative"
        
        # Ensure penalties are negative
        assert self.severe_hypoglycemia_penalty <= 0, "Penalties must be negative"
        assert self.emergency_visit_penalty <= 0, "Penalties must be negative"
        
        # Ensure thresholds make sense
        assert self.severe_hypoglycemia_threshold < self.moderate_hypoglycemia_threshold
        assert self.moderate_hyperglycemia_threshold < self.severe_hyperglycemia_threshold
        
        # Ensure gamma in valid range
        assert 0 <= self.gamma <= 1, "Gamma must be in [0, 1]"
    
    def to_dict(self) -> Dict:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            'weights': {
                'adherence': self.w_adherence,
                'health': self.w_health,
                'safety': self.w_safety,
                'cost': self.w_cost
            },
            'adherence': {
                'improvement_multiplier': self.adherence_improvement_multiplier,
                'high_threshold': self.high_adherence_threshold,
                'high_bonus': self.high_adherence_bonus,
                'sustained_days': self.sustained_adherence_days,
                'sustained_bonus': self.sustained_adherence_bonus
            },
            'health': {
                'targets': {
                    'glucose': self.glucose_target,
                    'systolic_bp': self.systolic_bp_target,
                    'diastolic_bp': self.diastolic_bp_target,
                    'hba1c': self.hba1c_target,
                    'cholesterol': self.cholesterol_target
                },
                'weights': {
                    'glucose': self.glucose_weight,
                    'systolic_bp': self.systolic_bp_weight,
                    'diastolic_bp': self.diastolic_bp_weight,
                    'hba1c': self.hba1c_weight,
                    'weight': self.weight_weight,
                    'cholesterol': self.cholesterol_weight
                },
                'improvement_multiplier': self.health_improvement_multiplier
            },
            'safety': {
                'thresholds': {
                    'severe_hypoglycemia': self.severe_hypoglycemia_threshold,
                    'moderate_hypoglycemia': self.moderate_hypoglycemia_threshold,
                    'severe_hyperglycemia': self.severe_hyperglycemia_threshold,
                    'moderate_hyperglycemia': self.moderate_hyperglycemia_threshold,
                    'severe_hypertension': self.severe_hypertension_threshold,
                    'moderate_hypertension': self.moderate_hypertension_threshold,
                    'severe_hypotension': self.severe_hypotension_threshold
                },
                'penalties': {
                    'severe_hypoglycemia': self.severe_hypoglycemia_penalty,
                    'moderate_hypoglycemia': self.moderate_hypoglycemia_penalty,
                    'severe_hyperglycemia': self.severe_hyperglycemia_penalty,
                    'moderate_hyperglycemia': self.moderate_hyperglycemia_penalty,
                    'severe_hypertension': self.severe_hypertension_penalty,
                    'moderate_hypertension': self.moderate_hypertension_penalty,
                    'severe_hypotension': self.severe_hypotension_penalty,
                    'drug_interaction': self.drug_interaction_penalty,
                    'overdose': self.overdose_penalty,
                    'contraindication': self.contraindication_penalty,
                    'emergency_visit': self.emergency_visit_penalty,
                    'hospitalization': self.hospitalization_penalty,
                    'icu_admission': self.icu_admission_penalty
                },
                'max_safe_dosages': self.max_safe_dosages
            },
            'cost': {
                'medication_costs': self.medication_costs,
                'service_costs': {
                    'appointment': self.appointment_cost,
                    'lab_test': self.lab_test_cost,
                    'emergency_visit': self.emergency_visit_cost,
                    'hospitalization_per_day': self.hospitalization_cost_per_day,
                    'icu_per_day': self.icu_cost_per_day
                },
                'communication_costs': {
                    'reminder': self.reminder_cost,
                    'phone_call': self.phone_call_cost
                },
                'normalization_factor': self.cost_normalization_factor
            },
            'shaping': {
                'use_potential_shaping': self.use_potential_shaping,
                'gamma': self.gamma,
                'normalize_components': self.normalize_components
            },
            'advanced': {
                'use_curriculum': self.use_curriculum,
                'curriculum_type': self.curriculum_type,
                'use_curiosity_bonus': self.use_curiosity_bonus,
                'curiosity_bonus_scale': self.curiosity_bonus_scale,
                'use_online_normalization': self.use_online_normalization,
                'normalization_clip_range': self.normalization_clip_range
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RewardConfig':
        """
        Create config from dictionary.
        
        Args:
            config_dict: Dictionary with config parameters
            
        Returns:
            RewardConfig instance
        """
        # Flatten nested dictionary
        flat_config = {}
        
        # Extract weights
        if 'weights' in config_dict:
            flat_config['w_adherence'] = config_dict['weights'].get('adherence', 1.0)
            flat_config['w_health'] = config_dict['weights'].get('health', 2.0)
            flat_config['w_safety'] = config_dict['weights'].get('safety', 5.0)
            flat_config['w_cost'] = config_dict['weights'].get('cost', 0.1)
        
        # Extract other parameters (simplified for brevity)
        # In production, would fully parse all nested parameters
        
        return cls(**flat_config)
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save config (JSON or YAML)
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'RewardConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to load config from
            
        Returns:
            RewardConfig instance
        """
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


# Predefined configurations for different scenarios

@dataclass
class ConservativeRewardConfig(RewardConfig):
    """Conservative config prioritizing safety over everything."""
    w_safety: float = 10.0  # Very high safety weight
    w_health: float = 1.0
    w_adherence: float = 0.5
    w_cost: float = 0.0  # Don't worry about cost
    
    # More stringent safety thresholds
    severe_hypoglycemia_threshold: float = 70.0  # Higher threshold
    severe_hyperglycemia_threshold: float = 250.0  # Lower threshold


@dataclass
class AggressiveRewardConfig(RewardConfig):
    """Aggressive config prioritizing health outcomes."""
    w_safety: float = 2.0  # Lower safety weight
    w_health: float = 5.0  # High health weight
    w_adherence: float = 1.0
    w_cost: float = 0.1
    
    health_improvement_multiplier: float = 3.0  # Strong improvement bonus


@dataclass
class CostAwareRewardConfig(RewardConfig):
    """Config that also considers cost-effectiveness."""
    w_safety: float = 5.0
    w_health: float = 2.0
    w_adherence: float = 1.0
    w_cost: float = 1.0  # Significant cost consideration
