"""
Safety System Configuration
Defines configuration for all safety components
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class SafetyConfig:
    """Configuration for safety system"""
    
    # Dosage limits per drug (drug_name: (min_dose, max_dose))
    drug_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'insulin': (0.0, 100.0),  # units
        'metformin': (500.0, 2000.0),  # mg
        'glipizide': (2.5, 20.0),  # mg
        'lisinopril': (2.5, 40.0),  # mg
        'atorvastatin': (10.0, 80.0),  # mg
    })
    
    # Safe physiological ranges (variable: (min_value, max_value))
    safe_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'glucose': (70.0, 200.0),  # mg/dL
        'blood_pressure_systolic': (90.0, 140.0),  # mmHg
        'blood_pressure_diastolic': (60.0, 90.0),  # mmHg
        'heart_rate': (60.0, 100.0),  # bpm
        'temperature': (36.0, 38.0),  # Celsius
        'oxygen_saturation': (95.0, 100.0),  # %
    })
    
    # Contraindication database path
    contraindication_db_path: str = "data/contraindications.json"
    
    # Safety critic settings
    safety_threshold: float = 0.8
    critic_hidden_dim: int = 256
    critic_lr: float = 1e-4
    
    # Optimization settings
    max_optimization_iterations: int = 100
    optimization_tolerance: float = 1e-6
    
    # Action bounds for optimization (will be set based on action space)
    action_bounds: Optional[list] = None
    
    # Maximum frequency limits
    max_reminders_per_week: int = 7
    min_appointment_interval_days: int = 3
    
    def __post_init__(self):
        if self.action_bounds is None:
            # Default action bounds (5-dimensional action space)
            self.action_bounds = [(0.0, 1.0)] * 5
