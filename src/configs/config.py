"""
Evaluation Configuration for Healthcare RL System
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import yaml


@dataclass
class OPEConfig:
    """Off-Policy Evaluation configuration."""
    gamma: float = 0.99
    methods: List[str] = field(default_factory=lambda: ['wis', 'dr'])
    clip_ratio: float = 10.0
    n_bootstrap_samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class SafetyConfig:
    """Safety evaluation configuration."""
    safe_glucose_range: Tuple[float, float] = (70, 180)
    safe_bp_systolic_range: Tuple[float, float] = (90, 140)
    safe_bp_diastolic_range: Tuple[float, float] = (60, 90)
    critical_glucose_low: float = 54
    critical_glucose_high: float = 250
    minor_severity_threshold: float = 0.1
    moderate_severity_threshold: float = 0.3
    severe_severity_threshold: float = 0.5


@dataclass
class ClinicalConfig:
    """Clinical evaluation configuration."""
    target_glucose_range: Tuple[float, float] = (80, 130)
    target_bp_systolic_range: Tuple[float, float] = (90, 120)
    target_bp_diastolic_range: Tuple[float, float] = (60, 80)
    target_hba1c: float = 6.5
    adverse_events: List[str] = field(default_factory=lambda: [
        'hypoglycemia', 'severe_hyperglycemia', 'hypertensive_crisis'
    ])


@dataclass
class PerformanceConfig:
    """Performance evaluation configuration."""
    success_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'glucose_control': 0.7,
        'bp_control': 0.8,
        'adherence': 0.8
    })
    max_episode_length: int = 90
    min_episode_length: int = 7


@dataclass
class EvaluationConfig:
    """Master evaluation configuration."""
    ope: OPEConfig = field(default_factory=OPEConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    random_seed: int = 42
    n_eval_episodes: int = 1000
    verbose: bool = True
    save_results: bool = True
    results_dir: str = 'evaluation_results'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'EvaluationConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'ope': self.ope.__dict__,
            'safety': self.safety.__dict__,
            'clinical': self.clinical.__dict__,
            'performance': self.performance.__dict__,
            'random_seed': self.random_seed,
            'n_eval_episodes': self.n_eval_episodes,
            'verbose': self.verbose,
            'save_results': self.save_results,
            'results_dir': self.results_dir
        }
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
