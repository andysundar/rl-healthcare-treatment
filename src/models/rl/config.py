"""
Configuration Module for Offline RL

Centralized configuration management using dataclasses.
Provides type-safe configuration for all components.

Author: Anindya Bandopadhyay (M23CSA508)
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class CQLConfig:
    """Configuration for Conservative Q-Learning agent."""
    
    # Network architecture
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    
    # Training hyperparameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient
    q_lr: float = 3e-4  # Q-network learning rate
    policy_lr: float = 1e-4  # Policy learning rate
    
    # CQL-specific
    cql_alpha: float = 1.0  # CQL regularization weight
    target_update_freq: int = 2  # Target network update frequency
    num_random_actions: int = 10  # Number of random actions for CQL penalty
    
    # General
    batch_size: int = 256
    device: str = 'cpu'
    action_space: str = 'continuous'  # 'continuous' or 'discrete'
    
    def to_dict(self):
        """Convert to dictionary."""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save(self, yaml_path: str):
        """Save to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class BCQConfig:
    """Configuration for Batch-Constrained Q-Learning agent."""
    
    # Network architecture
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    latent_dim: int = 64  # VAE latent dimension
    
    # Training hyperparameters
    gamma: float = 0.99
    tau: float = 0.005
    q_lr: float = 3e-4
    vae_lr: float = 3e-4
    perturbation_lr: float = 3e-4
    
    # BCQ-specific
    n_action_samples: int = 10  # Number of actions to sample from VAE
    phi: float = 0.05  # Perturbation scaling factor
    
    # General
    batch_size: int = 256
    device: str = 'cpu'
    
    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save(self, yaml_path: str):
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class ReplayBufferConfig:
    """Configuration for replay buffer."""
    
    capacity: int
    state_dim: int
    action_dim: int
    device: str = 'cpu'
    
    # Prioritized replay
    use_prioritized: bool = False
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment: float = 0.001
    epsilon: float = 1e-6


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training iterations
    num_iterations: int = 10000
    batch_size: int = 256
    
    # Evaluation
    eval_freq: int = 1000
    eval_episodes: int = 10
    
    # Checkpointing
    save_freq: int = 5000
    save_dir: str = './checkpoints'
    
    # Logging
    log_freq: int = 100
    use_mlflow: bool = False
    use_wandb: bool = False
    experiment_name: str = 'offline_rl_healthcare'
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Random seed
    seed: int = 42
    
    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, yaml_path: str):
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class SafetyConfig:
    """Configuration for safety constraints."""
    
    # Safety thresholds
    enable_safety: bool = True
    safety_threshold: float = 0.5  # Minimum safety score
    
    # Constraint types
    use_hard_constraints: bool = True
    use_soft_constraints: bool = True
    
    # Safety critic
    safety_critic_hidden_dim: int = 256
    safety_critic_lr: float = 3e-4
    
    # Physiological limits
    glucose_min: float = 70.0  # mg/dL
    glucose_max: float = 200.0
    bp_systolic_min: float = 90.0  # mmHg
    bp_systolic_max: float = 140.0
    
    # Dosage limits
    max_insulin_dose: float = 100.0  # units
    max_medication_dosage: float = 1.0  # normalized


@dataclass
class HealthcareConfig:
    """Master configuration combining all components."""
    
    # Agent configuration
    agent_type: str = 'cql'  # 'cql' or 'bcq'
    agent_config: Optional[CQLConfig] = None
    
    # Buffer configuration
    buffer_config: Optional[ReplayBufferConfig] = None
    
    # Training configuration
    training_config: Optional[TrainingConfig] = None
    
    # Safety configuration
    safety_config: Optional[SafetyConfig] = None
    
    # Data paths
    data_dir: str = './data'
    mimic_data_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.agent_config is None:
            if self.agent_type == 'cql':
                self.agent_config = CQLConfig(state_dim=128, action_dim=5)
            elif self.agent_type == 'bcq':
                self.agent_config = BCQConfig(state_dim=128, action_dim=5)
        
        if self.buffer_config is None:
            self.buffer_config = ReplayBufferConfig(
                capacity=100000,
                state_dim=self.agent_config.state_dim,
                action_dim=self.agent_config.action_dim
            )
        
        if self.training_config is None:
            self.training_config = TrainingConfig()
        
        if self.safety_config is None:
            self.safety_config = SafetyConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load complete configuration from YAML."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        agent_type = config_dict.get('agent_type', 'cql')
        
        # Parse agent config
        if agent_type == 'cql':
            agent_config = CQLConfig.from_dict(config_dict.get('agent_config', {}))
        else:
            agent_config = BCQConfig.from_dict(config_dict.get('agent_config', {}))
        
        # Parse buffer config
        buffer_config = ReplayBufferConfig(**config_dict.get('buffer_config', {}))
        
        # Parse training config
        training_config = TrainingConfig(**config_dict.get('training_config', {}))
        
        # Parse safety config
        safety_config = SafetyConfig(**config_dict.get('safety_config', {}))
        
        return cls(
            agent_type=agent_type,
            agent_config=agent_config,
            buffer_config=buffer_config,
            training_config=training_config,
            safety_config=safety_config,
            data_dir=config_dict.get('data_dir', './data'),
            mimic_data_path=config_dict.get('mimic_data_path')
        )
    
    def save(self, yaml_path: str):
        """Save complete configuration to YAML."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'agent_type': self.agent_type,
            'agent_config': self.agent_config.to_dict(),
            'buffer_config': self.buffer_config.__dict__,
            'training_config': self.training_config.to_dict(),
            'safety_config': self.safety_config.__dict__,
            'data_dir': self.data_dir,
            'mimic_data_path': self.mimic_data_path
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Default configurations for common scenarios

def get_diabetes_management_config() -> HealthcareConfig:
    """Get default configuration for diabetes management task."""
    agent_config = CQLConfig(
        state_dim=128,  # Patient state representation
        action_dim=5,   # Insulin dosage, appointment scheduling, etc.
        hidden_dim=256,
        cql_alpha=2.0,  # Higher conservatism for safety
        gamma=0.99,
        batch_size=256
    )
    
    training_config = TrainingConfig(
        num_iterations=50000,
        eval_freq=1000,
        save_freq=5000,
        early_stopping_patience=10
    )
    
    safety_config = SafetyConfig(
        enable_safety=True,
        glucose_min=70.0,
        glucose_max=200.0
    )
    
    return HealthcareConfig(
        agent_type='cql',
        agent_config=agent_config,
        training_config=training_config,
        safety_config=safety_config
    )


def get_mimic_experiment_config() -> HealthcareConfig:
    """Get default configuration for MIMIC-III experiments."""
    agent_config = CQLConfig(
        state_dim=256,  # Larger state for MIMIC complexity
        action_dim=10,
        hidden_dim=512,  # Larger networks for complex data
        cql_alpha=1.0,
        gamma=0.99,
        batch_size=512  # Larger batches for stable learning
    )
    
    buffer_config = ReplayBufferConfig(
        capacity=1000000,  # Large buffer for MIMIC dataset
        state_dim=256,
        action_dim=10
    )
    
    training_config = TrainingConfig(
        num_iterations=100000,
        eval_freq=2000,
        save_freq=10000,
        use_mlflow=True,
        experiment_name='mimic_cql_experiment'
    )
    
    return HealthcareConfig(
        agent_type='cql',
        agent_config=agent_config,
        buffer_config=buffer_config,
        training_config=training_config,
        data_dir='./data/mimic'
    )
