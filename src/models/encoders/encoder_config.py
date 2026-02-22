"""
Configuration dataclass for patient state encoders.

This module defines the configuration parameters for all encoder types,
providing a unified interface for encoder initialization and hyperparameter tuning.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class EncoderConfig:
    """
    Configuration for patient state encoders.
    
    This dataclass centralizes all hyperparameters for different encoder architectures,
    making it easy to experiment with different configurations and maintain consistency
    across training runs.
    
    Attributes:
        encoder_type: Type of encoder ('transformer', 'autoencoder', 'clinical_bert', 'bio_gpt')
        hidden_dim: Hidden dimension size for intermediate layers
        state_dim: Output state embedding dimension
        num_layers: Number of layers (transformer/autoencoder)
        num_heads: Number of attention heads (transformer only)
        dropout: Dropout probability for regularization
        
        # Modality-specific dimensions
        lab_dim: Dimension of laboratory test features
        vital_dim: Dimension of vital sign features
        demo_dim: Dimension of demographic features
        med_vocab_size: Size of medication vocabulary for embedding
        diag_vocab_size: Size of diagnosis code vocabulary
        proc_vocab_size: Size of procedure code vocabulary
        
        # Training parameters
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization coefficient
        max_seq_len: Maximum sequence length for temporal data
        batch_size: Training batch size
        
        # Model-specific parameters
        feedforward_dim: Feedforward network dimension (transformer)
        pretrained_model: Path to pretrained model (BERT/GPT)
        freeze_pretrained: Whether to freeze pretrained model weights
        use_positional_encoding: Enable positional encoding for sequences
        
        # Device configuration
        device: Device to use ('cuda', 'mps', 'cpu')
    """
    
    # Core architecture parameters
    encoder_type: Literal['transformer', 'autoencoder', 'clinical_bert', 'bio_gpt'] = 'transformer'
    hidden_dim: int = 256
    state_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Input feature dimensions
    lab_dim: int = 20
    vital_dim: int = 10
    demo_dim: int = 8
    med_vocab_size: int = 500
    diag_vocab_size: int = 1000
    proc_vocab_size: int = 500
    
    # Training hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_seq_len: int = 100
    batch_size: int = 32
    
    # Transformer-specific parameters
    feedforward_dim: int = 1024
    use_positional_encoding: bool = True
    
    # Pretrained model parameters
    pretrained_model: Optional[str] = None
    freeze_pretrained: bool = False
    
    # Device configuration
    device: str = 'cpu'
    
    # Checkpoint configuration
    save_checkpoints: bool = True          # Enable/disable checkpoint saving
    checkpoint_dir: Optional[str] = './checkpoints/encoders'  # Where to save
    save_frequency: int = 1                # Save every N epochs
    keep_best_only: bool = False           # Keep only best checkpoint or all
    save_optimizer_state: bool = True      # Save optimizer for resuming
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        
        if self.encoder_type == 'transformer' and self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads}) for transformer encoder"
            )
        
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        
        # Auto-detect device if not specified or set to 'auto'
        if self.device == 'auto':
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
    
    def get_total_input_dim(self) -> int:
        """
        Calculate total input dimension across all modalities.
        
        Returns:
            Total input feature dimension
        """
        return self.lab_dim + self.vital_dim + self.demo_dim
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'EncoderConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            EncoderConfig instance
        """
        return cls(**config_dict)
