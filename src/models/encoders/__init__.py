"""
Patient State Encoders for Healthcare RL.

This package provides comprehensive encoder implementations for transforming
sequential patient data into fixed-size state representations for RL agents.
"""

from .encoder_config import EncoderConfig
from .base_encoder import BasePatientEncoder
from .transformer_encoder import TransformerPatientEncoder
from .autoencoder import PatientAutoencoder, DeepAutoencoder
from .train_encoder import (
    EncoderTrainer,
    PatientDataset,
    create_dummy_dataset
)

__version__ = '1.0.0'
__author__ = 'Healthcare RL Team'

__all__ = [
    # Configuration
    'EncoderConfig',
    
    # Base classes
    'BasePatientEncoder',
    
    # Encoder implementations
    'TransformerPatientEncoder',
    'PatientAutoencoder',
    'DeepAutoencoder',
    
    # Training utilities
    'EncoderTrainer',
    'PatientDataset',
    'create_dummy_dataset',
]


def get_encoder(encoder_type: str, config: EncoderConfig) -> BasePatientEncoder:
    """
    Factory function to create encoder by type.
    
    Args:
        encoder_type: Type of encoder ('transformer', 'autoencoder', 'deep_autoencoder', 'vae')
        config: Encoder configuration
    
    Returns:
        Encoder instance
    
    Example:
        >>> config = EncoderConfig(encoder_type='transformer')
        >>> encoder = get_encoder('transformer', config)
    """
    encoder_map = {
        'transformer': TransformerPatientEncoder,
        'autoencoder': lambda cfg: PatientAutoencoder(cfg, variational=False),
        'deep_autoencoder': lambda cfg: DeepAutoencoder(cfg, variational=False),
        'vae': lambda cfg: PatientAutoencoder(cfg, variational=True),
        'deep_vae': lambda cfg: DeepAutoencoder(cfg, variational=True),
    }
    
    if encoder_type not in encoder_map:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available types: {list(encoder_map.keys())}"
        )
    
    return encoder_map[encoder_type](config)


# Quick start example
def quick_start_example():
    """
    Quick start example for encoder usage.
    
    This demonstrates the minimal code needed to create and use an encoder.
    """
    import torch
    
    # Create configuration
    config = EncoderConfig(
        encoder_type='transformer',
        state_dim=128,
        device='cpu'
    )
    
    # Create encoder
    encoder = get_encoder('transformer', config)
    
    # Create sample data
    batch = {
        'labs': torch.randn(8, 30, config.lab_dim),
        'vitals': torch.randn(8, 30, config.vital_dim),
        'demographics': torch.randn(8, config.demo_dim),
        'medications': torch.randint(1, config.med_vocab_size, (8, 30)),
        'seq_lengths': torch.randint(20, 30, (8,))
    }
    
    # Encode
    embeddings = encoder(batch)
    
    print(f"Created {config.encoder_type} encoder")
    print(f"Input: batch_size=8, seq_len=30")
    print(f"Output: {embeddings.shape}")
    
    return encoder, embeddings


if __name__ == '__main__':
    # Run quick start example if module is run directly
    encoder, embeddings = quick_start_example()
