"""
Example usage of patient state encoders.

This script demonstrates how to use different encoder architectures,
train them, and integrate with RL agents.
"""

import torch
import numpy as np
from pathlib import Path
import logging

from encoder_config import EncoderConfig
from transformer_encoder import TransformerPatientEncoder
from autoencoder import PatientAutoencoder, DeepAutoencoder
from train_encoder import PatientDataset, EncoderTrainer, create_dummy_dataset
from torch.utils.data import DataLoader


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_transformer_encoder():
    """
    Example 1: Using Transformer encoder for sequential patient data.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Transformer Encoder")
    print("="*80 + "\n")
    
    # Create configuration
    config = EncoderConfig(
        encoder_type='transformer',
        hidden_dim=256,
        state_dim=128,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        lab_dim=20,
        vital_dim=10,
        demo_dim=8,
        med_vocab_size=500,
        diag_vocab_size=1000,
        max_seq_len=100,
        device='cpu'  # Change to 'mps' for M4 Mac, 'cuda' for NVIDIA GPU
    )
    
    # Create encoder
    encoder = TransformerPatientEncoder(config)
    print(f"Created encoder: {encoder}")
    print(f"Total parameters: {encoder.count_parameters():,}")
    
    # Create sample batch
    batch_size = 32
    seq_len = 50
    
    batch = {
        'labs': torch.randn(batch_size, seq_len, config.lab_dim),
        'vitals': torch.randn(batch_size, seq_len, config.vital_dim),
        'demographics': torch.randn(batch_size, config.demo_dim),
        'medications': torch.randint(1, config.med_vocab_size, (batch_size, seq_len)),
        'diagnoses': torch.randint(1, config.diag_vocab_size, (batch_size, seq_len)),
        'seq_lengths': torch.randint(20, seq_len, (batch_size,))
    }
    
    # Encode batch
    print("\nEncoding batch...")
    state_embeddings = encoder(batch)
    print(f"Input batch size: {batch_size}, seq_len: {seq_len}")
    print(f"Output embeddings shape: {state_embeddings.shape}")
    print(f"Sample embedding (first 10 dims): {state_embeddings[0, :10]}")
    
    # Save checkpoint
    checkpoint_dir = Path('./checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'transformer_encoder.pt'
    encoder.save_checkpoint(checkpoint_path, epoch=0)
    print(f"\nSaved checkpoint to: {checkpoint_path}")
    
    # Test different pooling strategies
    print("\nTesting different pooling strategies:")
    for pooling in ['mean', 'max', 'last']:
        encoder.set_pooling_type(pooling)
        embeddings = encoder(batch)
        print(f"  {pooling} pooling: {embeddings.shape}")
    
    return encoder


def example_autoencoder():
    """
    Example 2: Training autoencoder for patient representation learning.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Autoencoder Training")
    print("="*80 + "\n")
    
    # Create configuration
    config = EncoderConfig(
        encoder_type='autoencoder',
        state_dim=128,
        lab_dim=20,
        vital_dim=10,
        demo_dim=8,
        device='cpu'
    )
    
    # Create autoencoder
    autoencoder = PatientAutoencoder(config, variational=False)
    print(f"Created autoencoder: {autoencoder}")
    print(f"Input dimension: {autoencoder.input_dim}")
    print(f"Latent dimension: {config.state_dim}")
    
    # Create dummy dataset
    print("\nCreating dummy dataset...")
    train_data = create_dummy_dataset(n_samples=800, seq_len_range=(20, 50))
    val_data = create_dummy_dataset(n_samples=200, seq_len_range=(20, 50))
    
    # Create data loaders
    train_dataset = PatientDataset(train_data, max_seq_len=50, normalize=True)
    val_dataset = PatientDataset(val_data, max_seq_len=50, normalize=True)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create trainer
    trainer_config = {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10,
        'grad_clip': 1.0
    }
    
    trainer = EncoderTrainer(autoencoder, trainer_config)
    
    # Train (just 5 epochs for demo)
    print("\nTraining autoencoder...")
    history = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,  # Use more epochs in practice (50-100)
        save_dir=Path('./checkpoints')
    )
    
    print(f"\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Test encoding
    print("\nTesting trained encoder:")
    sample_batch = next(iter(val_loader))
    with torch.no_grad():
        latent = autoencoder.encode_batch(sample_batch)
    print(f"Encoded {sample_batch['labs'].size(0)} samples to shape: {latent.shape}")
    
    return autoencoder


def example_vae():
    """
    Example 3: Variational Autoencoder (VAE).
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Variational Autoencoder (VAE)")
    print("="*80 + "\n")
    
    # Create configuration
    config = EncoderConfig(
        encoder_type='autoencoder',
        state_dim=128,
        lab_dim=20,
        vital_dim=10,
        demo_dim=8,
        device='cpu'
    )
    
    # Create VAE
    vae = PatientAutoencoder(config, variational=True)
    print(f"Created VAE: {vae}")
    
    # Create sample data
    sample_data = {
        'labs': torch.randn(16, 30, config.lab_dim),
        'vitals': torch.randn(16, 30, config.vital_dim),
        'demographics': torch.randn(16, config.demo_dim)
    }
    
    # Forward pass
    print("\nVAE forward pass:")
    reconstruction, latent = vae(sample_data)
    
    # Get VAE parameters
    mu, logvar = vae.get_vae_params()
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Compute loss
    original = vae._flatten_patient_data(sample_data)
    losses = vae.compute_loss(original, reconstruction, latent, kl_weight=0.1)
    
    print(f"\nLoss components:")
    print(f"  Reconstruction: {losses['reconstruction_loss'].item():.4f}")
    print(f"  KL Divergence: {losses['kl_loss'].item():.4f}")
    print(f"  Total: {losses['total_loss'].item():.4f}")
    
    return vae


def example_rl_integration():
    """
    Example 4: Integrating encoder with RL agent.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: RL Integration")
    print("="*80 + "\n")
    
    # Create and load pre-trained encoder
    config = EncoderConfig(
        encoder_type='transformer',
        hidden_dim=256,
        state_dim=128,
        num_layers=3,
        num_heads=8,
        device='cpu'
    )
    
    encoder = TransformerPatientEncoder(config)
    print(f"Created encoder for RL: {encoder}")
    
    # Simulate RL environment interaction
    print("\nSimulating RL environment interaction:")
    
    class SimpleRLAgent:
        """Simple RL agent using encoder."""
        
        def __init__(self, encoder, action_dim=5):
            self.encoder = encoder
            self.encoder.eval()  # Set to eval mode
            
            # Policy network: state_embedding -> action
            self.policy = torch.nn.Sequential(
                torch.nn.Linear(encoder.get_embedding_dim(), 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, action_dim),
                torch.nn.Softmax(dim=-1)
            )
        
        def get_action(self, patient_data):
            """Get action from policy."""
            with torch.no_grad():
                # Encode patient state
                state_embedding = self.encoder(patient_data)
                
                # Get action probabilities
                action_probs = self.policy(state_embedding)
            
            return action_probs
    
    # Create RL agent
    agent = SimpleRLAgent(encoder, action_dim=5)
    
    # Simulate environment step
    patient_data = {
        'labs': torch.randn(1, 40, 20),
        'vitals': torch.randn(1, 40, 10),
        'demographics': torch.randn(1, 8),
        'medications': torch.randint(1, 500, (1, 40)),
        'seq_lengths': torch.tensor([40])
    }
    
    action_probs = agent.get_action(patient_data)
    print(f"Action probabilities: {action_probs}")
    print(f"Selected action: {action_probs.argmax().item()}")
    
    return agent


def example_encoder_comparison():
    """
    Example 5: Comparing different encoder architectures.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Encoder Architecture Comparison")
    print("="*80 + "\n")
    
    # Common config
    base_config = EncoderConfig(
        state_dim=128,
        lab_dim=20,
        vital_dim=10,
        demo_dim=8,
        device='cpu'
    )
    
    # Create different encoders
    encoders = {
        'Transformer': TransformerPatientEncoder(base_config),
        'Autoencoder': PatientAutoencoder(base_config),
        'Deep Autoencoder': DeepAutoencoder(base_config),
        'VAE': PatientAutoencoder(base_config, variational=True)
    }
    
    # Sample data
    sample_data = {
        'labs': torch.randn(16, 30, base_config.lab_dim),
        'vitals': torch.randn(16, 30, base_config.vital_dim),
        'demographics': torch.randn(16, base_config.demo_dim),
        'medications': torch.randint(1, 500, (16, 30)),
        'seq_lengths': torch.randint(20, 30, (16,))
    }
    
    # Compare encoders
    print("\nEncoder Comparison:")
    print("-" * 80)
    print(f"{'Architecture':<20} {'Parameters':<15} {'Output Shape':<15} {'Time (ms)':<10}")
    print("-" * 80)
    
    for name, encoder in encoders.items():
        import time
        
        # Measure encoding time
        start = time.time()
        with torch.no_grad():
            if 'Autoencoder' in name or 'VAE' in name:
                output = encoder.encode_batch(sample_data)
            else:
                output = encoder.encode_batch(sample_data)
        elapsed = (time.time() - start) * 1000
        
        params = encoder.count_parameters()
        
        print(f"{name:<20} {params:<15,} {str(output.shape):<15} {elapsed:<10.2f}")
    
    print("-" * 80)


def example_save_load():
    """
    Example 6: Saving and loading encoder checkpoints.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Save and Load Checkpoints")
    print("="*80 + "\n")
    
    # Create and save encoder
    config = EncoderConfig(
        encoder_type='transformer',
        hidden_dim=256,
        state_dim=128,
        device='cpu'
    )
    
    encoder = TransformerPatientEncoder(config)
    
    # Create dummy data and encode
    sample = {
        'labs': torch.randn(4, 20, config.lab_dim),
        'vitals': torch.randn(4, 20, config.vital_dim),
        'demographics': torch.randn(4, config.demo_dim),
        'medications': torch.randint(1, 500, (4, 20)),
        'seq_lengths': torch.tensor([15, 18, 20, 12])
    }
    
    output_before = encoder(sample)
    print(f"Output before save: {output_before[0, :5]}")
    
    # Save checkpoint
    checkpoint_path = Path('./checkpoints/test_encoder.pt')
    encoder.save_checkpoint(
        checkpoint_path,
        epoch=10,
        note="Test checkpoint"
    )
    print(f"\nSaved checkpoint to: {checkpoint_path}")
    
    # Create new encoder and load checkpoint
    new_encoder = TransformerPatientEncoder(config)
    metadata = new_encoder.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint with metadata: {metadata}")
    
    # Verify outputs match
    output_after = new_encoder(sample)
    print(f"Output after load: {output_after[0, :5]}")
    
    match = torch.allclose(output_before, output_after, atol=1e-6)
    print(f"\nOutputs match: {match}")
    
    return encoder


if __name__ == '__main__':
    print("\n" + "="*80)
    print("PATIENT STATE ENCODER - COMPREHENSIVE EXAMPLES")
    print("="*80)
    
    # Run all examples
    try:
        # Example 1: Transformer encoder
        transformer_encoder = example_transformer_encoder()
        
        # Example 2: Autoencoder training
        autoencoder = example_autoencoder()
        
        # Example 3: VAE
        vae = example_vae()
        
        # Example 4: RL integration
        rl_agent = example_rl_integration()
        
        # Example 5: Encoder comparison
        example_encoder_comparison()
        
        # Example 6: Save/load
        saved_encoder = example_save_load()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        raise
