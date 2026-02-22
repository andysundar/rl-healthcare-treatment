"""
Test script to verify encoder implementations.

This script tests all encoder architectures to ensure they work correctly
before integration into the main project.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from encoder_config import EncoderConfig
from transformer_encoder import TransformerPatientEncoder
from autoencoder import PatientAutoencoder, DeepAutoencoder
from train_encoder import PatientDataset, create_dummy_dataset


def test_config():
    """Test configuration creation and validation."""
    print("\n" + "="*60)
    print("TEST 1: Configuration")
    print("="*60)
    
    try:
        # Valid config
        config = EncoderConfig(
            encoder_type='transformer',
            hidden_dim=256,
            state_dim=128,
            num_layers=4,
            num_heads=8,
            device='cpu'
        )
        print("✓ Created valid configuration")
        print(f"  Total input dim: {config.get_total_input_dim()}")
        
        # Test auto device detection
        config_auto = EncoderConfig(device='auto')
        print(f"✓ Auto-detected device: {config_auto.device}")
        
        # Test invalid config
        try:
            invalid_config = EncoderConfig(
                hidden_dim=255,  # Not divisible by num_heads
                num_heads=8
            )
            print("✗ Should have raised error for invalid config")
            return False
        except ValueError:
            print("✓ Correctly caught invalid configuration")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_transformer_encoder():
    """Test Transformer encoder."""
    print("\n" + "="*60)
    print("TEST 2: Transformer Encoder")
    print("="*60)
    
    try:
        # Create config
        config = EncoderConfig(
            encoder_type='transformer',
            hidden_dim=128,
            state_dim=64,
            num_layers=2,
            num_heads=4,
            lab_dim=10,
            vital_dim=5,
            demo_dim=4,
            device='cpu'
        )
        
        # Create encoder
        encoder = TransformerPatientEncoder(config)
        print(f"✓ Created Transformer encoder")
        print(f"  Parameters: {encoder.count_parameters():,}")
        
        # Test forward pass
        batch_size = 8
        seq_len = 20
        
        batch = {
            'labs': torch.randn(batch_size, seq_len, config.lab_dim),
            'vitals': torch.randn(batch_size, seq_len, config.vital_dim),
            'demographics': torch.randn(batch_size, config.demo_dim),
            'medications': torch.randint(1, 100, (batch_size, seq_len)),
            'seq_lengths': torch.randint(10, seq_len, (batch_size,))
        }
        
        embeddings = encoder(batch)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {embeddings.shape}")
        
        assert embeddings.shape == (batch_size, config.state_dim), \
            f"Expected shape ({batch_size}, {config.state_dim}), got {embeddings.shape}"
        print("✓ Output shape correct")
        
        # Test different pooling strategies
        for pooling in ['mean', 'max', 'last']:
            encoder.set_pooling_type(pooling)
            emb = encoder(batch)
            assert emb.shape == (batch_size, config.state_dim)
        print("✓ All pooling strategies work")
        
        # Test encode_batch
        embeddings_batch = encoder.encode_batch(batch)
        assert embeddings_batch.shape == (batch_size, config.state_dim)
        print("✓ encode_batch works")
        
        # Test save/load
        checkpoint_path = Path('./test_checkpoint.pt')
        encoder.save_checkpoint(checkpoint_path, test=True)
        
        new_encoder = TransformerPatientEncoder(config)
        new_encoder.load_checkpoint(checkpoint_path)
        
        # Verify outputs match
        new_embeddings = new_encoder(batch)
        assert torch.allclose(embeddings, new_embeddings, atol=1e-6)
        print("✓ Save/load works correctly")
        
        # Cleanup
        checkpoint_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Transformer encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autoencoder():
    """Test Autoencoder."""
    print("\n" + "="*60)
    print("TEST 3: Autoencoder")
    print("="*60)
    
    try:
        # Create config
        config = EncoderConfig(
            encoder_type='autoencoder',
            state_dim=32,
            lab_dim=10,
            vital_dim=5,
            demo_dim=4,
            device='cpu'
        )
        
        # Test standard autoencoder
        ae = PatientAutoencoder(config, variational=False)
        print(f"✓ Created Autoencoder")
        print(f"  Parameters: {ae.count_parameters():,}")
        print(f"  Input dim: {ae.input_dim}")
        
        # Test forward pass
        batch_size = 16
        batch = {
            'labs': torch.randn(batch_size, 20, config.lab_dim),
            'vitals': torch.randn(batch_size, 20, config.vital_dim),
            'demographics': torch.randn(batch_size, config.demo_dim)
        }
        
        reconstruction, latent = ae(batch)
        print(f"✓ Forward pass successful")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Reconstruction shape: {reconstruction.shape}")
        
        assert latent.shape == (batch_size, config.state_dim)
        assert reconstruction.shape == (batch_size, ae.input_dim)
        print("✓ Output shapes correct")
        
        # Test encode only
        latent_only = ae.encode_batch(batch)
        assert latent_only.shape == (batch_size, config.state_dim)
        print("✓ Encode-only works")
        
        # Test loss computation
        original = ae._flatten_patient_data(batch)
        losses = ae.compute_loss(original, reconstruction, latent)
        
        assert 'reconstruction_loss' in losses
        assert 'total_loss' in losses
        print(f"✓ Loss computation works")
        print(f"  Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Autoencoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae():
    """Test Variational Autoencoder."""
    print("\n" + "="*60)
    print("TEST 4: Variational Autoencoder")
    print("="*60)
    
    try:
        # Create config
        config = EncoderConfig(
            encoder_type='autoencoder',
            state_dim=32,
            lab_dim=10,
            vital_dim=5,
            demo_dim=4,
            device='cpu'
        )
        
        # Create VAE
        vae = PatientAutoencoder(config, variational=True)
        print(f"✓ Created VAE")
        print(f"  Parameters: {vae.count_parameters():,}")
        
        # Test forward pass
        batch_size = 16
        batch = {
            'labs': torch.randn(batch_size, 15, config.lab_dim),
            'vitals': torch.randn(batch_size, 15, config.vital_dim),
            'demographics': torch.randn(batch_size, config.demo_dim)
        }
        
        reconstruction, latent = vae(batch)
        print(f"✓ VAE forward pass successful")
        
        # Get VAE parameters
        mu, logvar = vae.get_vae_params()
        assert mu.shape == (batch_size, config.state_dim)
        assert logvar.shape == (batch_size, config.state_dim)
        print(f"✓ VAE parameters extracted")
        print(f"  Mean shape: {mu.shape}")
        print(f"  Logvar shape: {logvar.shape}")
        
        # Test loss with KL divergence
        original = vae._flatten_patient_data(batch)
        losses = vae.compute_loss(original, reconstruction, latent, kl_weight=0.1)
        
        assert 'reconstruction_loss' in losses
        assert 'kl_loss' in losses
        assert 'total_loss' in losses
        print(f"✓ VAE loss computation works")
        print(f"  Reconstruction: {losses['reconstruction_loss'].item():.4f}")
        print(f"  KL divergence: {losses['kl_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ VAE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deep_autoencoder():
    """Test Deep Autoencoder."""
    print("\n" + "="*60)
    print("TEST 5: Deep Autoencoder")
    print("="*60)
    
    try:
        config = EncoderConfig(
            encoder_type='autoencoder',
            state_dim=64,
            lab_dim=15,
            vital_dim=8,
            demo_dim=6,
            device='cpu'
        )
        
        # Create deep autoencoder
        deep_ae = DeepAutoencoder(config)
        print(f"✓ Created Deep Autoencoder")
        print(f"  Parameters: {deep_ae.count_parameters():,}")
        
        # Test forward pass
        batch = {
            'labs': torch.randn(8, 25, config.lab_dim),
            'vitals': torch.randn(8, 25, config.vital_dim),
            'demographics': torch.randn(8, config.demo_dim)
        }
        
        reconstruction, latent = deep_ae(batch)
        assert latent.shape == (8, config.state_dim)
        print("✓ Deep autoencoder works")
        
        return True
        
    except Exception as e:
        print(f"✗ Deep autoencoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test PatientDataset."""
    print("\n" + "="*60)
    print("TEST 6: PatientDataset")
    print("="*60)
    
    try:
        # Create dummy data
        data = create_dummy_dataset(n_samples=50, seq_len_range=(10, 30))
        print(f"✓ Created dummy dataset with {len(data)} samples")
        
        # Create dataset
        dataset = PatientDataset(data, max_seq_len=40, normalize=True)
        print(f"✓ Created PatientDataset")
        
        # Test __getitem__
        sample = dataset[0]
        assert 'labs' in sample
        assert 'vitals' in sample
        assert 'seq_lengths' in sample
        print(f"✓ Sample retrieval works")
        print(f"  Sample keys: {list(sample.keys())}")
        
        # Test __len__
        assert len(dataset) == 50
        print(f"✓ Dataset length correct: {len(dataset)}")
        
        # Test with DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        batch = next(iter(loader))
        assert batch['labs'].shape[0] == 8
        print(f"✓ DataLoader works")
        print(f"  Batch shape: {batch['labs'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("PATIENT STATE ENCODER - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Configuration", test_config),
        ("Transformer Encoder", test_transformer_encoder),
        ("Autoencoder", test_autoencoder),
        ("VAE", test_vae),
        ("Deep Autoencoder", test_deep_autoencoder),
        ("PatientDataset", test_dataset),
    ]
    
    results = []
    for test_name, test_func in tests:
        passed = test_func()
        results.append((test_name, passed))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<25} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
