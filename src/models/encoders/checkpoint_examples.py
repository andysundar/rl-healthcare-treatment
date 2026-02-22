"""
Comprehensive examples for checkpoint management in encoder training.

This demonstrates all checkpoint-related features including:
- Automatic saving (default behavior)
- Disabling checkpoints
- Resuming training
- Different checkpoint strategies
"""

import torch
from pathlib import Path
import logging

from encoder_config import EncoderConfig
from autoencoder import PatientAutoencoder
from train_encoder import EncoderTrainer, create_dummy_dataset, PatientDataset
from torch.utils.data import DataLoader


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_default_checkpoint_saving():
    """
    Example 1: Default behavior - automatic checkpoint saving.
    
    By default, checkpoints are saved automatically during training.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Default Checkpoint Saving (ENABLED by default)")
    print("="*80 + "\n")
    
    # Create configuration - save_checkpoints=True by default
    config = EncoderConfig(
        encoder_type='autoencoder',
        state_dim=64,
        lab_dim=10,
        vital_dim=5,
        demo_dim=4,
        device='cpu',
        # Checkpoint configuration (these are defaults)
        save_checkpoints=True,
        checkpoint_dir='./checkpoints/example1',
        save_frequency=5,  # Save every 5 epochs
        keep_best_only=False,  # Keep all checkpoints
        save_optimizer_state=True  # Save optimizer for resuming
    )
    
    print(f"✓ Checkpoint saving: {config.save_checkpoints}")
    print(f"✓ Checkpoint directory: {config.checkpoint_dir}")
    print(f"✓ Save frequency: every {config.save_frequency} epochs")
    print(f"✓ Keep best only: {config.keep_best_only}")
    
    # Create encoder
    encoder = PatientAutoencoder(config)
    
    # Create dummy data
    train_data = create_dummy_dataset(n_samples=100, seq_len_range=(10, 20))
    val_data = create_dummy_dataset(n_samples=20, seq_len_range=(10, 20))
    
    train_dataset = PatientDataset(train_data, max_seq_len=30)
    val_dataset = PatientDataset(val_data, max_seq_len=30)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create trainer
    trainer_config = {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'early_stopping_patience': 5
    }
    
    trainer = EncoderTrainer(encoder, trainer_config)
    
    # Train - checkpoints saved automatically
    print("\n📚 Starting training (10 epochs)...")
    history = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10
    )
    
    # List saved checkpoints
    checkpoint_dir = Path(config.checkpoint_dir)
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        print(f"\n💾 Saved checkpoints ({len(checkpoints)}):")
        for cp in sorted(checkpoints):
            print(f"   - {cp.name}")
    
    print("\n✅ Training complete with automatic checkpoint saving!")


def example_2_disable_checkpoint_saving():
    """
    Example 2: Disable checkpoint saving.
    
    Set save_checkpoints=False to train without saving checkpoints.
    Useful for quick experiments or when storage is limited.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Disable Checkpoint Saving")
    print("="*80 + "\n")
    
    # Create configuration with checkpointing DISABLED
    config = EncoderConfig(
        encoder_type='autoencoder',
        state_dim=64,
        lab_dim=10,
        vital_dim=5,
        demo_dim=4,
        device='cpu',
        save_checkpoints=False  # ← Disable checkpoint saving
    )
    
    print(f"✓ Checkpoint saving: {config.save_checkpoints}")
    print("  → No checkpoints will be saved during training")
    
    # Create encoder and trainer
    encoder = PatientAutoencoder(config)
    
    trainer_config = {'learning_rate': 1e-3}
    trainer = EncoderTrainer(encoder, trainer_config)
    
    # Create minimal data
    train_data = create_dummy_dataset(n_samples=50)
    train_dataset = PatientDataset(train_data, max_seq_len=30)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Train without saving
    print("\n📚 Training without checkpoints...")
    history = trainer.train_autoencoder(
        train_loader=train_loader,
        epochs=5
    )
    
    print("\n✅ Training complete - no checkpoints saved (as configured)")


def example_3_keep_best_only():
    """
    Example 3: Keep only the best checkpoint.
    
    Set keep_best_only=True to save only the model with best validation loss.
    This saves disk space while keeping the most important checkpoint.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Keep Best Checkpoint Only")
    print("="*80 + "\n")
    
    # Configuration to keep only best checkpoint
    config = EncoderConfig(
        encoder_type='autoencoder',
        state_dim=64,
        device='cpu',
        save_checkpoints=True,
        checkpoint_dir='./checkpoints/example3',
        keep_best_only=True,  # ← Only save best model
        save_frequency=1  # Ignored when keep_best_only=True
    )
    
    print(f"✓ Keep best only: {config.keep_best_only}")
    print("  → Only best_encoder.pt and encoder_final.pt will be saved")
    
    # Create and train
    encoder = PatientAutoencoder(config)
    trainer = EncoderTrainer(encoder, {'learning_rate': 1e-3})
    
    train_data = create_dummy_dataset(n_samples=100)
    val_data = create_dummy_dataset(n_samples=20)
    
    train_loader = DataLoader(PatientDataset(train_data, 30), batch_size=16)
    val_loader = DataLoader(PatientDataset(val_data, 30), batch_size=16)
    
    print("\n📚 Training with keep_best_only strategy...")
    history = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=8
    )
    
    # Check saved files
    checkpoint_dir = Path(config.checkpoint_dir)
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        print(f"\n💾 Saved checkpoints ({len(checkpoints)}):")
        for cp in sorted(checkpoints):
            print(f"   - {cp.name}")
    
    print("\n✅ Only best and final checkpoints saved!")


def example_4_resume_training():
    """
    Example 4: Resume training from checkpoint.
    
    Load a checkpoint and continue training from where you left off.
    This is crucial for long training runs and handling interruptions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Resume Training from Checkpoint")
    print("="*80 + "\n")
    
    checkpoint_dir = Path('./checkpoints/example4')
    
    # Phase 1: Initial training
    print("📚 Phase 1: Initial training (10 epochs)...")
    
    config = EncoderConfig(
        encoder_type='autoencoder',
        state_dim=64,
        device='cpu',
        checkpoint_dir=str(checkpoint_dir),
        save_frequency=3
    )
    
    encoder = PatientAutoencoder(config)
    trainer = EncoderTrainer(encoder, {'learning_rate': 1e-3})
    
    train_data = create_dummy_dataset(n_samples=100)
    val_data = create_dummy_dataset(n_samples=20)
    
    train_loader = DataLoader(PatientDataset(train_data, 30), batch_size=16)
    val_loader = DataLoader(PatientDataset(val_data, 30), batch_size=16)
    
    history1 = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10
    )
    
    print(f"\n✓ Phase 1 complete: {len(history1['train_loss'])} epochs trained")
    print(f"  Final train loss: {history1['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history1['val_loss'][-1]:.4f}")
    
    # Phase 2: Resume training
    print("\n📚 Phase 2: Resuming training (10 more epochs)...")
    
    # Create NEW encoder and trainer
    new_encoder = PatientAutoencoder(config)
    new_trainer = EncoderTrainer(new_encoder, {'learning_rate': 1e-3})
    
    # Load checkpoint
    best_checkpoint = checkpoint_dir / 'best_encoder.pt'
    metadata = new_trainer.resume_from_checkpoint(best_checkpoint)
    
    print(f"\n✓ Loaded checkpoint from epoch {metadata['epoch']}")
    print(f"  Previous best val loss: {metadata['best_val_loss']:.4f}")
    print(f"  Training history length: {len(new_trainer.history['train_loss'])} epochs")
    
    # Continue training
    history2 = new_trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20  # Will train from epoch 11 to 20
    )
    
    print(f"\n✓ Phase 2 complete: {len(history2['train_loss'])} total epochs")
    print(f"  Final train loss: {history2['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history2['val_loss'][-1]:.4f}")
    
    print("\n✅ Successfully resumed and continued training!")


def example_5_custom_checkpoint_frequency():
    """
    Example 5: Custom checkpoint frequency.
    
    Control how often checkpoints are saved with save_frequency parameter.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Checkpoint Frequency")
    print("="*80 + "\n")
    
    # Save checkpoint every 2 epochs
    config = EncoderConfig(
        encoder_type='autoencoder',
        state_dim=64,
        device='cpu',
        checkpoint_dir='./checkpoints/example5',
        save_frequency=2,  # ← Save every 2 epochs
        keep_best_only=False
    )
    
    print(f"✓ Save frequency: every {config.save_frequency} epochs")
    print("  → Checkpoints will be saved at epochs 2, 4, 6, 8, 10...")
    
    encoder = PatientAutoencoder(config)
    trainer = EncoderTrainer(encoder, {'learning_rate': 1e-3})
    
    train_data = create_dummy_dataset(n_samples=80)
    val_data = create_dummy_dataset(n_samples=20)
    
    train_loader = DataLoader(PatientDataset(train_data, 30), batch_size=16)
    val_loader = DataLoader(PatientDataset(val_data, 30), batch_size=16)
    
    print("\n📚 Training with custom checkpoint frequency...")
    history = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10
    )
    
    # Show saved checkpoints
    checkpoint_dir = Path(config.checkpoint_dir)
    if checkpoint_dir.exists():
        periodic_checkpoints = sorted(checkpoint_dir.glob('encoder_epoch_*.pt'))
        print(f"\n💾 Periodic checkpoints saved ({len(periodic_checkpoints)}):")
        for cp in periodic_checkpoints:
            print(f"   - {cp.name}")
    
    print("\n✅ Training complete with custom checkpoint frequency!")


def example_6_production_workflow():
    """
    Example 6: Production workflow with checkpoint management.
    
    This demonstrates a complete workflow:
    1. Train with checkpoints
    2. Evaluate checkpoints
    3. Select best model
    4. Resume if needed
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Production Workflow")
    print("="*80 + "\n")
    
    checkpoint_dir = Path('./checkpoints/production')
    
    # Configuration for production
    config = EncoderConfig(
        encoder_type='autoencoder',
        state_dim=128,
        hidden_dim=256,
        device='cpu',
        checkpoint_dir=str(checkpoint_dir),
        save_checkpoints=True,
        save_frequency=5,
        keep_best_only=False,  # Keep all for analysis
        save_optimizer_state=True  # Essential for resuming
    )
    
    print("🏭 Production Training Configuration:")
    print(f"   ✓ Checkpoint dir: {config.checkpoint_dir}")
    print(f"   ✓ Save frequency: {config.save_frequency}")
    print(f"   ✓ Keep all checkpoints: {not config.keep_best_only}")
    print(f"   ✓ Save optimizer state: {config.save_optimizer_state}")
    
    # Step 1: Train model
    print("\n📚 Step 1: Training encoder...")
    encoder = PatientAutoencoder(config)
    trainer = EncoderTrainer(encoder, {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10
    })
    
    train_data = create_dummy_dataset(n_samples=200)
    val_data = create_dummy_dataset(n_samples=50)
    
    train_loader = DataLoader(PatientDataset(train_data, 50), batch_size=32)
    val_loader = DataLoader(PatientDataset(val_data, 50), batch_size=32)
    
    history = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15
    )
    
    # Step 2: Analyze checkpoints
    print("\n📊 Step 2: Analyzing checkpoints...")
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob('*.pt'))
        print(f"\nAvailable checkpoints ({len(checkpoints)}):")
        
        for cp in checkpoints:
            checkpoint = torch.load(cp, map_location='cpu')
            epoch = checkpoint.get('epoch', -1)
            val_loss = checkpoint.get('val_loss', None)
            is_best = checkpoint.get('is_best', False)
            
            status = "⭐ BEST" if is_best else ""
            if val_loss:
                print(f"   - {cp.name}: epoch={epoch}, val_loss={val_loss:.4f} {status}")
            else:
                print(f"   - {cp.name}: epoch={epoch} {status}")
    
    # Step 3: Load best model for deployment
    print("\n🚀 Step 3: Loading best model for deployment...")
    best_checkpoint = checkpoint_dir / 'best_encoder.pt'
    
    deployment_encoder = PatientAutoencoder(config)
    deployment_encoder.load_checkpoint(best_checkpoint)
    
    print("✓ Best model loaded and ready for deployment")
    
    # Step 4: Show how to resume if interrupted
    print("\n💡 Step 4: How to resume if training interrupted:")
    print("   >>> new_trainer = EncoderTrainer(encoder, config)")
    print("   >>> new_trainer.resume_from_checkpoint('checkpoints/production/best_encoder.pt')")
    print("   >>> new_trainer.train_autoencoder(train_loader, val_loader, epochs=50)")
    
    print("\n✅ Production workflow complete!")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("CHECKPOINT MANAGEMENT - COMPREHENSIVE EXAMPLES")
    print("="*80)
    
    # Run all examples
    example_1_default_checkpoint_saving()
    example_2_disable_checkpoint_saving()
    example_3_keep_best_only()
    example_4_resume_training()
    example_5_custom_checkpoint_frequency()
    example_6_production_workflow()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80)
    print("\n📝 Summary:")
    print("   ✓ Default: Checkpoints enabled automatically")
    print("   ✓ Disable: Set save_checkpoints=False")
    print("   ✓ Resume: Use trainer.resume_from_checkpoint()")
    print("   ✓ Best only: Set keep_best_only=True")
    print("   ✓ Frequency: Control with save_frequency parameter")
    print("\n")
