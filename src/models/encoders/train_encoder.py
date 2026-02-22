"""
Training utilities for patient state encoders.

This module provides training functions, data loaders, and utilities
for pre-training encoders on reconstruction or contrastive tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import logging
from tqdm import tqdm
import json

from base_encoder import BasePatientEncoder
from autoencoder import PatientAutoencoder
from transformer_encoder import TransformerPatientEncoder


logger = logging.getLogger(__name__)


class PatientDataset(Dataset):
    """
    Dataset class for patient sequential data.
    
    This dataset handles patient data in dictionary format with support
    for variable-length sequences, padding, and masking.
    
    Args:
        data: List of patient data dictionaries
        max_seq_len: Maximum sequence length for padding
        normalize: Whether to normalize continuous features
    """
    
    def __init__(
        self,
        data: List[Dict[str, np.ndarray]],
        max_seq_len: int = 100,
        normalize: bool = True
    ):
        """Initialize patient dataset."""
        self.data = data
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        
        if normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        # Collect all lab and vital values
        all_labs = []
        all_vitals = []
        
        for sample in self.data:
            if 'labs' in sample:
                all_labs.append(sample['labs'])
            if 'vitals' in sample:
                all_vitals.append(sample['vitals'])
        
        if all_labs:
            all_labs = np.concatenate(all_labs, axis=0)
            self.lab_mean = np.mean(all_labs, axis=0)
            self.lab_std = np.std(all_labs, axis=0) + 1e-8
        else:
            self.lab_mean = None
            self.lab_std = None
        
        if all_vitals:
            all_vitals = np.concatenate(all_vitals, axis=0)
            self.vital_mean = np.mean(all_vitals, axis=0)
            self.vital_std = np.std(all_vitals, axis=0) + 1e-8
        else:
            self.vital_mean = None
            self.vital_std = None
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patient sample.
        
        Returns:
            Dictionary with patient data as PyTorch tensors
        """
        sample = self.data[idx].copy()
        
        # Normalize continuous features
        if self.normalize:
            if 'labs' in sample and self.lab_mean is not None:
                sample['labs'] = (sample['labs'] - self.lab_mean) / self.lab_std
            
            if 'vitals' in sample and self.vital_mean is not None:
                sample['vitals'] = (sample['vitals'] - self.vital_mean) / self.vital_std
        
        # Pad sequences
        seq_len = 0
        if 'labs' in sample:
            seq_len = len(sample['labs'])
            if seq_len < self.max_seq_len:
                padding = np.zeros((self.max_seq_len - seq_len, sample['labs'].shape[1]))
                sample['labs'] = np.vstack([sample['labs'], padding])
            elif seq_len > self.max_seq_len:
                sample['labs'] = sample['labs'][:self.max_seq_len]
                seq_len = self.max_seq_len
        
        if 'vitals' in sample:
            curr_len = len(sample['vitals'])
            if curr_len < self.max_seq_len:
                padding = np.zeros((self.max_seq_len - curr_len, sample['vitals'].shape[1]))
                sample['vitals'] = np.vstack([sample['vitals'], padding])
            elif curr_len > self.max_seq_len:
                sample['vitals'] = sample['vitals'][:self.max_seq_len]
        
        # Pad categorical sequences
        for key in ['medications', 'diagnoses', 'procedures']:
            if key in sample:
                curr_len = len(sample[key])
                if curr_len < self.max_seq_len:
                    padding = np.zeros(self.max_seq_len - curr_len, dtype=np.int64)
                    sample[key] = np.concatenate([sample[key], padding])
                elif curr_len > self.max_seq_len:
                    sample[key] = sample[key][:self.max_seq_len]
        
        # Convert to tensors
        result = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value).float()
                if key in ['medications', 'diagnoses', 'procedures']:
                    result[key] = result[key].long()
        
        # Add sequence length
        result['seq_lengths'] = torch.tensor(min(seq_len, self.max_seq_len), dtype=torch.long)
        
        return result


class EncoderTrainer:
    """
    Trainer for patient state encoders.
    
    Handles training loops, validation, checkpointing, and logging
    for different encoder architectures.
    
    Args:
        encoder: Patient encoder model
        config: Training configuration dictionary
        device: Device for training
    """
    
    def __init__(
        self,
        encoder: BasePatientEncoder,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """Initialize encoder trainer."""
        self.encoder = encoder
        self.config = config
        self.device = device or encoder.get_device()
        
        # Move encoder to device
        self.encoder.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Checkpoint configuration from encoder config
        encoder_config = encoder.config
        self.save_checkpoints = encoder_config.save_checkpoints
        self.checkpoint_dir = Path(encoder_config.checkpoint_dir) if encoder_config.checkpoint_dir else None
        self.save_frequency = encoder_config.save_frequency
        self.keep_best_only = encoder_config.keep_best_only
        self.save_optimizer_state = encoder_config.save_optimizer_state
        
        # Create checkpoint directory if saving is enabled
        if self.save_checkpoints and self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        
        logger.info("Initialized EncoderTrainer")
        logger.info(f"Checkpoint saving: {'ENABLED' if self.save_checkpoints else 'DISABLED'}")
    
    def resume_from_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Resume training from a saved checkpoint.
        
        This loads the model, optimizer state, training history, and epoch counter,
        allowing you to continue training exactly where you left off.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Dictionary with checkpoint metadata
        
        Example:
            >>> trainer = EncoderTrainer(encoder, config)
            >>> metadata = trainer.resume_from_checkpoint('./checkpoints/encoder_epoch10.pt')
            >>> print(f"Resuming from epoch {metadata['epoch']}")
            >>> # Continue training
            >>> trainer.train_autoencoder(train_loader, val_loader, epochs=50)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available and configured
        if self.save_optimizer_state and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Loaded scheduler state")
        
        # Load training state
        self.start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            logger.info(f"Loaded training history with {len(self.history['train_loss'])} epochs")
        
        logger.info(f"Resuming from epoch {self.start_epoch}")
        logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")
        
        # Return metadata
        return {
            'epoch': checkpoint.get('epoch', 0),
            'train_loss': checkpoint.get('train_loss'),
            'val_loss': checkpoint.get('val_loss'),
            'best_val_loss': self.best_val_loss
        }
    
    def train_autoencoder(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_dir: Optional[Path] = None
    ) -> Dict:
        """
        Train autoencoder on reconstruction task.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            save_dir: Directory to save checkpoints (overrides config if provided)
        
        Returns:
            Training history dictionary
        """
        if not isinstance(self.encoder, PatientAutoencoder):
            raise TypeError("train_autoencoder requires PatientAutoencoder instance")
        
        # Use provided save_dir or fall back to config
        if save_dir is not None:
            checkpoint_dir = Path(save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        elif self.save_checkpoints and self.checkpoint_dir:
            checkpoint_dir = self.checkpoint_dir
        else:
            checkpoint_dir = None
        
        patience_counter = 0
        max_patience = self.config.get('early_stopping_patience', 10)
        
        logger.info(f"Starting training from epoch {self.start_epoch} to {epochs}")
        if checkpoint_dir:
            logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
        
        for epoch in range(self.start_epoch, epochs):
            # Training phase
            train_loss = self._train_autoencoder_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate_autoencoder_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Check for improvement
                improved = val_loss < self.best_val_loss
                if improved:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model if checkpointing is enabled
                    if self.save_checkpoints and checkpoint_dir:
                        checkpoint_path = checkpoint_dir / 'best_encoder.pt'
                        self._save_checkpoint(
                            checkpoint_path,
                            epoch=epoch,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            is_best=True
                        )
                        logger.info(f"💾 Saved best model (val_loss={val_loss:.4f})")
                else:
                    patience_counter += 1
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    f"{' ⭐ NEW BEST' if improved else ''}"
                )
                
                # Early stopping
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}"
                )
            
            # Record learning rate
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save periodic checkpoint (if not keeping best only)
            if (self.save_checkpoints and checkpoint_dir and 
                not self.keep_best_only and 
                (epoch + 1) % self.save_frequency == 0):
                checkpoint_path = checkpoint_dir / f'encoder_epoch_{epoch+1}.pt'
                self._save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss if val_loader else None
                )
                logger.info(f"💾 Saved checkpoint: epoch_{epoch+1}.pt")
        
        # Save final checkpoint
        if self.save_checkpoints and checkpoint_dir:
            final_path = checkpoint_dir / 'encoder_final.pt'
            self._save_checkpoint(
                final_path,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss if val_loader else None,
                is_final=True
            )
            logger.info(f"💾 Saved final checkpoint")
        
        return self.history
    
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        is_best: bool = False,
        is_final: bool = False
    ):
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (if available)
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.encoder.state_dict(),
            'train_loss': train_loss,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.encoder.config.to_dict(),
            'encoder_type': self.encoder.__class__.__name__,
            'is_best': is_best,
            'is_final': is_final
        }
        
        # Add validation loss if available
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        
        # Save optimizer state if configured
        if self.save_optimizer_state:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def _train_autoencoder_epoch(self, train_loader: DataLoader) -> float:
        """Train autoencoder for one epoch."""
        self.encoder.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            reconstruction, latent = self.encoder(batch)
            
            # Flatten original data
            if isinstance(self.encoder, PatientAutoencoder):
                original = self.encoder._flatten_patient_data(batch)
            else:
                original = batch
            
            # Compute loss
            losses = self.encoder.compute_loss(original, reconstruction, latent)
            loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(),
                max_norm=self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_autoencoder_epoch(self, val_loader: DataLoader) -> float:
        """Validate autoencoder for one epoch."""
        self.encoder.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                reconstruction, latent = self.encoder(batch)
                
                # Flatten original data
                if isinstance(self.encoder, PatientAutoencoder):
                    original = self.encoder._flatten_patient_data(batch)
                else:
                    original = batch
                
                # Compute loss
                losses = self.encoder.compute_loss(original, reconstruction, latent)
                loss = losses['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_history(self, path: Path):
        """Save training history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Saved training history to {path}")


def create_dummy_dataset(
    n_samples: int = 1000,
    seq_len_range: Tuple[int, int] = (20, 50),
    lab_dim: int = 20,
    vital_dim: int = 10,
    demo_dim: int = 8
) -> List[Dict[str, np.ndarray]]:
    """
    Create dummy patient dataset for testing.
    
    Args:
        n_samples: Number of patient samples
        seq_len_range: Range of sequence lengths
        lab_dim: Dimension of lab features
        vital_dim: Dimension of vital features
        demo_dim: Dimension of demographic features
    
    Returns:
        List of patient data dictionaries
    """
    dataset = []
    
    for i in range(n_samples):
        seq_len = np.random.randint(seq_len_range[0], seq_len_range[1])
        
        sample = {
            'labs': np.random.randn(seq_len, lab_dim).astype(np.float32),
            'vitals': np.random.randn(seq_len, vital_dim).astype(np.float32),
            'demographics': np.random.randn(demo_dim).astype(np.float32),
            'medications': np.random.randint(1, 500, size=seq_len).astype(np.int64),
            'diagnoses': np.random.randint(1, 1000, size=seq_len).astype(np.int64)
        }
        
        dataset.append(sample)
    
    return dataset


if __name__ == '__main__':
    # Example usage
    from encoder_config import EncoderConfig
    from autoencoder import PatientAutoencoder
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = EncoderConfig(
        encoder_type='autoencoder',
        state_dim=128,
        lab_dim=20,
        vital_dim=10,
        demo_dim=8,
        device='cpu'
    )
    
    # Create encoder
    encoder = PatientAutoencoder(config)
    
    # Create dummy dataset
    train_data = create_dummy_dataset(n_samples=800)
    val_data = create_dummy_dataset(n_samples=200)
    
    # Create datasets and loaders
    train_dataset = PatientDataset(train_data, max_seq_len=50)
    val_dataset = PatientDataset(val_data, max_seq_len=50)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create trainer
    trainer_config = {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10,
        'grad_clip': 1.0
    }
    
    trainer = EncoderTrainer(encoder, trainer_config)
    
    # Train
    print("Starting training...")
    history = trainer.train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        save_dir=Path('./checkpoints')
    )
    
    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
