# Checkpoint Management - Quick Reference

## 🚀 Quick Start

### Default Behavior (Checkpoints ENABLED)
```python
from encoder_config import EncoderConfig
from autoencoder import PatientAutoencoder
from train_encoder import EncoderTrainer

# Create config - checkpoints enabled by default
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    device='mps'  # Your M4 Mac
)

# Train - checkpoints saved automatically
encoder = PatientAutoencoder(config)
trainer = EncoderTrainer(encoder, {'learning_rate': 1e-3})
history = trainer.train_autoencoder(train_loader, val_loader, epochs=50)

# Checkpoints saved to: ./checkpoints/encoders/
# - best_encoder.pt (best validation loss)
# - encoder_final.pt (final model)
# - encoder_epoch_N.pt (periodic checkpoints)
```

## 📋 Configuration Options

### All Checkpoint Parameters
```python
config = EncoderConfig(
    # ... other parameters ...
    
    # Checkpoint configuration
    save_checkpoints=True,                      # Enable/disable (default: True)
    checkpoint_dir='./checkpoints/encoders',    # Where to save (default)
    save_frequency=1,                           # Save every N epochs (default: 1)
    keep_best_only=False,                       # Only keep best checkpoint (default: False)
    save_optimizer_state=True,                  # Save optimizer for resuming (default: True)
)
```

## 🎛️ Common Use Cases

### 1. Default: Save Everything
```python
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    # That's it! Checkpoints enabled by default
)
```
**Result**: Saves best, final, and periodic checkpoints with optimizer state.

---

### 2. Disable Checkpoints (Quick Experiments)
```python
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    save_checkpoints=False  # ← Add this line
)
```
**Result**: No checkpoints saved. Use for quick tests.

---

### 3. Save Only Best Model (Save Space)
```python
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    keep_best_only=True  # ← Add this line
)
```
**Result**: Only `best_encoder.pt` and `encoder_final.pt` saved.

---

### 4. Custom Checkpoint Frequency
```python
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    save_frequency=5  # ← Save every 5 epochs
)
```
**Result**: Checkpoints at epochs 5, 10, 15, 20, ... + best + final.

---

### 5. Custom Directory
```python
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    checkpoint_dir='./my_custom_path/checkpoints'  # ← Custom path
)
```
**Result**: Checkpoints saved to your custom directory.

---

## 🔄 Resume Training

### Method 1: Resume from Best Checkpoint
```python
# Create new encoder and trainer
config = EncoderConfig(encoder_type='autoencoder', state_dim=128)
encoder = PatientAutoencoder(config)
trainer = EncoderTrainer(encoder, {'learning_rate': 1e-3})

# Load checkpoint
metadata = trainer.resume_from_checkpoint('./checkpoints/encoders/best_encoder.pt')
print(f"Resuming from epoch {metadata['epoch']}")

# Continue training
history = trainer.train_autoencoder(train_loader, val_loader, epochs=100)
```

### Method 2: Resume from Specific Epoch
```python
# Load specific epoch checkpoint
metadata = trainer.resume_from_checkpoint('./checkpoints/encoders/encoder_epoch_25.pt')

# Continue training from epoch 26
history = trainer.train_autoencoder(train_loader, val_loader, epochs=100)
```

## 📂 Checkpoint Contents

Each checkpoint file contains:
```python
{
    'epoch': 10,                          # Epoch number
    'model_state_dict': {...},            # Model weights
    'optimizer_state_dict': {...},        # Optimizer state (if enabled)
    'scheduler_state_dict': {...},        # LR scheduler state
    'train_loss': 0.123,                  # Training loss
    'val_loss': 0.145,                    # Validation loss
    'best_val_loss': 0.142,               # Best validation loss so far
    'history': {...},                     # Full training history
    'config': {...},                      # Encoder configuration
    'encoder_type': 'PatientAutoencoder', # Encoder class name
    'is_best': False,                     # Is this the best model?
    'is_final': False                     # Is this the final checkpoint?
}
```

## 💡 Best Practices

### For Long Training Runs
```python
config = EncoderConfig(
    save_checkpoints=True,
    save_frequency=5,          # Save every 5 epochs
    keep_best_only=False,      # Keep all for analysis
    save_optimizer_state=True  # Essential for resuming
)
```

### For Production
```python
config = EncoderConfig(
    save_checkpoints=True,
    checkpoint_dir='./models/production',
    keep_best_only=True,       # Only keep best model
    save_optimizer_state=False # Don't need optimizer for deployment
)
```

### For Experiments
```python
config = EncoderConfig(
    save_checkpoints=False  # Don't save for quick tests
)
```

## 🔍 Inspect Checkpoint

```python
import torch

# Load checkpoint
checkpoint = torch.load('./checkpoints/encoders/best_encoder.pt')

# Inspect contents
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train loss: {checkpoint['train_loss']:.4f}")
print(f"Val loss: {checkpoint['val_loss']:.4f}")
print(f"Is best: {checkpoint['is_best']}")

# See training history
history = checkpoint['history']
print(f"Training history: {len(history['train_loss'])} epochs")
```

## 🛠️ Troubleshooting

### Issue: "FileNotFoundError: Checkpoint not found"
```python
# Check if checkpoint exists
from pathlib import Path
checkpoint_path = Path('./checkpoints/encoders/best_encoder.pt')
print(f"Exists: {checkpoint_path.exists()}")

# List available checkpoints
checkpoint_dir = Path('./checkpoints/encoders')
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    print(f"Available checkpoints: {[c.name for c in checkpoints]}")
```

### Issue: "RuntimeError: Error loading checkpoint"
```python
# Load with error handling
try:
    metadata = trainer.resume_from_checkpoint(checkpoint_path)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    print("Starting training from scratch")
    history = trainer.train_autoencoder(...)
```

### Issue: Checkpoint Files Too Large
```python
# Option 1: Don't save optimizer state (reduces size by ~50%)
config = EncoderConfig(save_optimizer_state=False)

# Option 2: Keep only best checkpoint
config = EncoderConfig(keep_best_only=True)

# Option 3: Reduce checkpoint frequency
config = EncoderConfig(save_frequency=10)  # Save every 10 epochs
```

## 📊 Example: Complete Training Pipeline

```python
from encoder_config import EncoderConfig
from autoencoder import PatientAutoencoder
from train_encoder import EncoderTrainer, PatientDataset
from torch.utils.data import DataLoader

# 1. Configure with checkpointing
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    hidden_dim=256,
    device='mps',
    # Checkpoint config
    save_checkpoints=True,
    checkpoint_dir='./checkpoints/my_encoder',
    save_frequency=10,
    save_optimizer_state=True
)

# 2. Create encoder and trainer
encoder = PatientAutoencoder(config)
trainer = EncoderTrainer(encoder, {
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10
})

# 3. Train (checkpoints saved automatically)
print("Training...")
history = trainer.train_autoencoder(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50
)

# 4. If training interrupted, resume later
print("\nResuming training...")
new_encoder = PatientAutoencoder(config)
new_trainer = EncoderTrainer(new_encoder, {'learning_rate': 1e-3})
new_trainer.resume_from_checkpoint('./checkpoints/my_encoder/best_encoder.pt')
new_trainer.train_autoencoder(train_loader, val_loader, epochs=100)

# 5. Load best model for evaluation/deployment
best_encoder = PatientAutoencoder(config)
best_encoder.load_checkpoint('./checkpoints/my_encoder/best_encoder.pt')
print("Best model loaded!")
```

## ✅ Summary

| Feature | Parameter | Default | Use Case |
|---------|-----------|---------|----------|
| Enable checkpoints | `save_checkpoints` | `True` | Always on by default |
| Checkpoint directory | `checkpoint_dir` | `'./checkpoints/encoders'` | Where to save files |
| Save frequency | `save_frequency` | `1` | How often to save |
| Keep best only | `keep_best_only` | `False` | Save disk space |
| Save optimizer | `save_optimizer_state` | `True` | Resume training |

**Default behavior**: Checkpoints enabled, saved to `./checkpoints/encoders/`, includes optimizer state for resuming.

**To disable**: Set `save_checkpoints=False` in `EncoderConfig`.

**To resume**: Use `trainer.resume_from_checkpoint(path)` before calling `train_autoencoder()`.
