# Patient State Encoders for Healthcare RL

Comprehensive implementation of patient state encoders for transforming raw patient data (vitals, labs, demographics, medication history, diagnoses, procedures) into compact state representations for reinforcement learning agents.

## 📋 Overview

This package provides multiple encoder architectures specifically designed for healthcare sequential data:

- **Transformer Encoder**: Multi-head self-attention for temporal sequences
- **Autoencoder**: Unsupervised representation learning
- **Deep Autoencoder**: Deeper architecture for complex representations
- **Variational Autoencoder (VAE)**: Probabilistic latent representations

## 🚀 Features

- ✅ Multiple encoder architectures (Transformer, Autoencoder, VAE)
- ✅ Multi-modal input handling (continuous and categorical features)
- ✅ Variable-length sequence support with proper masking
- ✅ **Automatic checkpoint saving (enabled by default)**
- ✅ **Resume training from any checkpoint**
- ✅ Pre-training capability on reconstruction tasks
- ✅ Integration with PyTorch RL agents
- ✅ Attention weight extraction for interpretability
- ✅ MacBook M4 MPS support
- ✅ Comprehensive training utilities
- ✅ Production-ready code with type hints and docstrings

## 📦 Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0

# Optional (for examples)
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

### Setup

1. Copy all encoder files to your project:

```bash
src/models/encoders/
├── __init__.py
├── encoder_config.py          # Configuration dataclass
├── base_encoder.py             # Abstract base class
├── encoder_utils.py            # Utility functions
├── transformer_encoder.py      # Transformer implementation
├── autoencoder.py              # Autoencoder implementations
├── train_encoder.py            # Training utilities
└── example_usage.py            # Usage examples
```

2. Update `__init__.py`:

```python
from .encoder_config import EncoderConfig
from .base_encoder import BasePatientEncoder
from .transformer_encoder import TransformerPatientEncoder
from .autoencoder import PatientAutoencoder, DeepAutoencoder
from .train_encoder import EncoderTrainer, PatientDataset

__all__ = [
    'EncoderConfig',
    'BasePatientEncoder',
    'TransformerPatientEncoder',
    'PatientAutoencoder',
    'DeepAutoencoder',
    'EncoderTrainer',
    'PatientDataset'
]
```

## 🎯 Quick Start

### 1. Basic Transformer Encoder

```python
from encoder_config import EncoderConfig
from transformer_encoder import TransformerPatientEncoder
import torch

# Create configuration
config = EncoderConfig(
    encoder_type='transformer',
    hidden_dim=256,
    state_dim=128,
    num_layers=4,
    num_heads=8,
    lab_dim=20,
    vital_dim=10,
    demo_dim=8,
    med_vocab_size=500,
    device='mps'  # Use 'mps' for M4 Mac, 'cuda' for NVIDIA GPU, 'cpu' for CPU
)

# Create encoder
encoder = TransformerPatientEncoder(config)

# Prepare patient data
batch = {
    'labs': torch.randn(32, 50, 20),          # [batch, seq_len, lab_dim]
    'vitals': torch.randn(32, 50, 10),        # [batch, seq_len, vital_dim]
    'demographics': torch.randn(32, 8),        # [batch, demo_dim]
    'medications': torch.randint(1, 500, (32, 50)),  # [batch, seq_len]
    'seq_lengths': torch.randint(20, 50, (32,))      # actual lengths
}

# Encode to fixed-size embeddings
state_embeddings = encoder(batch)  # [32, 128]
```

## 💾 Checkpoint Management (NEW!)

**Checkpoints are automatically saved by default!** You never have to worry about losing training progress.

### Default Behavior (Recommended)
```python
# Checkpoints saved automatically - no configuration needed!
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    device='mps'
)

encoder = PatientAutoencoder(config)
trainer = EncoderTrainer(encoder, {'learning_rate': 1e-3})

# Train - checkpoints saved to ./checkpoints/encoders/
history = trainer.train_autoencoder(train_loader, val_loader, epochs=50)

# Saved files:
# - best_encoder.pt (best validation loss)
# - encoder_final.pt (final model)
# - encoder_epoch_N.pt (periodic checkpoints)
```

### Disable Checkpoints (Quick Experiments)
```python
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    save_checkpoints=False  # ← Disable checkpoint saving
)
```

### Resume Training
```python
# Load checkpoint and continue training
encoder = PatientAutoencoder(config)
trainer = EncoderTrainer(encoder, {'learning_rate': 1e-3})

# Resume from where you left off
metadata = trainer.resume_from_checkpoint('./checkpoints/encoders/best_encoder.pt')
print(f"Resuming from epoch {metadata['epoch']}")

# Continue training (will start from next epoch)
history = trainer.train_autoencoder(train_loader, val_loader, epochs=100)
```

### Advanced Checkpoint Options
```python
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    # Checkpoint configuration
    save_checkpoints=True,              # Enable/disable (default: True)
    checkpoint_dir='./my_checkpoints',  # Custom directory
    save_frequency=5,                   # Save every 5 epochs
    keep_best_only=True,                # Only keep best checkpoint
    save_optimizer_state=True           # Save optimizer for resuming
)
```

**See [CHECKPOINT_GUIDE.md](./CHECKPOINT_GUIDE.md) for complete documentation.**

---

### 2. Training Autoencoder

```python
from encoder_config import EncoderConfig
from autoencoder import PatientAutoencoder
from train_encoder import EncoderTrainer, PatientDataset, create_dummy_dataset
from torch.utils.data import DataLoader

# Create configuration
config = EncoderConfig(
    encoder_type='autoencoder',
    state_dim=128,
    lab_dim=20,
    vital_dim=10,
    demo_dim=8,
    device='mps'
)

# Create autoencoder
autoencoder = PatientAutoencoder(config, variational=False)

# Prepare dataset (use your MIMIC-III data here)
train_data = create_dummy_dataset(n_samples=800)
val_data = create_dummy_dataset(n_samples=200)

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

trainer = EncoderTrainer(autoencoder, trainer_config)

# Train
history = trainer.train_autoencoder(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    save_dir='./checkpoints'
)
```

### 3. Integration with RL Agent

```python
import torch
import torch.nn as nn
from encoder_config import EncoderConfig
from transformer_encoder import TransformerPatientEncoder

# Create encoder
config = EncoderConfig(state_dim=128, device='mps')
encoder = TransformerPatientEncoder(config)

# Load pre-trained weights (optional)
encoder.load_checkpoint('./checkpoints/best_encoder.pt')
encoder.eval()

# Create RL policy network
class HealthcarePolicy(nn.Module):
    def __init__(self, encoder, action_dim):
        super().__init__()
        self.encoder = encoder
        
        # Freeze encoder if pre-trained
        # self.encoder.freeze()
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(encoder.get_embedding_dim(), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, patient_data):
        # Encode patient state
        state_embedding = self.encoder(patient_data)
        
        # Get action logits
        action_logits = self.policy_head(state_embedding)
        
        return action_logits

# Create policy
policy = HealthcarePolicy(encoder, action_dim=5)

# Use in RL training loop
patient_data = {...}  # Your patient data
action_logits = policy(patient_data)
```

## 🔧 Integration with Your Healthcare RL Project

### Step 1: Prepare MIMIC-III Data

```python
import pandas as pd
import numpy as np
from pathlib import Path

def load_mimic_data(mimic_path: Path) -> list:
    """Load and preprocess MIMIC-III data."""
    
    # Load your MIMIC-III CSVs
    admissions = pd.read_csv(mimic_path / 'ADMISSIONS.csv')
    chartevents = pd.read_csv(mimic_path / 'CHARTEVENTS.csv')
    labevents = pd.read_csv(mimic_path / 'LABEVENTS.csv')
    prescriptions = pd.read_csv(mimic_path / 'PRESCRIPTIONS.csv')
    
    # Process into patient sequences
    # (Use your existing preprocessing from healthcare_rl/data/)
    patient_sequences = []
    
    for patient_id in admissions['SUBJECT_ID'].unique():
        # Extract patient timeline
        labs = extract_labs(labevents, patient_id)
        vitals = extract_vitals(chartevents, patient_id)
        meds = extract_medications(prescriptions, patient_id)
        demographics = extract_demographics(admissions, patient_id)
        
        patient_sequences.append({
            'labs': labs,
            'vitals': vitals,
            'medications': meds,
            'demographics': demographics
        })
    
    return patient_sequences
```

### Step 2: Create Data Loaders

```python
from train_encoder import PatientDataset
from torch.utils.data import DataLoader

# Load your data
mimic_path = Path('/path/to/mimic-iii/')
patient_data = load_mimic_data(mimic_path)

# Split train/val
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(patient_data, test_size=0.2)

# Create datasets
train_dataset = PatientDataset(train_data, max_seq_len=100, normalize=True)
val_dataset = PatientDataset(val_data, max_seq_len=100, normalize=True)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Step 3: Integrate with Your RL Environment

```python
from your_project.environments import HealthcareMDP
from your_project.agents import CQLAgent

# Create encoder
encoder_config = EncoderConfig(
    encoder_type='transformer',
    state_dim=128,
    hidden_dim=256,
    num_layers=4,
    num_heads=8,
    device='mps'
)

encoder = TransformerPatientEncoder(encoder_config)

# Load pre-trained encoder
encoder.load_checkpoint('./checkpoints/best_encoder.pt')

# Create RL environment with encoder
env = HealthcareMDP(
    state_encoder=encoder,
    reward_config=reward_config,
    safety_constraints=safety_constraints
)

# Create RL agent
agent = CQLAgent(
    state_dim=encoder.get_embedding_dim(),
    action_dim=env.action_space.n,
    encoder=encoder
)

# Train RL agent
agent.train(env, num_episodes=1000)
```

## 📊 Configuration Options

### EncoderConfig Parameters

```python
@dataclass
class EncoderConfig:
    # Core architecture
    encoder_type: str = 'transformer'  # 'transformer', 'autoencoder', 'clinical_bert', 'bio_gpt'
    hidden_dim: int = 256              # Hidden layer size
    state_dim: int = 128               # Output embedding size
    num_layers: int = 4                # Number of layers
    num_heads: int = 8                 # Attention heads (transformer)
    dropout: float = 0.1               # Dropout rate
    
    # Input dimensions
    lab_dim: int = 20                  # Lab test dimension
    vital_dim: int = 10                # Vital signs dimension
    demo_dim: int = 8                  # Demographics dimension
    med_vocab_size: int = 500          # Medication vocabulary
    diag_vocab_size: int = 1000        # Diagnosis vocabulary
    proc_vocab_size: int = 500         # Procedure vocabulary
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_seq_len: int = 100
    batch_size: int = 32
    
    # Device
    device: str = 'mps'  # 'cuda', 'mps', or 'cpu'
```

## 🎨 Advanced Features

### 1. Different Pooling Strategies

```python
encoder = TransformerPatientEncoder(config)

# Change pooling method
encoder.set_pooling_type('mean')    # Mean pooling
encoder.set_pooling_type('max')     # Max pooling
encoder.set_pooling_type('last')    # Last timestep
encoder.set_pooling_type('attention')  # Attention-based pooling
```

### 2. Variational Autoencoder

```python
# Create VAE
vae = PatientAutoencoder(config, variational=True)

# Forward pass
reconstruction, latent = vae(patient_data)

# Get distribution parameters
mu, logvar = vae.get_vae_params()

# Compute loss with KL divergence
losses = vae.compute_loss(
    original=original_data,
    reconstruction=reconstruction,
    latent=latent,
    kl_weight=0.1  # Adjust KL weight
)
```

### 3. Extracting Attention Weights

```python
encoder = TransformerPatientEncoder(config)
embeddings = encoder(patient_data)

# Get attention weights for interpretability
attention_weights = encoder.get_attention_weights()
```

### 4. Saving and Loading

```python
# Save checkpoint
encoder.save_checkpoint(
    path='./checkpoints/encoder_epoch10.pt',
    epoch=10,
    train_loss=0.15,
    val_loss=0.18
)

# Load checkpoint
metadata = encoder.load_checkpoint('./checkpoints/encoder_epoch10.pt')
print(f"Loaded model from epoch {metadata['epoch']}")
```

## 🧪 Testing

Run the comprehensive example script:

```bash
python example_usage.py
```

This will demonstrate:
1. Transformer encoder usage
2. Autoencoder training
3. VAE usage
4. RL integration
5. Encoder comparison
6. Save/load functionality

## 📈 Performance Tips

### For MacBook M4 (MPS)

```python
# Enable MPS
config = EncoderConfig(device='mps')

# Use float32 for better MPS compatibility
torch.set_default_dtype(torch.float32)
```

### For Large Datasets

```python
# Use gradient accumulation
trainer_config = {
    'learning_rate': 1e-3,
    'grad_accumulation_steps': 4,  # Accumulate over 4 batches
    'grad_clip': 1.0
}

# Use DataLoader with multiple workers
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)
```

## 🔍 Architecture Details

### Transformer Encoder

- Multi-head self-attention for temporal dependencies
- Positional encoding for sequence order
- Layer normalization for training stability
- Multiple pooling strategies (mean/max/last/attention)
- Support for variable-length sequences with masking

### Autoencoder

- Bottleneck architecture: input → 512 → 256 → latent
- Dropout for regularization
- MSE reconstruction loss
- Optional VAE with KL divergence

## 📝 Example Output

```
EXAMPLE 1: Transformer Encoder
================================================================================

Created encoder: TransformerPatientEncoder(
  embedding_dim=128,
  trainable_params=1,234,567,
  total_params=1,234,567,
  device=mps
)

Encoding batch...
Input batch size: 32, seq_len: 50
Output embeddings shape: torch.Size([32, 128])
Sample embedding (first 10 dims): tensor([-0.1234, 0.5678, ...])
```

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```python
# Reduce batch size
config.batch_size = 16

# Use gradient checkpointing (if supported)
# Enable mixed precision training
```

### Issue: MPS Not Available

```python
# Fallback to CPU
if not torch.backends.mps.is_available():
    config.device = 'cpu'
```

### Issue: NaN Loss

```python
# Add gradient clipping
trainer_config['grad_clip'] = 1.0

# Reduce learning rate
config.learning_rate = 1e-4

# Check for NaN in input data
assert not torch.isnan(patient_data['labs']).any()
```

## 📚 References

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- "Reinforcement Learning in Healthcare: A Survey" (Yu et al., 2021)

## 🤝 Integration with Your Project Structure

```
healthcare_rl/
├── data/
│   └── preprocessing/      # Your existing preprocessing
├── models/
│   ├── encoders/          # ← Place encoder files here
│   │   ├── __init__.py
│   │   ├── encoder_config.py
│   │   ├── base_encoder.py
│   │   ├── transformer_encoder.py
│   │   ├── autoencoder.py
│   │   └── train_encoder.py
│   └── rl_agents/         # Your RL agents
├── environments/
│   └── healthcare_mdp.py  # Use encoder here
└── experiments/
    └── train_encoder.py   # Training script
```

## 📧 Support

For issues or questions:
1. Check `example_usage.py` for comprehensive examples
2. Review docstrings in each module
3. Verify input data shapes match expected formats

## ✅ Next Steps

1. **Pre-train encoder** on your MIMIC-III data:
   ```bash
   python experiments/pretrain_encoder.py
   ```

2. **Integrate with RL agent**:
   - Load pre-trained encoder
   - Freeze encoder weights (optional)
   - Add policy head
   - Train end-to-end or fine-tune

3. **Evaluate representations**:
   - Visualize embeddings (t-SNE/UMAP)
   - Check reconstruction quality
   - Measure downstream task performance

---

**Note**: This implementation is production-ready with proper error handling, type hints, and documentation. All code follows best practices for PyTorch development and is optimized for MacBook M4 (MPS) devices.
