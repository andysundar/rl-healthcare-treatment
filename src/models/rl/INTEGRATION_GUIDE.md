# Integration Guide: Offline RL with Your Existing MIMIC-III Pipeline

This guide shows how to integrate the offline RL package with your existing healthcare treatment recommendation system.

## Overview of Integration

Your existing project structure:
```
/Users/andy/Documents/Coding/Learning/mtp/healthcare_rl/
├── src/
│   ├── data_processing/       # Your existing data pipeline
│   ├── models/
│   │   ├── state_encoder/     # Transformer encoder
│   │   ├── reward/            # Reward functions
│   │   └── rl/               # ← NEW: Add offline RL here
│   └── safety/                # Safety validation
├── data/
│   └── mimic/                 # MIMIC-III processed data
└── experiments/               # Experiment configs
```

## Step 1: Copy Offline RL Package

Copy the offline RL modules to your project:

```bash
# From the offline_rl_package directory
cp -r src/models/rl/* /Users/andy/Documents/Coding/Learning/mtp/healthcare_rl/src/models/rl/
```

Your structure becomes:
```
src/models/rl/
├── __init__.py
├── base_agent.py
├── cql.py
├── bcq.py
├── networks.py
├── replay_buffer.py
├── trainer.py
└── config.py
```

## Step 2: Adapt State Encoder Integration

Modify CQL agent to use your existing state encoder:

```python
# In your project: src/models/rl/cql_with_encoder.py

from src.models.rl.cql import CQLAgent
from src.models.state_encoder import PatientStateEncoder  # Your existing encoder

class CQLWithEncoder(CQLAgent):
    """CQL agent integrated with your transformer encoder."""
    
    def __init__(
        self,
        state_encoder: PatientStateEncoder,
        action_dim: int,
        **kwargs
    ):
        # Get encoded state dimension from your encoder
        state_dim = state_encoder.state_dim
        
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )
        
        self.state_encoder = state_encoder
        self.state_encoder.eval()  # Freeze encoder during RL training
    
    def select_action(self, patient_sequence, deterministic=False):
        """
        Select action given patient sequence.
        
        Args:
            patient_sequence: Your patient data format
            deterministic: Whether to use deterministic policy
        
        Returns:
            action: Treatment action
        """
        # Encode patient sequence using your encoder
        with torch.no_grad():
            state = self.state_encoder(patient_sequence)
            state_np = state.cpu().numpy().flatten()
        
        # Use CQL policy
        action = super().select_action(state_np, deterministic)
        
        return action
```

## Step 3: Create Data Loading Script

Create a script to load your preprocessed MIMIC data:

```python
# src/data_processing/load_for_rl.py

import numpy as np
from pathlib import Path
from src.data_processing import DataProcessor  # Your existing processor

def load_mimic_for_offline_rl(
    data_path: str,
    cohort: str = "diabetes",
    state_encoder_path: str = None
) -> tuple:
    """
    Load MIMIC data formatted for offline RL.
    
    Args:
        data_path: Path to MIMIC processed data
        cohort: Cohort name
        state_encoder_path: Path to pretrained state encoder
    
    Returns:
        (states, actions, rewards, next_states, dones, eval_episodes)
    """
    # Load your processed MIMIC data
    processor = DataProcessor(data_path)
    cohort_data = processor.load_cohort(cohort)
    
    # Load state encoder
    if state_encoder_path:
        from src.models.state_encoder import PatientStateEncoder
        state_encoder = PatientStateEncoder.load(state_encoder_path)
        state_encoder.eval()
    
    # Extract trajectories
    trajectories = cohort_data['trajectories']
    
    # Convert to (s, a, r, s', done) format
    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []
    
    for traj in trajectories:
        # Encode states using your encoder
        patient_sequences = traj['patient_sequences']
        
        with torch.no_grad():
            # Batch encode all states in trajectory
            encoded_states = []
            for seq in patient_sequences:
                state = state_encoder(seq)
                encoded_states.append(state.cpu().numpy())
        
        encoded_states = np.array(encoded_states)
        
        # Extract actions (from your existing action extraction)
        actions = traj['actions']
        
        # Compute rewards (using your reward function)
        rewards = traj['rewards']
        
        # Create next states
        next_states = encoded_states[1:]
        states = encoded_states[:-1]
        
        # Create done flags
        dones = np.zeros(len(states))
        dones[-1] = 1.0  # Last state is terminal
        
        states_list.append(states)
        actions_list.append(actions[:-1])
        rewards_list.append(rewards[:-1])
        next_states_list.append(next_states)
        dones_list.append(dones)
    
    # Concatenate all trajectories
    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    next_states = np.concatenate(next_states_list, axis=0)
    dones = np.concatenate(dones_list, axis=0)
    
    # Create evaluation episodes (last 10% of trajectories)
    n_eval = int(0.1 * len(trajectories))
    eval_episodes = []
    for traj in trajectories[-n_eval:]:
        episode = {
            'states': encoded_states,
            'actions': traj['actions'],
            'rewards': traj['rewards']
        }
        eval_episodes.append(episode)
    
    print(f"Loaded {len(states)} transitions from {len(trajectories)} trajectories")
    print(f"State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")
    print(f"Created {len(eval_episodes)} evaluation episodes")
    
    return states, actions, rewards, next_states, dones, eval_episodes
```

## Step 4: Create Training Script

Create your main training script:

```python
# experiments/train_cql_mimic.py

import torch
import numpy as np
from pathlib import Path

from src.models.rl import CQLAgent, ReplayBuffer, OfflineRLTrainer, CQLConfig
from src.data_processing.load_for_rl import load_mimic_for_offline_rl
from src.models.state_encoder import PatientStateEncoder

def main():
    # Configuration
    config = CQLConfig(
        state_dim=256,  # From your transformer encoder
        action_dim=10,  # Your action space
        hidden_dim=512,
        cql_alpha=2.0,  # Higher for safety
        gamma=0.99,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load data
    print("Loading MIMIC-III data...")
    states, actions, rewards, next_states, dones, eval_episodes = \
        load_mimic_for_offline_rl(
            data_path="./data/mimic",
            cohort="diabetes",
            state_encoder_path="./checkpoints/state_encoder_best.pt"
        )
    
    # Initialize agent
    print("Initializing CQL agent...")
    agent = CQLAgent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        cql_alpha=config.cql_alpha,
        gamma=config.gamma,
        device=config.device
    )
    
    # Create replay buffer
    print("Creating replay buffer...")
    replay_buffer = ReplayBuffer(
        capacity=1000000,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        device=config.device
    )
    
    # Load data into buffer
    replay_buffer.load_from_dataset(states, actions, rewards, next_states, dones)
    print(f"Buffer size: {len(replay_buffer)}")
    print(f"Buffer statistics: {replay_buffer.get_statistics()}")
    
    # Create evaluation function
    def evaluate_policy(agent):
        from src.models.rl.trainer import EvaluationManager
        evaluator = EvaluationManager(agent)
        metrics = evaluator.evaluate_on_episodes(eval_episodes)
        
        # Add safety metrics
        # TODO: Integrate with your safety validation
        
        return metrics
    
    # Create trainer
    print("Creating trainer...")
    trainer = OfflineRLTrainer(
        agent=agent,
        replay_buffer=replay_buffer,
        save_dir='./checkpoints/cql_mimic',
        eval_freq=2000,
        save_freq=10000,
        log_freq=100,
        use_mlflow=True
    )
    
    # Train
    print("Starting training...")
    history = trainer.train(
        num_iterations=100000,
        batch_size=512,
        eval_fn=evaluate_policy,
        verbose=True
    )
    
    print("Training completed!")
    print(f"Best evaluation return: {trainer.best_eval_return:.4f}")

if __name__ == "__main__":
    main()
```

## Step 5: Run Training

```bash
# Activate your environment
cd /Users/andy/Documents/Coding/Learning/mtp/healthcare_rl

# Install requirements
pip install torch numpy pyyaml tqdm scikit-learn

# Run training
python experiments/train_cql_mimic.py
```

## Step 6: Integrate Safety Validation

Integrate with your existing safety validation:

```python
# src/safety/safe_cql_agent.py

from src.models.rl.cql import CQLAgent
from src.safety import SafetyValidator  # Your existing safety module

class SafeCQLAgent(CQLAgent):
    """CQL agent with integrated safety validation."""
    
    def __init__(self, safety_validator: SafetyValidator, **kwargs):
        super().__init__(**kwargs)
        self.safety_validator = safety_validator
    
    def select_action(self, state, deterministic=False):
        """Select action with safety validation."""
        # Get action from policy
        action = super().select_action(state, deterministic)
        
        # Validate safety
        is_safe, violations = self.safety_validator.validate_action(state, action)
        
        if not is_safe:
            # Get safe fallback action
            action = self.safety_validator.get_safe_action(state)
            print(f"Safety violation detected: {violations}. Using safe action.")
        
        return action
```

## Step 7: Monitor Training

Track key metrics in MLflow:

```python
# View training progress
mlflow ui

# Navigate to http://localhost:5000
```

Key metrics to monitor:
- train/q1_loss: Should decrease
- train/cql_penalty: Indicates conservatism
- eval/mean_return: Should improve
- eval/safety_violations: Should be zero

## Step 8: Evaluate Trained Policy

```python
# src/evaluation/evaluate_cql.py

from src.models.rl import CQLAgent
from src.data_processing.load_for_rl import load_mimic_for_offline_rl

def evaluate_trained_policy(checkpoint_path):
    """Evaluate trained CQL policy."""
    # Load agent
    agent = CQLAgent(state_dim=256, action_dim=10)
    agent.load(checkpoint_path)
    agent.eval_mode()
    
    # Load test data
    _, _, _, _, _, eval_episodes = load_mimic_for_offline_rl(
        "./data/mimic",
        cohort="diabetes_test"
    )
    
    # Evaluate
    from src.models.rl.trainer import EvaluationManager
    evaluator = EvaluationManager(agent)
    
    metrics = evaluator.evaluate_on_episodes(eval_episodes)
    
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Additional healthcare-specific metrics
    # TODO: Add your clinical metrics here
    
    return metrics

if __name__ == "__main__":
    evaluate_trained_policy("./checkpoints/cql_mimic/best_model.pt")
```

## Common Integration Issues

### Issue 1: State Dimension Mismatch

**Symptom:** Error about tensor shapes

**Solution:**
```python
# Ensure state encoder output matches CQL state_dim
state = state_encoder(patient_sequence)
print(f"Encoded state shape: {state.shape}")  # Should be [batch_size, state_dim]
```

### Issue 2: Action Space Mismatch

**Symptom:** Actions outside valid range

**Solution:**
```python
# Normalize actions to [-1, 1] for CQL
def normalize_action(action, action_min, action_max):
    return 2 * (action - action_min) / (action_max - action_min) - 1

def denormalize_action(normalized_action, action_min, action_max):
    return (normalized_action + 1) / 2 * (action_max - action_min) + action_min
```

### Issue 3: Reward Scale

**Symptom:** Training unstable or not converging

**Solution:**
```python
# Normalize rewards to reasonable scale
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

## Next Steps

1. Start with synthetic data to verify integration works
2. Train on small subset of MIMIC-III (1000 patients)
3. Gradually increase to full dataset
4. Tune hyperparameters (especially cql_alpha)
5. Implement safety validation
6. Run comprehensive evaluation

## Questions?

If you encounter issues:
1. Check logs in MLflow UI
2. Verify data shapes match expected dimensions
3. Test individual components in isolation
4. Review training metrics for anomalies
