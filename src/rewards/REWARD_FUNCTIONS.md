# Healthcare RL Reward Functions

Comprehensive reward function system for reinforcement learning in healthcare treatment recommendations.

## Overview

This module provides a modular, configurable reward function system that balances multiple clinical objectives:
- **Health Outcomes**: Improvement in clinical markers (glucose, blood pressure, HbA1c)
- **Safety**: Prevention of adverse events and dangerous physiological states
- **Adherence**: Medication and treatment plan compliance
- **Cost-Effectiveness**: Resource utilization and healthcare costs

## Quick Start

```python
from src.rewards import (
    CompositeRewardFunction,
    RewardConfig,
    AdherenceReward,
    HealthOutcomeReward,
    SafetyPenalty,
    CostEffectivenessReward
)

# Create configuration
config = RewardConfig(
    w_adherence=1.0,    # Adherence weight
    w_health=2.0,       # Health outcome weight
    w_safety=5.0,       # Safety weight (most important)
    w_cost=0.1          # Cost weight
)

# Create composite reward
reward_fn = CompositeRewardFunction(config)

# Add components
reward_fn.add_component('adherence', AdherenceReward(config), config.w_adherence)
reward_fn.add_component('health', HealthOutcomeReward(config), config.w_health)
reward_fn.add_component('safety', SafetyPenalty(config), config.w_safety)
reward_fn.add_component('cost', CostEffectivenessReward(config), config.w_cost)

# Compute reward for a transition
state = {'glucose': 140.0, 'adherence_score': 0.7}
action = {'insulin_dose': 20.0}
next_state = {'glucose': 110.0, 'adherence_score': 0.85}

reward = reward_fn.compute_reward(state, action, next_state)
print(f"Reward: {reward:.3f}")
```

## Components

### 1. Base Reward Function

All reward functions inherit from `BaseRewardFunction`:

```python
from src.rewards import BaseRewardFunction

class CustomReward(BaseRewardFunction):
    def compute_reward(self, state, action, next_state):
        # Your reward logic here
        return reward_value
```

### 2. Adherence Reward

Rewards medication adherence and improvements:

```python
adherence_reward = AdherenceReward(config)
reward = adherence_reward.compute_reward(state, action, next_state)

# Breakdown
components = adherence_reward.get_reward_components(state, action, next_state)
# Returns: {
#   'adherence_base': 0.85,
#   'adherence_improvement': 0.30,
#   'adherence_high_bonus': 1.0,
#   'adherence_sustained_bonus': 2.0,
#   'adherence_total': 4.15
# }
```

**Rewards:**
- Current adherence level (0-1)
- Improvement bonus (×2 multiplier)
- High adherence bonus (>0.8)
- Sustained adherence bonus (>30 days)

### 3. Health Outcome Reward

Rewards improvements in clinical markers:

```python
health_reward = HealthOutcomeReward(config)
reward = health_reward.compute_reward(state, action, next_state)
```

**Tracked Metrics:**
- Glucose (target: 80-130 mg/dL)
- Systolic BP (target: 90-120 mmHg)
- Diastolic BP (target: 60-80 mmHg)
- HbA1c (target: 4.0-6.5%)
- Cholesterol (target: <200 mg/dL)

**Rewards:**
- Being within target range (+1.0)
- Distance from target (scaled penalty)
- Improvement toward target (×1.5 bonus)

### 4. Safety Penalty

Penalizes unsafe states and actions:

```python
safety_penalty = SafetyPenalty(config)
penalty = safety_penalty.compute_reward(state, action, next_state)
```

**Physiological Safety:**
- Severe hypoglycemia (<60 mg/dL): -10.0
- Moderate hypoglycemia (<70 mg/dL): -3.0
- Severe hyperglycemia (>300 mg/dL): -5.0
- Severe hypertension (>180 mmHg): -8.0

**Medication Safety:**
- Drug interactions: -7.0
- Overdose: -8.0
- Contraindications: -10.0

**Adverse Events:**
- Emergency visit: -20.0
- Hospitalization: -15.0
- ICU admission: -25.0

### 5. Cost-Effectiveness Reward

Penalizes expensive interventions:

```python
cost_reward = CostEffectivenessReward(config)
penalty = cost_reward.compute_reward(state, action, next_state)
```

**Costs:**
- Medications (per unit/dose)
- Appointments ($100)
- Lab tests ($50)
- Emergency visits ($5000)
- Hospitalizations ($2000/day)
- ICU stays ($10000/day)

### 6. Composite Reward

Combines all components with configurable weights:

```python
composite = CompositeRewardFunction(config)
composite.add_component('adherence', AdherenceReward(config), 1.0)
composite.add_component('health', HealthOutcomeReward(config), 2.0)
composite.add_component('safety', SafetyPenalty(config), 5.0)
composite.add_component('cost', CostEffectivenessReward(config), 0.1)

# Total reward
reward = composite.compute_reward(state, action, next_state)

# Detailed breakdown
breakdown = composite.get_reward_components(state, action, next_state)
```

## Configuration

### Standard Configuration

```python
config = RewardConfig(
    # Component weights
    w_adherence=1.0,
    w_health=2.0,
    w_safety=5.0,
    w_cost=0.1,
    
    # Health targets
    glucose_target=(80.0, 130.0),
    systolic_bp_target=(90.0, 120.0),
    
    # Safety thresholds
    severe_hypoglycemia_threshold=60.0,
    severe_hyperglycemia_threshold=300.0,
    
    # Costs
    appointment_cost=100.0,
    emergency_visit_cost=5000.0
)
```

### Predefined Configurations

**Conservative (Safety-First):**
```python
config = ConservativeRewardConfig()
# w_safety=10.0, stricter thresholds
```

**Aggressive (Health-First):**
```python
config = AggressiveRewardConfig()
# w_health=5.0, higher improvement multiplier
```

**Cost-Aware:**
```python
config = CostAwareRewardConfig()
# w_cost=1.0, balanced cost consideration
```

## Advanced Features

### Reward Shaping

```python
from src.rewards import (
    potential_based_shaping,
    health_potential_function,
    normalize_reward,
    clip_reward
)

# Potential-based shaping (preserves optimal policy)
shaping = potential_based_shaping(
    state, 
    next_state, 
    health_potential_function, 
    gamma=0.99
)

# Normalization
normalized = normalize_reward(reward, min_val=-10.0, max_val=10.0)

# Clipping
clipped = clip_reward(reward, clip_range=(-5.0, 5.0))
```

### Online Normalization

```python
from src.rewards import RewardNormalizer

normalizer = RewardNormalizer(clip_range=(-10.0, 10.0))

for state, action, next_state in trajectory:
    raw_reward = reward_fn.compute_reward(state, action, next_state)
    
    # Update statistics
    normalizer.update(raw_reward)
    
    # Normalize
    normalized_reward = normalizer.normalize(raw_reward)
```

### Curiosity Bonus

```python
from src.rewards import curiosity_bonus

state_visit_counts = {}
bonus = curiosity_bonus(state, state_visit_counts, bonus_scale=0.1)
total_reward = base_reward + bonus
```

## Clinical Validation

### Safety Checks

```python
safety = SafetyPenalty(config)

# Check if state is safe
is_safe = safety.is_safe_state(next_state)

# Add custom drug interaction
safety.add_drug_interaction('metformin', 'glipizide')
```

### Personalized Targets

```python
health_reward = HealthOutcomeReward(config)

# Set patient-specific targets
health_reward.set_personalized_targets(
    patient_id='patient_123',
    metric_targets={
        'glucose': (90.0, 140.0),  # Adjusted for this patient
        'systolic_bp': (100.0, 130.0)
    }
)
```

## Testing

Run the test suite:

```bash
pytest src/rewards/test_rewards.py -v
```

Run the demo:

```bash
python examples/reward_functions_demo.py
```

## Example Scenarios

### Successful Diabetes Management

```python
state = {
    'glucose': 160.0,
    'hba1c': 8.0,
    'adherence_score': 0.5
}

action = {
    'insulin_dose': 25.0,
    'num_reminders': 2
}

next_state = {
    'glucose': 110.0,
    'hba1c': 7.0,
    'adherence_score': 0.85
}

reward = composite.compute_reward(state, action, next_state)
# Expected: High positive reward (~3-5)
```

### Hypoglycemia Event

```python
state = {'glucose': 80.0}
action = {'insulin_dose': 30.0}  # Overdose
next_state = {'glucose': 50.0}  # Severe hypoglycemia

reward = composite.compute_reward(state, action, next_state)
# Expected: Large negative reward (~-50) due to safety penalty
```

### Emergency Hospitalization

```python
state = {'glucose': 280.0, 'systolic_bp': 175.0}
action = {'insulin_dose': 40.0}
next_state = {
    'glucose': 320.0,
    'systolic_bp': 185.0,
    'emergency_visit': True,
    'hospitalized': True
}

reward = composite.compute_reward(state, action, next_state)
# Expected: Very large negative reward (~-100+) from safety + cost
```

## Integration with RL Algorithms

### With CQL Training

```python
from src.rl.cql import ConservativeQLearning
from src.rewards import CompositeRewardFunction, RewardConfig

# Setup reward function
config = RewardConfig()
reward_fn = CompositeRewardFunction(config)
# ... add components ...

# Use in CQL training
cql_agent = ConservativeQLearning(state_dim, action_dim, config)

for batch in dataloader:
    states, actions, next_states, dones = batch
    
    # Compute rewards
    rewards = []
    for s, a, ns in zip(states, actions, next_states):
        r = reward_fn.compute_reward(s, a, ns)
        rewards.append(r)
    
    # Train
    loss = cql_agent.train_step(states, actions, rewards, next_states, dones)
```

### With Environment

```python
from src.environment import HealthcareMDP

class RewardedHealthcareMDP(HealthcareMDP):
    def __init__(self, config, reward_fn):
        super().__init__(config)
        self.reward_fn = reward_fn
    
    def step(self, action):
        state = self.get_state()
        next_state = self._simulate_dynamics(state, action)
        
        # Use composite reward
        reward = self.reward_fn.compute_reward(state, action, next_state)
        
        done = self._check_done(next_state)
        info = {}
        
        return next_state, reward, done, info
```

## Best Practices

1. **Safety First**: Always weight safety penalties highest (w_safety >= 5.0)

2. **Clinical Validation**: Validate reward functions with clinical experts

3. **Interpretability**: Use `get_reward_components()` to understand reward breakdown

4. **Personalization**: Adjust target ranges for individual patients

5. **Monitoring**: Track reward statistics during training

6. **Testing**: Test edge cases (hypoglycemia, emergencies, drug interactions)

7. **Normalization**: Use component normalization for fair weighting

## Troubleshooting

**Q: Rewards are too large/small?**
- Adjust component weights
- Use normalization: `config.normalize_components = True`
- Check reward statistics: `composite.get_component_statistics()`

**Q: Agent ignores safety?**
- Increase safety weight: `config.w_safety = 10.0`
- Use ConservativeRewardConfig
- Verify safety penalties are working

**Q: Training is unstable?**
- Use reward normalization: `RewardNormalizer`
- Clip extreme rewards: `clip_reward()`
- Smooth rewards: `reward_smoothing_filter()`

**Q: Need patient-specific targets?**
```python
health_reward.set_personalized_targets(patient_id, targets)
```

## References

Clinical reward design based on:
- Diabetes treatment guidelines (ADA 2025)
- Hypertension management (JNC 8)
- Healthcare cost-effectiveness analysis
- Clinical decision support systems

## License

MIT License - See LICENSE file

## Citation

```bibtex
@thesis{bandopadhyay2026healthcare,
  title={Reinforcement Learning for Healthcare Treatment Recommendations},
  author={Bandopadhyay, Anindya},
  school={IIT Jodhpur},
  year={2026}
}
```

## Contact

For questions or issues, please contact:
- Andy Bandopadhyay (m23csa508@iitj.ac.in)
- Supervisor: Dr. Pradip Sasmal (IIT Jodhpur)
