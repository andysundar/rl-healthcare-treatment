# Reward Functions - Installation Guide

## Files Included

You've downloaded a complete reward function system with 13 files:

### Documentation (2 files)
- `REWARD_IMPLEMENTATION_SUMMARY.md` - Quick overview
- `REWARD_FUNCTIONS.md` - Complete documentation

### Core Implementation (10 files in `src/rewards/`)
- `__init__.py` - Package initialization
- `base_reward.py` - Abstract base class
- `reward_config.py` - Configuration system
- `composite_reward.py` - Composite reward combiner
- `adherence_reward.py` - Adherence rewards
- `health_reward.py` - Health outcome rewards
- `safety_reward.py` - Safety penalties
- `cost_reward.py` - Cost penalties
- `reward_shaping.py` - Shaping utilities
- `test_rewards.py` - Test suite

### Examples (1 file)
- `reward_functions_demo.py` - Usage demonstrations

## Installation

### Step 1: Copy Files to Your Project

```bash
# Navigate to your project directory
cd /Users/andy/Documents/Coding/Learning/mtp/healthcare_rl

# The files are already in the correct locations:
# - src/rewards/ contains all Python modules
# - docs/ contains documentation
# - examples/ contains demo script
```

### Step 2: Verify Installation

```bash
# Test import
python3 -c "from src.rewards import CompositeRewardFunction; print('Success!')"
```

### Step 3: Run Tests

```bash
# Install pytest if needed
pip install pytest

# Run test suite
pytest src/rewards/test_rewards.py -v
```

### Step 4: Try the Demo

```bash
python3 examples/reward_functions_demo.py
```

## Quick Start

### Basic Usage

```python
from src.rewards import (
    CompositeRewardFunction,
    RewardConfig,
    AdherenceReward,
    HealthOutcomeReward,
    SafetyPenalty,
    CostEffectivenessReward
)

# 1. Create configuration
config = RewardConfig(
    w_adherence=1.0,
    w_health=2.0,
    w_safety=5.0,  # Safety most important
    w_cost=0.1
)

# 2. Build composite reward
reward_fn = CompositeRewardFunction(config)
reward_fn.add_component('adherence', AdherenceReward(config), config.w_adherence)
reward_fn.add_component('health', HealthOutcomeReward(config), config.w_health)
reward_fn.add_component('safety', SafetyPenalty(config), config.w_safety)
reward_fn.add_component('cost', CostEffectivenessReward(config), config.w_cost)

# 3. Use in your RL loop
state = {
    'glucose': 140.0,
    'systolic_bp': 125.0,
    'adherence_score': 0.7
}

action = {
    'insulin_dose': 20.0,
    'num_reminders': 2
}

next_state = {
    'glucose': 110.0,
    'systolic_bp': 118.0,
    'adherence_score': 0.85
}

# Compute reward
reward = reward_fn.compute_reward(state, action, next_state)
print(f"Reward: {reward:.3f}")

# Get detailed breakdown
breakdown = reward_fn.get_reward_components(state, action, next_state)
for key, value in breakdown.items():
    print(f"  {key}: {value:.3f}")
```

## Integration with Your CQL Implementation

### Option 1: Modify Your MDP Environment

```python
# In your HealthcareMDP class
from src.rewards import CompositeRewardFunction, RewardConfig

class HealthcareMDP:
    def __init__(self, config):
        # ... existing initialization ...
        
        # Add reward function
        reward_config = RewardConfig()
        self.reward_fn = CompositeRewardFunction(reward_config)
        # ... add components ...
    
    def step(self, action):
        # ... existing dynamics simulation ...
        
        # Use composite reward instead of hand-coded reward
        reward = self.reward_fn.compute_reward(
            self.state, 
            action, 
            next_state
        )
        
        return next_state, reward, done, info
```

### Option 2: Use in CQL Training Loop

```python
# In your CQL training script
from src.rewards import CompositeRewardFunction, RewardConfig

# Setup reward function
config = RewardConfig()
reward_fn = CompositeRewardFunction(config)
# ... add components ...

# In training loop
for batch in dataloader:
    states, actions, next_states, dones = batch
    
    # Compute rewards
    rewards = []
    for s, a, ns in zip(states, actions, next_states):
        r = reward_fn.compute_reward(s, a, ns)
        rewards.append(r)
    
    rewards = torch.tensor(rewards)
    
    # Train CQL
    loss = cql_agent.train_step(states, actions, rewards, next_states, dones)
```

## Configuration Options

### Predefined Configurations

```python
from src.rewards import (
    ConservativeRewardConfig,  # Safety-first
    AggressiveRewardConfig,    # Health-first
    CostAwareRewardConfig      # Cost-conscious
)

# Use conservative config (prioritizes safety)
config = ConservativeRewardConfig()
# w_safety = 10.0, stricter safety thresholds
```

### Custom Configuration

```python
config = RewardConfig(
    # Weights
    w_adherence=1.5,
    w_health=3.0,
    w_safety=8.0,
    w_cost=0.2,
    
    # Glucose targets (adjust for patient population)
    glucose_target=(90.0, 140.0),
    
    # Safety thresholds
    severe_hypoglycemia_threshold=65.0,
    
    # Penalties
    severe_hypoglycemia_penalty=-15.0,
    emergency_visit_penalty=-30.0
)
```

### Save/Load Configuration

```python
# Save configuration
config.save('my_reward_config.json')

# Load configuration
loaded_config = RewardConfig.load('my_reward_config.json')
```

## Testing

### Run Full Test Suite

```bash
pytest src/rewards/test_rewards.py -v
```

### Test Specific Scenarios

```python
# Test on your own data
from src.rewards.test_rewards import test_composite_reward_integration

# Create your test case
test_case = {
    'state': {...},
    'action': {...},
    'next_state': {...}
}

# Compute reward
reward = reward_fn.compute_reward(
    test_case['state'],
    test_case['action'],
    test_case['next_state']
)
```

## Troubleshooting

### Issue: Import errors

```bash
# Make sure you're in the project root
cd /Users/andy/Documents/Coding/Learning/mtp/healthcare_rl

# Try absolute import
python3 -c "import sys; sys.path.insert(0, '.'); from src.rewards import CompositeRewardFunction"
```

### Issue: Tests failing

```bash
# Install test dependencies
pip install pytest numpy

# Run with verbose output
pytest src/rewards/test_rewards.py -v -s
```

### Issue: Rewards seem wrong

```python
# Check reward breakdown
breakdown = reward_fn.get_reward_components(state, action, next_state)
print("Component breakdown:")
for key, value in breakdown.items():
    print(f"  {key}: {value:.3f}")

# Check if safety violations are detected
safety = SafetyPenalty(config)
is_safe = safety.is_safe_state(next_state)
print(f"State is safe: {is_safe}")
```

## Next Steps

1. **Verify Installation:**
   ```bash
   pytest src/rewards/test_rewards.py -v
   python3 examples/reward_functions_demo.py
   ```

2. **Customize for Your Use Case:**
   - Adjust reward weights in `RewardConfig`
   - Set patient-specific health targets
   - Add custom drug interactions to `SafetyPenalty`

3. **Integrate with CQL:**
   - Add reward function to your training loop
   - Track reward statistics during training
   - Compare different configurations

4. **Clinical Validation:**
   - Test on realistic patient scenarios
   - Show reward breakdowns to Dr. Sasmal
   - Validate safety penalties are working correctly

## Documentation

- **Quick Overview:** `REWARD_IMPLEMENTATION_SUMMARY.md`
- **Complete Guide:** `REWARD_FUNCTIONS.md`
- **Code Examples:** `examples/reward_functions_demo.py`
- **API Reference:** Docstrings in each module

## Support

For questions or issues:
1. Check the comprehensive documentation in `REWARD_FUNCTIONS.md`
2. Run the demo to see examples: `python3 examples/reward_functions_demo.py`
3. Look at test cases in `test_rewards.py` for usage examples

## File Structure

```
healthcare_rl/
├── src/
│   └── rewards/
│       ├── __init__.py                 # Package exports
│       ├── base_reward.py              # Base class
│       ├── adherence_reward.py         # Adherence rewards
│       ├── health_reward.py            # Health rewards
│       ├── safety_reward.py            # Safety penalties
│       ├── cost_reward.py              # Cost penalties
│       ├── composite_reward.py         # Composite combiner
│       ├── reward_shaping.py           # Utilities
│       ├── reward_config.py            # Configuration
│       └── test_rewards.py             # Tests
├── examples/
│   └── reward_functions_demo.py        # Demo script
└── docs/
    ├── REWARD_FUNCTIONS.md             # Full documentation
    └── REWARD_IMPLEMENTATION_SUMMARY.md # Quick summary
```

---

**Status: Ready for Integration**

All files are in place and tested. You can now integrate this reward system with your CQL implementation and start running experiments on MIMIC-III data!
