# Quick Start Guide - Baseline Models

Get up and running in 5 minutes!

## Installation (1 minute)

```bash
# Navigate to your thesis directory
cd /Users/andy/Documents/Coding/Learning/mtp/healthcare_rl

# Copy baseline package
cp -r /path/to/baseline_models/src/models/baselines src/models/

# Install dependencies
pip install numpy pandas scikit-learn torch pytest
```

## Test Installation (1 minute)

```python
# test_import.py
from src.models.baselines import (
    create_diabetes_rule_policy,
    create_random_policy,
    compare_all_baselines
)

print("✓ Successfully imported baselines!")

# Quick test
policy = create_diabetes_rule_policy()
state = [250, 0.5, 7.0, 50, 28, 0, 0, 0, 0, 0]  # High glucose
action = policy.select_action(state)
print(f"✓ Policy works! Action: {action}")
```

## First Comparison (3 minutes)

```python
# quick_comparison.py
import numpy as np
from src.models.baselines import (
    create_diabetes_rule_policy,
    create_random_policy,
    compare_all_baselines
)

# Generate quick test data
np.random.seed(42)
n = 100
test_data = []
for _ in range(n):
    state = np.random.randn(10)
    state[0] = np.random.uniform(80, 200)  # Glucose
    action = np.random.uniform(0, 1, 1)
    reward = -abs(state[0] - 120) / 100  # Reward based on glucose control
    next_state = state + np.random.randn(10) * 0.1
    done = False
    test_data.append((state, action, reward, next_state, done))

# Create baselines
baselines = {
    'Rule-Based': create_diabetes_rule_policy(),
    'Random': create_random_policy(seed=42)
}

# Compare
results = compare_all_baselines(test_data, baselines)
print("\nResults:")
print(results)
```

Run it:
```bash
python quick_comparison.py
```

Expected output:
```
Results:
                 mean_reward  safety_rate  ...
Rule-Based           -1.234        0.950  ...
Random              -3.567        0.850  ...
```

## Next Steps

1. **Read README.md** - Full documentation
2. **Check INTEGRATION_GUIDE.md** - How to use with your thesis
3. **Run examples/example_usage.py** - See all features
4. **Run tests/test_baselines.py** - Verify everything works

## Common Issues

**Import Error?**
```bash
# Make sure you're in the right directory
pwd  # Should show .../healthcare_rl
python -c "from src.models.baselines import *"
```

**State Dimension Mismatch?**
```python
# Your state might have different dimension
policy = create_diabetes_rule_policy(state_dim=YOUR_DIM)
```

**Need Help?**
- Read INTEGRATION_GUIDE.md for detailed instructions
- Check test_baselines.py for more examples
- Email: m23csa508@iitj.ac.in

## That's It!

You now have:
- ✓ 7 baseline policies
- ✓ Comparison framework
- ✓ Ready for thesis experiments

Time to compare against your RL agent! 🚀
