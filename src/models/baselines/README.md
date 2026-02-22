# Baseline Models for Healthcare RL

Comprehensive baseline policy implementations for comparison against Reinforcement Learning approaches in healthcare treatment recommendations.

## Overview

This package provides production-ready implementations of standard baseline policies used in healthcare RL research:

1. **Rule-Based Policies** - Clinical guideline-based decision making
2. **Random Policies** - Lower bound baselines
3. **Behavior Cloning** - Supervised learning from expert demonstrations
4. **Statistical Baselines** - Mean action, regression, and KNN approaches
5. **Comparison Framework** - Tools for evaluating and comparing baselines

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# Run tests
pytest tests/test_baselines.py -v

# Run examples
python examples/example_usage.py
```

## Quick Start

### 1. Rule-Based Policy

```python
from src.models.baselines import create_diabetes_rule_policy

# Create policy with pre-configured diabetes management rules
policy = create_diabetes_rule_policy(state_dim=10, action_dim=1)

# Use policy
state = np.array([250, 0.5, 7.0, ...])  # High glucose
action = policy.select_action(state)

# View rule statistics
stats = policy.get_rule_statistics()
```

### 2. Behavior Cloning

```python
from src.models.baselines import create_behavior_cloning_policy

# Create policy
policy = create_behavior_cloning_policy(
    state_dim=10,
    action_dim=1,
    hidden_dims=[128, 128]
)

# Train on historical data
policy.train(
    states=train_states,
    actions=train_actions,
    epochs=50,
    batch_size=256
)

# Use for inference
action = policy.select_action(state)
```

### 3. Comprehensive Comparison

```python
from src.models.baselines import compare_all_baselines

# Create multiple baselines
baselines = {
    'Rule-Based': rule_policy,
    'Random': random_policy,
    'Behavior-Cloning': bc_policy,
    'KNN': knn_policy
}

# Compare on test data
results = compare_all_baselines(
    test_data=test_data,
    baselines_dict=baselines,
    output_path='comparison_report.md'
)

print(results)
```

## Baseline Policies

### Rule-Based Policy

Implements clinical guideline-based decision rules. Ideal for encoding expert knowledge and comparing against learned policies.

**Features:**
- Threshold-based decision rules
- Priority-ordered rule evaluation
- Configurable for different conditions (diabetes, hypertension)
- Rule usage statistics

**Example:**
```python
from src.models.baselines import RuleBasedPolicy

policy = RuleBasedPolicy(
    action_space={'dim': 1, 'low': [0], 'high': [1]},
    state_dim=10
)

# Add custom rule
policy.add_rule(
    name="high_glucose",
    condition=lambda state: state[0] > 200,
    action=lambda state: np.array([0.8]),
    priority=100,
    description="High glucose: increase insulin"
)
```

**Clinical Rationale:**
Represents current clinical practice using established treatment guidelines. Strong baseline as it encodes decades of medical expertise.

### Random Policy

Selects random actions within safe bounds. Serves as lower bound for performance.

**Features:**
- Uniform or Gaussian sampling
- Configurable action bounds
- Safe variant that samples multiple actions

**Example:**
```python
from src.models.baselines import create_random_policy, create_safe_random_policy

# Simple random
random_policy = create_random_policy(
    action_dim=1,
    seed=42,
    distribution='uniform'
)

# Safe random (samples 10 actions, picks safest)
safe_random = create_safe_random_policy(
    action_dim=1,
    num_samples=10
)
```

**Clinical Rationale:**
Lower bound baseline. Any reasonable policy should significantly outperform random action selection.

### Behavior Cloning

Learns to imitate expert behavior using supervised learning. Strong baseline representing what's achievable by replicating past decisions.

**Features:**
- Neural network architecture
- Train/validation split support
- Model save/load functionality
- Configurable network depth

**Example:**
```python
from src.models.baselines import BehaviorCloningPolicy

policy = BehaviorCloningPolicy(
    action_space={'dim': 1, 'low': [0], 'high': [1]},
    state_dim=10,
    hidden_dims=[256, 256],
    learning_rate=1e-3
)

# Train
history = policy.train(
    states=train_states,
    actions=train_actions,
    val_states=val_states,
    val_actions=val_actions,
    epochs=100
)

# Save model
policy.save('models/bc_policy.pt')
```

**Clinical Rationale:**
Represents the best we can do by simply replicating historical clinical decisions. Important to show RL can improve beyond this.

### Statistical Baselines

Simple statistical methods for action prediction.

#### Mean Action Policy
Always returns the mean action from training data.

```python
from src.models.baselines import create_mean_action_policy

policy = create_mean_action_policy(action_dim=1, state_dim=10)
policy.fit(train_states, train_actions)
```

#### Regression Policy
Uses linear or ridge regression to map states to actions.

```python
from src.models.baselines import create_regression_policy

policy = create_regression_policy(
    state_dim=10,
    action_dim=1,
    regression_type='ridge',
    alpha=1.0
)
policy.fit(train_states, train_actions)
```

#### KNN Policy
Finds k nearest neighbors and averages their actions.

```python
from src.models.baselines import create_knn_policy

policy = create_knn_policy(
    state_dim=10,
    action_dim=1,
    k=5,
    metric='euclidean'
)
policy.fit(train_states, train_actions)
```

**Clinical Rationale:**
Simple, interpretable methods that provide additional comparison points. KNN is particularly interesting as it adapts to local patterns.

## Comparison Framework

### BaselineComparator

Comprehensive framework for comparing multiple baselines.

```python
from src.models.baselines import BaselineComparator

comparator = BaselineComparator(test_data)

# Add baselines
comparator.add_baseline('rule', rule_policy)
comparator.add_baseline('bc', bc_policy)

# Evaluate
results = comparator.evaluate_all()

# Get best
best_name, best_policy, best_score = comparator.get_best_baseline('mean_reward')

# Generate report
comparator.generate_report('report.md')
```

### Custom Metrics

Add domain-specific evaluation metrics:

```python
from src.models.baselines import (
    compute_action_stability,
    compute_safety_margin,
    compute_expected_return
)

# Define custom metrics
custom_metrics = {
    'action_stability': compute_action_stability,
    'safety_margin': lambda p, d: compute_safety_margin(p, d, 0.95),
    'expected_return': lambda p, d: compute_expected_return(p, d, 0.99)
}

# Use in comparison
results = compare_all_baselines(
    test_data=test_data,
    baselines_dict=baselines,
    custom_metrics=custom_metrics
)
```

## Evaluation Metrics

All baselines report standard metrics:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `mean_reward` | Average reward over test episodes | Higher is better |
| `safety_violations` | Number of unsafe states | Lower is better |
| `total_steps` | Total time steps evaluated | - |
| `safety_rate` | Fraction of safe states (1 - violations/steps) | Higher is better (target: >0.95) |
| `mean_action_value` | Average action magnitude | Depends on action space |
| `std_action_value` | Action variability | Lower = more stable |

Additional metrics can be computed:
- **Action Stability**: Variance of actions (lower = more consistent)
- **Safety Margin**: Distance above safety threshold
- **Expected Return**: Discounted cumulative reward

## Testing

Comprehensive test suite covering all baselines:

```bash
# Run all tests
pytest tests/test_baselines.py -v

# Run specific test
pytest tests/test_baselines.py::test_behavior_cloning_training -v

# Generate coverage report
pytest tests/test_baselines.py --cov=src/models/baselines --cov-report=html
```

## Examples

### Diabetes Management

```python
from src.models.baselines import create_diabetes_rule_policy

policy = create_diabetes_rule_policy(state_dim=10, action_dim=1)

# Test states
high_glucose_state = np.array([250, 0.5, ...])  # Glucose > 200
low_glucose_state = np.array([60, 0.5, ...])    # Glucose < 70
normal_state = np.array([120, 0.5, ...])        # 70 <= Glucose <= 200

# Actions
action_high = policy.select_action(high_glucose_state)  # Increases insulin
action_low = policy.select_action(low_glucose_state)    # Decreases insulin
action_normal = policy.select_action(normal_state)      # Maintains dosage
```

### Hypertension Management

```python
from src.models.baselines import create_hypertension_rule_policy

policy = create_hypertension_rule_policy(state_dim=10, action_dim=1)

# Based on ACC/AHA guidelines
# Stage 2: BP >= 140 mmHg
# Stage 1: 130 <= BP < 140 mmHg  
# Normal: BP < 120 mmHg
```

## Integration with RL Training

Compare baselines against your RL agent:

```python
from src.models.baselines import BaselineComparator

# After training your RL agent
rl_metrics = {
    'mean_reward': rl_mean_reward,
    'safety_rate': rl_safety_rate,
    # ... other metrics
}

comparator = BaselineComparator(test_data)
# Add baselines...

# Compare with RL
comparison = comparator.compare_against_rl(
    rl_results=rl_metrics,
    rl_name="CQL-Agent"
)

print(comparison)
```

## Project Structure

```
baseline_models/
├── src/
│   └── models/
│       └── baselines/
│           ├── __init__.py
│           ├── base_baseline.py          # Abstract base class
│           ├── rule_based.py             # Rule-based policies
│           ├── random_policy.py          # Random baselines
│           ├── behavior_cloning.py       # Behavior cloning
│           ├── statistical_baseline.py   # Statistical methods
│           └── compare_baselines.py      # Comparison framework
├── tests/
│   └── test_baselines.py                 # Comprehensive tests
├── examples/
│   └── example_usage.py                  # Usage examples
├── requirements.txt                      # Dependencies
└── README.md                            # This file
```

## Clinical Rationale for Baselines

### Why These Baselines?

1. **Rule-Based (Essential):**
   - Represents current clinical practice
   - Encodes expert knowledge
   - Strong baseline - must outperform to justify RL

2. **Behavior Cloning (Important):**
   - Shows what's achievable by mimicking past decisions
   - If BC performs well, RL must demonstrate clear advantage
   - Common in healthcare AI

3. **Random (Required):**
   - Lower bound - any reasonable policy should beat this
   - Important for statistical significance testing

4. **Statistical Methods (Useful):**
   - Simple, interpretable alternatives
   - Show that complex methods are justified
   - Quick to train and evaluate

### Expected Performance Hierarchy

```
RL Agent (Target)
    ↑
Behavior Cloning
    ↑
Rule-Based
    ↑
KNN / Regression
    ↑
Mean Action
    ↑
Random
```

## Limitations

- **Synthetic Data**: Examples use synthetic data; real clinical data has different characteristics
- **Safety Constraints**: Simple safety checks; real clinical deployment requires comprehensive validation
- **State Representation**: Assumes fixed-size state vectors; may need adaptation for variable-length histories
- **Action Space**: Continuous actions; discrete action spaces need minor modifications

## Future Extensions

Potential additions:
- Imitation learning with demonstrations
- Constrained behavior cloning
- Meta-learning baselines
- Multi-task baselines
- Uncertainty-aware policies

## Citation

If you use these baselines in your research, please cite:

```bibtex
@misc{healthcare_rl_baselines,
  author = {Bandopadhyay, Anindya},
  title = {Baseline Models for Healthcare RL Treatment Recommendations},
  year = {2025},
  institution = {IIT Jodhpur}
}
```

## Contributing

Contributions welcome! Please:
1. Add tests for new baselines
2. Follow existing code style
3. Update documentation
4. Provide clinical rationale

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues:
- Email: m23csa508@iitj.ac.in
- Supervisor: Dr. Pradip Sasmal, IIT Jodhpur

## Acknowledgments

Based on best practices from:
- Healthcare RL literature (Komorowski et al., 2018)
- Clinical decision support guidelines
- Offline RL benchmarking methods
