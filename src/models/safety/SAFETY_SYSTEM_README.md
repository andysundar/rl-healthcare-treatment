# Safety System for Healthcare RL

Complete safety system implementation for healthcare reinforcement learning with multiple constraint types, learned safety prediction, and comprehensive evaluation metrics.

## Overview

This safety system provides:
- **Hard Constraints**: Rule-based safety checks that are never violated
- **Soft Constraints**: Learned safety prediction via neural network
- **Real-time Safety Checking**: During action selection
- **Safety Enforcement**: Automatic correction of unsafe actions
- **Comprehensive Metrics**: For evaluation and reporting

## Components

### 1. Constraints (`constraints.py`)

Four types of safety constraints:

#### DosageConstraint
- Checks medication dosages are within safe FDA guidelines
- Supports age-adjusted limits (pediatric, geriatric)
- Per-drug dosage ranges

#### PhysiologicalConstraint
- Predicts next patient state
- Ensures vital signs remain in safe ranges
- Glucose: [70, 200] mg/dL
- Blood pressure: [90/60, 140/90] mmHg
- Heart rate, temperature, oxygen saturation

#### ContraindicationConstraint
- Checks drug allergies
- Detects drug-drug interactions
- Enforces age restrictions
- Validates against patient conditions

#### FrequencyConstraint
- Limits reminder frequency (max 7/week)
- Enforces minimum appointment intervals (3+ days)

### 2. Safety Critic (`safety_critic.py`)

Neural network for learned safety prediction:
- Architecture: [state+action → 256 → 128 → 1] with sigmoid
- Output: safety score ∈ [0, 1]
- Training: Binary classification on safe/unsafe examples
- Integration with constraint-based checking

### 3. Constraint Optimizer (`constraint_optimizer.py`)

Finds nearest safe action when constraints are violated:
- Uses scipy.optimize with SLSQP
- Minimizes: ||a_safe - a_proposed||²
- Subject to: all constraints satisfied
- Fallback: conservative safe action if optimization fails

### 4. Safety Layer (`safety_layer.py`)

Primary interface for all safety operations:
- `check_action_safety(state, action)` → (is_safe, violations)
- `enforce_safety(state, unsafe_action)` → safe_action
- `get_safe_action_bounds(state)` → (min_action, max_action)

### 5. Safety Metrics (`safety_metrics.py`)

Comprehensive evaluation metrics:
- **Safety Index**: 1 - (unsafe_states / total_states)
- **Violation Rate**: Per constraint type
- **Violation Severity**: Distance from safe bounds
- **Temporal Analysis**: Violations over time
- **Patient-specific**: Individual safety profiles

### 6. SafeRLAgent (`safety_layer.py`)

Wrapper for any RL agent:
- Transparent integration
- Automatic safety enforcement
- Tracks override statistics
- No changes to base agent needed

## Installation

```bash
# Copy safety package to your project
cp -r src/models/safety /path/to/your/project/src/models/

# Install dependencies
pip install numpy torch scipy
```

## Quick Start

### Basic Usage

```python
from src.models.safety import SafetyConfig, SafetyLayer

# 1. Create configuration
config = SafetyConfig()

# 2. Initialize safety layer
safety_layer = SafetyLayer(config)

# 3. Check action safety
state = {
    'glucose': 150,
    'age': 45,
    'allergies': [],
    'current_medications': []
}

action = {
    'medication_type': 'insulin',
    'dosage': 20.0
}

is_safe, violations = safety_layer.check_action_safety(state, action)

if is_safe:
    print("Action is safe!")
else:
    print(f"Violations: {violations}")
```

### Integration with RL Agent

```python
from src.models.safety import SafeRLAgent

# Wrap your existing agent
safe_agent = SafeRLAgent(
    rl_agent=your_cql_agent,
    safety_layer=safety_layer
)

# Use safe agent instead of base agent
action = safe_agent.select_action(state)  # Automatically enforces safety
```

### Training Safety Critic

```python
from src.models.safety import SafetyCritic, train_safety_critic

# Create critic
critic = SafetyCritic(state_dim=10, action_dim=5)

# Prepare data (from your environment)
safe_dataset = [...]  # List of (state, action) tuples that were safe
unsafe_dataset = [...] # List of (state, action) tuples that were unsafe

# Train
history = train_safety_critic(
    critic,
    safe_dataset,
    unsafe_dataset,
    num_epochs=100
)

# Integrate with safety layer
safety_layer.set_safety_critic(critic)
```

### Evaluating Safety

```python
from src.models.safety import generate_safety_report, print_safety_report

# Generate report
report = generate_safety_report(
    trajectories=test_trajectories,
    constraints=safety_layer.constraints,
    safe_ranges=config.safe_ranges
)

# Display report
print_safety_report(report, verbose=True)

# Access metrics
safety_index = report['overall_metrics']['safety_index']
```

## Configuration

Customize safety parameters:

```python
config = SafetyConfig(
    # Drug limits
    drug_limits={
        'insulin': (0.0, 100.0),
        'metformin': (500.0, 2000.0),
    },
    
    # Physiological ranges
    safe_ranges={
        'glucose': (70.0, 200.0),
        'blood_pressure_systolic': (90.0, 140.0),
    },
    
    # Safety critic
    safety_threshold=0.8,
    critic_hidden_dim=256,
    
    # Optimization
    max_optimization_iterations=100,
    
    # Frequency limits
    max_reminders_per_week=7,
    min_appointment_interval_days=3
)
```

## File Structure

```
src/models/safety/
├── __init__.py                 # Package exports
├── config.py                   # SafetyConfig dataclass
├── constraints.py              # All constraint classes
├── safety_critic.py            # Neural safety predictor
├── constraint_optimizer.py     # Safe action optimization
├── safety_layer.py             # Main interface + SafeRLAgent
└── safety_metrics.py           # Evaluation metrics

tests/
└── test_safety_system.py       # Comprehensive tests

examples/
└── safety_system_examples.py   # Usage examples
```

## Testing

Run comprehensive tests:

```bash
cd tests
python test_safety_system.py
```

Tests cover:
- Each constraint type individually
- Safety layer integration
- SafeRLAgent wrapper
- Safety critic training
- Metrics computation

## Examples

Run examples:

```bash
cd examples
python safety_system_examples.py
```

Examples demonstrate:
1. Basic safety checking
2. Safety enforcement
3. RL agent integration
4. Safety critic training
5. Safety evaluation
6. Custom configuration

## API Reference

### SafetyLayer

**Methods:**
- `check_action_safety(state, action)` → (is_safe, violations)
- `enforce_safety(state, unsafe_action)` → safe_action
- `get_safe_action_bounds(state)` → (min_action, max_action)
- `get_violation_statistics()` → stats
- `set_safety_critic(critic)` → None

### SafeRLAgent

**Methods:**
- `select_action(state)` → safe_action
- `get_override_rate()` → float
- `get_statistics()` → stats

### Constraints

**Interface:**
- `check(state, action)` → (is_satisfied, message)
- `get_constraint_name()` → str

### SafetyCritic

**Methods:**
- `forward(state, action)` → safety_score
- `train_step(safe_pairs, unsafe_pairs, optimizer)` → loss
- `evaluate(safe_pairs, unsafe_pairs)` → metrics
- `predict_safety(state, action, threshold)` → (is_safe, score)

### Metrics Functions

- `safety_index(trajectories, safe_ranges)` → float
- `violation_rate(trajectories, constraints)` → dict
- `violation_severity(trajectories, safe_ranges)` → float
- `generate_safety_report(...)` → report
- `print_safety_report(report, verbose)` → None

## Integration with Your Thesis

### Thesis Section 4: Implementation

Add to your implementation chapter:

```
4.X Safety System

We implemented a comprehensive safety system with multiple layers:

1. Constraint-Based Safety (Hard Constraints):
   - DosageConstraint: Ensures medication dosages within FDA guidelines
   - PhysiologicalConstraint: Predicts and validates next patient state
   - ContraindicationConstraint: Checks allergies and drug interactions
   - FrequencyConstraint: Enforces appointment/reminder limits

2. Learned Safety (Soft Constraints):
   - SafetyCritic: Neural network C(s,a) → [0,1]
   - Architecture: [state+action → 256 → 128 → 1]
   - Trained on safe/unsafe examples from environment

3. Safety Enforcement:
   - Constrained optimization: min ||a_safe - a_proposed||²
   - Subject to all constraints satisfied
   - Uses scipy.optimize SLSQP solver
```

### Thesis Section 5: Results

Report these metrics:

```python
report = generate_safety_report(trajectories, constraints, safe_ranges)

# For your results section:
safety_index = report['overall_metrics']['safety_index']
constraint_satisfaction = report['constraint_metrics']['satisfaction_rates']
violation_severity = report['overall_metrics']['violation_severity']
```

Expected values:
- Safety Index: > 0.95 (target)
- Constraint Satisfaction: > 0.95 for all constraint types
- Violation Severity: < 0.1 (low severity)
- Override Rate: 0.10-0.20 (10-20% of actions corrected)

## Performance

Computational overhead:
- Constraint checking: < 1ms per action
- Safety critic inference: < 5ms per action
- Constraint optimization: < 100ms per unsafe action
- Total overhead: negligible for episodic RL

## Troubleshooting

**Q: Optimization fails to find safe action**
A: Increase `max_optimization_iterations` or adjust action bounds

**Q: Too many safety overrides**
A: Check if constraints are too strict, adjust thresholds in SafetyConfig

**Q: Safety critic not learning**
A: Ensure balanced safe/unsafe examples, increase training epochs

**Q: Import errors**
A: Verify file structure matches expected layout, check __init__.py

## Citation

If you use this safety system in your research:

```
@mastersthesis{bandopadhyay2026healthcare,
  title={Reinforcement Learning for Healthcare Treatment Recommendations},
  author={Bandopadhyay, Anindya},
  year={2026},
  school={IIT Jodhpur}
}
```

## License

MIT License - Free to use for research and educational purposes.

## Contact

Author: Anindya Bandopadhyay (M23CSA508)  
Supervisor: Dr. Pradip Sasmal  
Institution: IIT Jodhpur

For questions or issues, contact: m23csa508@iitj.ac.in

---

**Version:** 1.0.0  
**Last Updated:** January 29, 2026
