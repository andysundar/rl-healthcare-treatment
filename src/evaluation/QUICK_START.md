# Quick Start Guide - 5 Minutes to Running Evaluation

## Installation (1 minute)

```bash
# Extract package
tar -xzf healthcare_rl_evaluation.tar.gz
cd healthcare_rl_evaluation

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_installation.py
```

## First Evaluation (2 minutes)

```bash
# Run complete example
cd examples
python complete_evaluation_example.py
```

This will:
- Generate synthetic healthcare trajectories
- Run all evaluation metrics (Safety, Clinical, OPE)
- Create visualizations
- Display comprehensive results

## Integration with Your Project (2 minutes)

```python
# In your training script
from src.evaluation import SafetyEvaluator, EvaluationConfig

# After training
config = EvaluationConfig()
safety_eval = SafetyEvaluator(config)

# Convert your data
trajectories = [
    {
        'states': [{'glucose': 100, 'bp_systolic': 120}, ...],
        'actions': [{'dosage': 0.5}, ...],
        'rewards': [1.0, ...],
        'next_states': [{}, ...],
        'dones': [False, ..., True]
    }
]

# Evaluate
result = safety_eval.evaluate(trajectories)
print(f"Safety Index: {result.safety_index:.3f}")
```

## For Your Thesis

Essential evaluations to run:

```python
from src.evaluation import (
    SafetyEvaluator,
    ClinicalEvaluator,
    OffPolicyEvaluator
)

# 1. Safety (critical for healthcare)
safety_result = SafetyEvaluator(config).evaluate(trajectories)

# 2. Clinical outcomes (compare with baseline)
clinical_result = ClinicalEvaluator(config).evaluate(
    cql_trajectories,
    baseline_trajectories
)

# 3. Off-policy evaluation (WIS recommended)
ope_result = OffPolicyEvaluator(gamma=0.99).evaluate(
    cql_policy,
    behavior_policy,
    trajectories,
    methods=['wis', 'dr']
)
```

## Key Files

- `INTEGRATION_GUIDE.md` - Detailed integration steps
- `SUMMARY.md` - Complete overview
- `examples/` - Working examples
- `src/evaluation/` - Main framework

## Need Help?

1. Check `INTEGRATION_GUIDE.md` for step-by-step instructions
2. Run examples to see usage patterns
3. Review `SUMMARY.md` for overview

## Timeline

- Day 1: Extract, install, test examples
- Day 2: Integrate with MIMIC-III pipeline
- Day 3-4: Run full evaluation
- Day 5: Generate thesis figures and tables

That's it! You're ready to complete your thesis evaluation.
