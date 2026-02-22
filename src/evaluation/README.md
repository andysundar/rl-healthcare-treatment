# Healthcare RL Evaluation Framework

Comprehensive evaluation framework for offline reinforcement learning in healthcare settings.

## Features

### Off-Policy Evaluation (OPE)
- **Importance Sampling (IS)**: Standard IS with clipping
- **Weighted Importance Sampling (WIS)**: Lower variance estimator
- **Doubly Robust (DR)**: Combines IS with Q-function estimates
- **Direct Method (DM)**: Model-based evaluation

### Safety Metrics
- **Safety Index**: Fraction of time in safe states
- **Violation Rate**: Frequency of constraint violations  
- **Violation Severity**: Magnitude of safety violations
- **Critical Events**: Tracking of severe safety incidents

### Clinical Metrics
- **Health Improvement**: Comparison vs baseline
- **Time in Target Range**: Percentage of time at clinical goals
- **Adverse Event Rate**: Clinical safety outcomes
- **Goal Achievement**: Multi-metric success rate

### Performance Metrics
- **Average Return**: Standard RL metric
- **Success Rate**: Task-specific success criteria
- **Episode Statistics**: Length, variance analysis

## Installation

```bash
cd healthcare_rl_evaluation
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from evaluation import (
    EvaluationConfig,
    OffPolicyEvaluator,
    SafetyEvaluator,
    ClinicalEvaluator,
    PerformanceEvaluator
)

# Load configuration
config = EvaluationConfig()

# Evaluate safety
safety_evaluator = SafetyEvaluator(config)
safety_result = safety_evaluator.evaluate(trajectories)

# Evaluate clinical outcomes
clinical_evaluator = ClinicalEvaluator(config)
clinical_result = clinical_evaluator.evaluate(
    policy_trajectories,
    baseline_trajectories
)

# Off-policy evaluation
ope_evaluator = OffPolicyEvaluator(gamma=0.99, q_function=q_net)
ope_results = ope_evaluator.evaluate(
    policy,
    behavior_policy,
    trajectories,
    methods=['wis', 'dr']
)
```

## Examples

Complete examples are in the `examples/` directory:

```bash
# Run complete evaluation pipeline
python examples/complete_evaluation_example.py

# OPE methods comparison
python examples/ope_comparison.py

# Safety evaluation
python examples/safety_evaluation.py
```

## Configuration

Create custom configuration:

```python
from evaluation.config import EvaluationConfig, SafetyConfig

# Custom safety ranges
safety_config = SafetyConfig(
    safe_glucose_range=(70, 180),
    safe_bp_systolic_range=(90, 140)
)

config = EvaluationConfig(safety=safety_config)
```

Or load from YAML:

```python
config = EvaluationConfig.from_yaml('config.yaml')
```

## Integration with Your Project

To integrate with your healthcare RL project:

1. Copy `src/evaluation/` to your project
2. Install dependencies: `pip install -r requirements.txt`
3. Import evaluators in your code
4. Call evaluation after training

```python
# In your training script
from evaluation import OffPolicyEvaluator, SafetyEvaluator

# After training
trajectories = collect_offline_data()
safety_result = SafetyEvaluator(config).evaluate(trajectories)
ope_results = OffPolicyEvaluator(config.ope.gamma).evaluate(
    policy, behavior_policy, trajectories
)
```

## Directory Structure

```
healthcare_rl_evaluation/
├── src/
│   └── evaluation/
│       ├── __init__.py
│       ├── config.py              # Configuration
│       ├── off_policy_eval.py     # OPE methods
│       ├── safety_metrics.py      # Safety evaluation
│       ├── clinical_metrics.py    # Clinical outcomes
│       ├── performance_metrics.py # Performance metrics
│       ├── comparison.py          # Policy comparison
│       └── visualizations.py      # Plotting utilities
├── examples/
│   ├── complete_evaluation_example.py
│   ├── ope_comparison.py
│   └── safety_evaluation.py
├── tests/
│   └── test_evaluation.py
├── docs/
│   └── API.md
├── README.md
└── requirements.txt
```

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{healthcare_rl_eval2025,
  title={Comprehensive Evaluation Framework for Healthcare Reinforcement Learning},
  author={Anindya Bandopadhyay},
  year={2025},
  institution={IIT Jodhpur}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please contact:
- Email: m23csa508@iitj.ac.in
- GitHub: [Create issues for bugs/features]
