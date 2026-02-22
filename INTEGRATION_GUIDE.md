# RL Healthcare Treatment - Integrated Solution Guide

**Author:** Anindya Bandopadhyay (M23CSA508)  
**Project:** Reinforcement Learning for Healthcare Treatment Recommendations  
**Supervisor:** Dr. Pradip Sasmal, IIT Jodhpur

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Architecture](#project-architecture)
3. [Installation](#installation)
4. [Data Pipeline](#data-pipeline)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Complete Integration Workflow](#complete-integration-workflow)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Working Example (5 minutes)

```bash
# 1. Install dependencies
cd /home/claude/rl-healthcare-treatment
pip install -r requirements.txt --break-system-packages

# 2. Run quick demo with synthetic data
python src/environments/quick_start_demo.py

# 3. Run baseline comparison
python examples/example_usage.py
```

### Full Pipeline (30-60 minutes)

```bash
# Run the integrated solution
python run_integrated_solution.py --mode full
```

---

## Project Architecture

```
rl-healthcare-treatment/
├── src/
│   ├── data/              # Data processing pipeline
│   │   ├── mimic_loader.py       # Load MIMIC-III/IV data
│   │   ├── preprocessor.py       # Data preprocessing
│   │   ├── feature_engineering.py # Feature extraction
│   │   ├── cohort_builder.py     # Patient cohort selection
│   │   ├── data_validator.py     # Quality validation
│   │   └── synthetic_generator.py # Generate synthetic data
│   │
│   ├── environments/      # Healthcare simulation environments
│   │   ├── base_env.py           # Base environment class
│   │   ├── diabetes_env.py       # Diabetes management env
│   │   ├── adherence_env.py      # Medication adherence env
│   │   ├── patient_simulator.py  # Patient dynamics simulator
│   │   └── disease_models.py     # Clinical models (Bergman, etc.)
│   │
│   ├── models/            # RL algorithms and baselines
│   │   ├── rl/                   # Conservative Q-Learning (CQL), etc.
│   │   ├── baselines/            # Rule-based, BC, statistical baselines
│   │   ├── encoders/             # Patient state encoders
│   │   └── safety/               # Safety constraints
│   │
│   ├── rewards/           # Reward function implementations
│   │   ├── composite_reward.py   # Multi-objective reward
│   │   ├── health_reward.py      # Health improvement rewards
│   │   ├── adherence_reward.py   # Adherence rewards
│   │   ├── safety_reward.py      # Safety penalties
│   │   └── cost_reward.py        # Cost-effectiveness
│   │
│   ├── evaluation/        # Evaluation framework
│   │   ├── off_policy_eval.py    # OPE methods (WIS, DR, DM)
│   │   ├── clinical_metrics.py   # Clinical validation
│   │   ├── safety_metrics.py     # Safety index calculation
│   │   └── comparison.py         # Baseline comparison
│   │
│   └── configs/           # Configuration files
│
├── examples/              # Example scripts
│   ├── example_pipeline.py       # Complete data pipeline
│   ├── example_usage.py          # Baseline usage examples
│   ├── run_evaluation.py         # Evaluation examples
│   └── diabetes_env_example.py   # Environment examples
│
└── tests/                 # Unit tests
    ├── test_env.py
    ├── test_baselines.py
    ├── test_rewards.py
    └── test_framework.py
```

---

## Installation

### 1. System Requirements

- **OS:** macOS (M4), Linux, or Windows
- **Python:** 3.8+
- **GPU:** Optional (PyTorch with MPS/CUDA support)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 5GB for dependencies, 10-50GB for MIMIC data

### 2. Install Dependencies

```bash
cd /home/claude/rl-healthcare-treatment

# Install core dependencies
pip install -r requirements.txt --break-system-packages

# Optional: Install GPU support for PyTorch (macOS M4)
# PyTorch should automatically detect MPS
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 3. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Quick environment check
python -c "
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from src.data import MIMICLoader
from src.environments import DiabetesEnv
from src.models.baselines import create_diabetes_rule_policy
print('✓ All imports successful!')
"
```

---

## Data Pipeline

### Option A: Using MIMIC-III Data (Real Clinical Data)

#### Prerequisites

1. **Complete MIMIC-III CITI Training**
   - Visit: https://physionet.org/content/mimiciii/
   - Complete required ethics training
   - Request access and download data

2. **Organize MIMIC-III Data**

```bash
# Expected directory structure
data/raw/mimic-iii/
├── PATIENTS.csv
├── ADMISSIONS.csv
├── DIAGNOSES_ICD.csv
├── LABEVENTS.csv
├── CHARTEVENTS.csv
└── PRESCRIPTIONS.csv
```

#### Run Data Pipeline

```bash
# Full pipeline with MIMIC-III data
python examples/example_pipeline.py \
    --mimic-dir data/raw/mimic-iii \
    --output-dir data/processed

# Sample mode (for testing)
python examples/example_pipeline.py \
    --mimic-dir data/raw/mimic-iii \
    --output-dir data/processed \
    --sample \
    --sample-size 100
```

### Option B: Using Synthetic Data (No Prerequisites)

```python
from src.data import SyntheticDataGenerator

# Generate synthetic diabetes patient data
generator = SyntheticDataGenerator()

# Generate patient cohort
patients = generator.generate_diabetes_cohort(
    n_patients=1000,
    time_horizon=365  # 1 year trajectories
)

# Save to disk
generator.save_cohort(patients, 'data/processed/synthetic_cohort.pkl')
```

### Data Processing Steps

The pipeline automatically handles:

1. **Cohort Selection**
   - Define inclusion/exclusion criteria
   - Filter by age, diagnosis, admission history
   - Generate cohort statistics

2. **Feature Engineering**
   - Extract demographics (age, gender, BMI)
   - Process lab values (glucose, HbA1c, BP)
   - Encode medications (one-hot, embeddings)
   - Create temporal features (trends, time gaps)

3. **Data Preprocessing**
   - Handle missing values (median imputation)
   - Detect and clip outliers (z-score > 3)
   - Normalize features (robust scaling)
   - Validate physiological ranges

4. **Quality Validation**
   - Completeness check (missing rate < 30%)
   - Consistency check (temporal ordering)
   - Generate quality report

5. **Train/Val/Test Split**
   - 70% train, 15% validation, 15% test
   - Patient-level stratification
   - Save as Parquet files

---

## Model Training

### 1. Baseline Policies

Run all baseline comparisons:

```bash
python examples/example_usage.py
```

This trains and evaluates:
- **Rule-Based Policy:** Clinical guidelines for diabetes
- **Random Policy:** Uniform random actions
- **Safe Random Policy:** Constrained random actions
- **Mean Action Policy:** Dataset mean action
- **Ridge Regression:** Linear policy
- **K-Nearest Neighbors:** Non-parametric policy
- **Behavior Cloning:** Neural network supervised learning

### 2. Conservative Q-Learning (CQL)

```python
from src.models.rl import ConservativeQLearning, CQLConfig
from src.environments import DiabetesEnv

# Configure CQL
config = CQLConfig(
    state_dim=10,
    action_dim=1,
    hidden_dim=256,
    q_lr=3e-4,
    policy_lr=1e-4,
    cql_alpha=5.0,  # Conservative penalty weight
    gamma=0.99
)

# Initialize CQL agent
agent = ConservativeQLearning(config)

# Create environment
env = DiabetesEnv(patient_params={'insulin_sensitivity': 1.0})

# Train offline (from dataset)
agent.train_offline(
    dataset=train_data,
    n_epochs=100,
    batch_size=256,
    log_interval=10
)

# Save trained model
agent.save_model('models/cql_diabetes.pth')
```

### 3. Patient State Encoder

```python
from src.models.encoders import PatientTransformerEncoder

# Configure encoder
encoder = PatientTransformerEncoder(
    config={
        'lab_dim': 20,
        'vital_dim': 10,
        'med_vocab_size': 100,
        'demo_dim': 5,
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'state_dim': 128
    }
)

# Train encoder (autoencoder or supervised)
encoder.train(
    patient_sequences=train_sequences,
    epochs=50,
    batch_size=32
)

# Use in RL pipeline
state_embedding = encoder(patient_sequence)
```

---

## Evaluation

### 1. Off-Policy Evaluation (OPE)

```python
from src.evaluation import OffPolicyEvaluator

evaluator = OffPolicyEvaluator()

# Evaluate trained policy
results = evaluator.evaluate_policy(
    policy=cql_agent,
    dataset=test_data,
    methods=['DR', 'WIS', 'DM']  # Doubly Robust, WIS, Direct Method
)

print(f"Policy Value (DR): {results['doubly_robust']:.4f}")
print(f"Policy Value (WIS): {results['weighted_importance_sampling']:.4f}")
print(f"Policy Value (DM): {results['direct_method']:.4f}")
```

### 2. Safety Metrics

```python
from src.evaluation import SafetyMetrics

safety = SafetyMetrics()

# Compute safety index
safety_index = safety.compute_safety_index(
    policy=cql_agent,
    test_data=test_data,
    safe_ranges={
        'glucose': (70, 180),
        'blood_pressure_systolic': (90, 140)
    }
)

print(f"Safety Index: {safety_index:.2%}")
```

### 3. Clinical Validation

```python
from src.evaluation import ClinicalMetrics

clinical = ClinicalMetrics()

# Validate against clinical guidelines
validation = clinical.validate_recommendations(
    policy=cql_agent,
    test_cases=clinical_test_cases,
    guidelines=diabetes_guidelines
)

print(f"Guideline Compliance: {validation['guideline_compliance']:.2%}")
print(f"Contraindication Detection: {validation['contraindication_detection']:.2%}")
```

### 4. Baseline Comparison

```bash
# Run comprehensive baseline comparison
python examples/run_evaluation.py

# Generate comparison report
python examples/complete_evaluation_example.py
```

Outputs:
- `baseline_comparison_report.md` - Markdown summary
- `baseline_comparison_report.json` - Raw results
- Visualizations (reward curves, safety plots)

---

## Complete Integration Workflow

### End-to-End Pipeline

I've created a comprehensive integration script that runs the entire pipeline:

```bash
# Run full integrated solution
python run_integrated_solution.py --mode full

# Run with synthetic data only (no MIMIC access needed)
python run_integrated_solution.py --mode synthetic

# Run only training and evaluation
python run_integrated_solution.py --mode train-eval

# Run quick test mode
python run_integrated_solution.py --mode test
```

### Workflow Stages

**Stage 1: Data Preparation**
- Load MIMIC-III or generate synthetic data
- Build patient cohort
- Feature engineering
- Data preprocessing
- Quality validation
- Train/val/test split

**Stage 2: Environment Setup**
- Initialize diabetes/adherence environments
- Configure reward functions
- Set up patient simulators

**Stage 3: Baseline Training**
- Train all baseline policies
- Evaluate on test data
- Generate comparison reports

**Stage 4: CQL Training**
- Train Conservative Q-Learning agent
- Monitor training metrics
- Save checkpoints

**Stage 5: Evaluation**
- Off-policy evaluation (WIS, DR, DM)
- Safety metrics computation
- Clinical validation
- Generate visualizations

**Stage 6: Results & Reports**
- LaTeX tables for thesis
- Markdown summaries
- Performance plots
- Safety analysis

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Run from project root
cd /home/claude/rl-healthcare-treatment
export PYTHONPATH=$PYTHONPATH:/home/claude/rl-healthcare-treatment
```

#### 2. MIMIC Data Not Found

```bash
# Error: FileNotFoundError: MIMIC-III data not found
# Solution: Use synthetic data mode
python run_integrated_solution.py --mode synthetic
```

#### 3. Memory Issues

```python
# Error: Out of memory during training
# Solution: Reduce batch size or use sample mode

# In run_integrated_solution.py
config.batch_size = 128  # Reduce from 256
config.use_sample = True
config.sample_size = 100
```

#### 4. PyTorch MPS Issues

```bash
# Error: RuntimeError: MPS backend not available
# Solution: Force CPU mode

export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### 5. Age Calculation Overflow

```python
# Error: OverflowError in date calculations
# Solution: Already fixed with AgeBucketing class
# Uses categorical age groups instead of datetime arithmetic
```

### Getting Help

For issues specific to:
- **Data Processing:** Check `src/data/README.md`
- **Environments:** Check `src/environments/CODE_OVERVIEW.md`
- **Rewards:** Check `src/rewards/REWARD_IMPLEMENTATION_SUMMARY.md`
- **Evaluation:** Check `src/evaluation/INTEGRATION_GUIDE.md`

---

## Next Steps for Thesis

### Immediate Tasks (This Week)

1. **Verify Ethics Approval**
   - Confirm IRB/ethics clearance status
   - Document in thesis methodology section

2. **Run Full Pipeline**
   - Execute `run_integrated_solution.py --mode full`
   - Collect baseline results for thesis

3. **Generate Initial Results**
   - Compare CQL vs baselines
   - Document safety metrics
   - Create visualization plots

### Short-Term (Next 2 Weeks)

1. **Large-Scale Experiments**
   - Run on full MIMIC-III cohort
   - Hyperparameter tuning
   - Cross-validation

2. **Interpretability Analysis**
   - Extract decision rules
   - Generate counterfactual explanations
   - Clinical validation with experts

3. **Thesis Writing**
   - Results section with LaTeX tables
   - Discussion of findings
   - Limitations and future work

### Medium-Term (1 Month)

1. **Final Experiments**
   - Transfer learning validation
   - Population adaptation tests
   - Comprehensive safety analysis

2. **Thesis Completion**
   - Complete all chapters
   - Generate final figures
   - Prepare defense presentation

3. **Documentation**
   - Code documentation
   - README updates
   - Usage examples

---

## Project Status Summary

✅ **Completed Components (70%)**
- Data processing pipeline (100%)
- Healthcare environments (100%)
- Reward functions (100%)
- Baseline policies (100%)
- CQL implementation (90%)
- OPE methods (100%)
- Safety metrics (100%)
- Integration framework (100%)

⏳ **In Progress (20%)**
- Large-scale MIMIC-III experiments
- Clinical validation with experts
- Interpretability tools (partially complete)

📋 **Remaining Tasks (10%)**
- Ethics approval confirmation
- Final thesis experiments
- Defense preparation

---

## Contact

**Student:** Anindya Bandopadhyay  
**Email:** m23csa508@iitj.ac.in  
**Supervisor:** Dr. Pradip Sasmal, IIT Jodhpur  
**GitHub:** https://github.com/andysundar/rl-healthcare-treatment

**Thesis Deadline:** March 25, 2026
