# Reinforcement Learning for Healthcare Treatment Recommendations

**M.Tech Thesis Project**  
**Author:** Anindya Bandopadhyay (M23CSA508)  
**Supervisor:** Dr. Pradip Sasmal  
**Institution:** IIT Jodhpur  
**Deadline:** March 25, 2026

---

## Project Overview

This project develops a **safety-constrained offline reinforcement learning framework** for personalized healthcare treatment recommendations. The system uses Conservative Q-Learning (CQL) on MIMIC-III clinical data to optimize medication dosing, appointment scheduling, and adherence interventions while maintaining strict safety guarantees.

### Key Features

- **Offline RL Algorithms:** Conservative Q-Learning (CQL), Batch Constrained Q-Learning (BCQ)
- **Safety-First Design:** Multi-layered safety validation, hard constraints, clinical guideline compliance
- **Comprehensive Baselines:** Rule-based, statistical, behavior cloning policies for comparison
- **Off-Policy Evaluation:** Weighted Importance Sampling (WIS), Doubly Robust (DR), Direct Method
- **Clinical Validation:** Safety metrics, guideline compliance, interpretability tools
- **Production-Ready Code:** Modular architecture, extensive testing, documentation

---

## Quick Start

### Option 1: Interactive Quick Start (Recommended)

```bash
./src/quick_start.sh
```

This script guides you through:
1. Quick Demo (5 min) - Synthetic data + baseline comparison
2. Full Synthetic (30 min) - Complete pipeline
3. MIMIC-III Sample (1-2 hrs) - Small real data cohort
4. Full MIMIC-III (several hrs) - Complete analysis

### Option 2: Manual Execution

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# Run with synthetic data (no prerequisites)
python src/run_integrated_solution.py --mode synthetic

# Run with MIMIC-III data
python src/run_integrated_solution.py \
    --mode full \
    --mimic-dir data/raw/mimic-iii \
    --train-cql
```

### Option 3: Individual Components

```bash
# Data pipeline only
python src/examples/example_pipeline.py --sample --sample-size 100

# Baseline comparison
python src/examples/example_usage.py

# Environment demos
python src/examples/diabetes_env_example.py
python src/examples/adherence_env_example.py

# Evaluation
python src/examples/run_evaluation.py
```

---

## Project Structure

```
rl-healthcare-treatment/
├── src/
│   ├── data/              # Data processing pipeline (MIMIC-III loader, preprocessing, etc.)
│   ├── environments/      # Healthcare simulation environments (diabetes, adherence)
│   ├── models/            # RL algorithms (CQL, baselines, encoders, safety)
│   ├── rewards/           # Reward function implementations
│   ├── evaluation/        # Evaluation framework (OPE, safety, clinical metrics)
│   ├── configs/           # Configuration files
|   ├── run_integrated_solution.py    # Main integrated runner
|   └── quick_start.sh                # Interactive setup script
│
├── examples/              # Runnable examples
│   ├── example_pipeline.py       # Complete data pipeline
│   ├── example_usage.py          # Baseline usage
│   └── run_evaluation.py         # Evaluation examples
│
├── tests/                 # Unit tests
│
├── INTEGRATION_GUIDE.md          # Detailed integration guide
└── README.md                     # This file
```

---

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Processing      │    │   RL Engine     │
│                 │    │  Pipeline        │    │                 │
│ • MIMIC-III/IV  │───▶│ • Preprocessing  │───▶│ • State Encoder │
│ • eICU          │    │ • Feature Eng.   │    │ • CQL Policy    │
│ • Synthetic     │    │ • Validation     │    │ • Reward System │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Evaluation     │    │  Safety &        │    │  Results        │
│  Framework      │◀───│  Interpretability│◀───│  Generation     │
│                 │    │                  │    │                 │
│ • OPE Metrics   │    │ • Constraints    │    │ • LaTeX Tables  │
│ • Safety Index  │    │ • Explainability │    │ • Visualizations│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## Methodology

### 1. Patient Representation

Patients are represented as sequences of clinical observations:

- **Demographics:** Age (bucketed), gender, BMI
- **Vitals:** Blood pressure, heart rate, temperature
- **Labs:** Glucose, HbA1c, creatinine, etc.
- **Medications:** Current prescriptions (one-hot encoded)
- **Comorbidities:** ICD-9 diagnosis codes

State encoding uses:
- **Transformer-based encoders** for temporal sequences
- **Autoencoders** for dimensionality reduction
- **Clinical feature engineering** (trends, gaps, thresholds)

### 2. MDP Formulation

```python
MDP = (S, A, P, R, γ)

# State: Patient clinical state at time t
S = [demographics, vitals, labs, medications, history]

# Actions: Treatment decisions
A = {
    'medication_dosage': continuous [0, 1],
    'appointment_days': discrete {7, 14, 30, 60, 90},
    'reminder_frequency': discrete {0, 1, 2, 3}
}

# Reward: Multi-objective
R(s,a) = w₁·Adherence + w₂·Health - w₃·Adverse - w₄·Cost
```

### 3. Conservative Q-Learning (CQL)

CQL prevents overestimation in offline RL through conservative value estimates:

```
Q-Learning: max E[∑ γᵗ R(s,a)]
CQL: max E[∑ γᵗ R(s,a)] - α·Conservative_Penalty

where Conservative_Penalty penalizes Q-values on out-of-distribution actions
```

### 4. Safety Mechanisms

**Multi-layer safety validation:**

1. **Hard Constraints:** Absolute limits (dosage, contraindications)
2. **Soft Constraints:** Clinical guidelines with warnings
3. **Confidence Thresholds:** Low-confidence actions trigger review
4. **Anomaly Detection:** Flag unusual state-action pairs
5. **Safety Override:** Default to safe baseline action

---

## Evaluation Framework

### Off-Policy Evaluation (OPE)

Since we cannot deploy policies on real patients, we use OPE methods:

**1. Weighted Importance Sampling (WIS)**
```
V̂_WIS = (∑ wᵢrᵢ) / (∑ wᵢ)
where wᵢ = ∏ π(aₜ|sₜ) / b(aₜ|sₜ)
```

**2. Doubly Robust (DR)**
```
V̂_DR = (1/n) ∑[wᵢ(rᵢ - Q(sᵢ,aᵢ)) + ∑ π(a|sᵢ)Q(sᵢ,a)]
```

**3. Direct Method (DM)**
```
V̂_DM = E_π[Q(s,a)]
```

### Safety Metrics

**Safety Index:**
```
Safety_Index = 1 - (violations / total_steps)

where violations = steps with state outside safe physiological ranges
```

**Clinical Validation:**
- Guideline compliance rate
- Contraindication detection
- Dosage appropriateness
- Drug interaction warnings

---

## Experiments

### Baseline Comparisons

The system evaluates 7 baseline policies:

1. **Rule-Based:** Clinical guidelines (e.g., diabetes management protocols)
2. **Random-Uniform:** Uniform random actions
3. **Random-Safe:** Random actions with safety constraints
4. **Mean-Action:** Dataset mean action
5. **Ridge-Regression:** Linear policy
6. **K-Nearest Neighbors:** Non-parametric policy  
7. **Behavior-Cloning:** Neural network supervised learning

### Expected Results

From preliminary experiments:

| Policy | Mean Reward | Safety Index | Guideline Compliance |
|--------|-------------|--------------|---------------------|
| CQL | **-0.15** | **0.99** | **0.92** |
| Rule-Based | -0.22 | 0.98 | 0.95 |
| Behavior Cloning | -0.18 | 0.97 | 0.88 |
| Ridge Regression | -0.25 | 0.96 | 0.85 |
| Random-Safe | -0.35 | 0.94 | 0.75 |

*Note: Negative rewards indicate distance from optimal glucose/BP targets*

---

## Development Setup

### System Requirements

- **Python:** 3.8+
- **RAM:** 16GB recommended (8GB minimum)
- **GPU:** Optional (PyTorch MPS/CUDA)
- **Storage:** 5GB for code, 10-50GB for MIMIC data

### Installation

```bash
# Clone repository
git clone https://github.com/andysundar/rl-healthcare-treatment.git
cd rl-healthcare-treatment

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Verify installation
pytest tests/ -v
```

### MIMIC-III Access

1. Complete CITI training: https://physionet.org/content/mimiciii/
2. Request access (requires ethics certification)
3. Download and extract to `data/raw/mimic-iii/`

Expected files:
- `PATIENTS.csv`
- `ADMISSIONS.csv`
- `DIAGNOSES_ICD.csv`
- `LABEVENTS.csv`
- `PRESCRIPTIONS.csv`

---

## Usage Examples

### Example 1: Quick Baseline Comparison

```python
from src.models.baselines import (
    create_diabetes_rule_policy,
    create_behavior_cloning_policy,
    compare_all_baselines
)

# Create policies
rule_policy = create_diabetes_rule_policy(state_dim=10, action_dim=1)
bc_policy = create_behavior_cloning_policy(state_dim=10, action_dim=1)

# Train BC on historical data
bc_policy.train(train_states, train_actions, epochs=50)

# Compare
baselines = {'Rule-Based': rule_policy, 'BC': bc_policy}
results = compare_all_baselines(test_data, baselines)
print(results)
```

### Example 2: Train CQL Agent

```python
from src.models.rl import ConservativeQLearning, CQLConfig
from src.environments import DiabetesEnv

# Configure
config = CQLConfig(state_dim=10, action_dim=1, cql_alpha=5.0)
agent = ConservativeQLearning(config)

# Train offline
agent.train_offline(
    dataset=offline_data,
    n_epochs=100,
    batch_size=256
)

# Evaluate
results = evaluator.evaluate_policy(agent, test_data)
```

### Example 3: Simulate Patient Trajectory

```python
from src.environments import DiabetesEnv, PatientSimulator

# Create environment
env = DiabetesEnv(patient_params={'insulin_sensitivity': 1.0})

# Simulate
simulator = PatientSimulator(env)
trajectory = simulator.simulate_trajectory(
    policy=trained_policy,
    initial_state=patient_state,
    n_steps=90  # 90 days
)

# Analyze
print(f"Final glucose: {trajectory[-1]['glucose']:.1f} mg/dL")
print(f"HbA1c improvement: {trajectory[-1]['hba1c_change']:.2f}%")
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_baselines.py -v
pytest tests/test_env.py -v
pytest tests/test_rewards.py -v

# Run with coverage
pytest --cov=src tests/
```

---

## Documentation

- **Integration Guide:** `INTEGRATION_GUIDE.md` - Detailed setup and usage
- **Data Module:** `src/data/README.md` - Data processing documentation
- **Environments:** `src/environments/CODE_OVERVIEW.md` - Environment details
- **Rewards:** `src/rewards/REWARD_IMPLEMENTATION_SUMMARY.md` - Reward functions
- **Evaluation:** `src/evaluation/INTEGRATION_GUIDE.md` - Evaluation framework

---

## Results & Outputs

Running the integrated solution generates:

```
outputs/
├── results_summary.json              # Overall results summary
├── baseline_comparison_report.md     # Markdown comparison table
├── baseline_comparison_report.json   # Raw comparison data
├── baseline_comparison.png           # Visualization plots
├── results_table.tex                 # LaTeX table for thesis
├── quality_report.json               # Data quality metrics
└── cohort_definition.txt             # Patient cohort details
```

---

## Contributing

This is a thesis project, but feedback is welcome:

1. Open an issue for bugs or questions
2. Submit pull requests for improvements
3. Contact: m23csa508@iitj.ac.in

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{bandopadhyay2026rl,
  title={Reinforcement Learning for Healthcare Treatment Recommendations on Tabular Patient Data},
  author={Bandopadhyay, Anindya},
  year={2026},
  school={Indian Institute of Technology Jodhpur},
  type={M.Tech Thesis},
  supervisor={Sasmal, Pradip}
}
```

---

## License

This project is licensed under the Apache 2.0 License - see the `LICENSE` file for details.

---

## Acknowledgments

- **Supervisor:** Dr. Pradip Sasmal, IIT Jodhpur
- **Datasets:** MIMIC-III Critical Care Database (MIT-LCP)
- **References:** 
  - Komorowski et al. (2018) - AI Clinician
  - Kumar et al. (2020) - Conservative Q-Learning
  - Gottesman et al. (2019) - Guidelines for RL in Healthcare

---

## Contact

**Anindya Bandopadhyay**  
M.Tech CSE, IIT Jodhpur  
Email: m23csa508@iitj.ac.in / anindyabandopadhyay@gmail.com
GitHub: [@andysundar](https://github.com/andysundar)

**Project Supervisor**  
Dr. Pradip Sasmal  
Department of Mathematics, IIT Jodhpur

---

## Project Timeline

- **Start:** August 2024
- **Proposal Defense:** September 2024
- **Data Collection:** October-November 2024
- **Implementation:** December 2024 - January 2026
- **Experiments:** February 2026
- **Thesis Writing:** January-March 2026
- **Defense:** March 2026
- **Deadline:** **March 25, 2026** 

