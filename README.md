# Reinforcement Learning for Healthcare Treatment Recommendations

**M.Tech Thesis Project**
**Author:** Anindya Bandopadhyay (M23CSA508)
**Supervisor:** Dr. Pradip Sasmal
**Institution:** IIT Jodhpur
**Deadline:** March 25, 2026

---

## Overview

This project develops a **safety-constrained offline reinforcement learning framework** for personalised healthcare treatment recommendations. The system trains a Conservative Q-Learning (CQL) policy on retrospective patient trajectories (MIMIC-III or synthetic) to optimise medication dosing and adherence interventions while maintaining strict physiological safety guarantees.

### Core capabilities

| Capability | Component |
|---|---|
| Synthetic patient data generation | `src/data/synthetic_generator.py` |
| Glucose-insulin simulation (Bergman model) | `src/environments/diabetes_env.py` |
| Medication adherence simulation | `src/environments/adherence_env.py` |
| Composite multi-objective rewards | `src/rewards/composite_reward.py` |
| 7 baseline policies | `src/models/baselines/` |
| Autoencoder / VAE state encoding | `src/models/encoders/` |
| Offline CQL training (replay buffer) | `src/models/rl/` |
| Safety layer & constraint optimisation | `src/models/safety/` |
| Off-policy evaluation (WIS / DR / DM) | `src/evaluation/off_policy_eval.py` |
| Safety & clinical metrics | `src/evaluation/safety_metrics.py` |
| Counterfactual & decision-rule interpretability | `src/evaluation/interpretability.py` |
| Personalisation score (PDF §7.2) | `src/evaluation/interpretability.py` |
| Domain-adaptation policy transfer | `src/models/policy_transfer/transfer.py` |
| Extended state: vital signs from CHARTEVENTS | `src/data/mimic_loader.py` (`--use-vitals`) |
| Extended state: medication history | `src/data/mimic_loader.py` (`--use-med-history`) |
| Thesis-quality visualisation artifacts | `src/run_integrated_solution.py` |

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Use the interactive script (recommended)

```bash
chmod +x quick_start.sh && ./quick_start.sh
```

Presents a numbered menu of 7 run scenarios (synthetic → full MIMIC-III).
To run a specific scenario non-interactively:

```bash
./quick_start.sh --mode 1   # quick demo (<2 min)
./quick_start.sh --mode 4   # extended state (vitals + med-history)
./quick_start.sh --mode 5   # full pipeline (~60 min)
./quick_start.sh --help     # list all scenarios
```

### 3. Smoke test — verify installation (< 30 s)

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --n-synthetic-patients 50 \
    --trajectory-length 10 \
    --output-dir outputs/smoke
```

### Fast local iteration on laptop (Apple Silicon friendly)

Use fast-eval mode for quick debugging runs; this is not for final thesis metrics.

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --mimic-dir /path/to/mimic \
    --fast-eval \
    --skip-slow-baselines \
    --max-eval-samples 1000 \
    --light-report \
    --output-dir outputs/mimic_fast_debug
```

Full-fidelity run (final experiments):

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --mimic-dir /path/to/mimic \
    --train-cql \
    --output-dir outputs/mimic_full
```

### Resume and save-point workflows

Resume from existing stage checkpoints:

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --mimic-dir /path/to/mimic \
    --resume \
    --output-dir outputs/mimic_full
```

Start from a specific stage and stop after another:

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --resume \
    --start-from stage_3_baseline_training \
    --stop-after stage_5_evaluation \
    --output-dir outputs/mimic_full
```

Invalidate downstream checkpoints and recompute:

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --resume \
    --invalidate-from stage_2b_encoder_training \
    --force-stage stage_3_baseline_training \
    --output-dir outputs/mimic_full
```

### 3. Run the PDF alignment test suite

```bash
# Fast subset (skips slow CQL / full-pipeline tests)
python tests/test_alignment.py --quick

# Full suite — 23 tests, includes CQL training and subprocess test
python tests/test_alignment.py
```

Expected: `Ran 23 tests … OK`

---

## Run Scenarios

All scenarios use `src/run_integrated_solution.py`. New flags are **opt-in** — the default run (no extra flags) is unchanged.

### Scenario A — Baseline comparison only (fast, ~2 min)

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --n-synthetic-patients 200 \
    --trajectory-length 20 \
    --output-dir outputs/baseline_only
```

Evaluates 7 baselines (Rule-Based, Random-Uniform, Random-Safe, Mean-Action, Ridge-Regression, KNN-5, Behavior-Cloning). Generates `baseline_comparison.png` and `policy_dashboard.png`.

---

### Scenario B — Real CQL training (no encoder)

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --n-synthetic-patients 500 \
    --trajectory-length 30 \
    --train-cql \
    --cql-iterations 5000 \
    --cql-batch-size 256 \
    --output-dir outputs/cql_only
```

Trains a CQL agent via `ReplayBuffer` + `OfflineRLTrainer`. Generates `cql_training_curves.png`.

---

### Scenario C — Encoder pre-training + CQL (PDF §3.1 / §3.2)

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --n-synthetic-patients 500 \
    --trajectory-length 30 \
    --use-encoder \
    --encoder-state-dim 64 \
    --encoder-epochs 50 \
    --encoder-type autoencoder \
    --train-cql \
    --cql-iterations 10000 \
    --output-dir outputs/enc_cql
```

Pre-trains a `PatientAutoencoder`, re-encodes all transitions to 64-dim embeddings, then trains CQL in embedding space.
Add `--encoder-type vae` to use the variational autoencoder instead.
Add `--encoder-checkpoint path/to/encoder.pt` to skip training and load a saved encoder.

---

### Scenario D — Interpretability (PDF §6 step 8)

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --use-encoder \
    --train-cql \
    --use-interpretability \
    --n-counterfactuals 5 \
    --tree-max-depth 4 \
    --explain-n-samples 100 \
    --output-dir outputs/interpret
```

Runs after CQL training:
- `DecisionRuleExtractor` — fits a surrogate decision tree; saves `decision_rules.txt`
- `CounterfactualExplainer` — gradient-based perturbation; saves `counterfactuals.json`
- `PersonalizationScorer` — computes `(1/T) Σ cos(fθ(sₜ), fϕ(xₜ))`; saves `personalization_score.png`
- `feature_importance.png`

---

### Scenario E — Policy transfer / domain adaptation (PDF §3.3 / §6 step 6)

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --use-encoder \
    --train-cql \
    --use-transfer \
    --transfer-steps 1000 \
    --output-dir outputs/transfer
```

Trains a `PolicyTransferAdapter` (MLP) that maps target-population embeddings to source-policy space using BC loss + cosine alignment loss. Saves `transfer_adapter.pt`.

---

### Scenario F — Full pipeline (all modules)

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --n-synthetic-patients 1000 \
    --trajectory-length 30 \
    --use-encoder \
    --encoder-state-dim 64 \
    --encoder-epochs 50 \
    --train-cql \
    --cql-iterations 10000 \
    --cql-batch-size 256 \
    --use-interpretability \
    --n-counterfactuals 5 \
    --explain-n-samples 100 \
    --use-transfer \
    --transfer-steps 1000 \
    --output-dir outputs/full_run
```

---

### Scenario G — Extended state: vitals + medication history

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --n-synthetic-patients 500 \
    --trajectory-length 30 \
    --use-vitals \
    --use-med-history \
    --use-encoder \
    --encoder-state-dim 64 \
    --train-cql \
    --cql-iterations 10000 \
    --output-dir outputs/extended_state
```

Expands the state vector from **10 → 16 dimensions**:

| Flag | Added features | Dims |
|---|---|---|
| `--use-vitals` | `heart_rate`, `sbp`, `respiratory_rate`, `spo2` | +4 |
| `--use-med-history` | `adherence_rate_7d`, `medication_count` | +2 |
| Both | — | **16-dim total** |

Vital features are physiologically correlated (HR rises during hypoglycaemia, SpO2 drops with hyperglycaemia, etc.).
All downstream components (encoder, CQL, baselines) adapt automatically.

---

### Scenario H — MIMIC-III data (full pipeline)

```bash
python src/run_integrated_solution.py \
    --mode train-eval \
    --mimic-dir data/raw/mimic-iii \
    --use-vitals \
    --use-med-history \
    --use-encoder \
    --encoder-state-dim 64 \
    --train-cql \
    --cql-iterations 20000 \
    --use-interpretability \
    --explain-n-samples 100 \
    --use-transfer \
    --transfer-steps 1000 \
    --output-dir outputs/mimic_full
```

Requires MIMIC-III credentialed access (see [PhysioNet](https://physionet.org/content/mimiciii/)).

**Required CSV files** (place in `data/raw/mimic-iii/`):

| File | Used for |
|---|---|
| `PATIENTS.csv` | Demographics, cohort selection |
| `ADMISSIONS.csv` | Admission records |
| `DIAGNOSES_ICD.csv` | ICD-9 diabetes cohort filter |
| `LABEVENTS.csv` | Glucose and lab measurements |
| `PRESCRIPTIONS.csv` | Insulin orders, medication history (`--use-med-history`) |
| `CHARTEVENTS.csv` *(optional)* | Vital signs — required for `--use-vitals` (~35 GB) |

> For a quick test without the full dataset, omit `--use-vitals` (skips CHARTEVENTS) or use a 100-patient sample:
>
> ```bash
> python src/run_integrated_solution.py --mode train-eval \
>     --mimic-dir data/raw/mimic-iii --use-sample --sample-size 100 \
>     --train-cql --cql-iterations 5000 --output-dir outputs/mimic_sample
> ```
>
> See `docs/MIMIC_DOWNLOAD_GUIDE.md` for download instructions.

---

## All CLI Flags

```
--mode               train-eval | eval-only (default: train-eval)
--data-source        synthetic | mimic (default: synthetic)
--mimic-dir          path to MIMIC-III CSVs
--n-synthetic-patients  (default: 100)
--trajectory-length  (default: 20)
--output-dir         (default: outputs/run_<timestamp>)

# Extended state dimensions
--use-vitals         add heart_rate, sbp, respiratory_rate, spo2 to state (+4 dims)
--use-med-history    add adherence_rate_7d, medication_count to state (+2 dims)
                     (MIMIC: vitals from CHARTEVENTS; synthetic: physiologically correlated)

# Encoder (PDF §3.1)
--use-encoder        pre-train autoencoder and use embeddings as RL state
--encoder-state-dim  latent dimension (default: 64)
--encoder-epochs     training epochs (default: 50)
--encoder-type       autoencoder | vae (default: autoencoder)
--encoder-checkpoint skip training, load encoder from this path

# CQL training (PDF §3.2)
--train-cql          run real offline CQL training
--cql-iterations     gradient steps (default: 10000)
--cql-batch-size     (default: 256)

# Policy transfer (PDF §3.3)
--use-transfer       run domain-adaptation policy transfer
--transfer-steps     adaptation gradient steps (default: 1000)
--target-data-dir    optional separate target-population data dir

# Interpretability (PDF §6 step 8)
--use-interpretability  run decision rules + counterfactuals + personalisation score
--n-counterfactuals  counterfactuals per sample (default: 5)
--tree-max-depth     surrogate decision tree depth (default: 4)
--explain-n-samples  test states to explain (default: 50)
```

---

## Generated Output artifacts

Every run produces the following under `--output-dir`:

| File | Description |
|---|---|
| `results_summary.json` | All metrics and module results |
| `baseline_comparison_report.json` | Per-baseline metric table |
| `results_table.tex` | LaTeX table for thesis |
| `baseline_comparison.png` | 2-panel bar chart (reward + safety rate) |
| `policy_dashboard.png` | 4-panel dashboard (reward, safety, action mean/std) |
| `cql_training_curves.png` | CQL loss and eval-return curves (if `--train-cql`) |
| `feature_importance.png` | Decision tree feature importances (if `--use-interpretability`) |
| `personalization_score.png` | PDF §7.2 personalisation score card (if `--use-encoder --use-interpretability`) |
| `safety_clinical_heatmap.png` | Safety index + guideline compliance heatmap |
| `thesis_figures.pdf` | All PNGs combined into one multi-page PDF |
| `decision_rules.txt` | Human-readable if-then rules (if `--use-interpretability`) |
| `counterfactuals.json` | Counterfactual explanations (if `--use-interpretability`) |
| `transfer_adapter.pt` | Saved transfer adapter weights (if `--use-transfer`) |

---

## Project Structure

```
rl-healthcare-treatment/
├── src/
│   ├── run_integrated_solution.py   # Main pipeline entry point
│   ├── configs/                     # Config dataclasses
│   ├── data/                        # MIMIC-III loader, synthetic generator, preprocessor
│   ├── environments/                # DiabetesEnv (Bergman), AdherenceEnv, patient simulator
│   ├── models/
│   │   ├── baselines/               # Rule-based, random, ridge, KNN, behavior cloning
│   │   ├── encoders/                # PatientAutoencoder, VAE, StateEncoderWrapper
│   │   ├── policy_transfer/         # PolicyTransferAdapter + PolicyTransferTrainer
│   │   ├── rl/                      # CQLAgent, BCQAgent, ReplayBuffer, OfflineRLTrainer
│   │   └── safety/                  # SafetyLayer, ConstraintOptimizer, SafetyCritic
│   ├── rewards/                     # CompositeReward, HealthReward, SafetyReward, etc.
│   └── evaluation/
│       ├── interpretability.py      # CounterfactualExplainer, DecisionRuleExtractor, PersonalizationScorer
│       ├── off_policy_eval.py       # WIS, DR, Direct Method OPE
│       ├── performance_metrics.py   # PerformanceEvaluator + PerformanceResult
│       ├── safety_metrics.py        # SafetyEvaluator
│       └── clinical_metrics.py      # ClinicalEvaluator
├── tests/
│   └── test_pdf_alignment.py        # 23-test suite covering all PDF requirements (T01–T23)
├── outputs/                         # Generated artifacts (git-ignored)
├── requirements.txt
└── README.md
```

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Data Layer     │    │  State Encoder   │    │    RL Engine        │
│                  │    │  (PDF §3.1)      │    │    (PDF §3.2)       │
│ • MIMIC-III      │───▶│ • Autoencoder    │───▶│ • CQLAgent          │
│ • Synthetic      │    │ • VAE            │    │ • ReplayBuffer      │
│   generator      │    │ • 10D → 64D      │    │ • OfflineRLTrainer  │
└──────────────────┘    └──────────────────┘    └─────────────────────┘
                                                          │
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Interpretability│    │  Policy Transfer │    │    Evaluation       │
│  (PDF §6 step 8) │    │  (PDF §3.3)      │◀───│                     │
│                  │    │                  │    │ • OPE (WIS/DR/DM)   │
│ • Decision rules │    │ • MLP adapter    │    │ • Safety index      │
│ • Counterfactuals│    │ • BC + cosine    │    │ • Clinical metrics  │
│ • Person. score  │    │   alignment loss │    │ • Baselines (7)     │
└──────────────────┘    └──────────────────┘    └─────────────────────┘
```

---

## MDP Formulation

```
State S (base, 10-dim):
  [glucose_mean, glucose_std, glucose_min, glucose_max,
   insulin_mean, medication_taken, reminder_sent,
   hypoglycemia, hyperglycemia, day]

  + --use-vitals    → heart_rate, sbp, respiratory_rate, spo2      (14-dim)
  + --use-med-history → adherence_rate_7d, medication_count        (12-dim)
  + both            →                                               (16-dim)
  + --use-encoder   → encoded latent                                (64-dim default)

Action A: continuous dosage ∈ [0, 1]

Reward R(s,a) = w₁·HealthReward + w₂·AdherenceReward
              − w₃·SafetyPenalty − w₄·CostPenalty

Discount γ = 0.99
```

**CQL objective:**
```
min_Q  α · E_{s~D}[log Σ_a exp Q(s,a) − Q(s,a_D)]
      + ½ · E_{(s,a,r,s')~D}[(Q(s,a) − r − γ·Q(s',a'))²]
```

---

## Personalisation Score (PDF §7.2)

```
PS = (1/T) Σ_{t=1}^{T} cos( fθ(sₜ), fϕ(xₜ) )

fθ = source patient encoder
fϕ = target patient encoder
sₜ, xₜ = source and target states at time t
```

A score near 1.0 indicates high alignment between source and target representations.

---

## Code Examples

### Train and query a CQL agent

```python
import sys; sys.path.insert(0, 'src')
from models.rl import CQLAgent, ReplayBuffer, OfflineRLTrainer
import numpy as np

state_dim, action_dim = 10, 1
agent   = CQLAgent(state_dim=state_dim, action_dim=action_dim, device='cpu')
buffer  = ReplayBuffer(capacity=10000, state_dim=state_dim, action_dim=action_dim)

# Load offline dataset (states, actions, rewards, next_states, dones are np.ndarrays)
buffer.load_from_dataset(states, actions, rewards, next_states, dones)

trainer = OfflineRLTrainer(agent, buffer, save_dir='outputs/cql')
history = trainer.train(num_iterations=5000, batch_size=256)

action = agent.select_action(state, deterministic=True)
```

### Encode patient states with autoencoder

```python
from models.encoders import PatientAutoencoder, EncoderConfig
from models.encoders.state_encoder_wrapper import StateEncoderWrapper

cfg     = EncoderConfig(lab_dim=10, vital_dim=0, demo_dim=0, state_dim=64)
encoder = PatientAutoencoder(cfg)
wrapper = StateEncoderWrapper(encoder, device='cpu', raw_state_dim=10)

# Train on transitions (list of (s, a, r, s', done) tuples)
wrapper.train_on_transitions(train_transitions, val_transitions,
                             epochs=50, save_dir='outputs/encoder')

encoded = wrapper.encode_state(raw_state)   # np.ndarray shape (64,)
```

### Explain a policy decision

```python
from evaluation.interpretability import (
    InterpretabilityConfig, DecisionRuleExtractor, CounterfactualExplainer
)

cfg       = InterpretabilityConfig(n_counterfactuals=3, tree_max_depth=4)
extractor = DecisionRuleExtractor(cfg)
extractor.fit(test_states, agent=agent)
print(extractor.extract_rules())

explainer = CounterfactualExplainer(agent, encoder_wrapper=wrapper, config=cfg)
cfs       = explainer.explain(raw_state)
for cf in cfs:
    print(cf['feature_changes'])
```

### Run 7-baseline comparison

```python
from models.baselines import (
    create_diabetes_rule_policy, create_behavior_cloning_policy,
    compare_all_baselines
)

rule_policy = create_diabetes_rule_policy(state_dim=10, action_dim=1)
bc_policy   = create_behavior_cloning_policy(state_dim=10, action_dim=1)
bc_policy.train(train_states, train_actions, epochs=50)

results = compare_all_baselines(test_data, {'Rule-Based': rule_policy, 'BC': bc_policy})
```

---

## Testing

```bash
# Full 23-test PDF alignment suite
python tests/test_pdf_alignment.py

# Quick subset (no CQL training or subprocess tests)
python tests/test_pdf_alignment.py --quick

# Run with pytest
pytest tests/test_pdf_alignment.py -v
```

The test suite covers (T01–T23):

| Range | Area |
|---|---|
| T01–T02 | Data sources (synthetic generator, MIMIC loader) |
| T03–T05 | Environments (DiabetesEnv Bergman model, AdherenceEnv) |
| T06 | Policy transfer adapter fit + inference |
| T07–T09 | Reward functions (composite, safety penalty, shaping) |
| T10 | All 7 baselines + compare_all_baselines |
| T11–T12 | CQL agent API + training loop |
| T13–T14 | Encoder shape + StateEncoderWrapper transitions |
| T15–T17 | Interpretability (rules, counterfactuals, personalisation score) |
| T18–T19 | Off-policy evaluation (WIS, multiple estimators) |
| T20 | PerformanceResult.personalization_score field |
| T21–T22 | SafetyEvaluator, ClinicalEvaluator |
| T23 | Full pipeline subprocess test |

---

## System Requirements

- Python 3.10+
- RAM: 8 GB minimum, 16 GB recommended
- GPU: optional — PyTorch MPS (Apple Silicon) or CUDA automatically detected
- Storage: ~200 MB for code; 10–50 GB for MIMIC-III

---

## MIMIC-III Access

1. Complete CITI training and request credentialed access at <https://physionet.org/content/mimiciii/>
2. Download and place CSV files in `data/raw/mimic-iii/`
3. See [docs/MIMIC_DOWNLOAD_GUIDE.md](docs/MIMIC_DOWNLOAD_GUIDE.md) for step-by-step download instructions

**Required CSV files** for a base MIMIC run:
`PATIENTS.csv`, `ADMISSIONS.csv`, `DIAGNOSES_ICD.csv`, `LABEVENTS.csv`, `PRESCRIPTIONS.csv`

**Additionally needed** for `--use-vitals` (vital signs — HR, SBP, RR, SpO2):
`CHARTEVENTS.csv` (~35 GB uncompressed)

**Quick testing without MIMIC-III** — the pipeline generates realistic synthetic patient data automatically:

```bash
./quick_start.sh --mode 1   # no MIMIC-III needed
```

---

## Citation

```bibtex
@mastersthesis{bandopadhyay2026rl,
  title     = {Reinforcement Learning for Healthcare Treatment Recommendations on Tabular Patient Data},
  author    = {Bandopadhyay, Anindya},
  year      = {2026},
  school    = {Indian Institute of Technology Jodhpur},
  type      = {M.Tech Thesis},
  supervisor= {Sasmal, Pradip}
}
```

---

## Contact

**Anindya Bandopadhyay** — M.Tech CSE, IIT Jodhpur
Email: m23csa508@iitj.ac.in / anindyabandopadhyay@gmail.com

**Supervisor:** Dr. Pradip Sasmal, Department of Mathematics, IIT Jodhpur
