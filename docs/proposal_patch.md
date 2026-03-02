# Proposal Patch (Reviewer-Proof)

## A1) Revised Abstract
This thesis studies **one primary decision problem**: offline reinforcement learning (RL) for treatment recommendation from tabular EHR trajectories. We define a daily-step Markov Decision Process (MDP) where each decision recommends a discrete insulin-adjustment bucket, learned purely from retrospective trajectories. The central hypothesis is that conservative offline RL can outperform imitation-only baselines in expected clinical utility while reducing unsafe actions under explicit safety constraints. To make “safety-first” operational, we formulate the problem as a constrained MDP (CMDP) with computable per-step safety costs and a pre-specified threshold. We evaluate policies using robust off-policy evaluation (OPE), including weighted importance sampling (WIS) and doubly robust (DR) estimators with effective sample size diagnostics, clipping, and bootstrap confidence intervals. We also include an interpretable policy distillation step using shallow decision trees to provide clinician-auditable rules and fidelity metrics. This work is explicitly positioned as **retrospective methodological research** and not a deployment study; conclusions are restricted to offline evaluation under available datasets and assumptions.

## A2) Revised Thesis Statement
A safe offline policy for daily treatment recommendations can be learned from tabular EHR trajectories using constrained offline RL with explicit safety costs and thresholds. The policy is considered successful only if it improves OPE-estimated return versus behavior cloning/rule-based/statistical baselines while satisfying predefined safety constraints. Claims are falsifiable through DR/WIS with uncertainty diagnostics and safety violation metrics.

## A3) Contributions (max 3)
1. **Deterministic offline RL trajectory pipeline** from EHR-like tables with leakage-safe splits and missingness-aware states; measured by deterministic reproducibility checks and split leakage tests; compared against prior ad-hoc split/trajectory construction.
2. **Safety-constrained offline RL benchmark (CQL + IQL baseline + safety layer)**; measured by OPE return (DR/WIS) and unsafe action rate/constraint satisfaction; compared against behavior cloning, rule-based, and statistical baselines.
3. **OPE reliability + interpretability package** (ESS, clipping, bootstrap CIs, support-mismatch warnings, policy distillation fidelity); measured by uncertainty diagnostics and fidelity/agreement rate; compared against OPE without diagnostics and black-box-only reporting.

## A4) MDP Specification
| Element | Specification |
|---|---|
| Timestep granularity | 1 day |
| Horizon | 14 days (MVP demo), extendable to 30 days in experiments |
| Terminal conditions | discharge/death/end-of-window/no-next-observation |
| State variables | demographics (if available), glucose summary stats, insulin summary, recent medication adherence, adverse-event flags, day index, missingness indicators per variable |
| Action space | discrete insulin adjustment bucket: {0=no/low adjustment, 1=moderate adjustment} for MVP; extensible to N buckets |
| Reward | computable clinical utility from observed data: in-range bonus, hypo/hyper penalties, adherence bonus |
| Safety cost c(s,a) | binary/weighted unsafe action indicator (e.g., high-dose bucket under low glucose, no-dose bucket under very high glucose) |
| Threshold τ | max allowable expected cost per episode (e.g., E[Σ c_t] ≤ τ, τ tuned by baseline clinician-policy cost percentiles) |

**Markov approximation justification:** Daily aggregation compresses short-term history into window features (e.g., rolling glucose/adherence summaries and adverse-event flags), and missingness indicators preserve observation process information. This provides a tractable approximate Markov state while keeping the model compatible with tabular hospital data.

## A5) Reward Design (2 alternatives)
### Option A: Sparse event-based
- **Formula:** \(r_t = +1\,\mathbb{1}[80\le G_t\le 180] -2\,\mathbb{1}[G_t<70] -1\,\mathbb{1}[G_t>180] +0.5\,\mathbb{1}[\text{adherent}_t]\)
- **Required fields:** glucose_mean/min/max, adherence/medication indicator.
- **Scaling:** bounded and interpretable in approximately [-3, 1.5], optional z-score for training stability.
- **Risk/Mitigation:** may over-optimize frequent easy states; mitigate with subgroup reporting and safety constraint monitoring.

### Option B: Shaped trend-based
- **Formula:** \(r_t = \alpha\cdot\Delta\text{dist}(G_t,\text{target band}) - \beta\cdot\text{volatility}_t - \gamma\cdot\mathbb{1}[\text{adverse event}]\)
- **Required fields:** serial glucose values, rolling variance, adverse-event labels.
- **Scaling:** normalize components to comparable ranges before weighted sum.
- **Risk/Mitigation:** reward hacking via short-term smoothing; mitigate with ablation against sparse reward and event-specific safety metrics.

## A6) Safety-Constrained Offline RL (CMDP)
- **Objective:** maximize expected discounted return \(J(\pi)\).
- **Constraint:** \(C(\pi)=\mathbb{E}_{\pi}[\sum_t \gamma^t c_t] \le \tau\).
- **Operational cost \(c(s,a)\):** indicator that selected action violates glucose-conditioned conservative rule set.
- **Primary enforcement method:** **action masking/clipping safety layer** before final action emission in training/evaluation loops.
- **If masked:** choose nearest feasible bucket from allowed set; log violations and corrections.
- **Offline safety metrics:** violation_rate, unsafe_action_rate, constraint_satisfaction_rate, and cost-return tradeoff curve.

## A7) Transfer Learning Evaluation (testable)
- **Source domain:** MIMIC-III cohort.
- **Target domain:** MIMIC-IV or eICU cohort (available benchmark shifts).
- **Expected shifts:** covariate shift (labs/demographics), action shift (treatment style).
- **Method:** representation adaptation (e.g., DANN) or importance weighting.
- **Protocol:** train on source, adapt/fine-tune with target offline data, evaluate on target with OPE + safety metrics; report shift diagnostics (feature/action distribution divergence and importance-weight diagnostics).
- **Indian-population statement:** retained as motivation only until an accessible target dataset is formally integrated.

## A8) Evaluation Protocol
| Category | Items |
|---|---|
| Baselines | Behavior Cloning (BC), Rule-based policy, Statistical baselines (mean/regression/KNN), Offline RL baselines: CQL + IQL |
| Ablations | without safety layer, reward A vs B, without adaptation, with/without encoder representation |
| OPE | DR + WIS, estimated behavior policy \(b(a\mid s)\), ratio clipping, ESS, bootstrap CIs, support mismatch/high variance warnings |

## A9) Ethics + Limitations
- Observational confounding can bias inferred policy value.
- Selection bias and missingness bias may reduce external validity.
- Dataset shift can break policy assumptions across institutions/time.
- OPE is fragile under poor support overlap; uncertainty diagnostics are mandatory.
- Distilled rules are approximations and may not capture all black-box behavior.
- **Non-deployment disclaimer:** this is a research prototype for retrospective analysis only; not intended for clinical deployment.
