# GitHub Copilot Instructions for rl-healthcare-treatment 🧭

**Purpose:** Give AI coding agents immediate, actionable knowledge to contribute safely and productively in this repo.

## Quick orientation (what this repo is)
- This repository implements an offline healthcare RL stack with three main domains: **environments** (physiological/patient sims), **data** (MIMIC loader, preprocess & cohort), and **evaluation** (OPE, safety, clinical & performance metrics). See: `src/environments/`, `src/data/`, `src/evaluation/`.
- Models/agents live under `src/models/` (CQL, BCQ, encoders, baselines). Training configs are defined in `src/models/rl/config.py`.

## Primary workflows & commands ✅
- Run examples to see intended usage (fast way to validate changes):
  - `python examples/complete_evaluation_example.py`
  - `python src/environments/quick_start_demo.py`
  - `python src/models/encoders/example_usage.py`
- Run the test script that exercises major evaluation components:
  - `python tests/test_framework.py`
  (The tests are simple scripts rather than pytest fixtures; they import modules via `sys.path` like examples.)
- Install dependencies per-subpackage when needed (some modules provide their own `requirements.txt`): check `src/models/baselines/` and `src/evaluation/` READMEs.

## Project conventions & patterns (do this when modifying/adding code) 🔧
- Configs are dataclasses with YAML helpers:
  - Examples: `src/models/rl/config.py`, `src/configs/config_template.py`.
  - Use `.from_yaml()` / `.save()` helpers to load/save reproducible configs.
- Trajectory & evaluator patterns:
  - Evaluators expect either dict-based trajectories (examples) or `Trajectory` objects (see `examples/complete_evaluation_example.py` and `src/evaluation/*`).
  - Off-policy evaluation methods are referenced as names like `'IS'`, `'WIS'`, `'DR'`, `'DM'` (case-insensitive in examples).
- Safety-first mindset: many components add safety checks (e.g., `SafetyEvaluator`, `SafeStateChecker`, safety thresholds in config). Preserve safety behavior unless explicitly changing tests.
- Data pipeline expectations:
  - Use `MIMICLoader` / `CohortBuilder` / `DataPreprocessor` helpers (see `src/data/README.md`) and prefer chunked reads and cache options for large files.

## Code-style & integration notes
- Follow the existing module-level structure and docstring style. Many modules include complete examples and quick-start scripts — add matching examples when changing public APIs.
- Prefer changing or adding small, well-tested functions. Add example usage in `examples/` for any new public feature.
- Tests are script-based (not a deep pytest suite). Use `tests/test_framework.py` as a smoke test to ensure imports and core evaluators work.

## Files to reference when implementing features or fixes 📚
- Architecture & evaluation: `src/evaluation/README.md`, `src/evaluation/off_policy_eval.py`, `src/evaluation/safety_metrics.py`.
- Environments & simulation: `src/environments/CODE_OVERVIEW.md`, `src/environments/diabetes_env.py`, `src/environments/patient_simulator.py`.
- Data pipeline: `src/data/README.md`, `src/data/mimic_loader.py`, `src/data/cohort_builder.py`.
- Training & RL config: `src/models/rl/config.py`, `src/models/rl/*`.
- Example scripts: `examples/complete_evaluation_example.py`, `examples/example_usage.py` in various submodules.

> Note: When in doubt about intended behavior, run the example(s) that exercise that code path; the examples act as executable documentation.

## Safety & ethics constraints (important)
- This project simulates clinical data and decisions. Do NOT fabricate clinical claims in docstrings; keep phrasing neutral and reference README ranges/thresholds that exist in code.
- Avoid adding real patient data or secrets. Use synthetic or cached data flows for examples.

## How to propose changes (PR guidance) ✍️
1. Make small, focused changes with one behavioral change per PR.
2. Add/modify an example that reproduces the new behavior (under `examples/`) and a small test or an addition to `tests/test_framework.py` when feasible.
3. Update the relevant module README or a top-level note explaining why the change was required.
4. If change affects performance experiments, include reproducible config YAML (use `src/configs/config_template.py` as a model).

---

If any of the above is unclear or you want richer examples for a specific area (e.g., adding a new OPE method or integrating MIMIC preprocessing), tell me which area and I will expand the file with short, concrete snippets. ✅