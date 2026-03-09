#!/usr/bin/env bash
# Robust scenario runner for RL Healthcare Treatment Project
# Usage:
#   ./quick_start.sh
#   ./quick_start.sh --mode smoke
#   ./quick_start.sh --mode 1
#   ./quick_start.sh --mode mimic-full --mimic-dir /path/to/mimic
#   ./quick_start.sh --install-deps --mode synthetic-fast

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

status()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()     { echo -e "${RED}[ERR]${NC} $*"; }
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
section() {
  echo ""
  echo "=================================================="
  echo "$*"
  echo "=================================================="
}

PYTHON_BIN="python3"
MODE=""
MIMIC_DIR=""
INSTALL_DEPS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'USAGE'
Usage:
  ./quick_start.sh
  ./quick_start.sh --mode <name_or_number>
  ./quick_start.sh --mode smoke
  ./quick_start.sh --mode resume-check
  ./quick_start.sh --mode synthetic-fast
  ./quick_start.sh --mode synthetic-medium
  ./quick_start.sh --mode synthetic-full
  ./quick_start.sh --mode mimic-sample --mimic-dir /path/to/mimic
  ./quick_start.sh --mode mimic-full --mimic-dir /path/to/mimic
  ./quick_start.sh --mode defense
  ./quick_start.sh --mode regression-pack

Options:
  --mode <value>         Scenario mode name or numeric alias
  --mimic-dir <path>     MIMIC-III CSV directory (optional; prompts if missing for mimic modes)
  --install-deps         Install dependencies from requirements.txt
  --help, -h             Show this help

Named modes:
  smoke
  resume-check
  synthetic-fast
  synthetic-medium
  synthetic-full
  mimic-sample
  mimic-full
  defense
  regression-pack

Backward-compatible numeric aliases:
  1 -> smoke
  2 -> synthetic-cql
  3 -> synthetic-encoder-cql
  4 -> synthetic-extended-state
  5 -> synthetic-full
  6 -> mimic-sample
  7 -> mimic-full
  8 -> defense
USAGE
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode)
        if [[ $# -lt 2 ]]; then
          err "--mode requires a value"
          exit 1
        fi
        MODE="$2"
        shift 2
        ;;
      --mimic-dir)
        if [[ $# -lt 2 ]]; then
          err "--mimic-dir requires a value"
          exit 1
        fi
        MIMIC_DIR="$2"
        shift 2
        ;;
      --install-deps)
        INSTALL_DEPS=1
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        err "Unknown argument: $1"
        usage
        exit 1
        ;;
    esac
  done
}

normalize_mode() {
  local raw="$1"
  case "$raw" in
    1) echo "smoke" ;;
    2) echo "synthetic-cql" ;;
    3) echo "synthetic-encoder-cql" ;;
    4) echo "synthetic-extended-state" ;;
    5) echo "synthetic-full" ;;
    6) echo "mimic-sample" ;;
    7) echo "mimic-full" ;;
    8) echo "defense" ;;
    *) echo "$raw" ;;
  esac
}

pick_python() {
  if [[ -x "./venv/bin/python" ]]; then
    PYTHON_BIN="./venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    err "python3 not found"
    exit 1
  fi
}

check_project_root() {
  if [[ ! -f "src/run_integrated_solution.py" ]]; then
    err "Run this script from the project root (missing src/run_integrated_solution.py)"
    exit 1
  fi
  if [[ ! -f "requirements.txt" ]]; then
    err "requirements.txt not found"
    exit 1
  fi
}

install_dependencies_if_requested() {
  if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    section "Installing Dependencies"
    "$PYTHON_BIN" -m pip install -r requirements.txt
    status "Dependencies installed"
  fi
}

check_runtime_deps() {
  local dep_check
  dep_check="$($PYTHON_BIN - <<'PY'
import importlib
mods = ["numpy", "pandas", "torch", "matplotlib", "sklearn"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
print(",".join(missing))
PY
)"
  if [[ -n "$dep_check" ]]; then
    err "Missing Python dependencies: $dep_check"
    info "Run: ./quick_start.sh --install-deps --mode <mode>"
    exit 1
  fi
  status "Runtime dependencies look available"
}

print_device_info() {
  "$PYTHON_BIN" - <<'PY'
import torch
if torch.backends.mps.is_available():
    print("[INFO] Device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    print("[INFO] Device: CUDA")
else:
    print("[INFO] Device: CPU")
PY
}

run_cmd() {
  local cmd=("$@")
  info "Running command:"
  printf '  '
  printf '%q ' "${cmd[@]}"
  echo ""
  "${cmd[@]}"
}

require_path() {
  local p="$1"
  local label="$2"
  if [[ ! -e "$p" ]]; then
    err "$label missing: $p"
    return 1
  fi
  return 0
}

check_mimic_dir() {
  local dir="$1"
  if [[ -z "$dir" ]]; then
    err "MIMIC directory is empty"
    return 1
  fi
  if [[ ! -d "$dir" ]]; then
    err "Directory not found: $dir"
    return 1
  fi

  local required=(PATIENTS.csv ADMISSIONS.csv DIAGNOSES_ICD.csv LABEVENTS.csv PRESCRIPTIONS.csv)
  local missing=()
  local f
  for f in "${required[@]}"; do
    if [[ ! -f "$dir/$f" ]]; then
      missing+=("$f")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    err "Missing required MIMIC files in $dir: ${missing[*]}"
    info "See docs/MIMIC_DOWNLOAD_GUIDE.md"
    return 1
  fi
  status "MIMIC directory validated: $dir"
}

resolve_mimic_dir() {
  if [[ -n "$MIMIC_DIR" ]]; then
    check_mimic_dir "$MIMIC_DIR" || exit 1
    return
  fi
  read -r -p "Path to MIMIC-III CSV directory: " MIMIC_DIR
  check_mimic_dir "$MIMIC_DIR" || exit 1
}

print_scenario_summary() {
  local scenario="$1"
  local requested_out_dir="$2"
  local out_dir
  out_dir="$(resolve_output_dir "$requested_out_dir")"
  section "Scenario Complete: $scenario"
  if [[ "$out_dir" != "$requested_out_dir" ]]; then
    info "Requested output dir: $requested_out_dir"
  fi
  info "Output dir: $out_dir"
  info "Checkpoint dir: $out_dir/checkpoints"

  if [[ -f "$out_dir/RUN_PROVENANCE.json" ]]; then
    status "Provenance: $out_dir/RUN_PROVENANCE.json"
  else
    warn "Provenance not found: $out_dir/RUN_PROVENANCE.json"
  fi

  if [[ -f "$out_dir/results_summary.json" ]]; then
    status "Summary: $out_dir/results_summary.json"
  fi
  if [[ -f "$out_dir/MASTER_RESULTS.csv" ]]; then
    status "Master results: $out_dir/MASTER_RESULTS.csv"
  fi
}

resolve_output_dir() {
  local requested="$1"
  if [[ -d "$requested" ]]; then
    echo "$requested"
    return
  fi
  if [[ -d "${requested}_synthetic" ]]; then
    echo "${requested}_synthetic"
    return
  fi
  if [[ -d "${requested}_mimic" ]]; then
    echo "${requested}_mimic"
    return
  fi
  echo "$requested"
}

validate_smoke_outputs() {
  local out_dir
  out_dir="$(resolve_output_dir "$1")"
  require_path "$out_dir" "Smoke output directory" || exit 1
  require_path "$out_dir/checkpoints" "Smoke checkpoint directory" || exit 1
  require_path "$out_dir/RUN_PROVENANCE.json" "Smoke provenance file" || exit 1
  status "Smoke validation checks passed"
}

run_smoke() {
  local requested_out_dir="outputs/smoke_test_synthetic"
  section "Scenario: smoke"
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --use-synthetic \
    --n-synthetic-patients 40 \
    --trajectory-length 8 \
    --fast-eval \
    --max-eval-samples 200 \
    --skip-slow-baselines \
    --light-report \
    --output-dir "$requested_out_dir"

  validate_smoke_outputs "$requested_out_dir"
  print_scenario_summary "smoke" "$requested_out_dir"
}

run_resume_check() {
  local requested_out_dir="outputs/smoke_test_synthetic"
  section "Scenario: resume-check"
  info "Step 1/2: initial smoke run"
  run_smoke

  local out_dir
  out_dir="$(resolve_output_dir "$requested_out_dir")"
  info "Step 2/2: rerun smoke with --resume"
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --use-synthetic \
    --n-synthetic-patients 40 \
    --trajectory-length 8 \
    --fast-eval \
    --max-eval-samples 200 \
    --skip-slow-baselines \
    --light-report \
    --resume \
    --output-dir "$out_dir"

  validate_smoke_outputs "$out_dir"
  status "Resume path exercised successfully"
  print_scenario_summary "resume-check" "$requested_out_dir"
}

run_synthetic_fast() {
  local out_dir="outputs/synthetic_fast_synthetic"
  section "Scenario: synthetic-fast"
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --use-synthetic \
    --n-synthetic-patients 120 \
    --trajectory-length 12 \
    --fast-eval \
    --max-eval-samples 1000 \
    --skip-slow-baselines \
    --light-report \
    --output-dir "$out_dir"
  print_scenario_summary "synthetic-fast" "$out_dir"
}

run_synthetic_medium() {
  local out_dir="outputs/synthetic_medium_synthetic"
  section "Scenario: synthetic-medium"
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --use-synthetic \
    --n-synthetic-patients 400 \
    --trajectory-length 20 \
    --use-encoder \
    --encoder-state-dim 64 \
    --encoder-epochs 30 \
    --train-cql \
    --cql-iterations 3000 \
    --cql-batch-size 256 \
    --fast-eval \
    --max-eval-samples 3000 \
    --output-dir "$out_dir"
  print_scenario_summary "synthetic-medium" "$out_dir"
}

run_synthetic_full() {
  local out_dir="outputs/synthetic_full_synthetic"
  section "Scenario: synthetic-full"
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --use-synthetic \
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
    --tree-max-depth 4 \
    --explain-n-samples 100 \
    --use-transfer \
    --transfer-steps 1000 \
    --output-dir "$out_dir"
  print_scenario_summary "synthetic-full" "$out_dir"
}

run_synthetic_cql() {
  local out_dir="outputs/cql_only_synthetic"
  section "Scenario: synthetic-cql"
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --use-synthetic \
    --n-synthetic-patients 500 \
    --trajectory-length 30 \
    --train-cql \
    --cql-iterations 5000 \
    --cql-batch-size 256 \
    --output-dir "$out_dir"
  print_scenario_summary "synthetic-cql" "$out_dir"
}

run_synthetic_encoder_cql() {
  local out_dir="outputs/enc_cql_synthetic"
  section "Scenario: synthetic-encoder-cql"
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --use-synthetic \
    --n-synthetic-patients 500 \
    --trajectory-length 30 \
    --use-encoder \
    --encoder-state-dim 64 \
    --encoder-epochs 50 \
    --encoder-type autoencoder \
    --train-cql \
    --cql-iterations 10000 \
    --output-dir "$out_dir"
  print_scenario_summary "synthetic-encoder-cql" "$out_dir"
}

run_synthetic_extended_state() {
  local out_dir="outputs/extended_state_synthetic"
  section "Scenario: synthetic-extended-state"
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --use-synthetic \
    --n-synthetic-patients 500 \
    --trajectory-length 30 \
    --use-vitals \
    --use-med-history \
    --use-encoder \
    --encoder-state-dim 64 \
    --encoder-epochs 50 \
    --train-cql \
    --cql-iterations 10000 \
    --output-dir "$out_dir"
  print_scenario_summary "synthetic-extended-state" "$out_dir"
}

run_mimic_sample() {
  local out_dir="outputs/mimic_sample"
  section "Scenario: mimic-sample"
  resolve_mimic_dir
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --mimic-dir "$MIMIC_DIR" \
    --use-sample \
    --sample-size 100 \
    --train-cql \
    --cql-iterations 5000 \
    --cql-batch-size 256 \
    --output-dir "$out_dir"
  print_scenario_summary "mimic-sample" "$out_dir"
}

run_mimic_full() {
  local out_dir="outputs/mimic_full"
  section "Scenario: mimic-full"
  resolve_mimic_dir

  local vitals_flag=()
  if [[ -f "$MIMIC_DIR/CHARTEVENTS.csv" ]]; then
    status "CHARTEVENTS.csv detected: enabling --use-vitals"
    vitals_flag=(--use-vitals)
  else
    warn "CHARTEVENTS.csv not found: vitals disabled"
  fi

  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --mode train-eval \
    --mimic-dir "$MIMIC_DIR" \
    "${vitals_flag[@]}" \
    --use-med-history \
    --use-encoder \
    --encoder-state-dim 64 \
    --encoder-epochs 50 \
    --train-cql \
    --cql-iterations 20000 \
    --cql-batch-size 256 \
    --use-interpretability \
    --n-counterfactuals 5 \
    --explain-n-samples 100 \
    --use-transfer \
    --transfer-steps 1000 \
    --output-dir "$out_dir"
  print_scenario_summary "mimic-full" "$out_dir"
}

run_defense() {
  section "Scenario: defense"
  local run_id="defense_$(date +%Y%m%d_%H%M%S)"
  run_cmd "$PYTHON_BIN" src/run_integrated_solution.py \
    --defense-bundle \
    --run-id "$run_id" \
    --seed 42

  local out_dir="outputs/$run_id"
  print_scenario_summary "defense" "$out_dir"
}

run_regression_pack() {
  section "Scenario: regression-pack"
  info "Step 1/4: smoke"
  run_smoke
  info "Step 2/4: resume-check"
  run_resume_check
  info "Step 3/4: synthetic-fast"
  run_synthetic_fast
  info "Step 4/4: defense"
  run_defense
  status "Regression pack completed successfully"
}

interactive_menu() {
  section "RL Healthcare Treatment — Scenario Runner"
  cat <<'MENU'
Select a run scenario:

  Core scenarios
  1) smoke
  2) synthetic-cql
  3) synthetic-encoder-cql
  4) synthetic-extended-state
  5) synthetic-full
  6) mimic-sample
  7) mimic-full
  8) defense

  Additional scenarios
  9) resume-check
 10) synthetic-fast
 11) synthetic-medium
 12) regression-pack
MENU
  echo ""
  read -r -p "Enter choice (1-12): " MODE
  case "$MODE" in
    9) MODE="resume-check" ;;
    10) MODE="synthetic-fast" ;;
    11) MODE="synthetic-medium" ;;
    12) MODE="regression-pack" ;;
    *) MODE="$(normalize_mode "$MODE")" ;;
  esac
}

dispatch_mode() {
  local m="$1"
  case "$m" in
    smoke) run_smoke ;;
    resume-check) run_resume_check ;;
    synthetic-fast) run_synthetic_fast ;;
    synthetic-medium) run_synthetic_medium ;;
    synthetic-full) run_synthetic_full ;;
    synthetic-cql) run_synthetic_cql ;;
    synthetic-encoder-cql) run_synthetic_encoder_cql ;;
    synthetic-extended-state) run_synthetic_extended_state ;;
    mimic-sample) run_mimic_sample ;;
    mimic-full) run_mimic_full ;;
    defense) run_defense ;;
    regression-pack) run_regression_pack ;;
    *)
      err "Unknown mode: $m"
      usage
      exit 1
      ;;
  esac
}

main() {
  parse_args "$@"
  pick_python
  check_project_root
  install_dependencies_if_requested
  check_runtime_deps
  print_device_info

  if [[ -z "$MODE" ]]; then
    interactive_menu
  else
    MODE="$(normalize_mode "$MODE")"
  fi

  dispatch_mode "$MODE"
}

main "$@"
