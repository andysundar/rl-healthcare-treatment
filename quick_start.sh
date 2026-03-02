#!/bin/bash
# Quick Start Script — RL Healthcare Treatment Project
# Author: Anindya Bandopadhyay (M23CSA508)
#
# Usage:
#   ./quick_start.sh              # interactive scenario menu
#   ./quick_start.sh --mode N     # run scenario N directly (N = 1-8)
#   ./quick_start.sh --help       # print scenario descriptions

set -e

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status()  { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error()   { echo -e "${RED}✗${NC} $1"; }
print_info()    { echo -e "${BLUE}ℹ${NC} $1"; }

echo "=========================================="
echo " RL Healthcare Treatment — Quick Start"
echo "=========================================="
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "src/run_integrated_solution.py" ]; then
    print_error "Run this script from the project root directory"
    exit 1
fi

python_ver=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python $python_ver"

# ── Install dependencies ───────────────────────────────────────────────────────
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --break-system-packages --quiet
print_status "Dependencies installed"

# ── Device check ──────────────────────────────────────────────────────────────
echo ""
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✓ MPS (Apple Silicon GPU) available')
elif torch.cuda.is_available():
    print('✓ CUDA GPU available')
else:
    print('⚠  CPU only — training will be slower')
"

# ── Parse --mode / --help arguments ──────────────────────────────────────────
CHOICE=""

if [ "$1" = "--mode" ] && [ -n "$2" ]; then
    CHOICE="$2"
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo ""
    echo "Usage:  ./quick_start.sh [--mode N]"
    echo ""
    echo "Synthetic data scenarios (no MIMIC-III required)"
    echo "  1  Quick demo          baseline comparison only                  (<2 min)"
    echo "  2  Baseline + CQL      adds real offline CQL training            (~10 min)"
    echo "  3  Encoder + CQL       adds state autoencoder pre-training       (~20 min)"
    echo "  4  Extended state      vitals + med-history (16-dim state)       (~20 min)"
    echo "  5  Full pipeline       encoder + CQL + interpretability + transfer (~60 min)"
    echo "  8  Defense bundle      complete thesis-defense artifact bundle     (~2-5 min)"
    echo ""
    echo "MIMIC-III scenarios (credentialed PhysioNet access required)"
    echo "  6  MIMIC sample        100-patient cohort, CQL training          (~30 min)"
    echo "  7  MIMIC full          complete pipeline incl. vitals if available (hours)"
    echo ""
    echo "Required MIMIC-III files (base run): PATIENTS.csv, ADMISSIONS.csv,"
    echo "  DIAGNOSES_ICD.csv, LABEVENTS.csv, PRESCRIPTIONS.csv"
    echo "Optional (for --use-vitals):         CHARTEVENTS.csv"
    echo ""
    echo "See docs/MIMIC_DOWNLOAD_GUIDE.md for download instructions."
    exit 0
fi

# ── Interactive menu ───────────────────────────────────────────────────────────
if [ -z "$CHOICE" ]; then
    echo ""
    echo "=========================================="
    echo "Select a run scenario:"
    echo "=========================================="
    echo ""
    echo "  Synthetic data (no MIMIC-III required)"
    echo "  ──────────────────────────────────────────────────────────────"
    echo "  1) Quick demo          baseline comparison only           (<2 min)"
    echo "  2) Baseline + CQL      adds real CQL training             (~10 min)"
    echo "  3) Encoder + CQL       adds state autoencoder             (~20 min)"
    echo "  4) Extended state      vitals + med-history (16-dim)      (~20 min)"
    echo "  5) Full pipeline       encoder+CQL+interp+transfer        (~60 min)"
    echo ""
    echo "  MIMIC-III data (PhysioNet credentialed access required)"
    echo "  ──────────────────────────────────────────────────────────────"
    echo "  6) MIMIC sample        100-patient cohort                 (~30 min)"
    echo "  7) MIMIC full          complete pipeline w/ vitals        (hours)"
    echo "  8) Defense bundle      full evidence/report artifact run  (~2-5 min)"
    echo ""
    read -p "Enter choice (1-8): " CHOICE
fi

# ── Helper: verify MIMIC-III directory ────────────────────────────────────────
check_mimic_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        print_error "Directory not found: $dir"
        exit 1
    fi
    local missing=""
    for f in PATIENTS.csv ADMISSIONS.csv DIAGNOSES_ICD.csv LABEVENTS.csv PRESCRIPTIONS.csv; do
        [ ! -f "$dir/$f" ] && missing="$missing  $f"
    done
    if [ -n "$missing" ]; then
        print_error "Missing required MIMIC-III files in $dir:"
        echo "$missing"
        print_info "See docs/MIMIC_DOWNLOAD_GUIDE.md for download instructions"
        exit 1
    fi
    print_status "All required MIMIC-III files found"
}

# ── Scenarios ─────────────────────────────────────────────────────────────────
case "$CHOICE" in

    # ── 1: Quick demo — baseline comparison only ──────────────────────────────
    1)
        echo ""
        print_status "Scenario 1: Quick demo — baseline comparison only (<2 min)"
        python3 src/run_integrated_solution.py \
            --mode train-eval \
            --n-synthetic-patients 200 \
            --trajectory-length 20 \
            --output-dir outputs/quick_demo
        ;;

    # ── 2: Baseline + CQL ─────────────────────────────────────────────────────
    2)
        echo ""
        print_status "Scenario 2: Baseline comparison + real CQL training (~10 min)"
        python3 src/run_integrated_solution.py \
            --mode train-eval \
            --n-synthetic-patients 500 \
            --trajectory-length 30 \
            --train-cql \
            --cql-iterations 5000 \
            --cql-batch-size 256 \
            --output-dir outputs/cql_only
        ;;

    # ── 3: Encoder + CQL ──────────────────────────────────────────────────────
    3)
        echo ""
        print_status "Scenario 3: State encoder pre-training + CQL (~20 min)"
        python3 src/run_integrated_solution.py \
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
        ;;

    # ── 4: Extended state — vitals + med-history ──────────────────────────────
    4)
        echo ""
        print_status "Scenario 4: Extended 16-dim state — vitals + med-history (~20 min)"
        print_info "Adds heart_rate, sbp, respiratory_rate, spo2, adherence_rate_7d, medication_count"
        python3 src/run_integrated_solution.py \
            --mode train-eval \
            --n-synthetic-patients 500 \
            --trajectory-length 30 \
            --use-vitals \
            --use-med-history \
            --use-encoder \
            --encoder-state-dim 64 \
            --encoder-epochs 50 \
            --train-cql \
            --cql-iterations 10000 \
            --output-dir outputs/extended_state
        ;;

    # ── 5: Full pipeline — all modules ────────────────────────────────────────
    5)
        echo ""
        print_status "Scenario 5: Full pipeline — encoder + CQL + interpretability + transfer (~60 min)"
        python3 src/run_integrated_solution.py \
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
            --tree-max-depth 4 \
            --explain-n-samples 100 \
            --use-transfer \
            --transfer-steps 1000 \
            --output-dir outputs/full_run
        ;;

    # ── 6: MIMIC-III sample ───────────────────────────────────────────────────
    6)
        echo ""
        print_warning "Requires MIMIC-III data — see docs/MIMIC_DOWNLOAD_GUIDE.md"
        read -p "Path to MIMIC-III CSV directory: " MIMIC_DIR
        check_mimic_dir "$MIMIC_DIR"
        print_status "Scenario 6: MIMIC-III 100-patient sample pipeline (~30 min)"
        python3 src/run_integrated_solution.py \
            --mode train-eval \
            --mimic-dir "$MIMIC_DIR" \
            --use-sample \
            --sample-size 100 \
            --train-cql \
            --cql-iterations 5000 \
            --cql-batch-size 256 \
            --output-dir outputs/mimic_sample
        ;;

    # ── 7: MIMIC-III full pipeline ────────────────────────────────────────────
    7)
        echo ""
        print_warning "Requires MIMIC-III data — see docs/MIMIC_DOWNLOAD_GUIDE.md"
        print_warning "CHARTEVENTS.csv (~35 GB) is needed for --use-vitals"
        read -p "Path to MIMIC-III CSV directory: " MIMIC_DIR
        check_mimic_dir "$MIMIC_DIR"

        # Auto-detect CHARTEVENTS for --use-vitals
        VITALS_FLAG=""
        if [ -f "$MIMIC_DIR/CHARTEVENTS.csv" ]; then
            print_status "CHARTEVENTS.csv found — enabling vital signs (HR, SBP, RR, SpO2)"
            VITALS_FLAG="--use-vitals"
        else
            print_warning "CHARTEVENTS.csv not found — vital signs disabled"
            print_info "Download CHARTEVENTS.csv and re-run to enable --use-vitals"
        fi

        print_status "Scenario 7: Full MIMIC-III pipeline"
        python3 src/run_integrated_solution.py \
            --mode train-eval \
            --mimic-dir "$MIMIC_DIR" \
            $VITALS_FLAG \
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
            --output-dir outputs/mimic_full
        ;;

    # ── 8: Defense bundle ─────────────────────────────────────────────────────
    8)
        echo ""
        print_status "Scenario 8: Thesis defense bundle (~2-5 min CPU)"
        RUN_ID="defense_$(date +%Y%m%d_%H%M%S)"
        python3 src/run_integrated_solution.py \
            --defense-bundle \
            --run-id "$RUN_ID" \
            --seed 42
        ;;

    *)
        print_error "Invalid choice: $CHOICE"
        echo "Run  ./quick_start.sh --help  to see available scenarios"
        exit 1
        ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
print_status "Pipeline complete!"
echo "=========================================="
echo ""
echo "Generated artefacts (outputs/<run_dir>/):"
echo "  results_summary.json           all metrics and module results"
echo "  baseline_comparison_report.md  per-baseline metric table"
echo "  baseline_comparison_report.json  machine-readable metrics"
echo "  baseline_comparison.png        reward + safety rate bar chart"
echo "  policy_dashboard.png           4-panel policy dashboard"
echo "  comparison.png                 multi-policy reward comparison"
echo "  health_metrics.png             time-in-range and safety index"
echo "  safety_clinical_heatmap.png    safety + clinical compliance heatmap"
echo "  results_table.tex              LaTeX table for thesis"
echo "  outputs/defense_<timestamp>/   full defense artifact bundle tree  (--mode 8)"
echo ""
echo "Optional artefacts (present when matching flags were used):"
echo "  cql_training_curves.png        CQL loss/return curves    (--train-cql)"
echo "  feature_importance.png         decision tree importances  (--use-interpretability)"
echo "  personalization_score.png      PDF §7.2 score card        (--use-encoder --use-interpretability)"
echo "  counterfactuals.json           counterfactual explanations (--use-interpretability)"
echo "  decision_rules.txt / .json     if-then rules              (--use-interpretability)"
echo "  transfer_adapter.pt            adapter weights            (--use-transfer)"
echo ""
