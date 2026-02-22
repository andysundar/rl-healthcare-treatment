#!/bin/bash
# Quick Start Script for RL Healthcare Treatment Project
# Author: Anindya Bandopadhyay (M23CSA508)

set -e  # Exit on error

echo "=========================================="
echo "RL Healthcare Treatment - Quick Start"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python version: $python_version"

# Check if we're in the correct directory
if [ ! -f "run_integrated_solution.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --break-system-packages --quiet
print_status "Dependencies installed"

# Check PyTorch availability
echo ""
echo "Checking PyTorch device availability..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✓ MPS (Apple Silicon GPU) available')
elif torch.cuda.is_available():
    print('✓ CUDA GPU available')
else:
    print('⚠ Using CPU (slower training)')
"

# Ask user for mode
echo ""
echo "=========================================="
echo "Select execution mode:"
echo "=========================================="
echo "1) Quick Demo (5 min) - Synthetic data, baseline comparison"
echo "2) Full Synthetic (30 min) - Complete pipeline with synthetic data"
echo "3) MIMIC-III Sample (1-2 hours) - Small MIMIC cohort"
echo "4) Full MIMIC-III (several hours) - Complete MIMIC pipeline"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        print_status "Running Quick Demo..."
        python3 run_integrated_solution.py \
            --mode synthetic \
            --n-synthetic-patients 100 \
            --trajectory-length 20 \
            --output-dir outputs/quick_demo
        ;;
    2)
        echo ""
        print_status "Running Full Synthetic Pipeline..."
        python3 run_integrated_solution.py \
            --mode synthetic \
            --n-synthetic-patients 1000 \
            --trajectory-length 30 \
            --output-dir outputs/synthetic_full
        ;;
    3)
        echo ""
        print_warning "This requires MIMIC-III data access"
        read -p "Path to MIMIC-III data directory: " mimic_dir
        
        if [ ! -d "$mimic_dir" ]; then
            print_error "Directory not found: $mimic_dir"
            exit 1
        fi
        
        print_status "Running MIMIC-III Sample Pipeline..."
        python3 run_integrated_solution.py \
            --mode full \
            --mimic-dir "$mimic_dir" \
            --use-sample \
            --sample-size 100 \
            --output-dir outputs/mimic_sample
        ;;
    4)
        echo ""
        print_warning "This requires MIMIC-III data access and will take several hours"
        read -p "Path to MIMIC-III data directory: " mimic_dir
        
        if [ ! -d "$mimic_dir" ]; then
            print_error "Directory not found: $mimic_dir"
            exit 1
        fi
        
        read -p "Train CQL agent? (y/n): " train_cql
        
        cql_flag=""
        if [ "$train_cql" = "y" ]; then
            cql_flag="--train-cql"
        fi
        
        print_status "Running Full MIMIC-III Pipeline..."
        python3 run_integrated_solution.py \
            --mode full \
            --mimic-dir "$mimic_dir" \
            $cql_flag \
            --output-dir outputs/mimic_full
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Execution complete!"
echo "=========================================="
echo ""
echo "Check outputs directory for results"
echo ""
echo "Key files:"
echo "  - results_summary.json"
echo "  - baseline_comparison_report.md"
echo "  - baseline_comparison.png"
echo "  - results_table.tex (for thesis)"
echo ""
print_status "Done!"
