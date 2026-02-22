#!/bin/bash

# Simple Test Runner for Healthcare RL Data Pipeline
# Runs all module examples sequentially
#

set -e  # Exit on error

echo "=================================="
echo "Healthcare RL Pipeline Test Suite"
echo "=================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

# Function to run a test
run_test() {
    local module=$1
    local name=$2
    
    echo "Testing: $name"
    echo "Running: python $module"
    echo ""
    
    if python "$module"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ $name failed${NC}"
        ((FAILED++))
    fi
    
    echo ""
    echo "---"
    echo ""
}

# Change to data directory if needed
if [ -d "src/data" ]; then
    cd src/data
elif [ ! -f "mimic_loader.py" ]; then
    echo -e "${RED}Error: Cannot find data modules${NC}"
    echo "Please run from project root or src/data directory"
    exit 1
fi

# Run all tests
echo "Starting tests..."
echo ""

run_test "mimic_loader.py" "MIMIC-III Loader"
run_test "preprocessor.py" "Data Preprocessor"
run_test "feature_engineering.py" "Feature Engineering"
run_test "cohort_builder.py" "Cohort Builder"
run_test "synthetic_generator.py" "Synthetic Data Generator"
run_test "data_validator.py" "Data Quality Validator"
run_test "utils.py" "Utility Functions"

# Print summary
echo "=================================="
echo "Test Summary"
echo "=================================="
echo ""
echo "Total tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
