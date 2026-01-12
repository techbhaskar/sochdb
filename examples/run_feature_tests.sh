#!/bin/bash
# SochDB Feature Validation Test Runner
# Usage: ./examples/run_feature_tests.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  SochDB Feature Validation Test Suite"
echo "=============================================="

export PYTHONPATH="${PROJECT_DIR}/sochdb-python-sdk/src:${PYTHONPATH}"
export SOCHDB_LIB_PATH="${PROJECT_DIR}/target/release"

case "${1:-}" in
    --quick)
        echo "Running quick smoke tests..."
        python3 "${SCRIPT_DIR}/python/sochdb_feature_validation.py" 2>&1 | head -20
        ;;
    *)
        python3 "${SCRIPT_DIR}/python/sochdb_feature_validation.py"
        ;;
esac

