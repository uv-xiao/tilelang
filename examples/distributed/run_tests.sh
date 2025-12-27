#!/bin/bash
# Copyright (c) Tile-AI Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Run distributed tests using the venv environment.
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh check        # Run NVSHMEM check only
#   ./run_tests.sh allreduce    # Run AllReduce test only
#   ./run_tests.sh pingpong     # Run ping-pong test only

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILELANG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate venv
if [ -f "$TILELANG_ROOT/.venv/bin/activate" ]; then
    source "$TILELANG_ROOT/.venv/bin/activate"
    echo "Activated venv at $TILELANG_ROOT/.venv"
else
    echo "Warning: venv not found at $TILELANG_ROOT/.venv"
    echo "Please create it with: uv venv .venv --python 3.10"
fi

# Add tilelang to PYTHONPATH
export PYTHONPATH="$TILELANG_ROOT:$PYTHONPATH"

# Detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=0
fi

echo "Detected $NUM_GPUS GPU(s)"
echo ""

run_check() {
    echo "============================================================"
    echo "Running NVSHMEM check..."
    echo "============================================================"
    python "$SCRIPT_DIR/check_nvshmem.py"
    echo ""
}

run_allreduce() {
    echo "============================================================"
    echo "Running AllReduce test..."
    echo "============================================================"
    if [ "$NUM_GPUS" -ge 2 ]; then
        torchrun --nproc_per_node=2 "$SCRIPT_DIR/simple_allreduce.py"
    else
        echo "Skipping: requires at least 2 GPUs"
    fi
    echo ""
}

run_pingpong() {
    echo "============================================================"
    echo "Running Ping-Pong test..."
    echo "============================================================"
    if [ "$NUM_GPUS" -ge 2 ]; then
        torchrun --nproc_per_node=2 "$SCRIPT_DIR/ping_pong.py" --bandwidth
    else
        echo "Skipping: requires at least 2 GPUs"
    fi
    echo ""
}

# Parse arguments
case "${1:-all}" in
    check)
        run_check
        ;;
    allreduce)
        run_allreduce
        ;;
    pingpong)
        run_pingpong
        ;;
    all)
        run_check
        run_allreduce
        run_pingpong
        ;;
    *)
        echo "Usage: $0 [check|allreduce|pingpong|all]"
        exit 1
        ;;
esac

echo "============================================================"
echo "All tests completed"
echo "============================================================"
