#!/usr/bin/env bash
# Use an activated GPU Python environment on an allocated compute node.
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$experiment_dir/../.." && pwd)"
mode="${1:-}"
case "$mode" in
    train|infer|all) ;;
    *) echo "Usage: bash run.sh {train|infer|all}" >&2; exit 2 ;;
esac

cd "$repo_root"
export SAVE_ROOT_DIR="$experiment_dir/results"
export MPLCONFIGDIR="$SAVE_ROOT_DIR/matplotlib"
mkdir -p "$SAVE_ROOT_DIR"

# The original entry point falls back to CPU when CUDA is unavailable.
# Fail early here to avoid accidentally launching full training on a CPU.
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable. Activate a GPU environment on an allocated compute node."'

if [[ "$mode" == train || "$mode" == all ]]; then
    python src/main.py -s "$experiment_dir/train.yaml" -g 0
fi
if [[ "$mode" == infer || "$mode" == all ]]; then
    python src/main.py -s "$experiment_dir/infer.yaml" -g 0
fi
