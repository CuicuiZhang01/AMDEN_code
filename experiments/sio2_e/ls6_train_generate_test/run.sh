#!/usr/bin/env bash
# Run inside an allocated GPU compute session with a Linux GPU environment active.
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$experiment_dir/../../.." && pwd)"
mode="${1:-}"
case "$mode" in
    train|infer|all) ;;
    *) echo "Usage: bash run.sh {train|infer|all}" >&2; exit 2 ;;
esac

cd "$repo_root"
export SAVE_ROOT_DIR="$experiment_dir/results"
export MPLCONFIGDIR="$SAVE_ROOT_DIR/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Avoid the original entry point's automatic CPU fallback.
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable. Activate a GPU environment on an allocated compute node."; print("GPU:", torch.cuda.get_device_name(0))'
mkdir -p "$SAVE_ROOT_DIR"

if [[ "$mode" == train || "$mode" == all ]]; then
    python src/main.py -s "$experiment_dir/train.yaml" -g 0
fi
if [[ "$mode" == infer || "$mode" == all ]]; then
    python src/main.py -s "$experiment_dir/infer.yaml" -g 0
fi
