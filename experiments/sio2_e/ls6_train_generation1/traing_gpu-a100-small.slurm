#!/bin/bash -l
#SBATCH --job-name=sio2-gpu-a100-small
#SBATCH --account=DMR21072
#SBATCH --partition=gpu-a100-small
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=experiments/sio2_e/ls6_train_generation1/slurm-gpu-a100-small-%j.out
#SBATCH --error=experiments/sio2_e/ls6_train_generation1/slurm-gpu-a100-small-%j.err

# Submit from the repository root. Slurm logs use the existing experiment directory.
# The script creates a separate output directory for each partition and job. Environment follows the smoke test.
set -eo pipefail
module unload python3
module load python/3.12.11
source "${SCRATCH:?SCRATCH is not set}/venvs/amden/bin/activate"
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f src/main.py
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
date -Is
# Slurm executes a spooled script, so resolve paths from the submission directory.
experiment_dir="$PWD/experiments/sio2_e/ls6_train_generation1"
export SAVE_ROOT_DIR="$experiment_dir/results/gpu-a100-small/${SLURM_JOB_ID}"
export MPLCONFIGDIR="$SAVE_ROOT_DIR/matplotlib"
mkdir -p "$SAVE_ROOT_DIR"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable. Activate a GPU environment on an allocated compute node."; print("GPU:", torch.cuda.get_device_name(0))'
python src/main.py -s "$experiment_dir/train.yaml" -g 0
echo "Training completed successfully."
date -Is
