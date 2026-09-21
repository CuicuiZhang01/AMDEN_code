#!/bin/bash -l
#SBATCH --job-name=sio2-infer-ep800_steps200_E70
#SBATCH --account=DMR21072
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=experiments/sio2_e/ls6_train_generation1/slurm-infer-ep800_steps200_E70-%j.out
#SBATCH --error=experiments/sio2_e/ls6_train_generation1/slurm-infer-ep800_steps200_E70-%j.err

# Submit from the repository root. Use the completed training job's result root.
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

experiment_dir="$PWD/experiments/sio2_e/ls6_train_generation1"
export SAVE_ROOT_DIR="$experiment_dir/results/gpu-a100-small/3453724"
export MPLCONFIGDIR="$SAVE_ROOT_DIR/matplotlib"
checkpoint="$SAVE_ROOT_DIR/models/SiO2/egnn-E/00800.pt"
if [[ ! -s "$checkpoint" ]]; then
    echo "Checkpoint missing or empty: $checkpoint" >&2
    exit 1
fi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable. Activate a GPU environment on an allocated compute node."; print("GPU:", torch.cuda.get_device_name(0))'
python src/main.py -s "$experiment_dir/infer_ep800_steps200_E70.yaml" -g 0
echo "Inference completed successfully."
date -Is
