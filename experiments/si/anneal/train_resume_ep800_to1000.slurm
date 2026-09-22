#!/bin/bash -l
#SBATCH --job-name=si_800_1000
#SBATCH --account=DMR21072
#SBATCH --partition=gpu-a100-small
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=experiments/si/anneal/slurm-resume800-1000-%j.out
#SBATCH --error=experiments/si/anneal/slurm-resume800-1000-%j.err

# Submit from the repository root. Slurm logs use the existing experiment directory.
# The script creates a separate output directory for each partition and job. Environment follows the smoke test.
set -eo pipefail
module unload python3
module load python/3.12.11
source "${SCRATCH:?SCRATCH is not set}/venvs/amden/bin/activate"
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f src/main.py
test -s datasets/Si/data/Si-anneal.extxyz
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
date -Is
# Slurm executes a spooled script, so resolve paths from the submission directory.
experiment_dir="$PWD/experiments/si/anneal"
export SAVE_ROOT_DIR="$experiment_dir/results/gpu-a100-small/${SLURM_JOB_ID}"
export MPLCONFIGDIR="$SAVE_ROOT_DIR/matplotlib"
# Default: completed 400-to-800 job. Override only when resuming from another job.
export RESUME_JOB_ID="${RESUME_JOB_ID:-3455794}"
export RESUME_EPOCH=800
export EXPERIMENT_DIR="$experiment_dir"
test "$SLURM_JOB_ID" != "$RESUME_JOB_ID"
mkdir -p "$SAVE_ROOT_DIR/models"
# Materialize the exact configuration alongside this job's outputs.
python - <<'PYCONFIG'
import os
from pathlib import Path
import yaml

experiment = Path(os.environ['EXPERIMENT_DIR'])
root = Path(os.environ['SAVE_ROOT_DIR'])
source_job = os.environ['RESUME_JOB_ID']
assert source_job.isdigit(), 'RESUME_JOB_ID must be a numeric job ID'
epoch = int(os.environ['RESUME_EPOCH'])
with (experiment / 'train_resume_ep800_to1000.yaml').open() as stream:
    config = yaml.safe_load(stream)
assert 0 < epoch < config['train']['params']['num_epochs'], 'Resume epoch must be below the target'
model_dir = experiment / 'results/gpu-a100-small' / source_job / 'models/a-Si/anneal/egnn'
checkpoint = model_dir / f'{epoch:05d}.pt'
if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    raise FileNotFoundError(f'Missing or empty checkpoint: {checkpoint}')
if config['load'].get('training_state', False):
    full_checkpoint = model_dir / f'{epoch:05d}.train.pt'
    if not full_checkpoint.is_file() or full_checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f'Missing full training state: {full_checkpoint}')
config['load'].update(enabled=True, name=str(model_dir), epoch=epoch)
with (root / 'train_resume.yaml').open('w') as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
print(f'Resuming {checkpoint}; target: 1000 epochs; save every 100 epochs', flush=True)
print(f'Outputs: {root}', flush=True)
PYCONFIG
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable. Activate a GPU environment on an allocated compute node."; print("GPU:", torch.cuda.get_device_name(0))'
python src/main.py -s "$SAVE_ROOT_DIR/train_resume.yaml" -g 0
echo "Training completed successfully."
date -Is
