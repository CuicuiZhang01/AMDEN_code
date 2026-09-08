#!/bin/bash -l
#SBATCH --job-name=sio2-smoke
#SBATCH --account=DMR21072
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/11579/cuicuizhang7879/ls6/AMDEN_code
#SBATCH --output=experiments/sio2_e/ls6_train_generate_test/results/slurm-%j.out
#SBATCH --error=experiments/sio2_e/ls6_train_generate_test/results/slurm-%j.err

# Create the results directory before submitting this job.
# LS6 allocates GPUs through the partition; do not add --gres or --gpus-per-task.
set -eo pipefail
module unload python3
module load python/3.12.11
source "${SCRATCH:?SCRATCH is not set}/venvs/amden/bin/activate"
set -u

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
date -Is

# Run one Python process on GPU 0: train first, then load epoch 2 and generate.
bash experiments/sio2_e/ls6_train_generate_test/run.sh all

echo "Training and generation completed successfully."
date -Is
