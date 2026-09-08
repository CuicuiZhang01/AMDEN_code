# LS6 GPU smoke test

Transfer this directory along with the repository's `src/` and
`datasets/SiO2/data/` directories, keeping their relative layout. Create and
activate a Linux environment with CUDA-enabled PyTorch and project dependencies;
do not copy the macOS `.venv` directory.

Inside an allocated GPU compute session, run from the repository root:

```bash
bash experiments/sio2_e/ls6_train_generate_test/run.sh train
bash experiments/sio2_e/ls6_train_generate_test/run.sh infer
```

Alternatively, use `run.sh all` to train and then generate. The runner does not
allocate GPUs or submit Slurm jobs; it may also be called from an allocated Slurm
batch job after environment activation. It stops if CUDA is unavailable.

Settings: original network architecture, first two training samples, batch size
1, two epochs, and checkpoint saving every epoch. The dataset loader pairs the
full structure and property lists before selecting the two training samples.
Inference loads epoch 2, uses 10 denoising steps, and disables charge-balance
restarts. The input template has 300 atoms, and the requested E is 70 GPa.
This tests execution, not physical quality or exact final composition.

The runner sets `SAVE_ROOT_DIR` before starting Python. All new outputs go to
this experiment's `results/` directory:

- `models/sio2_e/ls6-smoke/00002.pt`: epoch 2 checkpoint.
- `models/sio2_e/ls6-smoke/final.pt`: final weights.
- `infer/sio2_e/ls6-smoke/inferred.extxyz`: generated structures.
- `infer/sio2_e/ls6-smoke/traj-00000.extxyz`: denoising trajectory.
- `infer/sio2_e/ls6-smoke/given_properties.json`: requested conditions.
- `log/`: training logs and TensorBoard events.

The existing `cache/` directory contains earlier results and is left untouched.
Repeated runs can overwrite checkpoints and generated structures in `results/`.
Running `src/main.py` directly requires setting the same `SAVE_ROOT_DIR` manually.
Cluster execution has not yet been verified.
