# SiO2 E-conditioned training and generation

Transfer this experiment together with the repository's `src/` directory and
`datasets/SiO2/data/`. Paths in the YAML files are relative to the repository root.
Do not copy a macOS virtual environment to Linux; create a compatible GPU Python
environment on the server. The repository runtime files describe its dependencies.

On an allocated GPU compute node, activate that environment and run from the
repository root:

```bash
bash experiments/ls6_train_generation_test/run.sh train
bash experiments/ls6_train_generation_test/run.sh infer
```

Use `all` to run training followed by inference in one invocation. This runner
does not request resources or submit a Slurm job. Call it from a Slurm job after
setting the account, GPU resources, time limit, and Python environment for your
allocation. Cluster execution has not been validated locally.

The configuration retains the full dataset and original network, batch size 4,
learning rate 0.001, and 800 training epochs. Checkpoints are saved every 100
epochs; inference loads epoch 800. Inference uses 200 denoising steps and up to
4 charge-balance restarts. These are full experiment settings, not a short GPU
smoke test.

The input template contains 100 Si and 200 O atoms; its target E is 70 GPa.
Random initialization and the configured ghost atoms mean that the generated
atom count and composition are not guaranteed to match the template exactly.
The requested E must be checked by subsequent property evaluation.

`run.sh` exports `SAVE_ROOT_DIR` before importing the original Python code.
Outputs are written under this experiment's `results/` directory:

- `models/SiO2/egnn-E/00800.pt`: epoch 800 weights.
- `models/SiO2/egnn-E/final.pt`: final weights.
- `infer/SiO2/egnn-E/inferred.extxyz`: generated structures.
- `infer/SiO2/egnn-E/traj-00000.extxyz`: denoising trajectory.
- `infer/SiO2/egnn-E/given_properties.json`: requested conditions.
- `log/`: training logs and TensorBoard events.

Repeated runs can overwrite weights and generated structures. Existing local
`cache/` results are not moved. Running `src/main.py` directly requires exporting
the same `SAVE_ROOT_DIR` to use this output location.
