# SiO2 E-conditioned training and generation

当前阶段只准备 LS6 训练，操作步骤见 [TRAINING.md](TRAINING.md)。

- `train.slurm`：申请资源、激活环境、检查 CUDA、设置输出目录并直接启动 `src/main.py`。
- `train.yaml`：训练配置，全部当前 SiO2 数据、800 轮、batch size 4，每 100 轮保存。
- `results/`：模型权重及日志输出目录。

原 `run.sh` 的训练逻辑已合并进 `train.slurm`，无需单独上传或调用 run.sh。
请从仓库根目录使用 sbatch 提交，并指定 GPU 分区和时长；提交前创建 results 目录。

`infer.yaml` 和 `inputs/` 留待训练完成后整理；当前生成输入路径尚未更新。
当前训练数据为 `datasets/SiO2/data/`，尚未确认与原实验的 SiO2-mix 数据一致。
