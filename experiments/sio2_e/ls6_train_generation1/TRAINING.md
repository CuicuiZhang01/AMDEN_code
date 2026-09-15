# LS6 正式训练准备

本阶段只训练，不运行 infer。使用当前 SiO2/data 数据和原始模型超参数，
不代表已确认与原实验 SiO2-mix 数据相同。

## 上传文件

保持仓库相对路径：

```text
AMDEN_code/
├── src/                         # 完整目录
├── datasets/SiO2/data/
│   ├── SiO2.extxyz
│   └── SiO2_properties.json
└── experiments/sio2_e/ls6_train_generation1/
    ├── train.yaml
    ├── train.slurm
    └── TRAINING.md
```

使用 smoke test 已验证的 LS6 Python 环境；依赖清单位于 runtime/requirements.txt，
服务器已有的 requirements-ls6.lock.txt 可作为该环境的版本记录。
无需上传 Mac 虚拟环境、旧权重、旧日志或字体缓存。
train.yaml 的 infer.enabled 为 false，因此本阶段不需要 inputs/ 或 infer.yaml。

## 训练设置

- 全部 datasets/SiO2/data/SiO2.extxyz 结构及对应属性，未设置 :2 截断。
- 从头训练：load.enabled=false。
- batch_size=4，学习率 0.001，800 轮，每 100 轮保存。
- 保存名称 SiO2/egnn-E；结果在本实验 results/ 中。

## 提交

从 LS6 的 AMDEN_code 根目录执行：

```bash
mkdir -p experiments/sio2_e/ls6_train_generation1/results
# 先查看集群分区、时限，并确认账号权限
sinfo -o "%P %a %l %D %G"
# 将下面两个占位值替换成已确认的 GPU 分区和运行时长
sbatch --partition=<GPU分区> --time=<HH:MM:SS> experiments/sio2_e/ls6_train_generation1/train.slurm
```

占位符不能原样执行。正式分区和时长需要根据 LS6 当前可用资源以及训练耗时确定；
脚本不沿用 smoke test 的开发分区及 30 分钟时限。
账号默认沿用 DMR21072；如需更换，用 sbatch --account=你的账号 覆盖。
脚本沿用 python/3.12.11 和 $SCRATCH/venvs/amden 环境。
使用 sbatch 提交 train.slurm，不要在登录节点用 bash 直接运行它。

## 检查结果

```bash
squeue -u "$USER"
tail -f experiments/sio2_e/ls6_train_generation1/results/slurm-作业号.out
```

成功结束应有：

- results/models/SiO2/egnn-E/00800.pt
- results/models/SiO2/egnn-E/final.pt
- results/log/ 下的训练日志

定期权重从 00100.pt 开始保存。重复训练会使用相同输出路径，可能覆盖旧权重；
重新提交前先归档已有 results。保存的是模型权重，不是完整优化器恢复状态。
训练完成后再准备 infer；当前 infer 输入路径尚未整理，本阶段不运行生成。
