# Mac 小样运行

在仓库根目录运行。使用 Python 3.11，Apple Silicon CPU。

首次安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/mac-smoke/requirements.lock.txt
```

训练与生成（两个独立进程，生成时加载第 2 epoch 的权重）：

```bash
source .venv/bin/activate
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export OMP_NUM_THREADS=1
python experiments/mac-smoke/src/main.py -s experiments/mac-smoke/train.yaml -g -1
python experiments/mac-smoke/src/main.py -s experiments/mac-smoke/infer.yaml -g -1
```

配置读取 `datasets/Si/data/Si-anneal.extxyz` 的前两个结构，每个结构保留全部
256 个原子；batch size 为 1，训练 2 epochs。网络缩小为 hidden_nf=16、
n_layers=2、n_coords=2。生成使用一个结构的晶胞和原子数，从随机噪声开始，
只运行 10 个去噪步。

默认输出：

- `cache/models/mac-smoke/si/00002.pt`：生成所加载的权重。
- `cache/models/mac-smoke/si/final.pt`：训练最终权重。
- `cache/infer/mac-smoke/si/inferred.extxyz`：生成的结构。
- `cache/infer/mac-smoke/si/traj-00000.extxyz`：生成轨迹。
- `cache/log/`：训练日志和 TensorBoard 数据。

这是流程验证，生成结构不代表论文质量，也不能用于物性结论。
重复生成会覆盖该目录中的同名结构文件。

小数据训练时，日志对未采样到的扩散时间分箱计算平均值，会出现
`Mean of empty slice` 警告及分箱 `nan`；这与训练总 loss/权重出现非有限值
不同。本次验证训练总 loss 和保存的权重均为有限值。

本目录的 `src/` 是独立源码副本，仓库原始 `src/` 保持不变。
仅在此副本中修正了两个问题：`-g -1` 在有 CUDA 的机器上也选择 CPU；
无条件硅样品的 `properties=None` 在生成时按空字典导出。

## 后续迁移

同步代码、数据、配置及需要保留的权重；在服务器上重新创建环境，不复制
Mac 的 `.venv`。这里的 lock 文件记录 Mac 验证环境，不应直接作为 CUDA
或 Linux ARM 环境的安装清单。

先在 GPU 上用同一小样配置测试（`-g 0`），再使用正式网络配置、完整数据集
和正式训练/采样步数。小网络的权重不兼容正式的大网络，应重新训练。
正式配置中的数据路径需要逐项核对仓库实际目录。

Lonestar6 与 Vista 的环境和作业脚本应分别按 TACC 文档配置：

- https://docs.tacc.utexas.edu/hpc/lonestar6/
- https://docs.tacc.utexas.edu/hpc/vista/

GPU 训练通过计算节点/Slurm 作业运行。确定使用的系统和 allocation 后，
再配置该系统的 CUDA/PyTorch 环境和提交脚本。
