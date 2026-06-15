# 脚本目录

## `train/`

- `main_train.py`：训练 QCFS ANN。

## `evaluate/`

- `main_test.py`：评估原始 IF/QCFS 模型；
- `main_test_signed.py`：评估 SignedIF，并输出脉冲统计。

## `experiments/`

- `run_ablation.py`：六种 SNM/R0/FTBC 配置的准确率消融；
- `run_stats_ablation.py`：六配置的 SOPs、正负脉冲率和稀疏率消融；
- `run_state_ftbc_ablation.py`：完整 FTBC、状态低秩 FTBC 和 SOPs 约束低秩 FTBC 对比。

所有脚本会自动定位仓库根目录，因此既可以从仓库根目录运行，也可以使用绝对路径运行。
