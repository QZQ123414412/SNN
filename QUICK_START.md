# 快速开始

以下命令均从仓库根目录执行。

## 单模型测试

```powershell
# 原始 QCFS/IF
python scripts/evaluate/main_test.py -data cifar100 -arch vgg16 `
  -id cifar100-vgg16-l8-example -T 8 -dev 0

# SNM/R0 SignedIF，并统计 SOPs、正负脉冲率和稀疏率
python scripts/evaluate/main_test_signed.py -data cifar100 `
  -arch vgg16_signed -id cifar100-vgg16-l8-example -T 8 -dev 0
```

## 六配置消融

```powershell
python scripts/experiments/run_stats_ablation.py -data cifar100 `
  -id cifar100-vgg16-l8-example -dev 0 `
  --time_steps 1 2 4 8 16 32
```

默认结果保存到：

```text
docs/results/baseline/spike_statistics/cifar100/STATS_ABLATION_cifar100.md
```

## 状态条件低秩 FTBC

```powershell
python scripts/experiments/run_state_ftbc_ablation.py -data cifar100 `
  -id cifar100-vgg16-l8-example -dev 0 `
  --time_steps 1 2 4 8 16 32
```

配置含义：

- `F_FULL_FTBC`：完整逐时间步 FTBC；
- `G_STATE_LR`：对称损失的状态条件低秩 FTBC；
- `H_STATE_LR_SOPS`：提高过发放惩罚的最终低秩方案。

`T=1/2` 时低秩方案自动退化为完整 FTBC；`T>=4` 时使用 `3C` 参数的低秩 bias。

## 文档入口

- [实验协议](docs/methodology/experiment_protocol.md)
- [指标定义](docs/methodology/metric_definitions.md)
- [正式实验结果](docs/results/)
- [开发过程归档](docs/archive/)
