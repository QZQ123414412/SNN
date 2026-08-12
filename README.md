# QCFS ANN-to-SNN Conversion

本仓库基于 QCFS 实现 ANN-to-SNN 转换，并扩展了 Signed Neuron Model（SNM）、R0、FTBC、状态条件低秩 FTBC，以及 SOPs、正负脉冲率和脉冲稀疏率统计。

## 目录

```text
QCFS/
├─ models/                  模型和脉冲神经元实现
├─ preprocess/              数据加载与增强
├─ scripts/
│  ├─ train/                训练入口
│  ├─ evaluate/             单模型评估入口
│  └─ experiments/          消融和统计实验
├─ tests/                   单元测试
├─ docs/
│  ├─ methodology/          指标定义和实验协议
│  ├─ results/              正式实验结果
│  ├─ design/               方法设计
│  └─ archive/              开发过程和失败尝试
├─ figures/                 论文和实验图
├─ calibration.py           FTBC 校准方法
├─ spike_stats.py           脉冲和 SOPs 统计
└─ utils.py                 训练与验证工具
```

详细目录说明见 [docs/README.md](docs/README.md) 和 [scripts/README.md](scripts/README.md)。

## 常用命令

```powershell
# 训练 QCFS ANN
python scripts/train/main_train.py --epochs 300 -dev 0 -L 8 -data cifar100

# 测试原始 IF/QCFS 模型
python scripts/evaluate/main_test.py -data cifar100 -arch vgg16 `
  -id cifar100-vgg16-l8-example -T 8 -dev 0

# 测试 SignedIF 并输出脉冲统计
python scripts/evaluate/main_test_signed.py -data cifar100 `
  -arch vgg16_signed -id cifar100-vgg16-l8-example -T 8 -dev 0

# 六配置模块消融
python scripts/experiments/run_stats_ablation.py -data cifar100 `
  -id cifar100-vgg16-l8-example -dev 0

# 完整 FTBC 与状态条件低秩 FTBC 对比
python scripts/experiments/run_state_ftbc_ablation.py -data cifar100 `
  -id cifar100-vgg16-l8-example -dev 0
```

## 当前推荐结果

状态条件低秩 FTBC 的正式结果位于：

- [最终结论](docs/results/state_low_rank_ftbc/final/STATE_LOW_RANK_FTBC_FINAL.md)
- [完整 CIFAR-100/VGG16 数据](docs/results/state_low_rank_ftbc/final/cifar100/STATE_LOW_RANK_FTBC_cifar100_full.md)

SOPs、脉冲率和稀疏率的定义见 [指标定义](docs/methodology/metric_definitions.md)。

## Checkpoint

checkpoint 放在 `<dataset>-checkpoints/` 中，例如：

```text
cifar100-checkpoints/cifar100-vgg16-l8-example.pth
```

这些目录和模型权重已由 `.gitignore` 排除，不会提交到 Git。
