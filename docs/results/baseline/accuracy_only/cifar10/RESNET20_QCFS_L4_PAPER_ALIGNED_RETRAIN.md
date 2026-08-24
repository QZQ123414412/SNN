# CIFAR-10/ResNet20 QCFS-L4 论文口径重训记录

## 结论

本次训练完成了一个独立的 CIFAR-10/ResNet20 QCFS-\(L=4\) 模型。300 个 epoch 中的最高测试准确率为 **90.72%**，出现在零基 epoch 291（按自然计数为第 292 轮）。训练结束后重新构造模型、严格加载 checkpoint，并在完整 10,000 张 CIFAR-10 测试集上独立复测，仍得到 **90.72%**。

该结果比 QCFS 论文报告的 CIFAR-10/ResNet20、\(L=4\) ANN 准确率 91.77% 低 **1.05pp**。因此它适合作为当前仓库中的“论文口径对齐重训模型”，但不应宣称为论文数值的严格复现。

## 训练配置

| 项目 | 本次设置 |
|---|---|
| 数据集 / 网络 | CIFAR-10 / ResNet20 |
| QCFS 量化级数 | \(L=4\) |
| QCFS 训练 profile | `paper_era`（先量化，再裁剪） |
| batch size | 300 |
| epochs | 300 |
| 优化器 | SGD |
| momentum | 0.9 |
| weight decay | \(5\times10^{-4}\) |
| 初始学习率 | 0.1 |
| 学习率调度 | CosineAnnealingLR |
| 随机种子 | 42 |
| 数据增强 | RandomCrop、RandomHorizontalFlip、CIFAR10Policy、Cutout |
| checkpoint 选择 | 每轮测试，保存测试准确率最高者 |
| R0 / SNM / FTBC / A-SNM | 均不参与 ANN 训练 |

实际命令：

```powershell
D:\Anaconda\envs\ann2snn\python.exe -B scripts\train\main_train.py `
  --dataset cifar10 --model resnet20 --L 4 --batch_size 300 `
  --epochs 300 --lr 0.1 --weight_decay 0.0005 --seed 42 `
  --augmentation_profile fixed_repo --qcfs_training_profile paper_era `
  --device 0 --suffix paper_L4_bs300_seed42_testbest
```

## 结果与产物

| 项目 | 值 |
|---|---|
| 完成状态 | complete |
| 已完成 epochs | 300 |
| 最佳 epoch（零基） | 291 |
| 最佳 epoch（自然计数） | 292 |
| 训练内最佳测试准确率 | 90.72% |
| 严格重载测试准确率 | 90.72% |
| 严格重载测试样本数 | 10,000 |
| QCFS 激活层数 | 19 |
| checkpoint 大小 | 1,156,086 bytes |
| checkpoint SHA256 | `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3` |

产物：

- `cifar10-checkpoints/resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth`
- `cifar10-checkpoints/resnet20_L[4]_paper_L4_bs300_seed42_testbest.train_state.pth`
- `cifar10-checkpoints/resnet20_L[4]_paper_L4_bs300_seed42_testbest.log`
- `cifar10-checkpoints/resnet20_L[4]_paper_L4_bs300_seed42_testbest.metadata.json`

严格重载时没有使用 `strict=False`，没有缺失或多余参数，也没有执行旧版 `.up` 到 `.thresh` 的兼容转换。

## 与论文口径的关系

对齐项包括 CIFAR-10/ResNet20、\(L=4\)、300 epochs、SGD、momentum 0.9、weight decay \(5\times10^{-4}\)、初始学习率 0.1、余弦退火，以及 Crop、Flip、AutoAugment、Cutout 数据增强。

仍需保留两项限制：

1. QCFS 论文没有明确给出 batch size；这里使用作者更新后公开代码的默认值 300，所以不能证明它与论文原始训练完全相同。
2. checkpoint 是用测试集逐 epoch 选择的，存在测试集模型选择偏差。该做法用于匹配作者公开实现，不等价于使用独立验证集选择模型。

因此推荐名称为：

> CIFAR-10/ResNet20 QCFS-\(L=4\) 论文口径对齐重训模型

不推荐名称为：

> QCFS 论文 CIFAR-10/ResNet20 主实验的严格复现模型

## 下游消融

该 checkpoint 的 Full-FTBC + A-SNM 六组正式消融见：

- `docs/results/comparative_ablation/cifar10/FULL_FTBC_ASNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.md`
