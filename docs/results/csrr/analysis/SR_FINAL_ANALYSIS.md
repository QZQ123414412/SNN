# 方向一最终实验分析

## 1. 最终方法

实验后的方向一收敛为：

**Credit-aware Signed Residual Refinement（信用约束有符号残差精化，CSRR）**

固定配置为：

```text
time-scale ratio = 1.0
positive margin = 0.55
negative margin = 1.30
refinement R0 = credit_only
T=1 = 自动回退 rate coding
组合状态低秩FTBC时 w_over = 2.5
```

神经元保留负残差，但只有在已有足够净正累计输出时才允许负脉冲撤销。正、负决策边界使用非对称 hysteresis，抑制频繁的正负来回修正。

`ratio=1.0` 表示最终方案不执行时间尺度乘法，因此：

```text
新增模型参数 = 0
新增FTBC存储 = 0
Time-scale operations = 0
```

几何递减时间权重仍保留为消融模式。实验表明它可以明显提高低时间步准确率，但会增加负脉冲、SOPs 和尺度处理开销，因此不作为最终配置。

## 2. CIFAR-100 / VGG16

### 方向一单独使用：L 相对 C

| T | Accuracy变化 | SOPs变化 | Sparsity变化 |
|---:|---:|---:|---:|
| 1 | +0.00 pp | +0.000% | +0.0000 pp |
| 2 | +1.84 pp | -10.995% | +1.1875 pp |
| 4 | +1.21 pp | -5.902% | +0.6133 pp |
| 8 | +0.53 pp | -3.458% | +0.3360 pp |
| 16 | +0.17 pp | -1.953% | +0.1816 pp |
| 32 | +0.19 pp | -1.101% | +0.0998 pp |

除无逐次过程的 `T=1` 自动回退外，方向一在所有时间步同时提高准确率、降低 SOPs 并提高稀疏度。

### 方向一与方向二组合：M 相对 H

| T | Accuracy变化 | SOPs变化 | Sparsity变化 |
|---:|---:|---:|---:|
| 1 | +0.00 pp | +0.000% | +0.0000 pp |
| 2 | +0.03 pp | -1.877% | +0.1859 pp |
| 4 | +0.27 pp | -0.657% | +0.1125 pp |
| 8 | +0.35 pp | -0.385% | +0.0681 pp |
| 16 | -0.01 pp | -0.652% | +0.0621 pp |
| 32 | -0.13 pp | -0.643% | +0.0515 pp |

组合方法在 `T=2/4/8` 同时改善三项核心指标；`T=16` 精度基本不变；`T=32` 精度下降 `0.13` 个百分点，但 SOPs 和稀疏度仍改善。

完整原始结果见：

- `SR_ABLATION_cifar100.md`

## 3. CIFAR-10 / VGG16

所有参数直接复用 CIFAR-100 的最终配置，没有重新调参。

### 方向一单独使用：L 相对 C

| T | Accuracy变化 | SOPs变化 | Sparsity变化 |
|---:|---:|---:|---:|
| 2 | +0.29 pp | -11.700% | +1.0966 pp |
| 4 | +0.27 pp | -6.301% | +0.5825 pp |
| 8 | +0.16 pp | -3.420% | +0.3117 pp |

### 方向一与方向二组合：M 相对 H

| T | Accuracy变化 | SOPs变化 | Sparsity变化 |
|---:|---:|---:|---:|
| 2 | +0.02 pp | -2.344% | +0.2343 pp |
| 4 | +0.30 pp | -1.715% | +0.2069 pp |
| 8 | +0.11 pp | -1.100% | +0.1309 pp |

方向一及组合方法均在三个时间步同时改善准确率、SOPs 和稀疏度，证明该机制能够跨数据集迁移。

完整原始结果见：

- `SR_ABLATION_cifar10_T2_T8.md`

## 4. 关键消融结论

1. `legacy_clamp R0` 会清除逐次精化需要保留的负残差，导致准确率严重下降。
2. `credit_only R0` 只禁止无信用负输出，不清除残差，是方向一成立的关键。
3. 几何时间尺度 `ratio>1` 能提高低时间步准确率，但负脉冲和 SOPs 明显增加。
4. `ratio=1.0` 消除了尺度操作，说明主要收益来自信用状态、残差记忆和非对称 hysteresis。
5. 较大的负 margin 抑制负脉冲撤销抖动；适度提高正 margin 进一步减少正脉冲。
6. 状态低秩 FTBC 能补偿 hysteresis 带来的欠发放，使组合方法形成更好的 Accuracy-SOPs Pareto 点。

## 5. 论文表述边界

可以声明：

- 方向一不增加模型参数和 FTBC 存储；
- 最终配置不增加时间尺度乘法；
- CIFAR-10/100 的多个固定时间步上取得一致 Accuracy-SOPs-sparsity 改善；
- 方向一与状态条件低秩 FTBC 可以叠加。

不能声明：

- 几何递减时间权重是最终有效配置；
- 所有时间步准确率均严格提高；
- PyTorch 延迟下降。当前 refinement 状态判断使软件推理延迟略有增加，硬件收益需要进一步实测。

## 6. 复现实验

Windows：

```powershell
$env:QCFS_DATA_ROOT="D:\master_degree_paper\datasets"
D:\Anaconda\envs\ann2snn\python.exe `
  scripts\experiments\run_successive_refinement_ablation.py `
  -data cifar100 `
  -id cifar100-vgg16-l8-example `
  -dev 0 `
  --time_steps 1 2 4 8 16 32 `
  --configs C_RATE_SNM_R0 H_RATE_STATE_LR `
            L_SR_GEOM_SNM_R0 M_SR_GEOM_STATE_LR `
  --output docs\results\csrr\final\cifar100\SR_ABLATION_cifar100.md
```

脚本默认值就是最终固定配置：

```text
ratio=1.0
positive_margin=0.55
negative_margin=1.30
refinement w_over=2.5
```

H 基线在代码中固定使用原方向二的 `w_over=2.0`，不会被方向一参数覆盖。
