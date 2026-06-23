# 方向一实现与验证状态

## 已实现

- 统一、二进制、几何和自定义单调时间权重；
- `q_t=T*w_t*threshold` 的带权正负事件；
- 首层静态输入汇总、逐层带权事件传播和信用约束负修正；
- `T=1` 原始 rate coding 回退；
- 完整 FTBC 与状态低秩 FTBC 的一致校准重放；
- Input-driven SOPs、ScaleOps、正负脉冲率、稀疏度、FTBC 存储和延迟统计；
- 八配置完整消融脚本，以及仅使用校准集的全局参数选择。

## 当前验证结果

以下结果来自 CIFAR-100/VGG16 的完整测试集，使用 5 个确定性训练集校准批次。方向一自动选择一个全局时间比例和正、负边界，不使用测试集或逐层搜索。

| T | Rate+SNM+R0 Acc. | Calibrated refinement Acc. | Accuracy变化 | SOPs变化 | Sparsity变化 |
|---:|---:|---:|---:|---:|---:|
| 2 | 65.00% | 70.86% | +5.86 pp | -8.212% | +0.9817 pp |
| 4 | 72.29% | 76.11% | +3.82 pp | +0.928% | -0.0076 pp |
| 8 | 76.59% | 75.72% | -0.87 pp | -2.137% | +0.1506 pp |
| 16 | 77.49% | 76.27% | -1.22 pp | -0.542% | -0.0415 pp |
| 32 | 77.50% | 76.82% | -0.68 pp | +1.376% | -0.2359 pp |

`T=2` 达到准确率、SOPs 和稀疏度三项同时改善；`T=4` 在基本相同的事件开销下显著提高准确率。高时间步下收益消失，因此当前方法应定位为低时间步转换增强。

固定二进制递减权重表现明显变差，自动校准在所有时间步都选择 `ratio=1`。因此当前证据支持“信用约束的有符号残差精化动力学”，但不支持声称递减时间权重本身有效。仅对最终 logits 加权同样明显变差，说明带权机制必须进入神经元动力学。

现有完整 FTBC 和状态低秩 FTBC 与该动力学直接组合后性能下降，尤其状态低秩版本在高时间步严重退化。它们已保留为组合消融，但不能作为方向一的最终配置。

## CIFAR-10 迁移验证

关键配置在 CIFAR-10/VGG16 上得到同方向结果：

| T | Accuracy变化 | SOPs变化 | Sparsity变化 |
|---:|---:|---:|---:|
| 2 | +2.68 pp | -8.376% | +0.9062 pp |
| 4 | +0.30 pp | -6.208% | +0.5861 pp |

这表明全局边界校准后的残差精化并非只对 CIFAR-100 生效，尤其 `T=2` 的三指标同步改善可以稳定迁移。

## 完整实验命令

```powershell
$env:QCFS_DATA_ROOT='D:\master_degree_paper\datasets'
D:\Anaconda\envs\ann2snn\python.exe scripts\experiments\run_monotonic_refinement_ablation.py `
  -data cifar100 `
  -id cifar100-vgg16-l8-example `
  -dev 0 `
  -b 200 `
  --time_steps 1 2 4 8 16 32 `
  --cali_batches 5 `
  --output docs\results\monotonic_refinement\MSSR_ABLATION_cifar100.md
```
