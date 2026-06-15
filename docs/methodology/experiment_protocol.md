# 实验协议

本文档说明如何针对新增的三个统计量运行四配置消融实验。

新增统计量包括：

```text
Input-driven SOPs
正/负脉冲率
每层 spike sparsity
Elapsed wall-clock runtime
```

其中 `Elapsed` 是本地命令运行耗时，用于记录实验成本，但不等同于真实硬件推理延迟。

## 1. 消融配置

实验采用与论文图表一致的四种配置：

```text
A_QCFS
B_QCFS+SNM
C_QCFS+FTBC
D_QCFS+SNM+FTBC
```

对应关系如下：

| Config | Signed spikes | FTBC |
|---|---:|---:|
| A_QCFS | 否 | 否 |
| B_QCFS+SNM | 是 | 否 |
| C_QCFS+FTBC | 否 | 是 |
| D_QCFS+SNM+FTBC | 是 | 是 |

注意：该协议不包含 R0。原因是论文主线和图中展示的是 QCFS、SNM、FTBC 及其组合，R0 可以作为附录或额外消融。

## 2. 运行环境

当前可用环境为：

```text
D:\Anaconda\envs\ann2snn\python.exe
```

该环境包含 PyTorch 和 torchvision，并支持 CUDA。

## 3. 完整运行命令

在仓库根目录下运行：

```powershell
D:\Anaconda\envs\ann2snn\python.exe scripts\experiments\run_stats_ablation.py `
  -data=cifar100 `
  -id=cifar100-vgg16-l8-example `
  -dev=0 `
  --time_steps 1 2 4 8 16 32 `
  --cali_batches 5 `
  --output docs\results\spike_stats\STATS_ABLATION_cifar100.md
```

其中：

```text
-data                  数据集名称
-id                    checkpoint 文件名，不包含 .pth
-dev                   GPU 编号
--time_steps           要测试的 SNN 时间步
--cali_batches         FTBC 校准使用的 batch 数
--output               输出 Markdown 文件
```

## 4. 建议的续跑策略

完整实验耗时较长，尤其是高时间步和带 FTBC 的配置。为了避免中途超时导致结果丢失，可以分批运行。

例如，只运行 `T=32`：

```powershell
D:\Anaconda\envs\ann2snn\python.exe scripts\experiments\run_stats_ablation.py `
  -data=cifar100 `
  -id=cifar100-vgg16-l8-example `
  -dev=0 `
  --time_steps 32 `
  --cali_batches 5 `
  --output docs\results\spike_stats\STATS_ABLATION_cifar100_T32.md
```

后续可以进一步改进脚本，让它在每完成一个 `(config, T)` 后立即写入 CSV，这样即使实验被中断也不会丢失已有结果。

## 5. 输出指标

每个配置和每个时间步会输出：

```text
Accuracy
Input-driven SOPs
Positive spike rate
Negative spike rate
Overall spike sparsity
Per-layer spike sparsity
Elapsed wall-clock runtime
```

其中：

- `Accuracy`：分类准确率。
- `Input-driven SOPs`：按上一层输入脉冲触发当前层突触操作的主流口径计算。
- `Positive spike rate`：正脉冲率。
- `Negative spike rate`：负脉冲率。
- `Overall spike sparsity`：整体静默比例。
- `Per-layer spike sparsity`：每一层的静默比例。
- `Elapsed`：本地实验耗时。

## 6. 重要约定

1. FTBC 校准阶段会触发 SNN forward。脚本在 FTBC 校准后、正式 validation 前会重置脉冲统计量，因此报告中的 spike rate、sparsity 和 SOPs 不包含校准阶段的脉冲。
2. 原始图像输入没有被建模为 spike source，因此第一个 spiking layer 的 input-driven SOPs 记为 0。
3. 负脉冲被计入 spike rate 和 SOPs，因为负脉冲也是事件传输。
4. `Elapsed` 是本地 wall-clock runtime，包含框架开销、数据加载开销，以及启用 FTBC 时的校准开销。因此它不能直接作为硬件推理延迟指标。

## 7. 论文写作建议

建议在论文中同时报告：

```text
Accuracy
T
Input-driven SOPs
Positive / negative spike rate
Spike sparsity
```

这样可以更完整地说明方法在低时间步下是否真正具有低计算量和低功耗潜力。
