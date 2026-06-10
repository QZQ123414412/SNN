# 统计量定义

本文档定义本项目在精度和时间步 `T` 之外新增的三个统计量：

1. Input-driven SOPs
2. 正/负脉冲率
3. 每层 spike sparsity

这些统计量用于分析 ANN-to-SNN 转换后的事件驱动计算量、脉冲活动强度和稀疏性。

## 1. Input-driven SOPs

SOPs 是 Synaptic Operations，即突触操作次数。在 SNN 中，计算通常由脉冲事件触发：上一层神经元发出脉冲后，该脉冲会通过突触连接影响下一层神经元。

本项目采用主流的 **input-driven SOPs** 口径：

```text
SOP_l = input_spikes_{l-1} * fanout_l
Total SOPs = sum_l SOP_l
```

其中：

```text
input_spikes_{l-1} = 上一层输出的正脉冲数 + 上一层输出的负脉冲数
```

也就是说，当前层的 SOPs 由输入到当前层的脉冲数量决定，而不是由当前层自己的输出脉冲数量决定。

### 卷积层

对于卷积层：

```text
fanout_l = out_channels / groups * kernel_h * kernel_w
SOP_l = input_spikes_{l-1} * fanout_l
```

例如，如果上一层输出了 `1,000,000` 个脉冲，当前卷积层有 `128` 个输出通道，卷积核大小为 `3 x 3`，并且 `groups=1`，则：

```text
fanout_l = 128 * 3 * 3 = 1152
SOP_l = 1,000,000 * 1152 = 1,152,000,000
```

### 全连接层

对于全连接层：

```text
fanout_l = out_features
SOP_l = input_spikes_{l-1} * out_features
```

### 第一层约定

当前实现没有把原始 RGB 图像输入建模为脉冲源。因此，第一个 spiking layer 前面的图像输入不计入 input-driven SOPs。

所以当前代码中第一层 `SignedIF` 的 SOPs 为：

```text
layer1.2 SOPs = 0
```

如果后续加入显式 input encoder，例如 Poisson encoder 或 direct spike encoder，则可以重新定义第一层输入脉冲，并计算第一层 SOPs。

### SignedIF 中的负脉冲

对于 signed spiking neurons，正脉冲和负脉冲都被视为事件传输。因此：

```text
input_spikes = positive_input_spikes + negative_input_spikes
```

负脉冲同样会增加 SOPs，因为它也会触发通信和突触操作。

## 2. 正/负脉冲率

对于每个 `SignedIF` 层，正脉冲率和负脉冲率定义为：

```text
positive_spike_rate_l = positive_spikes_l / total_neuron_time_slots_l
negative_spike_rate_l = negative_spikes_l / total_neuron_time_slots_l
```

其中：

```text
total_neuron_time_slots_l = T * total_batch_size * number_of_neurons_per_sample_l
```

对于卷积层激活：

```text
number_of_neurons_per_sample_l = channels * height * width
```

对于全连接层激活：

```text
number_of_neurons_per_sample_l = features
```

整体正/负脉冲率通过聚合所有 `SignedIF` 层的脉冲数和 neuron-time slots 得到。

## 3. 每层 Spike Sparsity

Spike sparsity 表示某一层在所有 neuron-time slot 中保持静默的比例。

先定义总脉冲率：

```text
total_spike_rate_l = (positive_spikes_l + negative_spikes_l) / total_neuron_time_slots_l
```

则每层 spike sparsity 为：

```text
spike_sparsity_l = 1 - total_spike_rate_l
```

对于 signed spiking neurons，负脉冲同样计入总脉冲率，因为负脉冲也是一次实际事件。

## 代码位置

相关实现位于：

- `spike_stats.py`
  - `SpikeLayerStats`
  - `collect_signed_spike_stats`
  - `estimate_conv2d_fanout`
  - `estimate_linear_fanout`
- `main_test_signed.py`
  - 在 validation 前重置统计量
  - 在 validation 后打印每层统计结果
- `run_stats_ablation.py`
  - 运行四配置统计消融实验

## 解释建议

这些统计量应与 accuracy 和 `T` 一起报告。只报告 `T` 不能充分说明低功耗或低计算量，因为相同 `T` 下，不同模型的 spike rate 和 SOPs 可能不同。

分析时可以采用以下逻辑：

1. 如果 SNM 提高精度但也提高 SOPs，说明 SNM 带来了精度-计算量权衡。
2. 如果 FTBC 提高精度同时降低 SOPs 或提高 sparsity，说明 FTBC 可能抑制了冗余脉冲活动。
3. 如果负脉冲率很低但精度提升明显，说明少量负脉冲可能起到了有效校正作用。
