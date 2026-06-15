# Partial CIFAR-100 / VGG16 Results

This document records the currently collected results for the four-configuration
statistics ablation. The full run was interrupted during `C_QCFS+FTBC, T=64`,
so `D_QCFS+SNM+FTBC` has not been collected yet.

## Summary Table

| Config | T | Accuracy | Input-driven SOPs | PosRate | NegRate | Sparsity | Elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_QCFS | 1 | 58.82% | 624,247,119,488 | 14.5549% | 0.0000% | 85.4451% | 8.7s |
| A_QCFS | 2 | 64.83% | 1,303,733,907,712 | 15.1460% | 0.0000% | 84.8540% | 9.3s |
| A_QCFS | 4 | 70.55% | 2,582,967,259,264 | 15.1349% | 0.0000% | 84.8651% | 10.9s |
| A_QCFS | 8 | 74.67% | 5,119,241,360,192 | 15.0617% | 0.0000% | 84.9383% | 13.9s |
| A_QCFS | 16 | 76.68% | 10,197,099,598,272 | 15.0226% | 0.0000% | 84.9774% | 20.3s |
| A_QCFS | 32 | 77.58% | 20,359,590,690,368 | 15.0046% | 0.0000% | 84.9954% | 32.8s |
| A_QCFS | 64 | 77.68% | 40,687,305,414,656 | 14.9954% | 0.0000% | 85.0046% | 2177.3s |
| B_QCFS+SNM | 1 | 58.82% | 624,247,119,488 | 14.5549% | 0.0000% | 85.4451% | 9.9s |
| B_QCFS+SNM | 2 | 65.19% | 1,306,088,612,992 | 15.1573% | 0.0103% | 84.8323% | 10.5s |
| B_QCFS+SNM | 4 | 71.84% | 2,599,286,893,056 | 15.1528% | 0.0311% | 84.8161% | 12.4s |
| B_QCFS+SNM | 8 | 75.79% | 5,161,337,904,192 | 15.0765% | 0.0408% | 84.8827% | 16.5s |
| B_QCFS+SNM | 16 | 77.50% | 10,260,982,503,616 | 15.0312% | 0.0324% | 84.9364% | 23.5s |
| B_QCFS+SNM | 32 | 77.70% | 20,439,573,249,728 | 15.0100% | 0.0205% | 84.9695% | 39.0s |
| B_QCFS+SNM | 64 | 77.67% | 40,784,066,776,064 | 14.9993% | 0.0121% | 84.9886% | 2476.6s |
| C_QCFS+FTBC | 1 | 61.72% | 641,613,176,704 | 14.9624% | 0.0000% | 85.0376% | 31.9s |
| C_QCFS+FTBC | 2 | 67.61% | 1,257,782,968,704 | 14.8316% | 0.0000% | 85.1684% | 31.7s |
| C_QCFS+FTBC | 4 | 72.75% | 2,433,483,599,168 | 14.4922% | 0.0000% | 85.5078% | 36.2s |
| C_QCFS+FTBC | 8 | 76.11% | 4,747,826,399,360 | 14.2536% | 0.0000% | 85.7464% | 45.6s |
| C_QCFS+FTBC | 16 | 77.49% | 9,371,303,832,832 | 14.1040% | 0.0000% | 85.8960% | 63.1s |
| C_QCFS+FTBC | 32 | 77.79% | 18,604,856,992,000 | 14.0327% | 0.0000% | 85.9673% | 98.4s |

## Initial Observations

1. SOPs grow approximately linearly with T for `A_QCFS` and `B_QCFS+SNM`.
2. SNM introduces a small but measurable negative spike rate for T >= 2.
3. SNM slightly increases SOPs compared with QCFS because negative spikes are
   counted as events.
4. FTBC improves accuracy and, for T >= 2, reduces SOPs compared with QCFS.
5. FTBC also increases sparsity, suggesting that temporal calibration suppresses
   redundant spike activity.

## Remaining Runs

The following results are still missing:

```text
C_QCFS+FTBC, T=64
D_QCFS+SNM+FTBC, T=1,2,4,8,16,32,64
```

## Input-driven SOPs 计算过程

表中的 `Input-driven SOPs` 不是由 `PosRate` 或 `Sparsity` 直接反推得到的，而是在推理过程中逐层统计脉冲数后累加得到的。

当前采用的主流口径是：

```text
SOP_l = input_spikes_{l-1} * fanout_l
Total SOPs = sum_l SOP_l
```

其中：

```text
input_spikes_{l-1} = 上一层 SignedIF 输出的正脉冲数 + 负脉冲数
```

也就是说，当前层的 SOPs 由输入到当前层的脉冲事件数决定，而不是由当前层自己的输出脉冲数决定。

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

### VGG16 中的逐层累加方式

在 PyTorch 的 `nn.Sequential` 中，`layer1.6` 不是数学意义上的第 1.6 层，而是 `layer1` 这个 Sequential 中索引为 6 的模块。这里统计的是 `SignedIF` 激活层对应的事件活动。

例如：

```text
layer1.6 SOPs = spikes(layer1.2) * fanout(layer1.4 Conv2d)
layer2.2 SOPs = spikes(layer1.6) * fanout(layer2.0 Conv2d)
layer2.6 SOPs = spikes(layer2.2) * fanout(layer2.4 Conv2d)
...
classifier.2 SOPs = spikes(layer5.10) * fanout(classifier.1 Linear)
classifier.5 SOPs = spikes(classifier.2) * fanout(classifier.4 Linear)
```

其中：

```text
spikes(layer) = positive_spikes(layer) + negative_spikes(layer)
```

### 第一层约定

当前实现没有把原始 RGB 图像输入建模为 spike source。因此第一个 `SignedIF` 层 `layer1.2` 的 input-driven SOPs 记为 0：

```text
layer1.2 SOPs = 0
```

所以表中的总 SOPs 实际为：

```text
Total SOPs = layer1.6 SOPs
           + layer2.2 SOPs
           + layer2.6 SOPs
           + ...
           + classifier.5 SOPs
```

### SNM 配置中的负脉冲

对于 `B_QCFS+SNM` 和后续 signed spike 配置，负脉冲也计入 `input_spikes`：

```text
input_spikes = positive_spikes + negative_spikes
```

原因是负脉冲同样是一次事件传输，也会触发后续突触操作。因此，SNM 可能提升精度，但也可能略微增加 SOPs。

### 代码实现位置

相关实现位于：

```text
spike_stats.py
  estimate_conv2d_fanout()
  estimate_linear_fanout()
  collect_signed_spike_stats()
  SpikeLayerStats.sops
```
