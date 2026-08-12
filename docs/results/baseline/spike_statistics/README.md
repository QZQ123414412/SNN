# 脉冲统计正式结果

This folder documents the three newly added physical/statistical quantities for
the ANN-to-SNN conversion experiments:

1. Input-driven SOPs
2. Positive / negative spike rate
3. Per-layer spike sparsity

实现位置：

- `spike_stats.py`
- `scripts/evaluate/main_test_signed.py`
- `scripts/experiments/run_stats_ablation.py`

The current experiment branch is:

- `sops-spike-stats`

## 相关文档

- `../../../methodology/metric_definitions.md`：指标公式和实现约定；
- `../../../methodology/experiment_protocol.md`：六配置实验运行协议；
- `cifar100/ABLATION_RESULTS_VGG_cifar100_add_3_quantity_v2.md`：CIFAR-100/VGG16 正式结果。

## Main Convention

SOPs are computed using the mainstream input-driven convention:

```text
SOP_l = input_spikes_{l-1} * fanout_l
```

For signed spiking neurons, both positive and negative spikes are counted as
events. The raw image input before the first spiking layer is not modeled as a
spike source, so the first spiking layer reports 0 input-driven SOPs unless an
explicit input encoder is added later.
