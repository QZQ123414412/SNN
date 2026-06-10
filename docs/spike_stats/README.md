# Spike Statistics Documentation

This folder documents the three newly added physical/statistical quantities for
the ANN-to-SNN conversion experiments:

1. Input-driven SOPs
2. Positive / negative spike rate
3. Per-layer spike sparsity

The implementation is in:

- `spike_stats.py`
- `main_test_signed.py`
- `run_stats_ablation.py`

The current experiment branch is:

- `sops-spike-stats`

## Files

- `metric_definitions.md`: formal definitions, formulas, and implementation notes.
- `experiment_protocol.md`: how to run the four-configuration ablation experiment.
- `partial_cifar100_results.md`: currently collected CIFAR-100 / VGG16 results.

## Main Convention

SOPs are computed using the mainstream input-driven convention:

```text
SOP_l = input_spikes_{l-1} * fanout_l
```

For signed spiking neurons, both positive and negative spikes are counted as
events. The raw image input before the first spiking layer is not modeled as a
spike source, so the first spiking layer reports 0 input-driven SOPs unless an
explicit input encoder is added later.
