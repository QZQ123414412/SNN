# 验证脉冲统计公式是否正确
import unittest

import torch
import torch.nn as nn

from models.layer import SignedIF
from spike_stats import (
    SpikeLayerStats,
    collect_signed_spike_stats,
    estimate_conv2d_fanout,
    estimate_linear_fanout,
    set_signed_spike_stats_enabled,
)


class SpikeStatsTest(unittest.TestCase):
    def test_model_level_switch_updates_all_signed_neurons(self):
        model = nn.Sequential(SignedIF(T=1), nn.Identity(), SignedIF(T=1))

        set_signed_spike_stats_enabled(model, SignedIF, False)

        self.assertFalse(model[0].collect_spike_stats)
        self.assertFalse(model[2].collect_spike_stats)

    def test_layer_rates_and_sparsity_use_signed_spike_counts(self):
        stats = SpikeLayerStats(
            name="layer1.2",
            kind="Conv2d",
            time_steps=4,
            output_neurons_per_step=100,
            positive_spikes=30,
            negative_spikes=10,
            input_positive_spikes=20,
            input_negative_spikes=5,
            synaptic_ops_per_input_spike=576,
        )

        self.assertAlmostEqual(stats.positive_spike_rate, 0.075)
        self.assertAlmostEqual(stats.negative_spike_rate, 0.025)
        self.assertAlmostEqual(stats.total_spike_rate, 0.1)
        self.assertAlmostEqual(stats.spike_sparsity, 0.9)
        self.assertEqual(stats.sops, 14400)

    def test_conv2d_fanout_uses_output_channels_and_kernel(self):
        self.assertEqual(
            estimate_conv2d_fanout(in_channels=3, out_channels=64, kernel_size=(3, 3), groups=1),
            576,
        )

    def test_linear_fanout_uses_output_features(self):
        self.assertEqual(estimate_linear_fanout(in_features=512, out_features=4096), 4096)

    def test_refinement_records_per_time_events_and_scale_operations(self):
        neuron = SignedIF(T=4, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.is_input_layer = True
        neuron.set_coding_mode("monotonic_refinement", schedule="binary")
        neuron.eval()

        neuron(torch.full((4, 1), 0.4))
        stats = neuron.get_stats()

        self.assertEqual(sum(stats["positive_spikes_by_time"]), stats["pos_spike_count"])
        self.assertEqual(sum(stats["negative_spikes_by_time"]), stats["neg_spike_count"])
        self.assertEqual(
            stats["scale_operations"],
            stats["pos_spike_count"] + stats["neg_spike_count"],
        )

    def test_collected_layer_stats_keep_scale_ops_separate_from_sops(self):
        model = nn.Sequential(
            nn.Linear(1, 1, bias=False),
            SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True),
        )
        model[0].weight.data.fill_(1.0)
        neuron = model[1]
        neuron.set_ftbc_mode("none")
        neuron.set_coding_mode("monotonic_refinement", schedule="binary")
        neuron.eval()
        model.eval()
        model(torch.tensor([[1.0], [0.0]]))

        stats = collect_signed_spike_stats(model, SignedIF, nn.Conv2d, nn.Linear)[0]

        self.assertGreater(stats.scale_operations, 0)
        self.assertEqual(stats.sops, 0)
        self.assertEqual(sum(stats.positive_spikes_by_time), stats.positive_spikes)


if __name__ == "__main__":
    unittest.main()
