# 验证脉冲统计公式是否正确
import unittest

from spike_stats import SpikeLayerStats, estimate_conv2d_fanout, estimate_linear_fanout


class SpikeStatsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
