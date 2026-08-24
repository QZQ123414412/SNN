# 验证脉冲统计公式是否正确
import unittest

import torch
import torch.nn as nn

from models.layer import SignedIF
from scripts.experiments.run_stats_ablation import summarize_layer_stats
from spike_stats import (
    SpikeLayerStats,
    collect_resnet34_spike_stats,
    collect_resnet20_spike_stats,
    estimate_conv2d_fanout,
    estimate_linear_fanout,
    set_signed_spike_stats_enabled,
)
from models import modelpool


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

    def test_rate_coding_has_no_time_scale_operations(self):
        stats = SpikeLayerStats(
            name="layer1.2",
            kind="Conv2d",
            time_steps=2,
            output_neurons_per_step=10,
            positive_spikes=4,
            negative_spikes=0,
            input_positive_spikes=5,
            input_negative_spikes=0,
            synaptic_ops_per_input_spike=10,
            input_spikes_by_time=(2, 3),
            time_scales=(1.0, 1.0),
        )

        self.assertEqual(stats.scale_operations, 0)

    def test_refinement_counts_scaled_input_driven_synaptic_operations(self):
        stats = SpikeLayerStats(
            name="layer1.2",
            kind="Conv2d",
            time_steps=2,
            output_neurons_per_step=10,
            positive_spikes=4,
            negative_spikes=0,
            input_positive_spikes=5,
            input_negative_spikes=0,
            synaptic_ops_per_input_spike=10,
            input_spikes_by_time=(2, 3),
            time_scales=(4 / 3, 2 / 3),
        )

        self.assertEqual(stats.scale_operations, 50)

    def test_signed_neuron_reports_per_time_step_event_counts(self):
        neuron = SignedIF(T=2, thresh=1.0)
        neuron.set_ftbc_mode("none")
        neuron.eval()

        neuron(torch.tensor([[0.25], [0.25]]))
        stats = neuron.get_stats()

        self.assertEqual(stats["positive_spikes_by_time"], (0, 1))
        self.assertEqual(stats["negative_spikes_by_time"], (0, 0))

    def test_summary_reports_scale_operations_separately_from_sops(self):
        stats = SpikeLayerStats(
            name="layer1.2",
            kind="Conv2d",
            time_steps=2,
            output_neurons_per_step=10,
            positive_spikes=4,
            negative_spikes=0,
            input_positive_spikes=5,
            input_negative_spikes=0,
            synaptic_ops_per_input_spike=10,
            input_spikes_by_time=(2, 3),
            time_scales=(4 / 3, 2 / 3),
        )

        summary = summarize_layer_stats([stats])

        self.assertEqual(summary["sops"], 50)
        self.assertEqual(summary["scale_operations"], 50)

    def test_resnet_projection_shortcut_sops_are_counted_separately(self):
        model = modelpool("resnet20_signed", "cifar100")
        model.set_T(2)
        for module in model.modules():
            if isinstance(module, SignedIF):
                module.pos_spike_count = 1
                module.neg_spike_count = 0
                module.total_neurons = 2
                module.pos_spike_count_by_time = [1, 0]
                module.neg_spike_count_by_time = [0, 0]

        stats = collect_resnet20_spike_stats(model, SignedIF, nn.Conv2d)
        by_name = {item.name: item for item in stats}

        self.assertEqual(len(stats), 20)
        self.assertEqual(
            by_name["conv3_x.0.residual_function.2"].sops,
            32 * 3 * 3,
        )
        self.assertEqual(
            by_name["conv3_x.0.act"].sops,
            32 * 3 * 3 + 32,
        )
        self.assertEqual(by_name["conv3_x.0.act"].total_input_spikes, 2)
        self.assertEqual(by_name["fc"].sops, 100)
        self.assertFalse(by_name["fc"].has_spike_output)
        summary = summarize_layer_stats(stats)
        self.assertEqual(summary["positive_spikes"], 19)

    def test_resnet34_graph_aware_sops_cover_all_residual_stages(self):
        model = modelpool("resnet34_signed", "imagenet")
        model.set_T(2)
        for module in model.modules():
            if isinstance(module, SignedIF):
                module.pos_spike_count = 1
                module.neg_spike_count = 0
                module.total_neurons = 2
                module.pos_spike_count_by_time = [1, 0]
                module.neg_spike_count_by_time = [0, 0]

        stats = collect_resnet34_spike_stats(model, SignedIF, nn.Conv2d)
        by_name = {item.name: item for item in stats}

        self.assertEqual(len(stats), 34)
        self.assertIn("conv5_x.2.act", by_name)
        self.assertEqual(
            by_name["conv3_x.0.act"].sops,
            128 * 3 * 3 + 128,
        )
        self.assertEqual(by_name["fc"].sops, 1_000)
        self.assertFalse(by_name["fc"].has_spike_output)


if __name__ == "__main__":
    unittest.main()
