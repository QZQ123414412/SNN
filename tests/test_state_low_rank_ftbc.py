# 验证状态条件低秩FTBC的bias、拟合和存储量
import unittest

import torch
import torch.nn as nn

from calibration import (
    accumulate_state_low_rank_statistics,
    solve_state_low_rank_coefficients,
)
from models.layer import SignedIF
from spike_stats import summarize_ftbc_storage


class StateLowRankFTBCTest(unittest.TestCase):
    def test_spike_statistics_can_be_disabled_for_latency_measurement(self):
        neuron = SignedIF(T=1)
        neuron.eval()
        neuron.set_collect_spike_stats(False)

        neuron(torch.ones(1, 1))

        self.assertEqual(neuron.get_stats()["pos_spike_count"], 0)
        self.assertEqual(neuron.get_stats()["total_neurons"], 0)

    def test_spike_statistics_are_reported_as_host_integers(self):
        neuron = SignedIF(T=1)
        neuron.eval()
        neuron.set_collect_spike_stats(True)

        neuron(torch.ones(1, 1))
        stats = neuron.get_stats()

        self.assertIsInstance(stats["pos_spike_count"], int)
        self.assertEqual(stats["pos_spike_count"], 1)
        self.assertEqual(stats["neg_spike_count"], 0)
        self.assertEqual(stats["total_neurons"], 1)

    def test_state_conditioned_bias_uses_time_and_transmitted_state(self):
        neuron = SignedIF(T=4)
        neuron.set_ftbc_mode("state_low_rank")
        neuron._init_ftbc_bias(channels=2, device=torch.device("cpu"))
        neuron.bias_base.copy_(torch.tensor([0.10, 0.20]))
        neuron.bias_slope.copy_(torch.tensor([0.30, 0.40]))
        neuron.bias_state.copy_(torch.tensor([-0.05, -0.10]))

        reference = torch.zeros(1, 2, 1, 1)
        transmitted = torch.tensor([[[[0.0]], [[1.0]]]])
        bias = neuron.get_ftbc_bias(t=2, reference=reference, transmitted=transmitted)

        expected = torch.tensor([[[[0.30]], [[0.36666667]]]])
        self.assertTrue(torch.allclose(bias, expected, atol=1e-6))

    def test_ridge_solver_recovers_known_channelwise_coefficients(self):
        tau = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
        state = torch.tensor(
            [
                [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]],
                [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]],
            ]
        ).transpose(0, 1)
        beta = torch.tensor([[0.10, 0.20, -0.05], [0.30, -0.10, 0.08]])
        target = torch.empty(1, 2, 6)
        for channel in range(2):
            target[0, channel] = (
                beta[channel, 0]
                + beta[channel, 1] * tau
                + beta[channel, 2] * state[0, channel]
            )

        xtx, xty = accumulate_state_low_rank_statistics(
            target=target,
            state=state,
            tau=tau,
        )
        estimated = solve_state_low_rank_coefficients(xtx, xty, ridge=1e-8)

        self.assertTrue(torch.allclose(estimated, beta, atol=1e-5))

    def test_low_rank_parameter_count_is_three_per_channel(self):
        neuron = SignedIF(T=32)
        neuron.set_ftbc_mode("state_low_rank")
        neuron._init_ftbc_bias(channels=64, device=torch.device("cpu"))

        self.assertEqual(neuron.ftbc_parameter_count(), 3 * 64)
        self.assertEqual(neuron.ftbc_storage_bytes(), 3 * 64 * 4)

    def test_model_storage_summary_combines_full_and_low_rank_layers(self):
        full = SignedIF(T=4)
        full.set_ftbc_mode("full")
        full._init_ftbc_bias(channels=2, device=torch.device("cpu"))
        low_rank = SignedIF(T=8)
        low_rank.set_ftbc_mode("state_low_rank")
        low_rank._init_ftbc_bias(channels=3, device=torch.device("cpu"))
        model = nn.Sequential(full, low_rank)

        summary = summarize_ftbc_storage(model, SignedIF)

        self.assertEqual(summary["parameters"], 4 * 2 + 3 * 3)
        self.assertEqual(summary["bytes"], (4 * 2 + 3 * 3) * 4)


if __name__ == "__main__":
    unittest.main()
