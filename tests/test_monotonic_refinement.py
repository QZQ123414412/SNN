# 验证单调有符号逐次精化编码及其模型集成
import unittest

import torch
import torch.nn as nn

from calibration import (
    accumulate_state_low_rank_statistics,
    solve_state_low_rank_coefficients,
    state_low_rank_corr_step_by_step,
)
from models.layer import SignedIF
from models.temporal_coding import make_event_scales, make_time_weights
from models.VGG import vgg16_signed


class TemporalCodingTest(unittest.TestCase):
    def test_uniform_weights_sum_to_one(self):
        weights = make_time_weights(4, mode="uniform")

        self.assertTrue(torch.allclose(weights, torch.full((4,), 0.25)))
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)

    def test_binary_weights_are_normalized_and_monotonic(self):
        weights = make_time_weights(4, mode="binary")

        expected = torch.tensor([8 / 15, 4 / 15, 2 / 15, 1 / 15])
        self.assertTrue(torch.allclose(weights, expected, atol=1e-6))
        self.assertTrue(torch.all(weights[:-1] >= weights[1:]))

    def test_geometric_weights_follow_requested_ratio(self):
        weights = make_time_weights(3, mode="geometric", ratio=1.5)

        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
        self.assertAlmostEqual((weights[0] / weights[1]).item(), 1.5, places=6)
        self.assertAlmostEqual((weights[1] / weights[2]).item(), 1.5, places=6)

    def test_custom_weights_are_validated_and_normalized(self):
        weights = make_time_weights(
            3,
            mode="custom",
            custom_weights=[4.0, 2.0, 1.0],
        )

        self.assertTrue(torch.allclose(weights, torch.tensor([4 / 7, 2 / 7, 1 / 7])))

    def test_increasing_custom_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            make_time_weights(3, mode="custom", custom_weights=[1.0, 2.0, 1.0])

    def test_event_scales_have_mean_one(self):
        scales = make_event_scales(4, mode="binary")

        self.assertAlmostEqual(scales.mean().item(), 1.0, places=6)
        self.assertTrue(
            torch.allclose(
                scales,
                torch.tensor([32 / 15, 16 / 15, 8 / 15, 4 / 15]),
            )
        )

    def test_single_step_always_uses_unit_weight_and_scale(self):
        self.assertTrue(torch.equal(make_time_weights(1, mode="binary"), torch.ones(1)))
        self.assertTrue(torch.equal(make_event_scales(1, mode="binary"), torch.ones(1)))


class SignedRefinementTest(unittest.TestCase):
    def test_binary_refinement_encodes_static_target_from_coarse_to_fine(self):
        neuron = SignedIF(T=4, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.is_input_layer = True
        neuron.set_coding_mode(
            "monotonic_refinement",
            schedule="binary",
            positive_margin=0.5,
            negative_margin=0.5,
        )
        neuron.eval()

        output = neuron(torch.full((4, 1), 0.4))

        self.assertGreater(output[0].item(), 0.0)
        self.assertLess(output[1].item(), 0.0)
        self.assertGreater(output[2].item(), 0.0)
        self.assertAlmostEqual(output.mean().item(), 0.4, places=5)

    def test_first_refinement_layer_consolidates_repeated_static_drive(self):
        neuron = SignedIF(T=4, thresh=1.0)
        neuron.set_ftbc_mode("none")
        neuron.is_input_layer = True
        neuron.set_coding_mode("monotonic_refinement", schedule="binary")

        sequence = torch.arange(4.0).view(4, 1, 1)
        prepared = neuron.prepare_temporal_input(sequence)

        self.assertTrue(torch.equal(prepared[0], torch.tensor([[6.0]])))
        self.assertTrue(torch.equal(prepared[1:], torch.zeros(3, 1, 1)))

    def test_non_input_layer_keeps_event_sequence_unchanged(self):
        neuron = SignedIF(T=4, thresh=1.0)
        neuron.set_coding_mode("monotonic_refinement", schedule="binary")
        sequence = torch.arange(4.0).view(4, 1, 1)

        prepared = neuron.prepare_temporal_input(sequence)

        self.assertTrue(torch.equal(prepared, sequence))

    def test_negative_event_is_blocked_when_credit_is_smaller_than_quantum(self):
        neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_coding_mode(
            "monotonic_refinement",
            schedule="custom",
            custom_weights=[0.75, 0.25],
        )
        mem = torch.tensor([[-1.0]])
        transmitted = torch.tensor([[0.25]])

        spike, new_mem, new_transmitted = neuron.refinement_step(
            input_t=torch.zeros_like(mem),
            t=1,
            mem=mem,
            transmitted=transmitted,
            bias=torch.zeros_like(mem),
        )

        self.assertTrue(torch.equal(spike, torch.zeros_like(spike)))
        self.assertTrue(torch.equal(new_mem, mem))
        self.assertTrue(torch.equal(new_transmitted, transmitted))

    def test_credit_only_r0_preserves_negative_residual(self):
        neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_coding_mode(
            "monotonic_refinement",
            schedule="uniform",
            r0_mode="credit_only",
        )

        spike, mem, transmitted = neuron.refinement_step(
            input_t=torch.tensor([[-1.0]]),
            t=0,
            mem=torch.zeros(1, 1),
            transmitted=torch.zeros(1, 1),
            bias=torch.zeros(1, 1),
        )

        self.assertEqual(spike.item(), 0.0)
        self.assertLess(mem.item(), 0.0)
        self.assertEqual(transmitted.item(), 0.0)

    def test_single_step_refinement_falls_back_to_rate_dynamics(self):
        rate = SignedIF(T=1, thresh=1.0, enable_signed=True, enable_r0=True)
        rate.set_ftbc_mode("none")
        refinement = SignedIF(T=1, thresh=1.0, enable_signed=True, enable_r0=True)
        refinement.set_ftbc_mode("none")
        refinement.is_input_layer = True
        refinement.set_coding_mode("monotonic_refinement", schedule="binary")

        inputs = torch.tensor([[0.55]])

        self.assertTrue(torch.equal(refinement(inputs), rate(inputs)))

    def test_step_by_step_replay_matches_full_forward(self):
        neuron = SignedIF(T=4, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_coding_mode("monotonic_refinement", schedule="binary")
        inputs = torch.tensor([[1.3], [-0.4], [0.2], [-0.1]])

        expected = neuron(inputs)
        expected_mem = neuron.mem.clone()
        expected_transmitted = neuron.transmitted.clone()

        sequence = inputs.view(4, 1, 1)
        mem = torch.zeros_like(sequence[0])
        transmitted = torch.zeros_like(sequence[0])
        replayed = []
        for t in range(4):
            spike, mem, transmitted = neuron.refinement_step(
                input_t=sequence[t],
                t=t,
                mem=mem,
                transmitted=transmitted,
                bias=torch.zeros_like(sequence[t]),
            )
            replayed.append(spike)

        self.assertTrue(torch.equal(torch.stack(replayed), expected.view(4, 1, 1)))
        self.assertTrue(torch.allclose(mem, expected_mem))
        self.assertTrue(torch.allclose(transmitted, expected_transmitted))


class RefinementModelTest(unittest.TestCase):
    def test_vgg_marks_only_first_signed_neuron_as_input_layer(self):
        model = vgg16_signed(num_classes=10)
        neurons = [module for module in model.modules() if isinstance(module, SignedIF)]

        self.assertTrue(neurons[0].is_input_layer)
        self.assertTrue(all(not neuron.is_input_layer for neuron in neurons[1:]))

    def test_model_coding_configuration_updates_every_signed_layer(self):
        model = vgg16_signed(num_classes=10)

        model.set_coding_mode(
            "monotonic_refinement",
            schedule="geometric",
            ratio=1.5,
            positive_margin=0.6,
            negative_margin=0.7,
        )

        neurons = [module for module in model.modules() if isinstance(module, SignedIF)]
        self.assertTrue(
            all(neuron.coding_mode == "monotonic_refinement" for neuron in neurons)
        )
        self.assertTrue(all(neuron.refinement_ratio == 1.5 for neuron in neurons))
        self.assertTrue(all(neuron.positive_margin == 0.6 for neuron in neurons))
        self.assertTrue(all(neuron.negative_margin == 0.7 for neuron in neurons))

    def test_set_time_steps_invalidates_cached_event_scales(self):
        model = vgg16_signed(num_classes=10)
        model.set_T(4)
        model.set_coding_mode("monotonic_refinement", schedule="binary")
        neuron = next(module for module in model.modules() if isinstance(module, SignedIF))
        neuron.get_event_scales(torch.zeros(1))
        self.assertIsNotNone(neuron.time_scales)

        model.set_T(2)

        self.assertIsNone(neuron.time_scales)

    def test_set_time_steps_resizes_per_time_statistics(self):
        model = vgg16_signed(num_classes=10)

        model.set_T(4)

        neurons = [module for module in model.modules() if isinstance(module, SignedIF)]
        self.assertTrue(
            all(len(neuron.pos_spike_count_by_time) == 4 for neuron in neurons)
        )
        self.assertTrue(
            all(len(neuron.neg_spike_count_by_time) == 4 for neuron in neurons)
        )

    def test_weighted_readout_only_uses_requested_schedule(self):
        model = vgg16_signed(num_classes=10)
        model.set_T(4)
        model.set_readout_mode("weighted", schedule="binary")
        logits = torch.arange(1.0, 5.0).view(4, 1, 1)

        reduced = model.aggregate_temporal_output(logits)
        weights = make_time_weights(4, mode="binary")

        self.assertTrue(torch.allclose(reduced, (logits[:, 0] * weights[:, None]).sum(0)))

    def test_rate_and_all_layer_refinement_keep_mean_readout(self):
        model = vgg16_signed(num_classes=10)
        model.set_T(4)
        logits = torch.arange(1.0, 5.0).view(4, 1, 1)

        self.assertTrue(torch.equal(model.aggregate_temporal_output(logits), logits.mean(0)))

        model.set_coding_mode("monotonic_refinement", schedule="binary")
        model.set_readout_mode("event_mean")
        self.assertTrue(torch.equal(model.aggregate_temporal_output(logits), logits.mean(0)))


class _ToyTemporalModel(nn.Module):
    def __init__(self, neuron):
        super().__init__()
        self.neuron = neuron
        self.T = neuron.T

    def forward(self, inputs):
        if self.T <= 0:
            return self.neuron(inputs)
        sequence = inputs.unsqueeze(0).repeat(self.T, 1, 1)
        output = self.neuron(sequence.flatten(0, 1))
        return output.view(self.T, inputs.shape[0], -1)


class RefinementCalibrationTest(unittest.TestCase):
    def test_state_low_rank_target_uses_weighted_cumulative_transmission(self):
        ann_neuron = SignedIF(T=0, thresh=1.0, enable_signed=False)
        ann_neuron.set_ftbc_mode("none")
        ann = _ToyTemporalModel(ann_neuron)

        snn_neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        snn_neuron.set_coding_mode(
            "monotonic_refinement",
            schedule="binary",
        )
        snn_neuron.set_ftbc_mode("state_low_rank")
        snn = _ToyTemporalModel(snn_neuron)
        inputs = torch.tensor([[0.6]])

        state_low_rank_corr_step_by_step(
            ann,
            ann_neuron,
            snn,
            snn_neuron,
            T=2,
            train_data=inputs,
            curr_t_alpha=1.0,
            ridge=1e-4,
        )

        target = torch.tensor([[[-0.6, (2 / 3) / 2 - 0.6]]])
        state = torch.zeros_like(target)
        xtx, xty = accumulate_state_low_rank_statistics(
            target=target,
            state=state,
            tau=torch.tensor([0.0, 1.0]),
        )
        expected = solve_state_low_rank_coefficients(xtx, xty, ridge=1e-4)[0]
        actual = torch.stack(
            [
                snn_neuron.bias_base[0],
                snn_neuron.bias_slope[0],
                snn_neuron.bias_state[0],
            ]
        )

        self.assertTrue(torch.allclose(actual, expected, atol=5e-5))


if __name__ == "__main__":
    unittest.main()
