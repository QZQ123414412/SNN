# 验证逐次精化时间尺度、神经元状态和实验配置
import unittest

import torch
import torch.nn as nn

from calibration import (
    accumulate_state_low_rank_statistics,
    solve_state_low_rank_coefficients,
    state_low_rank_corr_step_by_step,
)
from models.layer import SignedIF
from models.temporal_coding import make_time_scales
from models.VGG import vgg16_signed
from scripts.experiments.run_successive_refinement_ablation import (
    BASE_CONFIGS,
    FINAL_NEGATIVE_MARGIN,
    FINAL_OVER_WEIGHT,
    FINAL_POSITIVE_MARGIN,
    FINAL_TIME_SCALE_RATIO,
    effective_ftbc_mode,
    expand_configurations,
)


class TemporalCodingTest(unittest.TestCase):
    def test_rate_schedule_is_all_ones(self):
        scales = make_time_scales(4, mode="rate")

        self.assertTrue(torch.equal(scales, torch.ones(4)))

    def test_geometric_schedule_is_positive_monotonic_and_mean_one(self):
        scales = make_time_scales(4, mode="geometric", ratio=2.0)

        self.assertTrue(torch.all(scales > 0))
        self.assertTrue(torch.all(scales[:-1] >= scales[1:]))
        self.assertAlmostEqual(scales.mean().item(), 1.0, places=6)
        expected = torch.tensor([32 / 15, 16 / 15, 8 / 15, 4 / 15])
        self.assertTrue(torch.allclose(scales, expected, atol=1e-6))

    def test_single_step_schedule_is_one(self):
        scales = make_time_scales(1, mode="geometric", ratio=2.0)

        self.assertTrue(torch.equal(scales, torch.ones(1)))

    def test_geometric_ratio_must_not_be_below_one(self):
        with self.assertRaisesRegex(ValueError, "ratio"):
            make_time_scales(4, mode="geometric", ratio=0.9)

    def test_time_steps_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "time_steps"):
            make_time_scales(0, mode="rate")


class SignedSuccessiveRefinementTest(unittest.TestCase):
    def test_rate_mode_preserves_legacy_dynamics(self):
        neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.eval()

        output = neuron(torch.tensor([[0.25], [0.25]]))

        self.assertTrue(torch.equal(output, torch.tensor([[0.0], [1.0]])))

    def test_single_step_refinement_matches_rate_output(self):
        rate = SignedIF(T=1, thresh=1.0)
        rate.set_ftbc_mode("none")
        refinement = SignedIF(T=1, thresh=1.0)
        refinement.set_ftbc_mode("none")
        refinement.set_coding_mode(
            "successive_refinement",
            ratio=2.0,
            positive_margin=0.6,
        )

        inputs = torch.tensor([[0.55]])

        self.assertTrue(torch.equal(refinement(inputs), rate(inputs)))

    def test_refinement_can_emit_positive_then_negative_correction(self):
        neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_coding_mode("successive_refinement", ratio=2.0)

        output = neuron(torch.tensor([[1.0], [-1.0]]))

        self.assertTrue(torch.equal(output, torch.tensor([[1.0], [-1.0]])))
        self.assertGreaterEqual(neuron.transmitted.item(), 0.0)

    def test_refinement_blocks_negative_correction_larger_than_credit(self):
        neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_coding_mode("successive_refinement", ratio=1.0)

        output = neuron(torch.tensor([[0.6], [-2.0]]))

        self.assertEqual(output[0].item(), 1.0)
        self.assertEqual(output[1].item(), -1.0)
        self.assertAlmostEqual(neuron.transmitted.item(), 0.0, places=6)

    def test_decode_transmitted_uses_cumulative_time_scale(self):
        neuron = SignedIF(T=4, thresh=1.0)
        neuron.set_coding_mode("successive_refinement", ratio=2.0)
        scales = make_time_scales(4, mode="geometric", ratio=2.0)
        transmitted = torch.tensor([[scales[0].item()]])

        decoded = neuron.decode_transmitted(transmitted, t=0)

        self.assertTrue(torch.allclose(decoded, torch.ones_like(decoded)))

    def test_refinement_r0_preserves_residual_but_blocks_negative_output(self):
        neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_coding_mode("successive_refinement", ratio=2.0)

        output = neuron(torch.tensor([[0.0], [-1.0]]))

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertLess(neuron.mem.item(), 0.0)
        self.assertTrue(
            torch.equal(neuron.transmitted, torch.zeros_like(neuron.transmitted))
        )

    def test_legacy_refinement_r0_clamps_negative_membrane(self):
        neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_coding_mode(
            "successive_refinement",
            ratio=2.0,
            r0_mode="legacy_clamp",
        )

        output = neuron(torch.tensor([[0.0], [-1.0]]))

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertTrue(torch.equal(neuron.mem, torch.zeros_like(neuron.mem)))

    def test_larger_negative_margin_suppresses_small_reversal(self):
        default = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        default.set_ftbc_mode("none")
        default.set_coding_mode(
            "successive_refinement",
            ratio=2.0,
            negative_margin=0.5,
        )
        hysteretic = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        hysteretic.set_ftbc_mode("none")
        hysteretic.set_coding_mode(
            "successive_refinement",
            ratio=2.0,
            negative_margin=0.75,
        )
        inputs = torch.tensor([[1.0], [-0.6]])

        default_output = default(inputs)
        hysteretic_output = hysteretic(inputs)

        self.assertEqual(default_output[1].item(), -1.0)
        self.assertEqual(hysteretic_output[1].item(), 0.0)

    def test_refinement_step_replay_matches_full_forward(self):
        neuron = SignedIF(T=3, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_coding_mode("successive_refinement", ratio=1.5)
        inputs = torch.tensor([[0.8], [-0.2], [0.4]])

        expected = neuron(inputs)
        expected_mem = neuron.mem.clone()
        expected_transmitted = neuron.transmitted.clone()

        sequence = inputs.view(3, 1, 1)
        mem = torch.zeros_like(sequence[0])
        transmitted = torch.zeros_like(sequence[0])
        replayed = []
        for t in range(neuron.T):
            spike, mem, transmitted = neuron.refinement_step(
                sequence[t],
                t,
                mem,
                transmitted,
                bias=torch.zeros_like(sequence[t]),
            )
            replayed.append(spike)

        self.assertTrue(torch.equal(torch.stack(replayed), expected.view(3, 1, 1)))
        self.assertTrue(torch.allclose(mem, expected_mem))
        self.assertTrue(torch.allclose(transmitted, expected_transmitted))


class SuccessiveRefinementVGGTest(unittest.TestCase):
    def test_model_coding_mode_updates_every_signed_neuron(self):
        model = vgg16_signed(num_classes=10)

        model.set_coding_mode(
            "successive_refinement",
            schedule="geometric",
            ratio=1.25,
        )

        neurons = [module for module in model.modules() if isinstance(module, SignedIF)]
        self.assertGreater(len(neurons), 0)
        self.assertTrue(
            all(module.coding_mode == "successive_refinement" for module in neurons)
        )
        self.assertTrue(
            all(module.refinement_ratio == 1.25 for module in neurons)
        )

    def test_set_time_steps_invalidates_neuron_scale_cache(self):
        model = vgg16_signed(num_classes=10)
        model.set_T(4)
        model.set_coding_mode("successive_refinement", ratio=2.0)
        neuron = next(
            module for module in model.modules() if isinstance(module, SignedIF)
        )
        neuron.get_time_scales(torch.zeros(1))
        self.assertIsNotNone(neuron.time_scales)

        model.set_T(2)

        self.assertIsNone(neuron.time_scales)

    def test_set_time_steps_resizes_per_time_statistics(self):
        model = vgg16_signed(num_classes=10)

        model.set_T(4)

        neurons = [module for module in model.modules() if isinstance(module, SignedIF)]
        self.assertTrue(
            all(len(module.pos_spike_count_by_time) == 4 for module in neurons)
        )
        self.assertTrue(
            all(len(module.neg_spike_count_by_time) == 4 for module in neurons)
        )

    def test_refinement_readout_scales_temporal_logits_with_mean_one(self):
        model = vgg16_signed(num_classes=10)
        model.set_T(4)
        model.set_coding_mode("successive_refinement", ratio=2.0)
        logits = torch.ones(4, 2, 3)

        scaled = model.apply_temporal_readout(logits)
        scales = make_time_scales(4, mode="geometric", ratio=2.0)

        self.assertTrue(torch.allclose(scaled[:, 0, 0], scales))
        self.assertTrue(torch.allclose(scaled.mean(0), torch.ones(2, 3)))

    def test_rate_readout_does_not_change_temporal_logits(self):
        model = vgg16_signed(num_classes=10)
        model.set_T(4)
        logits = torch.randn(4, 2, 3)

        scaled = model.apply_temporal_readout(logits)

        self.assertTrue(torch.equal(scaled, logits))


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


class SuccessiveRefinementCalibrationTest(unittest.TestCase):
    def test_state_low_rank_target_uses_refinement_decoded_output(self):
        ann_neuron = SignedIF(T=0, thresh=1.0, enable_signed=False)
        ann_neuron.set_ftbc_mode("none")
        ann = _ToyTemporalModel(ann_neuron)

        snn_neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        snn_neuron.set_coding_mode("successive_refinement", ratio=2.0)
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

        scales = make_time_scales(2, mode="geometric", ratio=2.0)
        target = torch.tensor(
            [[[1.0 - 0.6, scales[0].item() / scales.sum().item() - 0.6]]]
        )
        state = torch.tensor([[[0.0, 1.0]]])
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


class SuccessiveRefinementAblationConfigTest(unittest.TestCase):
    def test_three_way_rate_configs_are_controlled(self):
        full = BASE_CONFIGS["F_RATE_FULL_FTBC"]
        low_rank = BASE_CONFIGS["H_RATE_STATE_LR_MATCHED"]

        for config in (full, low_rank):
            self.assertEqual(config["coding_mode"], "rate")
            self.assertTrue(config["signed"])
            self.assertTrue(config["r0"])
            self.assertEqual(config["r0_mode"], "legacy_clamp")
            self.assertFalse(config["expand_ratios"])

        self.assertEqual(full["ftbc_mode"], "full")
        self.assertIsNone(full["over_weight"])
        self.assertEqual(low_rank["ftbc_mode"], "state_low_rank")
        self.assertEqual(low_rank["over_weight"], FINAL_OVER_WEIGHT)

    def test_final_configuration_constants_match_reported_method(self):
        self.assertEqual(FINAL_TIME_SCALE_RATIO, 1.0)
        self.assertEqual(FINAL_POSITIVE_MARGIN, 0.55)
        self.assertEqual(FINAL_NEGATIVE_MARGIN, 1.30)
        self.assertEqual(FINAL_OVER_WEIGHT, 2.5)

    def test_rate_configs_are_not_duplicated_across_ratios(self):
        configs = expand_configurations(
            ["C_RATE_SNM_R0", "L_SR_GEOM_SNM_R0"],
            ratios=[1.1, 2.0],
        )

        self.assertEqual(
            list(configs),
            [
                "C_RATE_SNM_R0",
                "L_SR_GEOM_SNM_R0_R1.1",
                "L_SR_GEOM_SNM_R0_R2",
            ],
        )

    def test_uniform_refinement_uses_ratio_one_once(self):
        configs = expand_configurations(
            ["I_SR_UNIFORM_SNM_R0"],
            ratios=[1.1, 2.0],
        )

        self.assertEqual(list(configs), ["I_SR_UNIFORM_SNM_R0"])
        self.assertEqual(configs["I_SR_UNIFORM_SNM_R0"]["ratio"], 1.0)

    def test_geometric_configs_expand_negative_hysteresis_margin(self):
        configs = expand_configurations(
            ["L_SR_GEOM_SNM_R0"],
            ratios=[1.1],
            negative_margins=[0.5, 0.75],
        )

        self.assertEqual(
            list(configs),
            [
                "L_SR_GEOM_SNM_R0_R1.1",
                "L_SR_GEOM_SNM_R0_R1.1_N0.75",
            ],
        )
        self.assertEqual(
            configs["L_SR_GEOM_SNM_R0_R1.1_N0.75"]["negative_margin"],
            0.75,
        )

    def test_geometric_configs_expand_positive_hysteresis_margin(self):
        configs = expand_configurations(
            ["M_SR_GEOM_STATE_LR"],
            ratios=[1.0],
            positive_margins=[0.5, 0.55],
            negative_margins=[1.3],
        )

        self.assertEqual(
            list(configs),
            [
                "M_SR_GEOM_STATE_LR_R1_N1.3",
                "M_SR_GEOM_STATE_LR_R1_P0.55_N1.3",
            ],
        )
        self.assertEqual(
            configs["M_SR_GEOM_STATE_LR_R1_P0.55_N1.3"]["positive_margin"],
            0.55,
        )

    def test_state_low_rank_falls_back_to_full_for_short_runs(self):
        self.assertEqual(effective_ftbc_mode("state_low_rank", 1), "full")
        self.assertEqual(effective_ftbc_mode("state_low_rank", 2), "full")
        self.assertEqual(
            effective_ftbc_mode("state_low_rank", 4),
            "state_low_rank",
        )

    def test_rate_baseline_keeps_original_overfiring_weight(self):
        configs = expand_configurations(
            ["H_RATE_STATE_LR", "M_SR_GEOM_STATE_LR"],
            ratios=[1.0],
        )

        self.assertEqual(configs["H_RATE_STATE_LR"]["over_weight"], 2.0)
        self.assertIsNone(
            configs["M_SR_GEOM_STATE_LR_R1"]["over_weight"]
        )


if __name__ == "__main__":
    unittest.main()
