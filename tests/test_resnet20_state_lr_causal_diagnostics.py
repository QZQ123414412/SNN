import unittest

import torch

from models import SignedIF, modelpool
from scripts.experiments.run_resnet20_state_lr_causal_diagnostics import (
    VARIANTS,
    clamp_final_state_lr_coefficients,
    coefficient_summary,
)


class StateBiasSwitchTest(unittest.TestCase):
    def test_state_term_can_be_disabled_without_changing_base_or_slope(self):
        neuron = SignedIF(T=4, thresh=2.0)
        neuron.set_ftbc_mode("state_low_rank")
        neuron._init_ftbc_bias(1, torch.device("cpu"))
        neuron.bias_base.fill_(0.2)
        neuron.bias_slope.fill_(0.4)
        neuron.bias_state.fill_(0.6)
        reference = torch.zeros(2, 1)
        state = torch.tensor([[True], [False]])

        enabled = neuron.get_ftbc_bias(0, reference, state)
        neuron.set_state_bias_enabled(False)
        disabled = neuron.get_ftbc_bias(0, reference, state)

        self.assertTrue(torch.allclose(enabled[:, 0], torch.tensor([0.8, 0.2])))
        self.assertTrue(torch.allclose(disabled[:, 0], torch.tensor([0.2, 0.2])))
        self.assertEqual(neuron.ftbc_parameter_count(), 2)
        self.assertEqual(neuron.ftbc_storage_bytes(), 8)

    def test_final_global_clip_limits_accumulated_coefficients(self):
        model = modelpool("resnet20_signed", "cifar100")
        model.set_T(4)
        model.set_ftbc_mode("state_low_rank")
        first = next(module for module in model.modules() if isinstance(module, SignedIF))
        first._init_ftbc_bias(3, torch.device("cpu"))
        threshold = float(first.thresh.detach().abs().item())
        first.bias_base.fill_(threshold * 0.5)
        first.bias_slope.fill_(-threshold * 0.4)
        first.bias_state.fill_(threshold * 0.1)

        stats = clamp_final_state_lr_coefficients(model, 0.25)
        summary = coefficient_summary(model, 0.25)

        self.assertEqual(stats["changed"], 6)
        self.assertEqual(stats["total"], 9)
        self.assertLessEqual(summary["max_coefficient_ratio"], 0.25)
        self.assertEqual(summary["fraction_over_global_limit"], 0.0)

    def test_variants_isolate_the_three_requested_interventions(self):
        self.assertEqual(
            list(VARIANTS),
            [
                "E_REFERENCE_STATE_LR",
                "F_REFERENCE_SNM_STATE_LR",
                "G_E_COEFFICIENTS_SNM_ON",
                "H_F_BIAS_STATE_OFF",
                "I_F_FINAL_GLOBAL_CLIP",
            ],
        )
        self.assertFalse(VARIANTS["G_E_COEFFICIENTS_SNM_ON"]["calibration_signed"])
        self.assertTrue(VARIANTS["G_E_COEFFICIENTS_SNM_ON"]["inference_signed"])
        self.assertFalse(VARIANTS["H_F_BIAS_STATE_OFF"]["state_bias"])
        self.assertTrue(VARIANTS["I_F_FINAL_GLOBAL_CLIP"]["global_clip"])


if __name__ == "__main__":
    unittest.main()
