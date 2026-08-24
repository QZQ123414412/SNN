import unittest

import torch

from models.layer import SignedIF


class HorizonAnnealedSNMTest(unittest.TestCase):
    def test_standard_mode_preserves_constant_threshold(self):
        neuron = SignedIF(T=4, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_snm_mode("standard")
        self.assertEqual(
            [neuron.snm_threshold_multiplier(t) for t in range(4)],
            [1.0, 1.0, 1.0, 1.0],
        )

    def test_horizon_anneals_from_start_to_end(self):
        neuron = SignedIF(T=5, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_snm_mode("horizon_annealed", start=1.5, end=0.5)
        self.assertEqual(
            [neuron.snm_threshold_multiplier(t) for t in range(5)],
            [1.5, 1.25, 1.0, 0.75, 0.5],
        )

    def test_long_horizon_converges_toward_standard_snm(self):
        neuron = SignedIF(T=32, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_snm_mode(
            "horizon_annealed", start=1.25, end=0.5, reference=8
        )
        self.assertAlmostEqual(neuron.snm_threshold_multiplier(0), 1.0625)
        self.assertAlmostEqual(neuron.snm_threshold_multiplier(31), 0.875)

    def test_early_high_confidence_gate_and_late_refinement(self):
        neuron = SignedIF(T=3, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_snm_mode("horizon_annealed", start=2.0, end=0.5)
        inputs = torch.tensor([[1.0], [-1.6], [-0.1]])
        spikes = neuron(inputs)
        self.assertEqual(spikes[:, 0].tolist(), [1.0, 0.0, -1.0])
        self.assertEqual(float(neuron.transmitted.item()), 0.0)

    def test_credit_rule_still_blocks_negative_debt(self):
        neuron = SignedIF(T=2, thresh=1.0, enable_signed=True, enable_r0=True)
        neuron.set_ftbc_mode("none")
        neuron.set_snm_mode("horizon_annealed", start=1.5, end=0.25)
        spikes = neuron(torch.tensor([[-2.0], [-2.0]]))
        self.assertEqual(spikes[:, 0].tolist(), [0.0, 0.0])
        self.assertEqual(float(neuron.transmitted.item()), 0.0)

    def test_invalid_schedule_is_rejected(self):
        neuron = SignedIF(T=4)
        with self.assertRaises(ValueError):
            neuron.set_snm_mode("unknown")
        with self.assertRaises(ValueError):
            neuron.set_snm_mode("horizon_annealed", start=0.5, end=1.0)


if __name__ == "__main__":
    unittest.main()
