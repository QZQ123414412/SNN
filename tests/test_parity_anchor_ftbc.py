import unittest

import torch
import torch.nn as nn

from models.layer import SignedIF
from parity_anchor_ftbc import (
    compress_full_ftbc_teacher,
    parity_anchor_basis,
)


class TinySNN(nn.Module):
    def __init__(self, time_steps=8, channels=3):
        super().__init__()
        self.neuron = SignedIF(T=time_steps, thresh=2.0)
        self.neuron.set_ftbc_mode("full")
        base = torch.arange(channels, dtype=torch.float32)
        parity = torch.arange(channels, dtype=torch.float32) + 0.5
        self.neuron.time_based_bias = [
            base + 10 if t == 0 else
            base - 10 if t == 1 else
            base + (parity if t % 2 == 0 else -parity)
            for t in range(time_steps)
        ]


class ParityAnchorFTBCTest(unittest.TestCase):
    def test_basis_has_expected_fixed_structure(self):
        basis = parity_anchor_basis(8)
        self.assertEqual(tuple(basis.shape), (8, 4))
        self.assertTrue(torch.equal(basis[0], torch.tensor([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(basis[1], torch.tensor([0.0, 1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(basis[2], torch.tensor([0.0, 0.0, 1.0, 1.0])))
        self.assertTrue(torch.equal(basis[3], torch.tensor([0.0, 0.0, 1.0, -1.0])))

    def test_structured_teacher_is_exact(self):
        model = TinySNN()
        expected = torch.stack(model.neuron.time_based_bias)
        report = compress_full_ftbc_teacher(model)
        self.assertAlmostEqual(report["explained_energy"], 1.0, places=6)
        for t in range(8):
            actual = model.neuron.get_ftbc_bias(t, expected[t])
            self.assertTrue(torch.allclose(actual, expected[t], atol=2e-5))

    def test_storage_and_synthesis_accounting(self):
        model = TinySNN(time_steps=8, channels=3)
        compress_full_ftbc_teacher(model)
        layer = model.neuron
        self.assertEqual(layer.ftbc_mode, "parity_anchor")
        self.assertEqual(layer.ftbc_parameter_count(), 12)
        self.assertEqual(layer.ftbc_storage_bytes(), 48)
        self.assertEqual(layer.ftbc_synthesis_macs(), 42)

    def test_short_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            parity_anchor_basis(2)
        with self.assertRaises(ValueError):
            compress_full_ftbc_teacher(TinySNN(time_steps=4))


if __name__ == "__main__":
    unittest.main()
