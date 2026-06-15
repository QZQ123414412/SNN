# 验证FTBC通道级校正张量与部署形状一致
import unittest

import torch

from calibration import reshape_channel_bias


class CalibrationTest(unittest.TestCase):
    def test_reshape_channel_bias_for_conv_output(self):
        bias_mean = torch.tensor([1.0, 2.0, 3.0])
        reference = torch.zeros(4, 3, 5, 5)

        correction = reshape_channel_bias(bias_mean, reference)

        self.assertEqual(correction.shape, (1, 3, 1, 1))
        self.assertTrue(torch.equal(correction[:, :, 0, 0], bias_mean.view(1, -1)))

    def test_reshape_channel_bias_for_linear_output(self):
        bias_mean = torch.tensor([1.0, 2.0, 3.0])
        reference = torch.zeros(4, 3)

        correction = reshape_channel_bias(bias_mean, reference)

        self.assertEqual(correction.shape, (1, 3))
        self.assertTrue(torch.equal(correction, bias_mean.view(1, -1)))


if __name__ == "__main__":
    unittest.main()
