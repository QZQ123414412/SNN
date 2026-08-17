import argparse
import io
import unittest

import numpy as np
import torch

from scripts.train.main_train import (
    capture_rng_state,
    restore_rng_state,
    training_config,
)


class TrainingResumeTest(unittest.TestCase):
    def test_rng_state_restores_numpy_and_torch_sequences(self):
        np.random.seed(7)
        torch.manual_seed(7)
        state = capture_rng_state()
        expected_numpy = np.random.rand(4)
        expected_torch = torch.rand(4)

        np.random.seed(99)
        torch.manual_seed(99)
        restore_rng_state(state)

        self.assertTrue(np.array_equal(np.random.rand(4), expected_numpy))
        self.assertTrue(torch.equal(torch.rand(4), expected_torch))
        serialized = capture_rng_state()
        self.assertIsInstance(serialized["numpy"]["keys"], torch.Tensor)
        buffer = io.BytesIO()
        torch.save({"rng_state": serialized}, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer)
        self.assertTrue(
            torch.equal(
                loaded["rng_state"]["numpy"]["keys"],
                serialized["numpy"]["keys"],
            )
        )

    def test_training_config_contains_reproduction_critical_fields(self):
        args = argparse.Namespace(
            dataset="cifar100",
            model="resnet20",
            L=8,
            time=0,
            batch_size=300,
            epochs=300,
            lr=0.02,
            weight_decay=5e-4,
            seed=42,
            augmentation_profile="paper_era",
            qcfs_training_profile="paper_era",
        )

        self.assertEqual(
            training_config(args),
            {
                "dataset": "cifar100",
                "model": "resnet20",
                "L": 8,
                "time": 0,
                "batch_size": 300,
                "epochs": 300,
                "lr": 0.02,
                "weight_decay": 5e-4,
                "seed": 42,
                "augmentation_profile": "paper_era",
                "qcfs_training_profile": "paper_era",
            },
        )


if __name__ == "__main__":
    unittest.main()
