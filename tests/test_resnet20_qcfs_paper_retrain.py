import argparse
import unittest

import torch

from models import IF
from scripts.train.retrain_resnet20_qcfs_paper import (
    RECIPES,
    initialize_model,
    recipe_config,
    validate_args,
)


class ResNet20QcfsPaperRetrainTest(unittest.TestCase):
    def _args(self, **changes):
        values = dict(
            epochs=300,
            workers=8,
            initial_lr=0.02,
            initial_threshold=4.0,
            cutout_length=8,
            seed=42,
            target_accuracy=69.94,
            weight_decay=5e-4,
        )
        values.update(changes)
        return argparse.Namespace(**values)

    def test_recipe_order_and_single_variable_changes_are_locked(self):
        self.assertEqual(
            list(RECIPES),
            [
                "R1_PAPER_FORMULA_EARLY_CODE",
                "R2_PUBLIC_CODE_GRADIENT",
                "R3_BATCH128_SENSITIVITY",
            ],
        )
        configs = [
            recipe_config(self._args(), name, recipe)
            for name, recipe in RECIPES.items()
        ]
        self.assertEqual(configs[0]["batch_size"], 200)
        self.assertEqual(configs[0]["qcfs_training_profile"], "paper_era")
        self.assertEqual(configs[1]["batch_size"], 200)
        self.assertEqual(configs[1]["qcfs_training_profile"], "fixed_repo")
        self.assertEqual(configs[2]["batch_size"], 128)
        self.assertEqual(configs[2]["qcfs_training_profile"], "paper_era")
        self.assertTrue(all(config["autoaugment"] for config in configs))
        self.assertTrue(all(config["cutout_length"] == 8 for config in configs))
        self.assertTrue(all(config["initial_threshold"] == 4.0 for config in configs))

    def test_initialization_sets_all_nineteen_trainable_thresholds_to_four(self):
        name, recipe = next(iter(RECIPES.items()))
        config = recipe_config(self._args(), name, recipe)
        model = initialize_model(config, torch.device("cpu"))
        layers = [module for module in model.modules() if isinstance(module, IF)]
        self.assertEqual(len(layers), 19)
        self.assertTrue(all(layer.thresh.requires_grad for layer in layers))
        self.assertTrue(
            all(torch.equal(layer.thresh, torch.tensor([4.0])) for layer in layers)
        )
        self.assertTrue(all(layer.L == 8 for layer in layers))
        self.assertTrue(
            all(layer.quantization_profile == "paper_era" for layer in layers)
        )

    def test_invalid_training_controls_are_rejected(self):
        for changes in (
            {"epochs": 0},
            {"workers": -1},
            {"initial_lr": 0.0},
            {"initial_threshold": 0.0},
            {"cutout_length": 0},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    validate_args(self._args(**changes))


if __name__ == "__main__":
    unittest.main()
