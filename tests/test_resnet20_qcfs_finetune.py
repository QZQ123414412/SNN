import argparse
import json
import unittest
import uuid
from pathlib import Path

import torch

from models import modelpool
from scripts.train.finetune_resnet20_qcfs import (
    TARGET_FILENAME,
    atomic_json_save,
    experiment_config,
    load_resume_state,
    save_first_target,
    validate_args,
    validate_qcfs_model,
)


class ResNet20QcfsFinetuneTest(unittest.TestCase):
    def _args(self, **changes):
        values = dict(
            source=Path("source.pth"),
            output_dir=Path("output"),
            seed=42,
            epochs=50,
            batch_size=128,
            learning_rates=[0.005, 0.002, 0.001],
            trajectory_names=["FT_LR005", "FT_LR002", "FT_LR001"],
            weight_decay=5e-4,
            target_accuracy=69.94,
            expected_source_accuracy=68.78,
            source_accuracy_tolerance=0.005,
        )
        values.update(changes)
        return argparse.Namespace(**values)

    def _path(self, name):
        return Path(__file__).resolve().parent / f".{name}_{uuid.uuid4().hex}"

    def test_default_protocol_is_fixed_three_by_fifty(self):
        args = self._args()
        validate_args(args)
        config = experiment_config(args, "a" * 64)
        self.assertEqual(config["epochs_per_trajectory"], 50)
        self.assertEqual(config["learning_rates"], [0.005, 0.002, 0.001])
        self.assertEqual(config["augmentation_profile"], "paper_era")
        self.assertEqual(config["qcfs_training_profile"], "fixed_repo")
        self.assertEqual(config["weight_origin"], "official_implementation_finetuned")

    def test_invalid_trajectory_pairing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            validate_args(self._args(trajectory_names=["one"]))

    def test_model_gate_requires_nineteen_l8_fixed_repo_layers(self):
        model = modelpool("resnet20", "cifar100")
        model.set_L(8)
        model.set_qcfs_training_profile("fixed_repo")
        validate_qcfs_model(model)
        model.conv1[2].L = 4
        with self.assertRaisesRegex(RuntimeError, "L=8"):
            validate_qcfs_model(model)

    def test_first_target_is_saved_once_without_overwrite(self):
        output_dir = self._path("target_dir")
        output_dir.mkdir()
        model = modelpool("resnet20", "cifar100")
        args = self._args(source=output_dir / "source.pth")
        config = experiment_config(args, "b" * 64)
        target_path = output_dir / TARGET_FILENAME
        metadata_path = target_path.with_suffix(".json")
        try:
            self.assertTrue(
                save_first_target(model, output_dir, config, "FT_LR005", 3, 69.95)
            )
            first_hash = metadata_path.read_text(encoding="utf-8")
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            self.assertFalse(
                save_first_target(model, output_dir, config, "FT_LR002", 4, 70.10)
            )
            self.assertEqual(metadata_path.read_text(encoding="utf-8"), first_hash)
            metadata = json.loads(first_hash)
            self.assertEqual(metadata["actual_accuracy"], 69.95)
            self.assertEqual(metadata["trajectory"], "FT_LR005")
        finally:
            if target_path.exists():
                target_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            output_dir.rmdir()

    def test_incomplete_target_pair_is_rejected(self):
        output_dir = self._path("incomplete_target_dir")
        output_dir.mkdir()
        model = modelpool("resnet20", "cifar100")
        args = self._args(source=output_dir / "source.pth")
        config = experiment_config(args, "d" * 64)
        metadata_path = (output_dir / TARGET_FILENAME).with_suffix(".json")
        try:
            atomic_json_save({"incomplete": True}, metadata_path)
            with self.assertRaisesRegex(RuntimeError, "both exist"):
                save_first_target(
                    model, output_dir, config, "FT_LR005", 3, 69.95
                )
        finally:
            if metadata_path.exists():
                metadata_path.unlink()
            output_dir.rmdir()

    def test_resume_rejects_changed_experiment_signature(self):
        path = self._path("resume").with_suffix(".pth")
        config = experiment_config(self._args(), "c" * 64)
        try:
            torch.save(
                {
                    "experiment_config": config,
                    "trajectory": "FT_LR005",
                    "learning_rate": 0.005,
                },
                path,
            )
            changed = dict(config, seed=7)
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                load_resume_state(path, changed, "FT_LR005", 0.005)
        finally:
            if path.exists():
                path.unlink()

    def test_atomic_json_is_valid(self):
        path = self._path("metadata").with_suffix(".json")
        try:
            atomic_json_save({"accuracy": 69.94}, path)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"accuracy": 69.94},
            )
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
