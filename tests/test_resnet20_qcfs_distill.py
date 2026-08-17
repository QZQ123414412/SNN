import argparse
import unittest
import uuid
from pathlib import Path

import torch

from models import modelpool
from scripts.train.distill_finetune_resnet20_qcfs import (
    DISTLoss,
    STUDENT_MEAN,
    STUDENT_STD,
    TEACHER_MEAN,
    TEACHER_STD,
    TeacherInputAdapter,
    build_teacher_model,
    experiment_config,
    load_resume_state,
    split_student_parameters,
    validate_args,
)


class ResNet20QcfsDistillTest(unittest.TestCase):
    def _args(self, **changes):
        values = dict(
            source=Path("source.pth"),
            teacher=Path("teacher.pth"),
            output_dir=Path("output"),
            device="0",
            seed=42,
            workers=8,
            epochs=100,
            batch_size=128,
            trajectory_names=[
                "KD_FT_WLR1E4_TLR1E5",
                "KD_FT_WLR5E5_TLR5E6",
            ],
            weight_learning_rates=[1e-4, 5e-5],
            threshold_learning_rates=[1e-5, 5e-6],
            weight_decay=5e-4,
            dist_tau=4.0,
            dist_beta=1.0,
            dist_gamma=1.0,
            ce_weight=1.0,
            dist_weight=2.0,
            target_accuracy=69.94,
            expected_source_accuracy=68.78,
            source_accuracy_tolerance=0.005,
            expected_teacher_accuracy=72.63,
            teacher_accuracy_tolerance=0.02,
            resume=False,
        )
        values.update(changes)
        return argparse.Namespace(**values)

    def _path(self, stem):
        return Path(__file__).resolve().parent / f".{stem}_{uuid.uuid4().hex}.pth"

    def test_stage_one_protocol_is_fixed(self):
        args = self._args()
        validate_args(args)
        teacher = {
            "path": "teacher.pth",
            "sha256": "b" * 64,
            "url": "https://example.invalid/teacher.pth",
            "architecture": "cifar100_resnet56",
            "origin": "chenyaofo_pytorch_cifar_models_pretrained",
        }
        config = experiment_config(args, "a" * 64, teacher)
        self.assertEqual(config["target_accuracy"], 69.94)
        self.assertEqual(config["epochs_per_trajectory"], 100)
        self.assertEqual(config["augmentation_profile"], "paper_era")
        self.assertFalse(config["autoaugment"])
        self.assertEqual(config["cutout_length"], 16)
        self.assertEqual(config["qcfs_training_profile"], "fixed_repo")
        self.assertEqual(config["dist_tau"], 4.0)
        self.assertEqual(config["ce_weight"], 1.0)
        self.assertEqual(config["dist_weight"], 2.0)
        self.assertTrue(config["cudnn_allow_tf32"])
        self.assertFalse(config["cuda_matmul_allow_tf32"])
        self.assertEqual(config["teacher_validation_batch_size"], 256)
        self.assertEqual(
            config["weight_origin"], "official_architecture_distilled_qat"
        )

    def test_invalid_trajectory_pairing_is_rejected(self):
        args = self._args(threshold_learning_rates=[1e-5])
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_args(args)

    def test_teacher_resnet56_shape(self):
        teacher = build_teacher_model().eval()
        self.assertEqual(len(teacher.layer1), 9)
        self.assertEqual(len(teacher.layer2), 9)
        self.assertEqual(len(teacher.layer3), 9)
        with torch.no_grad():
            outputs = teacher(torch.randn(2, 3, 32, 32))
        self.assertEqual(outputs.shape, (2, 100))

    def test_teacher_input_adapter_preserves_the_underlying_pixels(self):
        adapter = TeacherInputAdapter(torch.nn.Identity())
        raw = torch.rand(2, 3, 4, 4)
        student_mean = torch.tensor(STUDENT_MEAN).view(1, 3, 1, 1)
        student_std = torch.tensor(STUDENT_STD).view(1, 3, 1, 1)
        teacher_mean = torch.tensor(TEACHER_MEAN).view(1, 3, 1, 1)
        teacher_std = torch.tensor(TEACHER_STD).view(1, 3, 1, 1)
        student_normalized = (raw - student_mean) / student_std
        expected = (raw - teacher_mean) / teacher_std
        self.assertTrue(
            torch.allclose(adapter(student_normalized), expected, atol=1e-6)
        )

    def test_dist_is_zero_for_identical_logits_and_backpropagates(self):
        criterion = DISTLoss(beta=1.0, gamma=1.0, tau=4.0)
        teacher = torch.randn(8, 100)
        student = teacher.detach().clone().requires_grad_(True)
        identical_loss = criterion(student, teacher)
        # The official formula adds epsilon to the correlation denominator,
        # so identical distributions have a very small positive residual.
        self.assertLess(abs(float(identical_loss.detach())), 0.02)
        different_student = torch.randn(8, 100, requires_grad=True)
        loss = criterion(different_student, teacher)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(different_student.grad)
        self.assertTrue(torch.isfinite(different_student.grad).all())

    def test_optimizer_partition_has_nineteen_thresholds(self):
        student = modelpool("resnet20", "cifar100")
        weights, thresholds = split_student_parameters(student)
        self.assertEqual(len(thresholds), 19)
        self.assertTrue(weights)
        self.assertFalse({id(p) for p in weights} & {id(p) for p in thresholds})
        self.assertEqual(
            {id(p) for p in student.parameters()},
            {id(p) for p in weights + thresholds},
        )

    def test_resume_rejects_changed_experiment_signature(self):
        path = self._path("distill_resume")
        config = {"target_accuracy": 69.94, "seed": 42}
        try:
            torch.save(
                {
                    "experiment_config": config,
                    "trajectory": "KD_FT_WLR1E4_TLR1E5",
                    "weight_lr": 1e-4,
                    "threshold_lr": 1e-5,
                },
                path,
            )
            changed = dict(config, seed=7)
            with self.assertRaisesRegex(RuntimeError, "experiment_config mismatch"):
                load_resume_state(
                    path,
                    changed,
                    "KD_FT_WLR1E4_TLR1E5",
                    1e-4,
                    1e-5,
                )
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
