import unittest
from pathlib import Path

import torch

from models import modelpool
from scripts.experiments.run_temporal_lr_asnm_ablation import (
    A_SNM_CONFIGS,
    BASE_CONFIGS,
    CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL,
    CIFAR100_DEFAULT_OUTPUT,
    CONFIGS,
    build_parser,
    build_temporal_model,
    exact_metrics,
    resolve_protocol_args,
    selected_time_label,
    snapshot_full_ftbc,
    validate_args,
)
from temporal_lr import named_signed_layers


class TemporalLRASNMAblationTest(unittest.TestCase):
    def parse(self, *values):
        return resolve_protocol_args(build_parser().parse_args(list(values)))

    def make_teacher_and_schedule(self, time_steps):
        model = modelpool("resnet20_signed", "cifar100").eval()
        model.set_T(time_steps)
        model.set_coding_mode("rate", schedule="rate", ratio=1.0)
        model.set_signed(False)
        model.set_r0(True)
        model.set_ftbc_mode("full")
        with torch.no_grad():
            model(torch.zeros(2, 3, 32, 32))
        for layer_index, module in enumerate(named_signed_layers(model).values()):
            channels = module.time_based_bias[0].numel()
            channel_scale = torch.linspace(0.1, 1.0, channels)
            for time_index in range(time_steps):
                module.time_based_bias[time_index] = (
                    (layer_index + 1) * (time_index + 1) * channel_scale
                )
        return model, snapshot_full_ftbc(model, time_steps)

    def test_nine_way_configuration_has_three_independent_families(self):
        self.assertEqual(len(CONFIGS), 9)
        self.assertEqual(set(BASE_CONFIGS), {"qcfs", "full", "temporal"})
        self.assertEqual(set(A_SNM_CONFIGS), {"qcfs", "full", "temporal"})
        for family in BASE_CONFIGS:
            self.assertEqual(set(BASE_CONFIGS[family]), {"off", "on"})

    def test_temporal_deployment_falls_back_to_full_at_t4(self):
        teacher, schedule = self.make_teacher_and_schedule(time_steps=4)

        model, compression = build_temporal_model(
            teacher,
            schedule,
            time_steps=4,
            signed=True,
            device=torch.device("cpu"),
            architecture="resnet20",
        )

        self.assertTrue(compression["fallback_to_full"])
        self.assertEqual(compression["explained_energy"], 1.0)
        self.assertEqual(
            {module.ftbc_mode for module in named_signed_layers(model).values()},
            {"full"},
        )

    def test_temporal_deployment_uses_rank_four_after_t4(self):
        teacher, schedule = self.make_teacher_and_schedule(time_steps=8)

        model, compression = build_temporal_model(
            teacher,
            schedule,
            time_steps=8,
            signed=False,
            device=torch.device("cpu"),
            architecture="resnet20",
        )

        self.assertFalse(compression["fallback_to_full"])
        self.assertEqual(compression["effective_rank"], 4)
        self.assertTrue(compression["threshold_normalize"])
        self.assertGreater(compression["explained_energy"], 0.0)
        self.assertEqual(
            {module.ftbc_mode for module in named_signed_layers(model).values()},
            {"temporal_low_rank"},
        )
        self.assertGreater(compression["ftbc_synthesis_macs"], 0)

    def test_selected_time_label_uses_each_family_gate(self):
        gates = {
            "qcfs": {"1": False, "2": True},
            "full": {"1": True, "2": False},
            "temporal": {"1": False, "2": False},
        }

        self.assertEqual(selected_time_label("C_QCFS_ASNM_R0", gates, (1, 2)), "2")
        self.assertEqual(
            selected_time_label("F_QCFS_FULL_FTBC_ASNM_R0", gates, (1, 2)),
            "1",
        )
        self.assertEqual(
            selected_time_label(
                "I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0", gates, (1, 2)
            ),
            "none",
        )

    def test_exact_metrics_ignores_timing_but_detects_result_changes(self):
        left = {"acc": 70.0, "sops": 10, "elapsed": 1.0}
        right = {"acc": 70.0, "sops": 10, "elapsed": 2.0}
        self.assertTrue(exact_metrics(left, right, ("acc", "sops")))
        right["sops"] = 11
        self.assertFalse(exact_metrics(left, right, ("acc", "sops")))

    def test_cifar100_defaults_remain_frozen(self):
        args = self.parse()

        self.assertEqual(args.dataset, "cifar100")
        self.assertEqual(args.L, 8)
        self.assertEqual(args.output, CIFAR100_DEFAULT_OUTPUT)
        self.assertEqual(
            args.resnet20_checkpoint,
            Path(
                "cifar100-checkpoints/"
                "resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth"
            ),
        )
        validate_args(args)

    def test_cifar10_resnet20_l4_protocol_is_fully_locked(self):
        args = self.parse(
            "--dataset",
            "cifar10",
            "--architectures",
            "resnet20",
            "--L",
            "4",
        )

        self.assertEqual(
            args.resnet20_checkpoint,
            CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL["checkpoint"],
        )
        self.assertEqual(
            args.resnet20_checkpoint_sha256,
            CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL["expected_sha256"],
        )
        self.assertEqual(args.resnet20_eval_profile, "paper_era")
        self.assertEqual(
            args.output,
            CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL["default_output"],
        )
        validate_args(args)

    def test_cifar10_protocol_rejects_other_level_or_architecture(self):
        args = self.parse(
            "--dataset",
            "cifar10",
            "--architectures",
            "resnet20",
        )
        with self.assertRaisesRegex(ValueError, "ResNet20 L=4"):
            validate_args(args)

        args = self.parse(
            "--dataset",
            "cifar10",
            "--architectures",
            "vgg16",
            "--L",
            "4",
        )
        with self.assertRaisesRegex(ValueError, "ResNet20 L=4"):
            validate_args(args)

    def test_nonformal_cifar10_output_must_be_archived(self):
        args = self.parse(
            "--dataset",
            "cifar10",
            "--architectures",
            "resnet20",
            "--L",
            "4",
            "--time_steps",
            "4",
            "8",
            "--test_batches",
            "1",
        )
        with self.assertRaisesRegex(ValueError, "docs/archive"):
            validate_args(args)

        args.output = Path(
            "docs/archive/experiments/cifar10/TEMPORAL_LR_ASNM_SMOKE.md"
        )
        validate_args(args)


if __name__ == "__main__":
    unittest.main()
