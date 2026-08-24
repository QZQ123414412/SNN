import unittest
from pathlib import Path

import torch

from models import modelpool
from parity_anchor_ftbc import named_signed_layers
from scripts.experiments.run_full_ftbc_asnm_ablation import snapshot_full_ftbc
from scripts.experiments.run_pa_ftbc_asnm_ablation import (
    A_SNM_CONFIGS,
    BASE_CONFIGS,
    CONFIGS,
    DEFAULT_OUTPUTS,
    build_pa_model,
    build_parser,
    protocol_key,
    resolve_protocol_args,
    selected_time_label,
    validate_args,
)


class PAFTBCASNMAblationTest(unittest.TestCase):
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
            base = torch.linspace(0.1, 1.0, channels) * (layer_index + 1)
            parity = torch.linspace(0.05, 0.5, channels)
            module.time_based_bias = [
                base + 2 if t == 0 else
                base - 2 if t == 1 else
                base + (parity if t % 2 == 0 else -parity)
                for t in range(time_steps)
            ]
        return model, snapshot_full_ftbc(model, time_steps)

    def test_twelve_way_configuration_has_four_independent_families(self):
        self.assertEqual(len(CONFIGS), 12)
        self.assertEqual(set(BASE_CONFIGS), {"qcfs", "full", "temporal", "pa"})
        self.assertEqual(set(A_SNM_CONFIGS), set(BASE_CONFIGS))
        for family in BASE_CONFIGS:
            self.assertEqual(set(BASE_CONFIGS[family]), {"off", "on"})

    def test_pa_deployment_falls_back_to_full_at_t4(self):
        teacher, schedule = self.make_teacher_and_schedule(4)
        model, report = build_pa_model(
            teacher, schedule, 4, True, torch.device("cpu")
        )
        self.assertTrue(report["fallback_to_full"])
        self.assertEqual(report["explained_energy"], 1.0)
        self.assertEqual(
            {module.ftbc_mode for module in named_signed_layers(model).values()},
            {"full"},
        )

    def test_pa_deployment_uses_fixed_four_coefficients_after_t4(self):
        teacher, schedule = self.make_teacher_and_schedule(8)
        model, report = build_pa_model(
            teacher, schedule, 8, False, torch.device("cpu")
        )
        self.assertFalse(report["fallback_to_full"])
        self.assertEqual(report["coefficient_count"], 4)
        self.assertFalse(report["basis_stored"])
        self.assertAlmostEqual(report["explained_energy"], 1.0, places=5)
        self.assertEqual(
            {module.ftbc_mode for module in named_signed_layers(model).values()},
            {"parity_anchor"},
        )
        self.assertGreater(report["ftbc_synthesis_macs"], 0)

    def test_all_four_formal_protocols_are_locked(self):
        values = (
            ("cifar10", "resnet20", 4),
            ("cifar10", "vgg16", 8),
            ("cifar100", "resnet20", 8),
            ("cifar100", "vgg16", 8),
        )
        for dataset, architecture, qcfs_l in values:
            args = self.parse(
                "--dataset", dataset,
                "--architecture", architecture,
                "-L", str(qcfs_l),
            )
            self.assertEqual(protocol_key(args), (dataset, architecture, qcfs_l))
            self.assertEqual(args.output, DEFAULT_OUTPUTS[(dataset, architecture, qcfs_l)])
            self.assertTrue(Path(args.checkpoint).name.endswith(".pth"))
            self.assertEqual(len(args.checkpoint_sha256), 64)
            validate_args(args)

    def test_unsupported_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Supported protocols"):
            self.parse(
                "--dataset", "cifar10",
                "--architecture", "vgg16",
                "-L", "4",
            )

    def test_smoke_output_must_be_archived(self):
        args = self.parse(
            "--dataset", "cifar10",
            "--architecture", "resnet20",
            "-L", "4",
            "--time_steps", "4", "8",
            "--fit_batches", "1",
            "--validation_batches", "1",
            "--test_batches", "1",
        )
        with self.assertRaisesRegex(ValueError, "docs/archive"):
            validate_args(args)
        args.output = Path("docs/archive/experiments/pa_ftbc/smoke.md")
        validate_args(args)

    def test_selected_time_label_uses_pa_gate(self):
        gates = {
            "qcfs": {"1": False, "2": False},
            "full": {"1": False, "2": False},
            "temporal": {"1": False, "2": False},
            "pa": {"1": False, "2": True},
        }
        self.assertEqual(
            selected_time_label("L_QCFS_PA_FTBC_ASNM_R0", gates, (1, 2)),
            "2",
        )


if __name__ == "__main__":
    unittest.main()
