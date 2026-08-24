import unittest
from pathlib import Path

from scripts.experiments.run_full_ftbc_asnm_ablation import (
    CIFAR10_RESNET20_L4_PROTOCOL,
    DATASET_PROTOCOLS,
    build_parser,
    resolve_protocol_args,
    validate_args,
)


class FullFtbcAsnmProtocolTest(unittest.TestCase):
    def parse(self, *values):
        return resolve_protocol_args(build_parser().parse_args(list(values)))

    def test_cifar100_defaults_remain_frozen(self):
        args = self.parse("--architectures", "vgg16")

        self.assertEqual(args.dataset, "cifar100")
        self.assertEqual(
            args.vgg16_checkpoint,
            Path("cifar100-checkpoints/cifar100-vgg16-l8-example.pth"),
        )
        self.assertEqual(args.output, DATASET_PROTOCOLS["cifar100"]["default_output"])
        validate_args(args)

    def test_cifar10_vgg16_defaults_lock_legacy_checkpoint_and_hash(self):
        args = self.parse("--dataset", "cifar10", "--architectures", "vgg16")

        self.assertEqual(
            args.vgg16_checkpoint,
            Path("cifar10-checkpoints/cifar10-vgg16-example.pth"),
        )
        self.assertEqual(
            args.vgg16_checkpoint_sha256,
            "093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84",
        )
        self.assertEqual(args.output, DATASET_PROTOCOLS["cifar10"]["default_output"])
        validate_args(args)

    def test_cifar10_resnet20_defaults_lock_trained_checkpoint_hash(self):
        args = self.parse("--dataset", "cifar10", "--architectures", "resnet20")

        self.assertEqual(
            args.resnet20_checkpoint_sha256,
            "eb8301ebda8ae91e52f2f273306befa5d349931c05b829a9440dafa05df70631",
        )
        self.assertEqual(args.resnet20_eval_profile, "fixed_repo")
        validate_args(args)

    def test_smoke_output_must_be_archived(self):
        args = self.parse(
            "--dataset",
            "cifar10",
            "--architectures",
            "vgg16",
            "--time_steps",
            "1",
            "2",
            "--test_batches",
            "1",
        )
        with self.assertRaisesRegex(ValueError, "docs/archive"):
            validate_args(args)

        args.output = Path(
            "docs/archive/experiments/cifar10/FULL_FTBC_ASNM_SMOKE.md"
        )
        validate_args(args)

    def test_cifar10_resnet20_l4_protocol_is_fully_locked(self):
        args = self.parse(
            "--dataset", "cifar10", "--architectures", "resnet20", "--L", "4"
        )

        self.assertEqual(
            args.resnet20_checkpoint,
            CIFAR10_RESNET20_L4_PROTOCOL["checkpoint"],
        )
        self.assertEqual(
            args.resnet20_checkpoint_sha256,
            CIFAR10_RESNET20_L4_PROTOCOL["expected_sha256"],
        )
        self.assertEqual(args.resnet20_eval_profile, "paper_era")
        self.assertEqual(args.output, CIFAR10_RESNET20_L4_PROTOCOL["default_output"])
        validate_args(args)

    def test_l4_protocol_rejects_other_dataset_or_architecture(self):
        args = self.parse("--architectures", "resnet20", "--L", "4")
        with self.assertRaisesRegex(ValueError, "CIFAR-10/ResNet20"):
            validate_args(args)

        args = self.parse(
            "--dataset", "cifar10", "--architectures", "vgg16", "--L", "4"
        )
        with self.assertRaisesRegex(ValueError, "CIFAR-10/ResNet20"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
