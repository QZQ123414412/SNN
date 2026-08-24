import copy
import gc
import json
import unittest
import uuid
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from models import SignedIF
from preprocess.getdataloader import GetImageNet, GetImageNetDatasets
from scripts.experiments.qcfs_checkpoint import (
    _normalize_legacy_module_names,
    _normalize_legacy_threshold_keys,
    checkpoint_sha256,
    load_qcfs_pair,
)
from scripts.experiments.run_imagenet_ftbc_pa_ha_ablation import (
    CONFIGS,
    DEFAULT_TIME_STEPS,
    HA_SNM,
    PROTOCOLS,
    SMOKE_TIME_STEPS,
    batches_sha256,
    build_parser,
    copy_full_fallback,
    evaluation_batch_size,
    implementation_signature,
    inspect_imagenet_datasets,
    materialize_fit_batch,
    resolve_args,
    validate_args,
    validate_completed_payload,
)
from scripts.experiments.summarize_imagenet_ftbc_pa_ha import main as summary_main


REPO_ROOT = Path(__file__).resolve().parents[1]


class ImageNetAblationProtocolTest(unittest.TestCase):
    def parse(self, *values):
        return resolve_args(build_parser().parse_args(list(values)))

    def test_formal_protocol_is_locked(self):
        for architecture in PROTOCOLS:
            args = self.parse("--architecture", architecture)
            validate_args(args)
            self.assertEqual(args.time_steps, DEFAULT_TIME_STEPS)
            self.assertEqual(args.fit_batch_size, 2)
            self.assertEqual(args.calibration_iterations, 50)
            self.assertEqual(args.alpha, 0.5)
            self.assertEqual(args.validation_batches, 0)
            self.assertEqual(args.timing_samples, 1_000)
            self.assertIn("formal", args.cache_dir.parts)
            self.assertEqual(
                args.checkpoint_sha256,
                PROTOCOLS[architecture]["sha256"],
            )

    def test_smoke_protocol_is_isolated_and_locked(self):
        args = self.parse("--architecture", "resnet34", "--smoke")
        validate_args(args)
        self.assertEqual(args.time_steps, SMOKE_TIME_STEPS)
        self.assertEqual(args.calibration_iterations, 1)
        self.assertEqual(args.validation_batches, 2)
        self.assertEqual(args.timing_samples, 0)
        self.assertIn("smoke", args.cache_dir.parts)
        self.assertIn("archive", {part.lower() for part in args.output.parts})

    def test_configuration_matrix_and_ha_constants_are_exact(self):
        self.assertEqual(len(CONFIGS), 9)
        self.assertEqual(
            {(item["family"], item["mode"]) for item in CONFIGS.values()},
            {
                (family, mode)
                for family in ("qcfs", "full", "pa")
                for mode in ("off", "standard", "ha")
            },
        )
        self.assertEqual(
            HA_SNM,
            {"start": 1.25, "end": 0.5, "reference": 8.0},
        )

    def test_temporal_batch_budget_formula(self):
        self.assertEqual(
            [evaluation_batch_size(value, 32) for value in DEFAULT_TIME_STEPS],
            [32, 16, 8, 4, 2, 1],
        )

    def test_checkpoint_hash_and_device_cannot_escape_protocol(self):
        args = self.parse(
            "--architecture",
            "vgg16",
            "--checkpoint-sha256",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "requires checkpoint SHA256"):
            validate_args(args)

        args = self.parse("--architecture", "vgg16", "--device", "cuda:0")
        with self.assertRaisesRegex(ValueError, "CUDA device index"):
            validate_args(args)

    def test_implementation_signature_covers_core_formulas(self):
        signature = implementation_signature()
        self.assertEqual(len(signature["sha256"]), 64)
        self.assertIn("models/layer.py", signature["files"])
        self.assertIn("calibration.py", signature["files"])
        self.assertIn("parity_anchor_ftbc.py", signature["files"])

    def test_legacy_key_conversion_is_narrow_and_collision_safe(self):
        tensor = torch.tensor([1.0])
        normalized, count = _normalize_legacy_threshold_keys(
            OrderedDict(
                [
                    ("conv1.2.up", tensor),
                    ("conv1.0.weight", tensor),
                ]
            )
        )
        self.assertEqual(count, 1)
        self.assertIn("conv1.2.thresh", normalized)
        self.assertIn("conv1.0.weight", normalized)

        renamed, count = _normalize_legacy_module_names(
            OrderedDict([("conv2_x.0.relu.thresh", tensor)]),
            "resnet34",
        )
        self.assertEqual(count, 1)
        self.assertIn("conv2_x.0.act.thresh", renamed)

        with self.assertRaisesRegex(RuntimeError, "colliding threshold"):
            _normalize_legacy_threshold_keys(
                OrderedDict(
                    [
                        ("conv1.2.up", tensor),
                        ("conv1.2.thresh", tensor),
                    ]
                )
            )

    def test_pa_fallback_is_an_exact_deep_copy(self):
        payload = {
            "results": OrderedDict((name, OrderedDict()) for name in CONFIGS),
            "layers": OrderedDict((name, OrderedDict()) for name in CONFIGS),
            "fallback_checks": [],
        }
        source_result = {"top1": 12.5, "nested": {"count": 3}}
        source_layers = [{"name": "activation", "sops": 7}]
        payload["results"]["D_QCFS_FULL_FTBC_R0"]["4"] = source_result
        payload["layers"]["D_QCFS_FULL_FTBC_R0"]["4"] = source_layers

        copy_full_fallback(payload, 4, "off")

        copied = payload["results"]["G_QCFS_PA_FTBC_R0"]["4"]
        self.assertEqual(copied, source_result)
        self.assertIsNot(copied, source_result)
        self.assertTrue(payload["fallback_checks"][0]["exact"])

    def test_smoke_completion_validator_enforces_samples_fallback_and_storage(self):
        args = self.parse("--architecture", "resnet34", "--smoke")
        validate_args(args)
        payload = {
            "dataset": {"validation_samples": 50_000},
            "ann": {"samples": 64},
            "results": OrderedDict((name, OrderedDict()) for name in CONFIGS),
            "compression": {},
            "fallback_checks": [
                {"time_steps": 4, "mode": mode, "exact": True}
                for mode in ("off", "standard", "ha")
            ],
        }
        expected_samples = {4: 16, 8: 8, 32: 2}
        for time_steps in SMOKE_TIME_STEPS:
            key = str(time_steps)
            payload["compression"][key] = {
                "full_parameters": 8 * time_steps,
                "pa_parameters": 8 * time_steps if time_steps <= 4 else 32,
            }
            for name, config in CONFIGS.items():
                parameters = 0
                if config["family"] == "full" or (
                    config["family"] == "pa" and time_steps <= 4
                ):
                    parameters = 8 * time_steps
                elif config["family"] == "pa":
                    parameters = 32
                payload["results"][name][key] = {
                    "top1": 1.0,
                    "top5": 2.0,
                    "logit_mse": 3.0,
                    "positive_spikes": 4,
                    "negative_spikes": 0,
                    "positive_rate": 0.1,
                    "negative_rate": 0.0,
                    "sparsity": 0.9,
                    "sops": 5,
                    "evaluated_samples": expected_samples[time_steps],
                    "ftbc_parameters": parameters,
                    "ftbc_bytes": 4 * parameters,
                }
            if time_steps <= 4:
                for mode in ("off", "standard", "ha"):
                    full = next(
                        name
                        for name, config in CONFIGS.items()
                        if config == {"family": "full", "mode": mode}
                    )
                    pa = next(
                        name
                        for name, config in CONFIGS.items()
                        if config == {"family": "pa", "mode": mode}
                    )
                    payload["results"][pa][key] = copy.deepcopy(
                        payload["results"][full][key]
                    )

        checks = validate_completed_payload(payload, args)
        self.assertTrue(all(item["passed"] for item in checks))
        payload["results"]["A_QCFS_R0"]["8"]["evaluated_samples"] = 7
        with self.assertRaisesRegex(RuntimeError, "expected 8"):
            validate_completed_payload(payload, args)

    def test_temporary_imagefolder_loader_is_single_process_compatible(self):
        root = REPO_ROOT / "tests" / f"_imagenet_loader_{uuid.uuid4().hex}"
        root.mkdir()
        image_paths = []
        class_dirs = []
        split_dirs = []
        try:
            for split in ("train", "val"):
                split_dir = root / split
                split_dir.mkdir()
                split_dirs.append(split_dir)
                for class_name, color in (("n00000001", 32), ("n00000002", 224)):
                    class_dir = split_dir / class_name
                    class_dir.mkdir()
                    class_dirs.append(class_dir)
                    image_path = class_dir / "sample.png"
                    Image.new("RGB", (256, 256), (color, color, color)).save(
                        image_path
                    )
                    image_paths.append(image_path)

            train_data, validation_data = GetImageNetDatasets(root=root)
            train_loader, validation_loader = GetImageNet(1, root=root)
            self.assertIsInstance(train_loader.sampler, RandomSampler)
            self.assertIsInstance(validation_loader.sampler, SequentialSampler)
            signature = inspect_imagenet_datasets(
                train_data,
                validation_data,
                require_official=False,
            )
            self.assertEqual(signature["classes"], 2)
            self.assertEqual(signature["train_samples"], 2)
            self.assertEqual(signature["validation_samples"], 2)
            inputs, targets = materialize_fit_batch(train_data, 2, 42)[0]
            self.assertEqual(tuple(inputs.shape), (2, 3, 224, 224))
            self.assertEqual(tuple(targets.shape), (2,))
            first_hash = batches_sha256([(inputs, targets)])
            second_hash = batches_sha256(materialize_fit_batch(train_data, 2, 42))
            self.assertEqual(first_hash, second_hash)
        finally:
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)
            for class_dir in reversed(class_dirs):
                class_dir.rmdir()
            for split_dir in reversed(split_dirs):
                split_dir.rmdir()
            root.rmdir()


class ImageNetCheckpointIntegrationTest(unittest.TestCase):
    def test_downloaded_checkpoints_strictly_load_on_cpu(self):
        specifications = (
            (
                "resnet34",
                REPO_ROOT / "ImageNet-checkpoints/ImageNet-ResNet34-t8.pth",
            ),
            (
                "vgg16",
                REPO_ROOT / "ImageNet-checkpoints/ImageNet-VGG16-t16.pth",
            ),
        )
        for architecture, checkpoint in specifications:
            if not checkpoint.is_file():
                self.skipTest(f"Optional local checkpoint is absent: {checkpoint}")
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    checkpoint_sha256(checkpoint),
                    PROTOCOLS[architecture]["sha256"],
                )
                ann, snn, metadata = load_qcfs_pair(
                    checkpoint,
                    "imagenet",
                    architecture,
                    torch.device("cpu"),
                )
                ann_thresholds = {
                    name: module.thresh.detach().clone()
                    for name, module in ann.named_modules()
                    if type(module).__name__ == "IF"
                }
                snn_thresholds = {
                    name: module.thresh.detach().clone()
                    for name, module in snn.named_modules()
                    if isinstance(module, SignedIF)
                }
                self.assertEqual(ann_thresholds.keys(), snn_thresholds.keys())
                self.assertEqual(
                    len(snn_thresholds),
                    PROTOCOLS[architecture]["signed_layers"],
                )
                for name in ann_thresholds:
                    self.assertTrue(
                        torch.equal(ann_thresholds[name], snn_thresholds[name])
                    )
                self.assertEqual(metadata["sha256"], PROTOCOLS[architecture]["sha256"])
                del ann, snn, metadata, ann_thresholds, snn_thresholds
                gc.collect()


class ImageNetSummaryTest(unittest.TestCase):
    def make_payload(self, architecture):
        results = {
            name: {
                str(value): {
                    "top1": 70.0 + value / 100.0,
                    "top5": 90.0 + value / 100.0,
                    "ftbc_bytes": 16,
                }
                for value in DEFAULT_TIME_STEPS
            }
            for name in CONFIGS
        }
        compression = {
            str(value): {
                "full_parameters": 8 * value,
                "pa_parameters": 8 * value if value <= 4 else 32,
                "pa_bytes": 32 * value if value <= 4 else 128,
            }
            for value in DEFAULT_TIME_STEPS
        }
        return {
            "status": "complete",
            "protocol": {
                "architecture": architecture,
                "time_steps": list(DEFAULT_TIME_STEPS),
                "qcfs_L": PROTOCOLS[architecture]["L"],
            },
            "ann": {"top1": 74.3, "top5": 91.8},
            "runtime": {"gpu": "synthetic", "active_elapsed_seconds": 1.0},
            "results": results,
            "compression": compression,
        }

    def test_two_model_summary_requires_complete_inputs_and_refuses_overwrite(self):
        stem = uuid.uuid4().hex
        resnet_path = REPO_ROOT / "tests" / f"_resnet_{stem}.json"
        vgg_path = REPO_ROOT / "tests" / f"_vgg_{stem}.json"
        output_path = REPO_ROOT / "tests" / f"_summary_{stem}.md"
        try:
            resnet_path.write_text(
                json.dumps(self.make_payload("resnet34")),
                encoding="utf-8",
            )
            vgg_path.write_text(
                json.dumps(self.make_payload("vgg16")),
                encoding="utf-8",
            )
            summary_main(
                [
                    "--resnet34-progress",
                    str(resnet_path),
                    "--vgg16-progress",
                    str(vgg_path),
                    "--output",
                    str(output_path),
                ]
            )
            text = output_path.read_text(encoding="utf-8")
            self.assertIn("Status: complete", text)
            self.assertIn("ResNet34 accuracy", text)
            self.assertIn("VGG16 accuracy", text)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                summary_main(
                    [
                        "--resnet34-progress",
                        str(resnet_path),
                        "--vgg16-progress",
                        str(vgg_path),
                        "--output",
                        str(output_path),
                    ]
                )
        finally:
            output_path.unlink(missing_ok=True)
            vgg_path.unlink(missing_ok=True)
            resnet_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
