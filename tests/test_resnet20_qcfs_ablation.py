import argparse
import unittest
import uuid
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from calibration import match_calibration_layers
from models import IF, SignedIF, modelpool
from scripts.experiments.qcfs_checkpoint import load_qcfs_pair
from scripts.experiments.run_resnet20_qcfs_ablation import (
    CONFIGS,
    FORMAL_CHECKPOINT_SHA256,
    FORMAL_MIN_ANN_ACCURACY,
    FORMAL_TIME_STEPS,
    calibration_batches_sha256,
    configure_snn,
    effective_ftbc_mode,
    load_progress,
    protocol_signature,
    resolve_time_steps,
    resolve_calibration_batches,
    save_progress,
    validate_t32_conversion,
    validate_formal_checkpoint,
    validate_formal_protocol,
    build_parser,
)
from spike_stats import SpikeLayerStats


class ResNet20SignedTest(unittest.TestCase):
    def test_qcfs_training_profiles_have_equal_values_but_different_outer_gradients(self):
        fixed = IF(L=8, thresh=8.0, quantization_profile="fixed_repo")
        paper_era = IF(L=8, thresh=8.0, quantization_profile="paper_era")
        paper_era.load_state_dict(fixed.state_dict())
        fixed_input = torch.tensor([-0.1, 4.0, 8.1], requires_grad=True)
        paper_input = fixed_input.detach().clone().requires_grad_(True)

        fixed_output = fixed(fixed_input)
        paper_output = paper_era(paper_input)
        self.assertTrue(torch.equal(fixed_output, paper_output))
        fixed_output.sum().backward()
        paper_output.sum().backward()
        self.assertFalse(torch.equal(fixed_input.grad, paper_input.grad))
        self.assertEqual(fixed_input.grad[0].item(), 0.0)
        self.assertEqual(paper_input.grad[0].item(), 1.0)

    def test_signed_model_preserves_activation_names_and_parameter_shapes(self):
        ann = modelpool("resnet20", "cifar100")
        snn = modelpool("resnet20_signed", "cifar100")

        ann_activations = [
            name for name, module in ann.named_modules() if isinstance(module, IF)
        ]
        snn_activations = [
            name
            for name, module in snn.named_modules()
            if isinstance(module, SignedIF)
        ]

        self.assertEqual(ann_activations, snn_activations)
        self.assertEqual(len(ann_activations), 19)
        for name, tensor in ann.state_dict().items():
            self.assertIn(name, snn.state_dict())
            self.assertEqual(tensor.shape, snn.state_dict()[name].shape)

    def test_rate_controls_update_every_signed_activation(self):
        model = modelpool("resnet20_signed", "cifar100")
        mode = configure_snn(
            model,
            {"signed": False, "ftbc_mode": "state_low_rank"},
            time_steps=4,
        )
        neurons = [
            module for module in model.modules() if isinstance(module, SignedIF)
        ]

        self.assertEqual(mode, "state_low_rank")
        self.assertTrue(all(module.T == 4 for module in neurons))
        self.assertTrue(all(not module.enable_signed for module in neurons))
        self.assertTrue(all(module.enable_r0 for module in neurons))
        self.assertTrue(all(module.coding_mode == "rate" for module in neurons))
        self.assertTrue(all(module.ftbc_mode == "state_low_rank" for module in neurons))

    def test_forward_shapes_match_in_ann_and_snn_modes(self):
        ann = modelpool("resnet20", "cifar100").eval()
        snn = modelpool("resnet20_signed", "cifar100").eval()
        snn.set_T(2)
        snn.set_ftbc_mode("none")
        inputs = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            ann_output = ann(inputs)
            snn_output = snn(inputs)

        self.assertEqual(ann_output.shape, (2, 100))
        self.assertEqual(snn_output.shape, (2, 2, 100))


class QcfsCheckpointTest(unittest.TestCase):
    def _checkpoint_path(self, stem):
        return (
            Path(__file__).resolve().parent
            / f".{stem}_{uuid.uuid4().hex}.pth"
        )

    def test_exact_qcfs_checkpoint_loads_into_ann_and_signed_snn(self):
        source = modelpool("resnet20", "cifar100")
        path = self._checkpoint_path("resnet20_qcfs")
        try:
            torch.save(source.state_dict(), path)
            ann, snn, metadata = load_qcfs_pair(
                path, "cifar100", "resnet20", torch.device("cpu")
            )
        finally:
            if path.exists():
                path.unlink()

        pairs = match_calibration_layers(ann, snn)
        self.assertEqual(len(pairs), 19)
        self.assertEqual(metadata["qcfs_layers"], 19)
        self.assertEqual(len(metadata["sha256"]), 64)
        for name, ann_layer, snn_layer in pairs:
            self.assertEqual(name, name)
            self.assertTrue(torch.equal(ann_layer.thresh, snn_layer.thresh))
            self.assertTrue(torch.equal(snn_layer.neg_thresh, -ann_layer.thresh))

    def test_non_qcfs_checkpoint_is_rejected(self):
        path = self._checkpoint_path("wrong")
        try:
            torch.save({"fc.weight": torch.zeros(1)}, path)
            with self.assertRaisesRegex(RuntimeError, "not an exact"):
                load_qcfs_pair(
                    path, "cifar100", "resnet20", torch.device("cpu")
                )
        finally:
            if path.exists():
                path.unlink()


class ResNet20ProtocolTest(unittest.TestCase):
    def _args(self, **changes):
        values = dict(
            run_kind="formal",
            time_steps=list(FORMAL_TIME_STEPS),
            configs=list(CONFIGS),
            batch_size=200,
            cali_batches=5,
            seed=42,
            qcfs_L=8,
            alpha=0.4,
            ridge=1e-3,
            coefficient_clip=0.25,
            over_weight=2.5,
            under_weight=1.0,
            min_ann_accuracy=FORMAL_MIN_ANN_ACCURACY,
            max_t32_conversion_gap=2.0,
        )
        values.update(changes)
        return argparse.Namespace(**values)

    def test_six_groups_form_symmetric_two_by_three_matrix(self):
        self.assertEqual(
            list(CONFIGS),
            [
                "A_QCFS_R0",
                "B_QCFS_SNM_R0",
                "C_QCFS_R0_FULL_FTBC",
                "D_QCFS_SNM_R0_FULL_FTBC",
                "E_QCFS_R0_STATE_LR",
                "F_QCFS_SNM_R0_STATE_LR",
            ],
        )

    def test_time_steps_are_selected_by_run_kind_when_omitted(self):
        formal = build_parser().parse_args(
            ["--checkpoint", "model.pth", "--run_kind", "formal"]
        )
        smoke = build_parser().parse_args(["--checkpoint", "model.pth"])
        self.assertIsNone(formal.time_steps)
        self.assertIsNone(smoke.time_steps)
        self.assertIsNone(formal.cali_batches)
        self.assertIsNone(smoke.cali_batches)
        formal.time_steps = resolve_time_steps(formal.run_kind, formal.time_steps)
        smoke.time_steps = resolve_time_steps(smoke.run_kind, smoke.time_steps)
        self.assertEqual(formal.time_steps, [1, 2, 4, 8, 16, 32])
        self.assertEqual(smoke.time_steps, [2, 4])
        self.assertEqual(
            resolve_calibration_batches(formal.run_kind, formal.cali_batches),
            5,
        )
        self.assertEqual(
            resolve_calibration_batches(smoke.run_kind, smoke.cali_batches),
            1,
        )
        self.assertEqual(
            [(value["signed"], value["ftbc_mode"]) for value in CONFIGS.values()],
            [
                (False, "none"),
                (True, "none"),
                (False, "full"),
                (True, "full"),
                (False, "state_low_rank"),
                (True, "state_low_rank"),
            ],
        )

    def test_state_low_rank_falls_back_only_below_three_steps(self):
        self.assertEqual(effective_ftbc_mode("state_low_rank", 1), "full")
        self.assertEqual(effective_ftbc_mode("state_low_rank", 2), "full")
        self.assertEqual(
            effective_ftbc_mode("state_low_rank", 4), "state_low_rank"
        )

    def test_formal_protocol_accepts_only_fixed_design(self):
        validate_formal_protocol(self._args())
        with self.assertRaisesRegex(ValueError, "time_steps"):
            validate_formal_protocol(self._args(time_steps=[2, 4]))
        with self.assertRaisesRegex(ValueError, "all six"):
            validate_formal_protocol(self._args(configs=list(CONFIGS)[:5]))
        with self.assertRaisesRegex(ValueError, "alpha"):
            validate_formal_protocol(self._args(alpha=0.5))

    def test_formal_run_is_locked_to_selected_68_78_checkpoint(self):
        validate_formal_checkpoint(
            self._args(), {"sha256": FORMAL_CHECKPOINT_SHA256}
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
            validate_formal_checkpoint(self._args(), {"sha256": "0" * 64})
        validate_formal_checkpoint(
            self._args(run_kind="smoke"), {"sha256": "0" * 64}
        )

    def test_t32_gate_accepts_small_gap_and_rejects_large_gap(self):
        model = modelpool("resnet20_signed", "cifar100")
        loader = [(torch.zeros(1, 3, 32, 32), torch.zeros(1, dtype=torch.long))]
        accuracy, gap = validate_t32_conversion(
            model,
            loader,
            torch.device("cpu"),
            ann_accuracy=1.0,
            max_gap=2.0,
        )
        self.assertEqual(accuracy, 0.0)
        self.assertEqual(gap, 1.0)
        with self.assertRaisesRegex(RuntimeError, "conversion gate failed"):
            validate_t32_conversion(
                model,
                [(torch.zeros(1, 3, 32, 32), torch.zeros(1, dtype=torch.long))],
                torch.device("cpu"),
                ann_accuracy=10.0,
                max_gap=2.0,
            )

    def test_progress_round_trip_and_signature_gate(self):
        path = (
            Path(__file__).resolve().parent
            / f".progress_{uuid.uuid4().hex}.json"
        )
        configs = OrderedDict(
            [("A_QCFS_R0", CONFIGS["A_QCFS_R0"])]
        )
        args = argparse.Namespace(
            weight_origin="official_implementation_retrained",
            official_commit="eca136b",
            run_kind="smoke",
            time_steps=[2],
            batch_size=200,
            seed=42,
            qcfs_L=8,
            alpha=0.4,
            ridge=1e-3,
            coefficient_clip=0.25,
            over_weight=2.5,
            under_weight=1.0,
            cali_batches=1,
            min_ann_accuracy=69.0,
            max_t32_conversion_gap=2.0,
        )
        metadata = {"sha256": "a" * 64}
        signature = protocol_signature(
            args,
            configs,
            metadata,
            calibration_sha256="b" * 64,
        )
        results = {"A_QCFS_R0": {2: {"acc": 10.0}}}
        layer = SpikeLayerStats(
            name="conv1.2",
            kind="Conv2d",
            time_steps=2,
            output_neurons_per_step=10,
            positive_spikes=2,
            negative_spikes=0,
        )
        try:
            save_progress(
                path,
                signature,
                results,
                {"A_QCFS_R0": {2: [layer]}},
            )
            restored_results, restored_layers = load_progress(
                path, signature, configs
            )
            self.assertEqual(restored_results, results)
            self.assertEqual(restored_layers["A_QCFS_R0"][2][0].name, "conv1.2")
            invalid_signature = dict(signature, seed=7)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                load_progress(path, invalid_signature, configs)
        finally:
            if path.exists():
                path.unlink()

    def test_calibration_hash_is_content_sensitive_and_repeatable(self):
        first = [(torch.tensor([[1.0]]), torch.tensor([2]))]
        same = [(torch.tensor([[1.0]]), torch.tensor([2]))]
        changed = [(torch.tensor([[1.0]]), torch.tensor([3]))]
        self.assertEqual(
            calibration_batches_sha256(first),
            calibration_batches_sha256(same),
        )
        self.assertNotEqual(
            calibration_batches_sha256(first),
            calibration_batches_sha256(changed),
        )


if __name__ == "__main__":
    unittest.main()
