# 验证逐次精化消融配置和校准权重选择规则
import unittest
from types import SimpleNamespace

from scripts.experiments.run_monotonic_refinement_ablation import (
    CONFIGS,
    effective_ftbc_mode,
    select_calibrated_parameters,
    select_calibrated_ratio,
    use_deterministic_calibration_transform,
)


class MonotonicAblationTest(unittest.TestCase):
    def test_complete_direction_one_ablation_is_present(self):
        self.assertEqual(
            list(CONFIGS),
            [
                "A_RATE_QCFS",
                "B_RATE_SNM_R0",
                "C_UNIFORM_REFINEMENT",
                "D_READOUT_BINARY",
                "E_BINARY_REFINEMENT",
                "F_CALIBRATED_REFINEMENT",
                "G_CALIBRATED_FULL_FTBC",
                "H_CALIBRATED_STATE_LR",
            ],
        )

    def test_ratio_selection_uses_sops_within_accuracy_tolerance(self):
        candidates = {
            1.1: {"acc": 70.00, "sops": 1000},
            1.5: {"acc": 70.03, "sops": 1200},
            2.0: {"acc": 69.99, "sops": 800},
        }

        selected = select_calibrated_ratio(candidates, accuracy_tolerance=0.05)

        self.assertEqual(selected, 2.0)

    def test_ratio_selection_rejects_large_accuracy_loss(self):
        candidates = {
            1.1: {"acc": 70.00, "sops": 1000},
            2.0: {"acc": 69.80, "sops": 100},
        }

        selected = select_calibrated_ratio(candidates, accuracy_tolerance=0.05)

        self.assertEqual(selected, 1.1)

    def test_joint_parameter_selection_uses_accuracy_then_sops(self):
        candidates = {
            (1.0, 0.5, 0.5): {"acc": 70.00, "sops": 1000},
            (1.1, 0.55, 1.0): {"acc": 70.03, "sops": 1200},
            (1.05, 0.55, 1.3): {"acc": 69.99, "sops": 800},
        }

        selected = select_calibrated_parameters(
            candidates,
            accuracy_tolerance=0.05,
        )

        self.assertEqual(selected, (1.05, 0.55, 1.3))

    def test_candidate_exactly_on_accuracy_tolerance_is_eligible(self):
        candidates = {
            (1.0, 0.5, 0.5): {"acc": 96.2, "sops": 1000},
            (1.0, 0.55, 0.5): {"acc": 96.0, "sops": 800},
        }

        selected = select_calibrated_parameters(
            candidates,
            accuracy_tolerance=0.2,
        )

        self.assertEqual(selected, (1.0, 0.55, 0.5))

    def test_state_low_rank_falls_back_for_short_time_windows(self):
        self.assertEqual(effective_ftbc_mode("state_low_rank", 1), "full")
        self.assertEqual(effective_ftbc_mode("state_low_rank", 2), "full")
        self.assertEqual(
            effective_ftbc_mode("state_low_rank", 4),
            "state_low_rank",
        )

    def test_calibration_uses_training_samples_with_evaluation_transform(self):
        train_loader = SimpleNamespace(
            dataset=SimpleNamespace(transform="train-augmentation")
        )
        test_loader = SimpleNamespace(
            dataset=SimpleNamespace(transform="deterministic-eval")
        )

        use_deterministic_calibration_transform(train_loader, test_loader)

        self.assertEqual(train_loader.dataset.transform, "deterministic-eval")


if __name__ == "__main__":
    unittest.main()
