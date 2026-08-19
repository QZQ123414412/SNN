import unittest
from collections import OrderedDict

from horizon_gate import aggregate_subset_metrics, select_robust_candidate


def metrics(acc, sops=1000, negatives=100, mse=1.0):
    return {
        "acc": acc,
        "sops": sops,
        "negative_spikes": negatives,
        "logit_mse": mse,
    }


class HorizonGateTest(unittest.TestCase):
    def test_aggregate_penalizes_subset_instability(self):
        stable = aggregate_subset_metrics([metrics(80), metrics(80), metrics(80)])
        unstable = aggregate_subset_metrics([metrics(79), metrics(80), metrics(81)])
        self.assertEqual(stable["mean_acc"], unstable["mean_acc"])
        self.assertGreater(stable["robust_acc"], unstable["robust_acc"])

    def test_accuracy_gain_beats_lower_overhead(self):
        candidates = OrderedDict(
            off=aggregate_subset_metrics([metrics(70, 900, 0)] * 3),
            standard=aggregate_subset_metrics([metrics(71, 1000, 100)] * 3),
        )
        winner, _ = select_robust_candidate(candidates, accuracy_tolerance=0.1)
        self.assertEqual(winner, "standard")

    def test_true_off_wins_inside_accuracy_tolerance(self):
        candidates = OrderedDict(
            off=aggregate_subset_metrics([metrics(70.0, 900, 0)] * 3),
            standard=aggregate_subset_metrics([metrics(70.05, 1000, 100)] * 3),
        )
        winner, trace = select_robust_candidate(
            candidates,
            accuracy_tolerance=0.1,
        )
        self.assertEqual(winner, "off")
        self.assertTrue(trace["off"]["accuracy_eligible"])

    def test_teacher_mse_breaks_saturated_accuracy_tie(self):
        candidates = OrderedDict(
            off=aggregate_subset_metrics([metrics(100, 900, 0, mse=0.35)] * 3),
            standard=aggregate_subset_metrics(
                [metrics(100, 1000, 100, mse=0.18)] * 3
            ),
        )
        winner, _ = select_robust_candidate(candidates, accuracy_tolerance=0.1)
        self.assertEqual(winner, "standard")

    def test_empty_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_subset_metrics([])
        with self.assertRaises(ValueError):
            select_robust_candidate({})


if __name__ == "__main__":
    unittest.main()
