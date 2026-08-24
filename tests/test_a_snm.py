import unittest

from a_snm import a_snm_enabled, select_a_snm_modes


def metric(acc):
    return {"acc": acc}


class ASNMTest(unittest.TestCase):
    def test_snm_is_enabled_only_for_strict_accuracy_gain(self):
        off = {1: metric(70.0), 2: metric(71.0), 4: metric(72.0)}
        on = {1: metric(70.0), 2: metric(70.9), 4: metric(72.1)}

        selected, trace = select_a_snm_modes(off, on, time_steps=(1, 2, 4))

        self.assertEqual(selected, {1: False, 2: False, 4: True})
        self.assertEqual(trace["1"]["selected_mode"], "off")
        self.assertEqual(trace["2"]["selected_mode"], "off")
        self.assertEqual(trace["4"]["selected_mode"], "on")

    def test_non_monotonic_time_decisions_are_preserved(self):
        off = {1: metric(70.0), 2: metric(71.0), 4: metric(72.0)}
        on = {1: metric(70.1), 2: metric(70.9), 4: metric(72.2)}

        selected, _ = select_a_snm_modes(off, on, time_steps=(1, 2, 4))

        self.assertTrue(a_snm_enabled(selected, 1))
        self.assertFalse(a_snm_enabled(selected, 2))
        self.assertTrue(a_snm_enabled(selected, 4))

    def test_missing_time_step_is_rejected(self):
        with self.assertRaises(ValueError):
            select_a_snm_modes({1: metric(70.0)}, {}, time_steps=(1,))
        with self.assertRaises(ValueError):
            a_snm_enabled({1: False}, 2)

    def test_empty_time_steps_are_rejected(self):
        with self.assertRaises(ValueError):
            select_a_snm_modes({}, {}, time_steps=())


if __name__ == "__main__":
    unittest.main()
