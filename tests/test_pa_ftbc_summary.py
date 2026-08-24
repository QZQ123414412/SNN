import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

from scripts.experiments.summarize_pa_ftbc import (
    REPORTS,
    load_payload,
    regression_audit,
    write_summary,
)


class PAFTBCSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads = OrderedDict(
            (label, load_payload(path)) for label, path in REPORTS.items()
        )

    def test_all_four_formal_reports_are_complete(self):
        self.assertEqual(len(self.payloads), 4)
        for payload in self.payloads.values():
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(len(payload["results"]), 12)
            self.assertEqual(len(payload["equivalence_checks"]), 54)
            self.assertTrue(all(x["exact"] for x in payload["equivalence_checks"]))

    def test_existing_result_regression_is_exact(self):
        self.assertEqual(regression_audit(self.payloads), 1386)

    def test_summary_is_generated_from_payloads(self):
        with patch.object(Path, "write_text") as write_text:
            write_summary(Path("summary.md"), self.payloads, 1386)
        text = write_text.call_args.args[0]
        self.assertIn("Status: complete", text)
        self.assertIn("216/216 exact", text)
        self.assertIn("1386, mismatches: 0", text)
        self.assertIn("maximum absolute six-step mean-accuracy difference", text)
        self.assertIn("0.103pp", text)


if __name__ == "__main__":
    unittest.main()
