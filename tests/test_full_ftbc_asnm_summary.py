import json
import unittest
import uuid
from pathlib import Path

from scripts.experiments.summarize_full_ftbc_asnm import load_complete_progress


class FullFtbcAsnmSummaryTest(unittest.TestCase):
    def write_payload(self, **changes):
        payload = {
            "status": "complete",
            "protocol": {
                "dataset": "cifar10",
                "architecture": "resnet20",
                "test_samples": 10000,
            },
            "equivalence_checks": [{"exact": True}],
        }
        payload.update(changes)
        path = Path(__file__).resolve().parent / f".summary_{uuid.uuid4().hex}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_complete_formal_progress_is_accepted(self):
        path = self.write_payload()
        try:
            payload = load_complete_progress(path, "cifar10", "resnet20")
        finally:
            if path.exists():
                path.unlink()

        self.assertEqual(payload["protocol"]["test_samples"], 10000)

    def test_incomplete_or_nonformal_progress_is_rejected(self):
        path = self.write_payload(status="a_snm_modes_frozen_testing")
        try:
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                load_complete_progress(path, "cifar10", "resnet20")
        finally:
            if path.exists():
                path.unlink()

        path = self.write_payload(
            protocol={
                "dataset": "cifar10",
                "architecture": "resnet20",
                "test_samples": 200,
            },
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "10,000"):
                load_complete_progress(path, "cifar10", "resnet20")
        finally:
            if path.exists():
                path.unlink()

    def test_failed_cached_equivalence_is_rejected(self):
        path = self.write_payload(equivalence_checks=[{"exact": False}])
        try:
            with self.assertRaisesRegex(RuntimeError, "equivalence"):
                load_complete_progress(path, "cifar10", "resnet20")
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
