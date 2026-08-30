import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from fetch_osha_incidents import safe_csv_member


class FetchOshaIncidentTests(unittest.TestCase):
    def test_accepts_one_csv(self):
        self.assertEqual(safe_csv_member(["report.csv"]), "report.csv")

    def test_rejects_archive_traversal(self):
        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            safe_csv_member(["../report.csv"])

    def test_rejects_ambiguous_csvs(self):
        with self.assertRaisesRegex(ValueError, "exactly one CSV"):
            safe_csv_member(["one.csv", "two.csv"])


if __name__ == "__main__":
    unittest.main()
