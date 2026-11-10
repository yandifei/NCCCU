"""用临时样例核对测速前的输入检查、多视图顺序和不一致计数。"""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import cost


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "datasets"
        self.directory = self.data / "test/subject/study"
        self.directory.mkdir(parents=True)
        self.views = [self.directory / "a.jpg", self.directory / "b.jpg"]
        for path in self.views:
            path.touch()
        (self.data / "test.csv").write_text(
            "Subject_id,Study_id,Dicom_id\nsubject,study,a\nsubject,study,b\n", encoding="utf-8",
        )
        self.submission = self.root / "submission.csv"
        self.probabilities = np.linspace(0.1, 0.9, 10)
        self.rows = [
            ["subject", "study", label, float(value), 0.5]
            for label, value in enumerate(self.probabilities, start=1)
        ]
        self.config = {"gate": 0.5, "quantize": 32}
        self.write_reference(self.rows)

    def write_reference(self, rows: list[list[str | int | float]]) -> None:
        with self.submission.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("Subject_id", "Study_id", "Predict_class", "Probability", "Gate"))
            writer.writerows(rows)

    def test_empty_samples_fail_before_loading_model(self) -> None:
        with patch.object(cost, "_load_predictor") as load:
            with self.assertRaisesRegex(ValueError, "不能为空"):
                cost.benchmark({}, self.root, self.submission, [], [])
        load.assert_not_called()

    def test_invalid_reference_fails_before_loading_model(self) -> None:
        for field, value in ((3, "nan"), (3, "inf"), (3, -0.1), (4, "nan"), (4, 1.1), (2, 11)):
            with self.subTest(field=field, value=value):
                rows = [row.copy() for row in self.rows]
                rows[0][field] = value
                self.write_reference(rows)
                with patch.object(cost, "_load_predictor") as load:
                    with self.assertRaises(ValueError):
                        cost.benchmark(self.config, self.root, self.submission, self.views[:1], [])
                load.assert_not_called()
        for rows in (self.rows + self.rows[:1], self.rows[:-1]):
            with self.subTest(rows=len(rows)):
                self.write_reference(rows)
                with self.assertRaises(ValueError):
                    cost.benchmark(self.config, self.root, self.submission, self.views[:1], [])

    def test_multiview_warmup_and_three_round_measurement(self) -> None:
        with (
            patch.object(cost, "_load_predictor") as load,
            patch.object(cost, "perf_counter", side_effect=(0, 0.05, 1, 1.08, 2, 2.03)),
        ):
            infer = load.return_value
            infer.return_value = self.probabilities
            latency, mismatches = cost.benchmark(
                self.config, self.root, self.submission, self.views[:1], self.views[1:],
            )
        self.assertEqual(infer.call_count, 4)
        self.assertTrue(all(call.args == (self.views,) for call in infer.call_args_list))
        self.assertAlmostEqual(latency, (50 + 80 + 30) / 3)
        self.assertEqual(mismatches, 0)

    def test_repeated_mismatch_is_counted_once_per_sample(self) -> None:
        with patch.object(cost, "_load_predictor") as load:
            load.return_value.return_value = self.probabilities + 0.01
            _, mismatches = cost.benchmark(self.config, self.root, self.submission, self.views[:1], [])
        self.assertEqual(mismatches, 1)

    def test_nonfinite_predictions_are_never_consistent(self) -> None:
        rows = {label: (0.2, 0.5) for label in cost.CLASS_IDS}
        self.assertFalse(cost._matches_reference(np.full(10, np.nan), rows, 0.5))

    def test_ambiguous_study_requires_subject_column(self) -> None:
        (self.data / "test.csv").write_text(
            "Subject_id,Study_id,Dicom_id\nsubject,study,a\nother,study,c\n", encoding="utf-8",
        )
        self.submission.write_text("Study_id,Predict_class,Probability,Gate\nstudy,1,0.5,0.5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "跨患者重复"):
            cost.benchmark(self.config, self.root, self.submission, self.views[:1], [])


if __name__ == "__main__":
    unittest.main()
