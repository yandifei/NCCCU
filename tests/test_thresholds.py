"""用穷举核对阈值搜索，覆盖同分概率、边界和无效输入。"""
import unittest

import numpy as np

from ncccu.thresholds import f1_scores, optimize_thresholds, patient_folds


class ThresholdTests(unittest.TestCase):
    def test_search_matches_brute_force_with_ties(self):
        labels = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [0, 1]])
        scores = np.array([[0.1, 1.0], [0.2, 0.0], [0.2, 0.4], [0.9, 0.4], [0.0, 0.5]])
        result = optimize_thresholds(labels, scores)
        candidates = np.unique(np.r_[0.0, scores.ravel(), 1.0])
        brute = []
        for threshold in candidates:
            predicted = scores >= threshold
            tp = (predicted * labels).sum(0)
            denominator = predicted.sum(0) + labels.sum(0)
            brute.append(np.divide(2*tp, denominator, out=np.zeros(2), where=denominator != 0))
        brute = np.array(brute)
        expected_global = candidates[np.flatnonzero(brute.mean(1) == brute.mean(1).max())[-1]]
        self.assertEqual(result['global_threshold'], expected_global)
        np.testing.assert_allclose(result['curve_macro_f1'], brute.mean(1))
        for index in range(2):
            expected = candidates[np.flatnonzero(brute[:, index] == brute[:, index].max())[-1]]
            self.assertEqual(result['class_thresholds'][index], expected)
        np.testing.assert_allclose(f1_scores(labels, scores, result['class_thresholds']), brute.max(0))

    def test_all_equal_scores_and_rejected_inputs(self):
        labels = np.array([[1], [0]])
        self.assertEqual(optimize_thresholds(labels, np.full((2, 1), 0.3))['global_threshold'], 0.3)
        for scores in (np.array([[np.nan], [0.1]]), np.array([[1.1], [0.1]])):
            with self.assertRaises(ValueError):
                optimize_thresholds(labels, scores)
        with self.assertRaises(ValueError):
            optimize_thresholds(np.zeros((2, 1)), np.zeros((2, 1)))

    def test_patient_folds_never_split_one_patient(self):
        subjects = ['a', 'a', 'b', 'c', 'd', 'd', 'e', 'f']
        folds = patient_folds(subjects, count=3, seed=0)
        self.assertEqual(folds.tolist(), patient_folds(subjects, count=3, seed=0).tolist())
        self.assertEqual(set(folds), {0, 1, 2})
        for subject in set(subjects):
            self.assertEqual(len(set(folds[np.array(subjects) == subject])), 1)


if __name__ == '__main__':
    unittest.main()
