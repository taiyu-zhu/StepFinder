import unittest

import numpy as np

from feature_construction import TemporalFeatureExtractor


class MistakeLabelTest(unittest.TestCase):
    def test_original_whowhen_root_label(self):
        log = {
            "mistake_step": "1",
            "history": [{"content": "first"}, {"content": "second"}],
        }

        labels = TemporalFeatureExtractor._mistake_labels(log, log["history"])

        np.testing.assert_array_equal(labels, np.array([0, 1]))

    def test_step_labels_remain_supported(self):
        log = {
            "history": [
                {"content": "first", "is_mistake": 1},
                {"content": "second", "is_mistake": 0},
            ]
        }

        labels = TemporalFeatureExtractor._mistake_labels(log, log["history"])

        np.testing.assert_array_equal(labels, np.array([1, 0]))

    def test_disagreeing_label_formats_are_rejected(self):
        log = {
            "mistake_step": 1,
            "history": [
                {"content": "first", "is_mistake": 1},
                {"content": "second", "is_mistake": 0},
            ],
        }

        with self.assertRaisesRegex(ValueError, "disagrees"):
            TemporalFeatureExtractor._mistake_labels(log, log["history"])

    def test_missing_labels_are_rejected(self):
        log = {"history": [{"content": "first"}]}

        with self.assertRaisesRegex(ValueError, "Missing labels"):
            TemporalFeatureExtractor._mistake_labels(log, log["history"])

    def test_all_zero_step_labels_are_rejected(self):
        log = {
            "history": [
                {"content": "first", "is_mistake": 0},
                {"content": "second", "is_mistake": 0},
            ]
        }

        with self.assertRaisesRegex(ValueError, "exactly one"):
            TemporalFeatureExtractor._mistake_labels(log, log["history"])

    def test_out_of_range_root_label_is_rejected(self):
        log = {"mistake_step": 2, "history": [{"content": "first"}]}

        with self.assertRaisesRegex(ValueError, "outside history length"):
            TemporalFeatureExtractor._mistake_labels(log, log["history"])


if __name__ == "__main__":
    unittest.main()
