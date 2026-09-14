import json
import os
import tempfile
import unittest

import numpy as np
import torch

from main import load_data_from_dir


class DataLoadingTest(unittest.TestCase):
    def _write_json(self, directory):
        with open(os.path.join(directory, "sample.json"), "w", encoding="utf-8") as f:
            json.dump({"history": [{"content": "x"}], "mistake_step": 0}, f)

    def test_missing_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory)

            with self.assertRaisesRegex(FileNotFoundError, "Cache not found"):
                load_data_from_dir(directory)

    def test_all_zero_cached_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_json(directory)
            cache_dir = os.path.join(directory, "cache")
            os.makedirs(cache_dir)
            torch.save(
                {
                    "content_features": np.zeros((1, 128), dtype=np.float32),
                    "agent_features": np.zeros((1, 32), dtype=np.float32),
                    "mistake_labels": np.zeros(1, dtype=np.int64),
                },
                os.path.join(cache_dir, "sample_multi_features.pt"),
            )

            with self.assertRaisesRegex(ValueError, "Invalid cached labels"):
                load_data_from_dir(directory)


if __name__ == "__main__":
    unittest.main()
