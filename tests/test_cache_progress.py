"""缓存必须在首张读取前打印状态，长循环中定期打印进度。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ncccu.multilabel import ChestDataset


class CacheProgressTests(unittest.TestCase):
    def test_disk_cache_reports_start_progress_and_completion(self):
        dataset = ChestDataset.__new__(ChestDataset)
        dataset.rows = [None] * 3
        dataset.split = "train"
        dataset.memory_cache = {}
        events = []

        def read_image(index):
            events.append(("read", index))
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def log(message, **kwargs):
            self.assertTrue(kwargs["flush"])
            events.append(("log", message))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache"
            with patch.object(dataset, "read_image", side_effect=read_image),                     patch("ncccu.multilabel.time.monotonic", side_effect=[0, 1, 7, 8]),                     patch("builtins.print", side_effect=log):
                dataset.prepare_cache("disk", path)
            self.assertEqual(events[0][0], "log")
            messages = [value for kind, value in events if kind == "log"]
            self.assertTrue(any("2/3" in text for text in messages))
            self.assertTrue(any("3/3 (100.0%)" in text for text in messages))
            self.assertEqual(len(list(path.glob("*.npy"))), 3)
            np.testing.assert_array_equal(np.load(path / "0.npy"), np.zeros((2, 2, 3)))

    def test_disabled_cache_is_reported(self):
        dataset = ChestDataset.__new__(ChestDataset)
        dataset.split = "train"
        with patch("builtins.print") as log:
            dataset.prepare_cache(False, Path("unused"))
        self.assertIn("已关闭", log.call_args.args[0])
        self.assertTrue(log.call_args.kwargs["flush"])


if __name__ == "__main__":
    unittest.main()
