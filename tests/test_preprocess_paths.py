# 验证数据目录与DataLoader worker配置可跨平台使用
import os
import unittest
from pathlib import Path
from unittest import mock

from preprocess.getdataloader import resolve_dataset_root, resolve_num_workers


class PreprocessPathTest(unittest.TestCase):
    def test_dataset_specific_environment_variable_has_priority(self):
        with mock.patch.dict(
            os.environ,
            {
                "QCFS_DATA_ROOT": "D:/common",
                "QCFS_CIFAR100_ROOT": "D:/specific",
            },
            clear=False,
        ):
            self.assertEqual(resolve_dataset_root("CIFAR100"), "D:/specific")

    def test_common_environment_variable_is_used_as_fallback(self):
        with mock.patch.dict(
            os.environ,
            {"QCFS_DATA_ROOT": "D:/common"},
            clear=True,
        ):
            self.assertEqual(resolve_dataset_root("CIFAR10"), "D:/common")

    def test_worker_environment_variable_overrides_platform_default(self):
        with mock.patch.dict(os.environ, {"QCFS_NUM_WORKERS": "3"}, clear=False):
            self.assertEqual(resolve_num_workers(default=8), 3)

    def test_default_dataset_root_is_absolute(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            root = resolve_dataset_root("CIFAR100")

        self.assertTrue(Path(root).is_absolute(), root)


if __name__ == "__main__":
    unittest.main()
