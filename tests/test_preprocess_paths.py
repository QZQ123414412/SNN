# 验证数据集目录在Windows/Linux和环境变量下可移植
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from preprocess.augment import CIFAR10Policy, Cutout
from preprocess.getdataloader import (
    cifar100_train_transform,
    resolve_dataset_root,
    resolve_num_workers,
)


class DatasetRootResolutionTest(unittest.TestCase):
    def test_cifar100_augmentation_profiles_differ_only_by_autoaugment(self):
        fixed = cifar100_train_transform("fixed_repo").transforms
        paper_era = cifar100_train_transform("paper_era").transforms

        self.assertTrue(any(isinstance(item, CIFAR10Policy) for item in fixed))
        self.assertFalse(
            any(isinstance(item, CIFAR10Policy) for item in paper_era)
        )
        self.assertEqual(
            [type(item) for item in fixed if not isinstance(item, CIFAR10Policy)],
            [type(item) for item in paper_era],
        )

    def test_cifar100_cutout_length_is_explicit_without_changing_default(self):
        default = cifar100_train_transform("fixed_repo").transforms
        paper_reproduction = cifar100_train_transform(
            "fixed_repo", cutout_length=8
        ).transforms
        default_cutout = next(item for item in default if isinstance(item, Cutout))
        paper_cutout = next(
            item for item in paper_reproduction if isinstance(item, Cutout)
        )
        self.assertEqual(default_cutout.length, 16)
        self.assertEqual(paper_cutout.length, 8)

    def test_dataset_specific_environment_variable_has_priority(self):
        with patch.dict(
            os.environ,
            {
                "QCFS_DATA_ROOT": "D:/common-data",
                "QCFS_CIFAR100_ROOT": "D:/cifar100-data",
            },
            clear=False,
        ):
            root = resolve_dataset_root("CIFAR100")

        self.assertEqual(root, "D:/cifar100-data")

    def test_home_datasets_directory_is_default(self):
        home = Path("D:/portable-home")
        expected = home / "datasets"
        with patch.dict(os.environ, {}, clear=True):
            root = resolve_dataset_root(
                "CIFAR100",
                home=home,
            )

        self.assertEqual(Path(root), expected)

    def test_existing_linux_legacy_root_is_preserved(self):
        root = resolve_dataset_root(
            "CIFAR100",
            environ={},
            home=Path("/home/researcher"),
            platform_name="posix",
            path_exists=lambda path: path == "/root/autodl-tmp/datasets",
        )

        self.assertEqual(root, "/root/autodl-tmp/datasets")

    def test_windows_defaults_to_single_process_data_loading(self):
        workers = resolve_num_workers(
            default=8,
            environ={},
            platform_name="nt",
        )

        self.assertEqual(workers, 0)

    def test_worker_environment_override_has_priority(self):
        workers = resolve_num_workers(
            default=8,
            environ={"QCFS_NUM_WORKERS": "3"},
            platform_name="nt",
        )

        self.assertEqual(workers, 3)


if __name__ == "__main__":
    unittest.main()
