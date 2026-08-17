from textwrap import fill
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
from pathlib import Path
from preprocess.augment import Cutout, CIFAR10Policy


def resolve_dataset_root(
    dataset_name,
    environ=None,
    home=None,
    platform_name=None,
    path_exists=None,
):
    """Resolve a portable dataset root without embedding a user-specific path."""
    environ = os.environ if environ is None else environ
    dataset_name = str(dataset_name).upper()
    specific = environ.get(f"QCFS_{dataset_name}_ROOT")
    if specific:
        return specific
    common = environ.get("QCFS_DATA_ROOT")
    if common:
        return common
    if dataset_name == "IMAGENET":
        return "YOUR_IMAGENET_DIR"

    platform_name = os.name if platform_name is None else platform_name
    path_exists = os.path.exists if path_exists is None else path_exists
    legacy_root = "/root/autodl-tmp/datasets"
    if platform_name != "nt" and path_exists(legacy_root):
        return legacy_root

    home = Path.home() if home is None else Path(home)
    return str(home / "datasets")


def resolve_num_workers(default, environ=None, platform_name=None):
    """Use worker processes where supported, with an explicit override."""
    environ = os.environ if environ is None else environ
    override = environ.get("QCFS_NUM_WORKERS")
    if override is not None:
        workers = int(override)
        if workers < 0:
            raise ValueError("QCFS_NUM_WORKERS must be non-negative")
        return workers
    platform_name = os.name if platform_name is None else platform_name
    return 0 if platform_name == "nt" else int(default)

def GetCifar10(batchsize, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    root = resolve_dataset_root("CIFAR10")
    train_data = datasets.CIFAR10(root, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(root, train=False, transform=trans, download=True)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batchsize,
        shuffle=True,
        num_workers=resolve_num_workers(8),
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batchsize,
        shuffle=False,
        num_workers=resolve_num_workers(8),
    )
    return train_dataloader, test_dataloader

def cifar100_train_transform(
    augmentation_profile="fixed_repo",
    cutout_length=16,
):
    if augmentation_profile not in {"fixed_repo", "paper_era"}:
        raise ValueError(
            "augmentation_profile must be 'fixed_repo' or 'paper_era'"
        )
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation_profile == "fixed_repo":
        augmentations.append(CIFAR10Policy())
    augmentations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                std=[n / 255. for n in [68.2, 65.4, 70.4]],
            ),
            Cutout(n_holes=1, length=cutout_length),
        ]
    )
    return transforms.Compose(augmentations)


def GetCifar100(
    batchsize,
    augmentation_profile="fixed_repo",
    cutout_length=16,
):
    trans_t = cifar100_train_transform(
        augmentation_profile,
        cutout_length=cutout_length,
    )
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    root = resolve_dataset_root("CIFAR100")
    train_data = datasets.CIFAR100(root, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(root, train=False, transform=trans, download=True)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batchsize,
        shuffle=True,
        num_workers=resolve_num_workers(8),
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batchsize,
        shuffle=False,
        num_workers=resolve_num_workers(4),
        pin_memory=True,
    )
    return train_dataloader, test_dataloader

def GetImageNet(batchsize):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    
    trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    root = resolve_dataset_root("ImageNet")
    train_data = datasets.ImageFolder(root=os.path.join(root, 'train'), transform=trans_t)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader =DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8, sampler=train_sampler, pin_memory=True)

    test_data = datasets.ImageFolder(root=os.path.join(root, 'val'), transform=trans)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2, sampler=test_sampler) 
    return train_dataloader, test_dataloader
