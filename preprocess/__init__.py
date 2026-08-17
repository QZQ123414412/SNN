from .getdataloader import *

def datapool(
    DATANAME,
    batchsize,
    augmentation_profile="fixed_repo",
    cutout_length=16,
):
    if DATANAME.lower() == 'cifar10':
        return GetCifar10(batchsize)
    elif DATANAME.lower() == 'cifar100':
        return GetCifar100(
            batchsize,
            augmentation_profile=augmentation_profile,
            cutout_length=cutout_length,
        )
    elif DATANAME.lower() == 'imagenet':
        return GetImageNet(batchsize)
    else:
        print("still not support this model")
        exit(0)
