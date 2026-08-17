import argparse
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import numpy as np
from models import modelpool
from preprocess import datapool
from utils import train, val, seed_all, get_logger

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers', default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size', default=300, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix', default='', type=str,help='suffix')
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg16',type=str,help='model')

# training configuration
parser.add_argument('--epochs',default=300,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate') # 0.05 for cifar100 / 0.1 for cifar10
parser.add_argument('-wd','--weight_decay',default=5e-4, type=float, help='weight_decay')
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-L', '--L', default=8, type=int, help='Step L')
parser.add_argument(
    '--augmentation_profile',
    choices=('fixed_repo', 'paper_era'),
    default='fixed_repo',
    help='CIFAR-100 augmentation recipe; paper_era omits AutoAugment',
)
parser.add_argument(
    '--qcfs_training_profile',
    choices=('fixed_repo', 'paper_era'),
    default='fixed_repo',
    help='QCFS clamp/quantize order used during ANN training',
)
parser.add_argument('--resume_state', type=str, help='full training-state checkpoint')
parser.add_argument('--state_path', type=str, help='path for resumable training state')
parser.add_argument(
    '--run_epochs',
    type=int,
    help='maximum epochs to execute in this invocation (for exact chunked runs)',
)

args = None
device = None


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def training_config(args):
    return {
        "dataset": args.dataset,
        "model": args.model,
        "L": args.L,
        "time": args.time,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "augmentation_profile": args.augmentation_profile,
        "qcfs_training_profile": args.qcfs_training_profile,
    }

def main(cli_args=None):
    global args, device
    args = parser.parse_args(cli_args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    # preparing data
    train_loader, test_loader = datapool(
        args.dataset,
        args.batch_size,
        augmentation_profile=args.augmentation_profile,
    )
    # preparing model
    model = modelpool(args.model, args.dataset)
    model.set_L(args.L)
    model.set_qcfs_training_profile(args.qcfs_training_profile)

    log_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0
    start_epoch = 0

    identifier = args.model

    identifier += '_L[%d]'%(args.L)

    if not args.suffix == '':
        identifier += '_%s'%(args.suffix)

    state_path = Path(args.state_path) if args.state_path else Path(
        log_dir, f"{identifier}.train_state.pth"
    )
    resume = None
    if args.resume_state:
        resume = torch.load(args.resume_state, map_location="cpu")
        expected_config = training_config(args)
        if resume.get("config") != expected_config:
            raise RuntimeError(
                "Training-state configuration mismatch: "
                f"saved={resume.get('config')}, requested={expected_config}"
            )
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        best_acc = float(resume["best_acc"])
        start_epoch = int(resume["epoch"]) + 1
        restore_rng_state(resume["rng_state"])

    logger = get_logger(
        os.path.join(log_dir, '%s.log'%(identifier)),
        filemode="a" if resume is not None else "w",
    )
    if resume is None:
        logger.info('start training!')
    else:
        logger.info(
            'resume training from epoch %d with best acc %.3f',
            start_epoch,
            best_acc,
        )

    if args.run_epochs is not None and args.run_epochs <= 0:
        raise ValueError("run_epochs must be positive")
    stop_epoch = args.epochs
    if args.run_epochs is not None:
        stop_epoch = min(args.epochs, start_epoch + args.run_epochs)

    for epoch in range(start_epoch, stop_epoch):
        loss, acc = train(model, device, train_loader, criterion, optimizer, args.time)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc))
        scheduler.step()
        tmp = val(model, test_loader, device, args.time)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch , args.epochs, tmp))

        if best_acc < tmp:
            best_acc = tmp
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth'%(identifier)))

        atomic_torch_save(
            {
                "epoch": epoch,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": capture_rng_state(),
                "config": training_config(args),
            },
            state_path,
        )

    logger.info('Best Test acc={:.3f}'.format(best_acc))

if __name__ == "__main__":
    main()
