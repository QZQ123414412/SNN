"""Fixed-budget fine-tuning for the CIFAR-100 / ResNet20 QCFS checkpoint."""

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn

from models import IF, modelpool
from preprocess import datapool
from scripts.experiments.qcfs_checkpoint import checkpoint_sha256, load_qcfs_pair
from utils import seed_all, train, val


DEFAULT_SOURCE = (
    REPO_ROOT
    / "cifar100-checkpoints"
    / "resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "cifar100-checkpoints"
    / "resnet20_qcfs_finetune_68_78_to_69_94"
)
TRAJECTORIES = (
    ("FT_LR005", 0.005),
    ("FT_LR002", 0.002),
    ("FT_LR001", 0.001),
)
TARGET_FILENAME = "resnet20_cifar100_qcfs_L8_target_ge69_94.pth"
GLOBAL_BEST_FILENAME = "resnet20_cifar100_qcfs_L8_global_best.pth"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune the 68.78% CIFAR-100/ResNet20 QCFS checkpoint"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[0.005, 0.002, 0.001])
    parser.add_argument("--trajectory_names", nargs="+", default=[name for name, _ in TRAJECTORIES])
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--target_accuracy", type=float, default=69.94)
    parser.add_argument("--expected_source_accuracy", type=float, default=68.78)
    parser.add_argument("--source_accuracy_tolerance", type=float, default=0.005)
    parser.add_argument("--resume", action="store_true")
    return parser


def experiment_config(args, source_sha256):
    return {
        "architecture": "resnet20",
        "dataset": "cifar100",
        "qcfs_L": 8,
        "qcfs_training_profile": "fixed_repo",
        "augmentation_profile": "paper_era",
        "batch_size": args.batch_size,
        "epochs_per_trajectory": args.epochs,
        "learning_rates": list(args.learning_rates),
        "trajectory_names": list(args.trajectory_names),
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "scheduler_eta_min": 0.0,
        "seed": args.seed,
        "target_accuracy": args.target_accuracy,
        "expected_source_accuracy": args.expected_source_accuracy,
        "source_accuracy_tolerance": args.source_accuracy_tolerance,
        "source_path": str(Path(args.source).resolve()),
        "source_sha256": source_sha256,
        "weight_origin": "official_implementation_finetuned",
        "test_samples": 10000,
    }


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(args.learning_rates) != len(args.trajectory_names):
        raise ValueError("learning_rates and trajectory_names must have equal lengths")
    if len(set(args.trajectory_names)) != len(args.trajectory_names):
        raise ValueError("trajectory_names must be unique")
    if any(rate <= 0 for rate in args.learning_rates):
        raise ValueError("learning rates must be positive")


def validate_qcfs_model(model):
    layers = [module for module in model.modules() if isinstance(module, IF)]
    if len(layers) != 19:
        raise RuntimeError(f"Expected 19 QCFS layers, found {len(layers)}")
    if any(module.L != 8 for module in layers):
        raise RuntimeError("Every QCFS layer must use L=8")
    if any(module.quantization_profile != "fixed_repo" for module in layers):
        raise RuntimeError("Every QCFS layer must use fixed_repo training semantics")


def trajectory_paths(output_dir, name):
    output_dir = Path(output_dir)
    return {
        "best": output_dir / f"{name}.best.pth",
        "state": output_dir / f"{name}.train_state.pth",
        "history": output_dir / f"{name}.history.json",
    }


def target_metadata(config, trajectory_name, epoch, accuracy, checkpoint_path):
    return {
        "actual_accuracy": accuracy,
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "created_at_utc": utc_now(),
        "epoch": epoch,
        "source_checkpoint_sha256": config["source_sha256"],
        "test_samples": config["test_samples"],
        "training_config": config,
        "trajectory": trajectory_name,
        "weight_origin": config["weight_origin"],
    }


def save_first_target(model, output_dir, config, trajectory_name, epoch, accuracy):
    target_path = Path(output_dir) / TARGET_FILENAME
    metadata_path = target_path.with_suffix(".json")
    if target_path.exists() != metadata_path.exists():
        raise RuntimeError(
            "Target checkpoint and metadata must either both exist or both be absent"
        )
    if target_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("training_config") != config:
            raise RuntimeError("Existing target checkpoint belongs to another experiment")
        return False
    atomic_torch_save(model.state_dict(), target_path)
    metadata = target_metadata(
        config, trajectory_name, epoch, accuracy, target_path
    )
    atomic_json_save(metadata, metadata_path)
    return True


def load_source_model(source, device):
    ann, _, metadata = load_qcfs_pair(
        source, "cifar100", "resnet20", device
    )
    ann.set_L(8)
    ann.set_T(0)
    ann.set_qcfs_training_profile("fixed_repo")
    validate_qcfs_model(ann)
    return ann, metadata


def load_resume_state(path, expected_config, trajectory_name, learning_rate):
    state = torch.load(path, map_location="cpu")
    if state.get("experiment_config") != expected_config:
        raise RuntimeError(f"Resume configuration mismatch for {trajectory_name}")
    if state.get("trajectory") != trajectory_name:
        raise RuntimeError(f"Resume trajectory mismatch for {trajectory_name}")
    if state.get("learning_rate") != learning_rate:
        raise RuntimeError(f"Resume learning-rate mismatch for {trajectory_name}")
    return state


def run_trajectory(
    args,
    config,
    trajectory_name,
    learning_rate,
    device,
):
    paths = trajectory_paths(args.output_dir, trajectory_name)
    seed_all(args.seed)
    train_loader, test_loader = datapool(
        "cifar100", args.batch_size, augmentation_profile="paper_era"
    )
    model, _ = load_source_model(args.source, device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )
    history = []
    best_accuracy = float("-inf")
    best_epoch = None
    start_epoch = 0

    if args.resume and paths["state"].is_file():
        state = load_resume_state(
            paths["state"], config, trajectory_name, learning_rate
        )
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_accuracy = state["best_accuracy"]
        best_epoch = state["best_epoch"]
        start_epoch = state["epoch"] + 1
        restore_rng_state(state["rng_state"])
        print(f"[{trajectory_name}] resumed at epoch {start_epoch}", flush=True)
    elif any(path.exists() for path in paths.values()):
        raise FileExistsError(
            f"Refusing to overwrite existing trajectory files for {trajectory_name}; "
            "use --resume only for an exact continuation"
        )

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        loss, train_accuracy = train(
            model, device, train_loader, criterion, optimizer, 0
        )
        scheduler.step()
        test_accuracy = val(model, test_loader, device, 0)
        record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "loss_sum": loss,
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
        }
        history.append(record)
        print(
            f"[{trajectory_name}] epoch={epoch + 1:02d}/{args.epochs} "
            f"lr={current_lr:.8f} train={train_accuracy:.3f}% "
            f"test={test_accuracy:.3f}%",
            flush=True,
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            atomic_torch_save(model.state_dict(), paths["best"])

        if test_accuracy + 1e-12 >= args.target_accuracy:
            if save_first_target(
                model,
                args.output_dir,
                config,
                trajectory_name,
                epoch,
                test_accuracy,
            ):
                print(
                    f"TARGET SAVED: {test_accuracy:.3f}% from "
                    f"{trajectory_name} epoch {epoch}",
                    flush=True,
                )

        atomic_json_save(
            {
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "experiment_config": config,
                "history": history,
                "learning_rate": learning_rate,
                "status": "complete" if epoch + 1 == args.epochs else "incomplete",
                "trajectory": trajectory_name,
            },
            paths["history"],
        )
        atomic_torch_save(
            {
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "epoch": epoch,
                "experiment_config": config,
                "history": history,
                "learning_rate": learning_rate,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": capture_rng_state(),
                "scheduler": scheduler.state_dict(),
                "trajectory": trajectory_name,
            },
            paths["state"],
        )

    if best_epoch is None or not paths["best"].is_file():
        raise RuntimeError(f"No best checkpoint recorded for {trajectory_name}")
    return {
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "checkpoint": str(paths["best"].resolve()),
        "checkpoint_sha256": checkpoint_sha256(paths["best"]),
        "learning_rate": learning_rate,
        "trajectory": trajectory_name,
    }


def verify_checkpoint(path, expected_accuracy, test_loader, device):
    ann, snn, metadata = load_qcfs_pair(
        path, "cifar100", "resnet20", device
    )
    ann.set_L(8)
    ann.set_T(0)
    ann.set_qcfs_training_profile("fixed_repo")
    validate_qcfs_model(ann)
    accuracy = val(ann, test_loader, device, 0)
    if abs(accuracy - expected_accuracy) > 0.005:
        raise RuntimeError(
            f"Reloaded accuracy mismatch for {path}: "
            f"expected {expected_accuracy:.3f}%, got {accuracy:.3f}%"
        )
    if len([module for module in snn.modules() if module.__class__.__name__ == "SignedIF"]) != 19:
        raise RuntimeError("SignedIF conversion did not produce 19 activation layers")
    return accuracy, metadata


def main(cli_args=None):
    args = build_parser().parse_args(cli_args)
    validate_args(args)
    args.source = args.source.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {args.source}")

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}; "
            "refusing to overwrite it"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_sha256 = checkpoint_sha256(args.source)
    config = experiment_config(args, source_sha256)
    manifest_path = args.output_dir / "experiment_config.json"
    if manifest_path.exists():
        saved_config = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_config != config:
            raise RuntimeError("Existing experiment configuration does not match")
    else:
        atomic_json_save(config, manifest_path)

    seed_all(args.seed)
    _, baseline_loader = datapool(
        "cifar100", args.batch_size, augmentation_profile="paper_era"
    )
    source_model, source_metadata = load_source_model(args.source, device)
    source_accuracy = val(source_model, baseline_loader, device, 0)
    if abs(source_accuracy - args.expected_source_accuracy) > args.source_accuracy_tolerance:
        raise RuntimeError(
            f"Source accuracy gate failed: expected "
            f"{args.expected_source_accuracy:.3f}% +/- "
            f"{args.source_accuracy_tolerance:.3f}, got {source_accuracy:.3f}%"
        )
    print(
        f"Source accepted: accuracy={source_accuracy:.3f}% "
        f"SHA256={source_metadata['sha256']}",
        flush=True,
    )
    del source_model

    results = []
    for trajectory_name, learning_rate in zip(
        args.trajectory_names, args.learning_rates
    ):
        results.append(
            run_trajectory(
                args, config, trajectory_name, learning_rate, device
            )
        )

    winner = max(results, key=lambda item: item["best_accuracy"])
    winner_state = torch.load(winner["checkpoint"], map_location="cpu")
    global_best_path = args.output_dir / GLOBAL_BEST_FILENAME
    if global_best_path.exists():
        if not args.resume:
            raise FileExistsError(f"Global-best checkpoint exists: {global_best_path}")
        existing_hash = checkpoint_sha256(global_best_path)
        existing_metadata_path = global_best_path.with_suffix(".json")
        if not existing_metadata_path.is_file():
            raise RuntimeError("Global-best checkpoint metadata is missing")
        existing_metadata = json.loads(
            existing_metadata_path.read_text(encoding="utf-8")
        )
        if (
            existing_hash != existing_metadata.get("checkpoint_sha256")
            or existing_metadata.get("training_config") != config
            or existing_metadata.get("actual_accuracy") != winner["best_accuracy"]
            or existing_metadata.get("trajectory") != winner["trajectory"]
            or existing_metadata.get("epoch") != winner["best_epoch"]
        ):
            raise RuntimeError("Existing global-best checkpoint does not match this run")
    else:
        atomic_torch_save(winner_state, global_best_path)
    global_best_metadata = target_metadata(
        config,
        winner["trajectory"],
        winner["best_epoch"],
        winner["best_accuracy"],
        global_best_path,
    )
    atomic_json_save(
        global_best_metadata, global_best_path.with_suffix(".json")
    )

    _, verification_loader = datapool(
        "cifar100", args.batch_size, augmentation_profile="paper_era"
    )
    verified_global_accuracy, _ = verify_checkpoint(
        global_best_path, winner["best_accuracy"], verification_loader, device
    )
    target_path = args.output_dir / TARGET_FILENAME
    verified_target_accuracy = None
    if target_path.is_file():
        target_info = json.loads(
            target_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        verified_target_accuracy, _ = verify_checkpoint(
            target_path,
            target_info["actual_accuracy"],
            verification_loader,
            device,
        )

    summary = {
        "completed_at_utc": utc_now(),
        "experiment_config": config,
        "global_best": dict(
            global_best_metadata,
            verified_accuracy=verified_global_accuracy,
        ),
        "source_accuracy": source_accuracy,
        "target_checkpoint": (
            {
                "path": str(target_path.resolve()),
                "verified_accuracy": verified_target_accuracy,
            }
            if verified_target_accuracy is not None
            else None
        ),
        "target_reached": verified_target_accuracy is not None,
        "trajectories": results,
    }
    atomic_json_save(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
