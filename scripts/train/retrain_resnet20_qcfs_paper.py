"""Controlled reconstruction of the QCFS paper's CIFAR-100/ResNet20 ANN."""

import argparse
import json
import os
import platform
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch
import torch.nn as nn
import torchvision

from models import IF, modelpool
from preprocess import datapool
from scripts.experiments.qcfs_checkpoint import checkpoint_sha256, load_qcfs_pair
from scripts.experiments.run_resnet20_qcfs_ablation import validate_t32_conversion
from scripts.train.finetune_resnet20_qcfs import (
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)
from utils import seed_all, train, val


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "cifar100-checkpoints"
    / "resnet20_qcfs_paper_reproduction"
)
TARGET_FILENAME = "resnet20_cifar100_qcfs_L8_target_ge69_94.pth"
GLOBAL_BEST_FILENAME = "resnet20_cifar100_qcfs_L8_global_best.pth"
PAPER_ANN_ACCURACY = 69.94

RECIPES = OrderedDict(
    [
        (
            "R1_PAPER_FORMULA_EARLY_CODE",
            {
                "batch_size": 200,
                "qcfs_training_profile": "paper_era",
            },
        ),
        (
            "R2_PUBLIC_CODE_GRADIENT",
            {
                "batch_size": 200,
                "qcfs_training_profile": "fixed_repo",
            },
        ),
        (
            "R3_BATCH128_SENSITIVITY",
            {
                "batch_size": 128,
                "qcfs_training_profile": "paper_era",
            },
        ),
    ]
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reconstruct the QCFS paper CIFAR-100/ResNet20 training recipe"
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--initial_lr", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--initial_threshold", type=float, default=4.0)
    parser.add_argument("--cutout_length", type=int, default=8)
    parser.add_argument("--target_accuracy", type=float, default=PAPER_ANN_ACCURACY)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.initial_lr <= 0:
        raise ValueError("initial_lr must be positive")
    if args.initial_threshold <= 0:
        raise ValueError("initial_threshold must be positive")
    if args.cutout_length <= 0:
        raise ValueError("cutout_length must be positive")


def environment_metadata():
    return {
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def configure_numerics():
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def recipe_config(args, name, recipe):
    return {
        "architecture": "resnet20",
        "augmentation_profile": "fixed_repo",
        "autoaugment": True,
        "batch_size": recipe["batch_size"],
        "cutout_length": args.cutout_length,
        "dataset": "cifar100",
        "epochs": args.epochs,
        "initial_lr": args.initial_lr,
        "initial_threshold": args.initial_threshold,
        "momentum": 0.9,
        "optimizer": "SGD",
        "qcfs_L": 8,
        "qcfs_training_profile": recipe["qcfs_training_profile"],
        "recipe": name,
        "scheduler": "CosineAnnealingLR",
        "scheduler_eta_min": 0.0,
        "seed": args.seed,
        "target_accuracy": args.target_accuracy,
        "test_samples": 10000,
        "tf32": False,
        "train_samples": 50000,
        "weight_decay": args.weight_decay,
        "weight_origin": "official_implementation_retrained",
        "workers": args.workers,
    }


def recipe_paths(output_dir, name):
    directory = Path(output_dir) / name
    return {
        "directory": directory,
        "best": directory / f"{name}.best.pth",
        "history": directory / f"{name}.history.json",
        "initial": directory / f"{name}.initial.pth",
        "state": directory / f"{name}.train_state.pth",
        "summary": directory / f"{name}.summary.json",
    }


def initialize_model(config, device):
    model = modelpool("resnet20", "cifar100")
    model.set_L(config["qcfs_L"])
    model.set_T(0)
    model.set_qcfs_training_profile(config["qcfs_training_profile"])
    layers = [module for module in model.modules() if isinstance(module, IF)]
    if len(layers) != 19:
        raise RuntimeError(f"Expected 19 QCFS layers, found {len(layers)}")
    with torch.no_grad():
        for layer in layers:
            layer.thresh.fill_(config["initial_threshold"])
    if any(layer.L != 8 for layer in layers):
        raise RuntimeError("Every QCFS activation must use L=8")
    if any(
        layer.quantization_profile != config["qcfs_training_profile"]
        for layer in layers
    ):
        raise RuntimeError("QCFS gradient profile was not applied to every layer")
    return model.to(device)


def build_loaders(config):
    return datapool(
        "cifar100",
        config["batch_size"],
        augmentation_profile="fixed_repo",
        cutout_length=config["cutout_length"],
    )


def save_target_once(model, output_dir, config, epoch, accuracy):
    target_path = Path(output_dir) / TARGET_FILENAME
    metadata_path = target_path.with_suffix(".json")
    if target_path.exists() != metadata_path.exists():
        raise RuntimeError("Target checkpoint/metadata pair is incomplete")
    if target_path.exists():
        return False
    atomic_torch_save(model.state_dict(), target_path)
    atomic_json_save(
        {
            "actual_accuracy": accuracy,
            "checkpoint_sha256": checkpoint_sha256(target_path),
            "created_at_utc": utc_now(),
            "epoch": epoch,
            "recipe_config": config,
            "test_samples": config["test_samples"],
            "weight_origin": config["weight_origin"],
        },
        metadata_path,
    )
    return True


def load_resume_state(path, config, initial_sha256):
    state = torch.load(path, map_location="cpu")
    if state.get("recipe_config") != config:
        raise RuntimeError("Resume recipe configuration mismatch")
    if state.get("initial_checkpoint_sha256") != initial_sha256:
        raise RuntimeError("Resume initial-checkpoint SHA256 mismatch")
    return state


def run_recipe(args, name, recipe, device):
    config = recipe_config(args, name, recipe)
    paths = recipe_paths(args.output_dir, name)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    atomic_json_save(config, paths["directory"] / "recipe_config.json")

    seed_all(args.seed)
    configure_numerics()
    train_loader, test_loader = build_loaders(config)
    model = initialize_model(config, device)
    if paths["initial"].exists():
        if not args.resume:
            raise FileExistsError(f"Initial checkpoint exists: {paths['initial']}")
        model.load_state_dict(
            torch.load(paths["initial"], map_location="cpu"), strict=True
        )
    else:
        atomic_torch_save(model.state_dict(), paths["initial"])
    initial_sha256 = checkpoint_sha256(paths["initial"])

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["initial_lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=0.0
    )
    criterion = nn.CrossEntropyLoss().to(device)
    history = []
    best_accuracy = float("-inf")
    best_epoch = None
    start_epoch = 0

    if args.resume and paths["state"].is_file():
        state = load_resume_state(paths["state"], config, initial_sha256)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng_state"])
        history = state["history"]
        best_accuracy = state["best_accuracy"]
        best_epoch = state["best_epoch"]
        start_epoch = state["epoch"] + 1
        print(f"[{name}] resumed at epoch {start_epoch}", flush=True)
    elif any(paths[key].exists() for key in ("best", "history", "state", "summary")):
        raise FileExistsError(
            f"Refusing to overwrite existing outputs for {name}; use --resume"
        )

    for epoch in range(start_epoch, config["epochs"]):
        current_lr = optimizer.param_groups[0]["lr"]
        loss, train_accuracy = train(
            model, device, train_loader, criterion, optimizer, 0
        )
        scheduler.step()
        test_accuracy = val(model, test_loader, device, 0)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "loss_sum": loss,
                "test_accuracy": test_accuracy,
                "train_accuracy": train_accuracy,
            }
        )
        print(
            f"[{name}] epoch={epoch + 1:03d}/{config['epochs']} "
            f"lr={current_lr:.8f} train={train_accuracy:.3f}% "
            f"test={test_accuracy:.3f}% best={max(best_accuracy, test_accuracy):.3f}%",
            flush=True,
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            atomic_torch_save(model.state_dict(), paths["best"])
        if test_accuracy + 1e-12 >= config["target_accuracy"]:
            if save_target_once(
                model, args.output_dir, config, epoch, test_accuracy
            ):
                print(
                    f"TARGET SAVED: {test_accuracy:.3f}% from {name} epoch {epoch}",
                    flush=True,
                )

        status = "complete" if epoch + 1 == config["epochs"] else "incomplete"
        atomic_json_save(
            {
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "history": history,
                "initial_checkpoint_sha256": initial_sha256,
                "recipe_config": config,
                "status": status,
            },
            paths["history"],
        )
        atomic_torch_save(
            {
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "epoch": epoch,
                "history": history,
                "initial_checkpoint_sha256": initial_sha256,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "recipe_config": config,
                "rng_state": capture_rng_state(),
                "scheduler": scheduler.state_dict(),
            },
            paths["state"],
        )

    if best_epoch is None or not paths["best"].is_file():
        raise RuntimeError(f"No best checkpoint available for {name}")
    verification_model = initialize_model(config, device)
    verification_model.load_state_dict(
        torch.load(paths["best"], map_location="cpu"), strict=True
    )
    verified_accuracy = val(verification_model, test_loader, device, 0)
    if abs(verified_accuracy - best_accuracy) > 0.005:
        raise RuntimeError(
            f"Best-checkpoint verification failed for {name}: "
            f"saved={best_accuracy:.3f}%, reloaded={verified_accuracy:.3f}%"
        )
    result = {
        "best_accuracy": best_accuracy,
        "best_checkpoint": str(paths["best"].resolve()),
        "best_checkpoint_sha256": checkpoint_sha256(paths["best"]),
        "best_epoch": best_epoch,
        "completed_at_utc": utc_now(),
        "initial_checkpoint_sha256": initial_sha256,
        "passed_ann_gate": best_accuracy + 1e-12 >= config["target_accuracy"],
        "recipe_config": config,
        "verified_accuracy": verified_accuracy,
    }
    atomic_json_save(result, paths["summary"])
    return result, test_loader


def copy_global_best(result, output_dir, resume):
    global_path = Path(output_dir) / GLOBAL_BEST_FILENAME
    metadata_path = global_path.with_suffix(".json")
    source_state = torch.load(result["best_checkpoint"], map_location="cpu")
    if global_path.exists() or metadata_path.exists():
        if not resume or not (global_path.exists() and metadata_path.exists()):
            raise FileExistsError("Global-best checkpoint pair already exists")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("best_checkpoint_sha256")
            != result["best_checkpoint_sha256"]
            or metadata.get("best_accuracy") != result["best_accuracy"]
        ):
            raise RuntimeError("Existing global-best checkpoint metadata mismatch")
        return global_path
    atomic_torch_save(source_state, global_path)
    atomic_json_save(
        dict(
            result,
            global_checkpoint_sha256=checkpoint_sha256(global_path),
        ),
        metadata_path,
    )
    return global_path


def main(cli_args=None):
    args = build_parser().parse_args(cli_args)
    validate_args(args)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory exists: {args.output_dir}; refusing to overwrite it"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["QCFS_NUM_WORKERS"] = str(args.workers)
    configure_numerics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = {
        "environment": environment_metadata(),
        "paper_reference": {
            "ann_accuracy": PAPER_ANN_ACCURACY,
            "cifar100_resnet20_L": 8,
            "epochs": 300,
            "initial_lr": 0.02,
            "paper_pdf_sha256": "e23e1e9ae5dc6193b7908275c681cab371d2167208a5951f3867fc66580b9b07",
        },
        "planned_recipes": [
            recipe_config(args, name, recipe)
            for name, recipe in RECIPES.items()
        ],
    }
    experiment_path = args.output_dir / "experiment_config.json"
    if experiment_path.exists():
        saved = json.loads(experiment_path.read_text(encoding="utf-8"))
        if saved != experiment:
            raise RuntimeError("Experiment configuration mismatch")
    else:
        atomic_json_save(experiment, experiment_path)

    results = []
    final_test_loader = None
    for name, recipe in RECIPES.items():
        result, final_test_loader = run_recipe(args, name, recipe, device)
        results.append(result)
        if result["passed_ann_gate"]:
            break

    winner = max(results, key=lambda item: item["best_accuracy"])
    global_path = copy_global_best(winner, args.output_dir, args.resume)
    ann, snn, global_metadata = load_qcfs_pair(
        global_path, "cifar100", "resnet20", device
    )
    ann.set_L(8)
    ann.set_T(0)
    verified_global_accuracy = val(ann, final_test_loader, device, 0)
    if abs(verified_global_accuracy - winner["best_accuracy"]) > 0.005:
        raise RuntimeError("Global-best checkpoint verification failed")

    t32_accuracy = None
    t32_gap = None
    if winner["passed_ann_gate"]:
        t32_accuracy, t32_gap = validate_t32_conversion(
            snn,
            final_test_loader,
            device,
            verified_global_accuracy,
            max_gap=2.0,
        )

    target_path = args.output_dir / TARGET_FILENAME
    summary = {
        "completed_at_utc": utc_now(),
        "executed_recipes": results,
        "global_best": {
            "accuracy": verified_global_accuracy,
            "checkpoint": str(global_path),
            "checkpoint_sha256": global_metadata["sha256"],
            "recipe": winner["recipe_config"]["recipe"],
        },
        "target_checkpoint": str(target_path) if target_path.is_file() else None,
        "target_reached": target_path.is_file(),
        "t32_accuracy": t32_accuracy,
        "t32_conversion_gap": t32_gap,
        "t32_gate_passed": t32_gap is not None and t32_gap <= 2.0,
    }
    atomic_json_save(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
