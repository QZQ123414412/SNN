import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models import IF, modelpool
from utils import seed_all


def iter_qcfs_activations(model):
    for module in model.modules():
        if isinstance(module, IF):
            yield module


def set_refinement_proxy(
    model,
    enabled,
    time_steps=4,
    schedule="uniform",
    ratio=1.0,
    custom_weights=None,
    positive_margin=0.5,
    negative_margin=0.5,
    r0_mode="credit_only",
):
    for module in iter_qcfs_activations(model):
        module.set_refinement_proxy(
            enabled=enabled,
            time_steps=time_steps,
            schedule=schedule,
            ratio=ratio,
            custom_weights=custom_weights,
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            r0_mode=r0_mode,
        )


def reset_refinement_proxy_stats(model):
    for module in iter_qcfs_activations(model):
        module.reset_refinement_proxy_stats()


def _model_scalar(model):
    parameter = next(model.parameters(), None)
    if parameter is None:
        return torch.tensor(0.0)
    return parameter.new_tensor(0.0)


def get_refinement_event_rate(model):
    event_count = None
    event_total = 0
    for module in iter_qcfs_activations(model):
        if module.refinement_event_count is None:
            continue
        if event_count is None:
            event_count = module.refinement_event_count
        else:
            event_count = event_count + module.refinement_event_count
        event_total += module.refinement_event_total
    if event_count is None or event_total == 0:
        return _model_scalar(model)
    return event_count / float(event_total)


def sample_time_steps(time_steps, probabilities=None, generator=None):
    if not time_steps:
        raise ValueError("time_steps must not be empty")
    values = [int(value) for value in time_steps]
    if probabilities is None:
        index = torch.randint(len(values), (1,), generator=generator).item()
        return values[index]
    if len(probabilities) != len(values):
        raise ValueError("probabilities must have the same length as time_steps")
    weights = torch.tensor(probabilities, dtype=torch.float64)
    if torch.any(weights < 0) or weights.sum().item() <= 0:
        raise ValueError("probabilities must be non-negative and sum to a positive value")
    index = torch.multinomial(weights / weights.sum(), 1, generator=generator).item()
    return values[index]


def split_train_validation_loader(
    train_loader,
    evaluation_loader=None,
    val_fraction=0.1,
    seed=42,
    batch_size=None,
):
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    dataset = train_loader.dataset
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("training dataset must contain at least two samples")

    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_size = min(max(int(round(dataset_size * float(val_fraction))), 1), dataset_size - 1)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    validation_dataset = dataset
    if (
        evaluation_loader is not None
        and hasattr(dataset, "transform")
        and hasattr(evaluation_loader.dataset, "transform")
    ):
        validation_dataset = copy.copy(dataset)
        validation_dataset.transform = evaluation_loader.dataset.transform

    common_kwargs = {
        "batch_size": batch_size or train_loader.batch_size,
        "num_workers": getattr(train_loader, "num_workers", 0),
        "pin_memory": getattr(train_loader, "pin_memory", False),
        "collate_fn": train_loader.collate_fn,
    }
    split_train = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        drop_last=getattr(train_loader, "drop_last", False),
        **common_kwargs,
    )
    split_validation = DataLoader(
        Subset(validation_dataset, val_indices),
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return split_train, split_validation


def configure_trainable_stage(model, stage):
    stage = stage.upper()
    if stage not in {"A", "B"}:
        raise ValueError("stage must be 'A' or 'B'")
    if stage == "B":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, IF):
            module.thresh.requires_grad_(True)
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)

    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        for parameter in classifier.parameters():
            parameter.requires_grad_(True)


def freeze_batchnorm_running_stats(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def compute_dual_branch_loss(
    model,
    inputs,
    targets,
    time_steps,
    schedule="uniform",
    ratio=1.0,
    custom_weights=None,
    positive_margin=0.5,
    negative_margin=0.5,
    lambda_clean=0.5,
    lambda_cons=1.0,
    lambda_event=0.0,
    temperature=1.0,
    criterion=None,
):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    set_refinement_proxy(model, enabled=False)
    clean_logits = model(inputs)
    ce_clean = criterion(clean_logits, targets)

    set_refinement_proxy(
        model,
        enabled=True,
        time_steps=time_steps,
        schedule=schedule,
        ratio=ratio,
        custom_weights=custom_weights,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
    )
    reset_refinement_proxy_stats(model)
    refinement_logits = model(inputs)
    ce_refinement = criterion(refinement_logits, targets)
    kl_consistency = F.kl_div(
        F.log_softmax(refinement_logits / temperature, dim=1),
        F.softmax(clean_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    event_rate = get_refinement_event_rate(model)
    loss = (
        ce_refinement
        + float(lambda_clean) * ce_clean
        + float(lambda_cons) * kl_consistency
        + float(lambda_event) * event_rate
    )
    metrics = {
        "loss": loss.detach(),
        "ce_refinement": ce_refinement.detach(),
        "ce_clean": ce_clean.detach(),
        "kl_consistency": kl_consistency.detach(),
        "event_rate": event_rate.detach(),
    }
    return loss, metrics


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    time_steps,
    time_step_probabilities=None,
    schedule="uniform",
    ratio=1.0,
    custom_weights=None,
    positive_margin=0.5,
    negative_margin=0.5,
    lambda_clean=0.5,
    lambda_cons=1.0,
    lambda_event=0.0,
    freeze_bn_stats=True,
    max_batches=0,
):
    model.train()
    if freeze_bn_stats:
        freeze_batchnorm_running_stats(model)
    totals = {
        "loss": 0.0,
        "ce_refinement": 0.0,
        "ce_clean": 0.0,
        "kl_consistency": 0.0,
        "event_rate": 0.0,
        "samples": 0,
    }
    batches = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        current_T = sample_time_steps(time_steps, time_step_probabilities)
        optimizer.zero_grad()
        loss, metrics = compute_dual_branch_loss(
            model,
            inputs,
            targets,
            time_steps=current_T,
            schedule=schedule,
            ratio=ratio,
            custom_weights=custom_weights,
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            lambda_clean=lambda_clean,
            lambda_cons=lambda_cons,
            lambda_event=lambda_event,
        )
        loss.backward()
        optimizer.step()
        batches += 1
        batch_size = int(targets.numel())
        totals["samples"] += batch_size
        for key in ("loss", "ce_refinement", "ce_clean", "kl_consistency", "event_rate"):
            totals[key] += float(metrics[key].item()) * batch_size
    averaged = {
        key: value / max(totals["samples"], 1)
        for key, value in totals.items()
        if key != "samples"
    }
    averaged["samples"] = totals["samples"]
    averaged["batches"] = batches
    return averaged


@torch.no_grad()
def evaluate_clean(model, loader, device, max_batches=0, return_count=False):
    model.eval()
    set_refinement_proxy(model, enabled=False)
    correct = 0
    total = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        logits = model(inputs.to(device))
        predicted = logits.argmax(dim=1).cpu()
        total += int(targets.numel())
        correct += int(predicted.eq(targets).sum().item())
    accuracy = 100.0 * correct / max(total, 1)
    if return_count:
        return accuracy, total
    return accuracy


@torch.no_grad()
def evaluate_refinement(
    model,
    loader,
    device,
    time_steps,
    schedule="uniform",
    ratio=1.0,
    custom_weights=None,
    positive_margin=0.5,
    negative_margin=0.5,
    max_batches=0,
    return_count=False,
):
    model.eval()
    set_refinement_proxy(
        model,
        enabled=True,
        time_steps=time_steps,
        schedule=schedule,
        ratio=ratio,
        custom_weights=custom_weights,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
    )
    correct = 0
    total = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        logits = model(inputs.to(device))
        predicted = logits.argmax(dim=1).cpu()
        total += int(targets.numel())
        correct += int(predicted.eq(targets).sum().item())
    accuracy = 100.0 * correct / max(total, 1)
    if return_count:
        return accuracy, total
    return accuracy


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def append_epoch_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def load_qcfs_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def build_optimizer(model, lr, weight_decay):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no trainable parameters selected")
    return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)


def run_stage(
    model,
    train_loader,
    val_loader,
    device,
    args,
    stage,
    epochs,
    lr,
    best_score,
    best_path,
):
    configure_trainable_stage(model, stage)
    optimizer = build_optimizer(model, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    for epoch in range(epochs):
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            time_steps=args.time_steps,
            time_step_probabilities=args.time_step_probabilities,
            schedule=args.schedule,
            ratio=args.ratio,
            positive_margin=args.positive_margin,
            negative_margin=args.negative_margin,
            lambda_clean=args.lambda_clean,
            lambda_cons=args.lambda_cons,
            lambda_event=args.lambda_event,
            freeze_bn_stats=args.freeze_bn_stats,
            max_batches=args.max_train_batches,
        )
        scheduler.step()
        clean_acc, clean_samples = evaluate_clean(
            model,
            val_loader,
            device,
            max_batches=args.max_val_batches,
            return_count=True,
        )
        refinement_accs = [
            evaluate_refinement(
                model,
                val_loader,
                device,
                time_steps=T,
                schedule=args.schedule,
                ratio=args.ratio,
                positive_margin=args.positive_margin,
                negative_margin=args.negative_margin,
                max_batches=args.max_val_batches,
            )
            for T in args.time_steps
        ]
        score = sum(refinement_accs) / len(refinement_accs)
        latest_path = best_path.with_name(best_path.stem + "_latest.pth")
        torch.save(model.state_dict(), latest_path)
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            "epoch": epoch + 1,
            "epochs": epochs,
            "lr": scheduler.get_last_lr()[0],
            "train": metrics,
            "validation": {
                "clean_acc": clean_acc,
                "clean_samples": clean_samples,
                "refinement_acc_by_T": dict(zip(args.time_steps, refinement_accs)),
                "refinement_avg": score,
            },
            "paths": {
                "latest_checkpoint": latest_path,
                "best_checkpoint": best_path,
            },
        }
        append_epoch_record(args.metrics_jsonl, record)
        print(
            f"stage={stage} epoch={epoch + 1}/{epochs} "
            f"loss={metrics['loss']:.5f} event={metrics['event_rate']:.5f} "
            f"clean={clean_acc:.2f} refinement_avg={score:.2f} "
            f"refinement={dict(zip(args.time_steps, refinement_accs))}"
        )
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            append_epoch_record(
                args.metrics_jsonl,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "new_best",
                    "stage": stage,
                    "epoch": epoch + 1,
                    "score": best_score,
                    "best_checkpoint": best_path,
                },
            )
    return best_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deployment-consistent refinement-aware QCFS fine-tuning"
    )
    parser.add_argument("-data", "--dataset", default="cifar100")
    parser.add_argument("-arch", "--model", default="vgg16")
    parser.add_argument("-id", "--identifier", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("-dev", "--device", default="0")
    parser.add_argument("-b", "--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("-L", "--L", default=4, type=int)
    parser.add_argument("--time_steps", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument(
        "--time_step_probabilities",
        nargs="+",
        type=float,
        default=[0.45, 0.45, 0.10],
    )
    parser.add_argument("--schedule", default="geometric")
    parser.add_argument("--ratio", default=1.1, type=float)
    parser.add_argument("--positive_margin", default=0.5, type=float)
    parser.add_argument("--negative_margin", default=0.5, type=float)
    parser.add_argument("--lambda_clean", default=0.5, type=float)
    parser.add_argument("--lambda_cons", default=1.0, type=float)
    parser.add_argument("--lambda_event", default=0.0, type=float)
    parser.add_argument("--stage_a_epochs", default=5, type=int)
    parser.add_argument("--stage_b_epochs", default=20, type=int)
    parser.add_argument("--stage_a_lr", default=1e-3, type=float)
    parser.add_argument("--stage_b_lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--val_fraction", default=0.1, type=float)
    parser.add_argument("--max_train_batches", default=0, type=int)
    parser.add_argument("--max_val_batches", default=0, type=int)
    parser.add_argument("--freeze_bn_stats", action="store_true", default=True)
    parser.add_argument(
        "--output_dir",
        default="docs/results/refinement_finetune/checkpoints",
    )
    parser.add_argument("--metrics_jsonl", default="")
    parser.add_argument("--suffix", default="refinement_ft")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    if len(args.time_step_probabilities) != len(args.time_steps):
        raise ValueError("--time_step_probabilities must match --time_steps length")

    from preprocess import datapool

    raw_train_loader, evaluation_transform_loader = datapool(args.dataset, args.batch_size)
    train_loader, val_loader = split_train_validation_loader(
        raw_train_loader,
        evaluation_loader=evaluation_transform_loader,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    model = modelpool(args.model, args.dataset)
    model.set_L(args.L)
    checkpoint = args.checkpoint or os.path.join(
        f"{args.dataset}-checkpoints",
        args.identifier + ".pth",
    )
    load_qcfs_checkpoint(model, checkpoint, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / (
        f"{args.model}_{args.dataset}_L[{args.L}]_{args.schedule}_"
        f"ratio[{args.ratio:g}]_{args.suffix}.pth"
    )
    if not args.metrics_jsonl:
        args.metrics_jsonl = str(best_path.with_suffix(".jsonl"))
    append_epoch_record(
        args.metrics_jsonl,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "start",
            "dataset": args.dataset,
            "model": args.model,
            "checkpoint": checkpoint,
            "time_steps": args.time_steps,
            "time_step_probabilities": args.time_step_probabilities,
            "schedule": args.schedule,
            "ratio": args.ratio,
            "stage_a_epochs": args.stage_a_epochs,
            "stage_b_epochs": args.stage_b_epochs,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "best_checkpoint": best_path,
        },
    )

    best_score = -1.0
    if args.stage_a_epochs > 0:
        best_score = run_stage(
            model,
            train_loader,
            val_loader,
            device,
            args,
            stage="A",
            epochs=args.stage_a_epochs,
            lr=args.stage_a_lr,
            best_score=best_score,
            best_path=best_path,
        )
    if args.stage_b_epochs > 0:
        best_score = run_stage(
            model,
            train_loader,
            val_loader,
            device,
            args,
            stage="B",
            epochs=args.stage_b_epochs,
            lr=args.stage_b_lr,
            best_score=best_score,
            best_path=best_path,
        )
    append_epoch_record(
        args.metrics_jsonl,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "finished",
            "best_score": best_score,
            "best_checkpoint": best_path,
        },
    )
    print(
        f"Best refinement validation score={best_score:.2f}; "
        f"saved to {best_path}; metrics={args.metrics_jsonl}"
    )


if __name__ == "__main__":
    main()
