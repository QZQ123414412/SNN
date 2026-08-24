"""Cloud-side ImageNet QCFS/Full-FTBC/PA-FTBC/SNM/HA-SNM ablation.

This entry point is intentionally single-GPU.  Local development can exercise
its pure helpers and checkpoint loading without having an ImageNet copy; real
preflight, smoke, and formal runs are performed on the cloud server.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from calibration import bias_corr_model
from models import SignedIF
from parity_anchor_ftbc import named_signed_layers
from preprocess.getdataloader import GetImageNetDatasets, resolve_dataset_root
from scripts.experiments.qcfs_checkpoint import checkpoint_sha256, load_qcfs_pair
from scripts.experiments.run_full_ftbc_asnm_ablation import (
    build_full_model,
    build_plain_model,
    configure_snn,
    snapshot_full_ftbc,
)
from scripts.experiments.run_ha_snm_ablation import configure_snm
from scripts.experiments.run_pa_ftbc_asnm_ablation import build_pa_model
from scripts.experiments.run_stats_ablation import summarize_layer_stats
from scripts.experiments.run_temporal_lr_gated_snm import batches_sha256
from spike_stats import (
    collect_resnet34_spike_stats,
    collect_signed_spike_stats,
    reset_signed_spike_stats,
    set_signed_spike_stats_enabled,
    summarize_ftbc_storage,
)
from utils import seed_all


DEFAULT_TIME_STEPS = (1, 2, 4, 8, 16, 32)
SMOKE_TIME_STEPS = (4, 8, 32)
HA_SNM = {"start": 1.25, "end": 0.5, "reference": 8.0}
PROTOCOL_VERSION = "imagenet-full-pa-ha-v1"
EXPECTED_TRAIN_SAMPLES = 1_281_167
EXPECTED_VALIDATION_SAMPLES = 50_000
EXPECTED_CLASSES = 1_000
EXPECTED_ANN_TOLERANCE_PP = 0.2
ANN_REFERENCE_URL = "https://github.com/hzc1208/ANN2SNN_SRP"

PROTOCOLS = {
    "resnet34": {
        "L": 8,
        "checkpoint": Path("ImageNet-checkpoints/ImageNet-ResNet34-t8.pth"),
        "sha256": "8f98b197a943aee0a1cb8971a04a7e1d1fed0cb80f5d32a0dd89c9bd6ece6bb2",
        "expected_ann_top1": 74.32,
        "signed_layers": 33,
    },
    "vgg16": {
        "L": 16,
        "checkpoint": Path("ImageNet-checkpoints/ImageNet-VGG16-t16.pth"),
        "sha256": "4027d8f06497dd34718fb0e2be910768a22c64d116e4e6af4c58a80a4b5422c6",
        "expected_ann_top1": 74.29,
        "signed_layers": 15,
    },
}

FAMILIES = ("qcfs", "full", "pa")
MODES = ("off", "standard", "ha")
CONFIGS = OrderedDict(
    [
        ("A_QCFS_R0", {"family": "qcfs", "mode": "off"}),
        ("B_QCFS_STANDARD_SNM_R0", {"family": "qcfs", "mode": "standard"}),
        ("C_QCFS_HA_SNM_R0", {"family": "qcfs", "mode": "ha"}),
        ("D_QCFS_FULL_FTBC_R0", {"family": "full", "mode": "off"}),
        (
            "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
            {"family": "full", "mode": "standard"},
        ),
        ("F_QCFS_FULL_FTBC_HA_SNM_R0", {"family": "full", "mode": "ha"}),
        ("G_QCFS_PA_FTBC_R0", {"family": "pa", "mode": "off"}),
        (
            "H_QCFS_PA_FTBC_STANDARD_SNM_R0",
            {"family": "pa", "mode": "standard"},
        ),
        ("I_QCFS_PA_FTBC_HA_SNM_R0", {"family": "pa", "mode": "ha"}),
    ]
)
CONFIG_BY_FAMILY_MODE = {
    (item["family"], item["mode"]): name for name, item in CONFIGS.items()
}


def default_output(architecture, smoke=False):
    protocol = PROTOCOLS[architecture]
    stem = (
        f"IMAGENET_{architecture.upper()}_L{protocol['L']}_"
        "FULL_PA_HA_SNM"
    )
    if smoke:
        return Path("docs/archive/experiments/imagenet") / f"{stem}_SMOKE.md"
    return Path("docs/results/comparative_ablation/imagenet") / f"{stem}.md"


def resolve_args(args):
    protocol = PROTOCOLS[args.architecture]
    args.checkpoint = args.checkpoint or protocol["checkpoint"]
    args.checkpoint_sha256 = args.checkpoint_sha256 or protocol["sha256"]
    args.data_root = Path(
        args.data_root or resolve_dataset_root("ImageNet")
    ).expanduser()
    args.time_steps = tuple(
        args.time_steps
        if args.time_steps is not None
        else (SMOKE_TIME_STEPS if args.smoke else DEFAULT_TIME_STEPS)
    )
    args.calibration_iterations = (
        args.calibration_iterations
        if args.calibration_iterations is not None
        else (1 if args.smoke else 50)
    )
    args.validation_batches = (
        args.validation_batches
        if args.validation_batches is not None
        else (2 if args.smoke else 0)
    )
    args.timing_samples = (
        args.timing_samples
        if args.timing_samples is not None
        else (0 if args.smoke else 1_000)
    )
    args.output = args.output or default_output(args.architecture, args.smoke)
    args.cache_dir = args.cache_dir or (
        Path("runtime_cache/imagenet")
        / ("smoke" if args.smoke else "formal")
        / args.architecture
    )
    return args


def validate_args(args):
    protocol = PROTOCOLS[args.architecture]
    if tuple(sorted(set(args.time_steps))) != tuple(args.time_steps):
        raise ValueError("Time steps must be unique, positive, and sorted")
    if not args.time_steps or any(value <= 0 for value in args.time_steps):
        raise ValueError("Time steps must be positive")
    expected_steps = SMOKE_TIME_STEPS if args.smoke else DEFAULT_TIME_STEPS
    if tuple(args.time_steps) != expected_steps:
        raise ValueError(
            f"{'Smoke' if args.smoke else 'Formal'} protocol requires "
            f"time steps {expected_steps}"
        )
    if args.fit_batch_size != 2:
        raise ValueError("ImageNet Full-FTBC protocol fixes fit batch size at two")
    if args.alpha != 0.5:
        raise ValueError("ImageNet Full-FTBC protocol fixes alpha at 0.5")
    expected_iterations = 1 if args.smoke else 50
    if args.calibration_iterations != expected_iterations:
        raise ValueError(
            f"Calibration iterations must be {expected_iterations} for this run kind"
        )
    expected_validation_batches = 2 if args.smoke else 0
    if args.validation_batches != expected_validation_batches:
        raise ValueError(
            "Smoke evaluates two validation batches; formal evaluation uses all batches"
        )
    if args.eval_temporal_batch_budget <= 0:
        raise ValueError("Temporal batch budget must be positive")
    if args.timing_samples < 0 or args.num_workers < 0:
        raise ValueError("Timing samples and worker count must be non-negative")
    if args.smoke and args.timing_samples != 0:
        raise ValueError("Smoke runs do not perform a separate timing pass")
    if not args.smoke and args.timing_samples != 1_000:
        raise ValueError("Formal timing subset is fixed at 1,000 samples")
    if len(args.checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in args.checkpoint_sha256
    ):
        raise ValueError("A valid checkpoint SHA256 is required")
    if args.checkpoint_sha256.lower() != protocol["sha256"]:
        raise ValueError(
            f"{args.architecture} protocol requires checkpoint SHA256 "
            f"{protocol['sha256']}"
        )
    if not str(args.device).isdigit() or int(args.device) < 0:
        raise ValueError("--device must be one non-negative CUDA device index")
    output_parts = {part.lower() for part in Path(args.output).parts}
    if args.smoke and "archive" not in output_parts:
        raise ValueError("Smoke output must be written under docs/archive")
    if not args.smoke and "results" not in output_parts:
        raise ValueError("Formal output must be written under docs/results")
    if args.resume and args.preflight_only:
        raise ValueError("--resume and --preflight-only cannot be combined")


def evaluation_batch_size(time_steps, temporal_budget):
    return max(1, int(temporal_budget) // int(time_steps))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_signature():
    """Hash every local source file that defines this experiment protocol."""
    relative_paths = (
        "calibration.py",
        "models/ResNet.py",
        "models/VGG.py",
        "models/layer.py",
        "parity_anchor_ftbc.py",
        "preprocess/getdataloader.py",
        "scripts/experiments/qcfs_checkpoint.py",
        "scripts/experiments/run_full_ftbc_asnm_ablation.py",
        "scripts/experiments/run_ha_snm_ablation.py",
        "scripts/experiments/run_imagenet_ftbc_pa_ha_ablation.py",
        "scripts/experiments/run_pa_ftbc_asnm_ablation.py",
        "spike_stats.py",
    )
    files = OrderedDict(
        (relative_path, file_sha256(REPO_ROOT / relative_path))
        for relative_path in relative_paths
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "files": files,
        "sha256": canonical_sha256(files),
    }


def class_mapping_sha256(class_to_idx):
    return canonical_sha256(sorted(class_to_idx.items()))


def inspect_imagenet_datasets(train_data, validation_data, require_official=True):
    train_classes = len(train_data.classes)
    validation_classes = len(validation_data.classes)
    if train_data.class_to_idx != validation_data.class_to_idx:
        raise RuntimeError("ImageNet train/val class mappings do not match")
    if train_classes != validation_classes:
        raise RuntimeError("ImageNet train/val class counts do not match")
    if require_official:
        if train_classes != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Expected {EXPECTED_CLASSES} ImageNet classes, got {train_classes}"
            )
        if len(train_data) != EXPECTED_TRAIN_SAMPLES:
            raise RuntimeError(
                f"Expected {EXPECTED_TRAIN_SAMPLES} training images, got {len(train_data)}"
            )
        if len(validation_data) != EXPECTED_VALIDATION_SAMPLES:
            raise RuntimeError(
                "Expected 50,000 validation images, "
                f"got {len(validation_data)}"
            )
    return {
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "classes": train_classes,
        "class_mapping_sha256": class_mapping_sha256(train_data.class_to_idx),
    }


def materialize_fit_batch(train_data, batch_size, seed):
    seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=False,
    )
    inputs, targets = next(iter(loader))
    return [(inputs.contiguous(), targets.contiguous())]


def make_validation_loader(validation_data, batch_size, num_workers, batches=0):
    dataset = validation_data
    if batches:
        sample_count = min(len(dataset), int(batch_size) * int(batches))
        dataset = Subset(dataset, range(sample_count))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def atomic_json_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def schedule_sha256(schedule):
    digest = hashlib.sha256()
    for name, values in schedule.items():
        digest.update(name.encode("utf-8"))
        for tensor in values:
            value = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def protocol_payload(args, checkpoint, dataset, fit_sha256):
    protocol = PROTOCOLS[args.architecture]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "implementation": implementation_signature(),
        "dataset": "imagenet",
        "architecture": args.architecture,
        "qcfs_L": protocol["L"],
        "checkpoint_sha256": checkpoint["sha256"],
        "dataset_signature": dataset,
        "fit_sha256": fit_sha256,
        "time_steps": list(args.time_steps),
        "fit_batch_size": args.fit_batch_size,
        "calibration_iterations": args.calibration_iterations,
        "alpha": args.alpha,
        "eval_temporal_batch_budget": args.eval_temporal_batch_budget,
        "num_workers": args.num_workers,
        "device_argument": str(args.device),
        "validation_batches": args.validation_batches,
        "timing_samples": args.timing_samples,
        "seed": args.seed,
        "ha_snm": dict(HA_SNM),
        "r0": True,
        "pa_coefficients": 4,
        "pa_full_fallback_max_t": 4,
        "single_gpu": True,
        "ann_reference_url": ANN_REFERENCE_URL,
        "smoke": bool(args.smoke),
    }


def run_paths(args):
    output = Path(args.output)
    return {
        "output": output,
        "progress": output.with_suffix(".progress.json"),
        "cache_dir": Path(args.cache_dir),
    }


def check_output_collisions(args):
    paths = run_paths(args)
    collisions = [
        path
        for path in (paths["output"], paths["progress"], paths["cache_dir"])
        if path.exists()
    ]
    if collisions and not args.resume:
        raise FileExistsError(
            "Refusing to overwrite existing ImageNet artifacts: "
            + ", ".join(str(path) for path in collisions)
        )
    if args.resume and not paths["progress"].is_file():
        raise FileNotFoundError(
            f"Cannot resume without progress file: {paths['progress']}"
        )
    return paths


def build_variant(template, schedule, family, mode, time_steps, device):
    signed = mode != "off"
    compression = None
    if family == "qcfs":
        model = build_plain_model(template, time_steps, signed, device)
    elif family == "full":
        model = build_full_model(template, schedule, time_steps, signed, device)
    elif family == "pa":
        if int(time_steps) <= 4:
            model = build_full_model(template, schedule, time_steps, signed, device)
        else:
            model, compression = build_pa_model(
                template,
                schedule,
                time_steps,
                signed,
                device,
            )
    else:
        raise ValueError(f"Unknown family: {family}")
    configure_snm(model, mode, **HA_SNM)
    model.set_r0(True)
    return model, compression


def collect_architecture_stats(model, architecture):
    if architecture == "resnet34":
        return collect_resnet34_spike_stats(model, SignedIF, nn.Conv2d)
    return collect_signed_spike_stats(model, SignedIF, nn.Conv2d, nn.Linear)


@torch.no_grad()
def compute_ann_cache(model, validation_data, args, device):
    batch_size = max(1, int(args.eval_temporal_batch_budget))
    loader = make_validation_loader(
        validation_data,
        batch_size,
        args.num_workers,
        batches=args.validation_batches,
    )
    model.eval()
    logits = []
    labels = []
    correct1 = 0
    correct5 = 0
    total = 0
    synchronize(device)
    started = time.perf_counter()
    for inputs, targets in loader:
        output = model(inputs.to(device))
        target_device = targets.to(device)
        predictions = output.topk(5, dim=1).indices
        correct1 += int(predictions[:, 0].eq(target_device).sum().item())
        correct5 += int(
            predictions.eq(target_device.view(-1, 1)).any(dim=1).sum().item()
        )
        total += int(targets.numel())
        logits.append(output.detach().cpu())
        labels.append(targets.detach().cpu())
    synchronize(device)
    return {
        "logits": torch.cat(logits, dim=0),
        "labels": torch.cat(labels, dim=0),
        "top1": 100.0 * correct1 / max(total, 1),
        "top5": 100.0 * correct5 / max(total, 1),
        "samples": total,
        "elapsed": time.perf_counter() - started,
    }


def load_or_compute_ann_cache(
    ann,
    validation_data,
    args,
    device,
    cache_path,
    signature,
):
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("signature") != signature:
            raise RuntimeError("Existing ANN cache belongs to another protocol")
        return cached["metrics"]
    metrics = compute_ann_cache(ann, validation_data, args, device)
    atomic_torch_save(
        {"signature": signature, "metrics": metrics},
        cache_path,
    )
    return metrics


@torch.no_grad()
def time_inference(model, validation_data, batch_size, args, device):
    if args.timing_samples <= 0:
        return {"samples": 0, "elapsed": 0.0, "seconds_per_image": 0.0}
    sample_count = min(int(args.timing_samples), len(validation_data))
    loader = DataLoader(
        Subset(validation_data, range(sample_count)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model.eval()
    iterator = iter(loader)
    warmup = []
    for _ in range(2):
        try:
            warmup.append(next(iterator)[0])
        except StopIteration:
            break
    for inputs in warmup:
        model(inputs.to(device)).mean(0)
    del iterator
    synchronize(device)
    started = time.perf_counter()
    measured = 0
    for inputs, _ in loader:
        model(inputs.to(device)).mean(0)
        measured += int(inputs.shape[0])
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "samples": measured,
        "elapsed": elapsed,
        "seconds_per_image": elapsed / max(measured, 1),
    }


@torch.no_grad()
def evaluate_snn(
    model,
    validation_data,
    ann_cache,
    time_steps,
    architecture,
    args,
    device,
):
    batch_size = evaluation_batch_size(
        time_steps, args.eval_temporal_batch_budget
    )
    loader = make_validation_loader(
        validation_data,
        batch_size,
        args.num_workers,
        batches=args.validation_batches,
    )
    model.eval()
    set_signed_spike_stats_enabled(model, SignedIF, True)
    reset_signed_spike_stats(model, SignedIF)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    correct1 = 0
    correct5 = 0
    total = 0
    squared_error = 0.0
    logit_values = 0
    cursor = 0
    synchronize(device)
    started = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        target_device = targets.to(device)
        snn_logits = model(inputs).mean(0)
        count = int(targets.numel())
        cached_targets = ann_cache["labels"][cursor : cursor + count]
        if not torch.equal(cached_targets, targets.cpu()):
            raise RuntimeError("Validation order differs from the ANN cache")
        ann_logits = ann_cache["logits"][cursor : cursor + count].to(device)
        predictions = snn_logits.topk(5, dim=1).indices
        correct1 += int(predictions[:, 0].eq(target_device).sum().item())
        correct5 += int(
            predictions.eq(target_device.view(-1, 1)).any(dim=1).sum().item()
        )
        squared_error += float(
            torch.nn.functional.mse_loss(
                snn_logits,
                ann_logits,
                reduction="sum",
            ).item()
        )
        logit_values += int(snn_logits.numel())
        total += count
        cursor += count
    synchronize(device)
    evaluation_elapsed = time.perf_counter() - started
    if total != cursor or total > int(ann_cache["samples"]):
        raise RuntimeError("SNN evaluation count is incompatible with ANN cache")
    layer_stats = collect_architecture_stats(model, architecture)
    summary = summarize_layer_stats(layer_stats)
    storage = summarize_ftbc_storage(model, SignedIF)
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    set_signed_spike_stats_enabled(model, SignedIF, False)
    timing = time_inference(model, validation_data, batch_size, args, device)
    summary.update(
        {
            "top1": 100.0 * correct1 / max(total, 1),
            "top5": 100.0 * correct5 / max(total, 1),
            "logit_mse": squared_error / max(logit_values, 1),
            "ftbc_parameters": int(storage["parameters"]),
            "ftbc_bytes": int(storage["bytes"]),
            "ftbc_synthesis_macs": int(storage["synthesis_macs"]),
            "evaluated_samples": total,
            "batch_size": batch_size,
            "evaluation_elapsed": evaluation_elapsed,
            "peak_memory_bytes": peak_bytes,
            "timing_samples": timing["samples"],
            "timing_elapsed": timing["elapsed"],
            "seconds_per_image": timing["seconds_per_image"],
        }
    )
    return summary, [asdict(item) for item in layer_stats]


def fit_full_ftbc(
    ann,
    snn_template,
    fit_batches,
    time_steps,
    args,
    device,
):
    teacher = copy.deepcopy(snn_template).to(device)
    configure_snn(teacher, time_steps, signed=False, ftbc_mode="full")
    teacher.set_snm_mode("standard")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    for iteration in range(args.calibration_iterations):
        print(
            f"[calibration T={time_steps}] iteration "
            f"{iteration + 1}/{args.calibration_iterations}",
            flush=True,
        )
        bias_corr_model(
            ann=ann,
            snn=teacher,
            T=time_steps,
            train_loader=fit_batches,
            curr_t_alpha=args.alpha,
            num_cali_sample_batches=1,
            ftbc_mode="full",
        )
    synchronize(device)
    schedule = snapshot_full_ftbc(teacher, time_steps)
    storage = summarize_ftbc_storage(teacher, SignedIF)
    metadata = {
        "elapsed": time.perf_counter() - started,
        "iterations": args.calibration_iterations,
        "parameters": int(storage["parameters"]),
        "bytes": int(storage["bytes"]),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "schedule_sha256": schedule_sha256(schedule),
    }
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return schedule, metadata


def full_schedule_cache_path(cache_dir, time_steps):
    return Path(cache_dir) / f"full_ftbc_T{int(time_steps)}.pt"


def load_or_fit_full_schedule(
    ann,
    snn_template,
    fit_batches,
    time_steps,
    args,
    device,
    signature,
    payload,
):
    cache_path = full_schedule_cache_path(args.cache_dir, time_steps)
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("signature") != signature:
            raise RuntimeError(
                f"Full-FTBC cache for T={time_steps} belongs to another protocol"
            )
        schedule = cached["schedule"]
        if schedule_sha256(schedule) != cached.get("schedule_sha256"):
            raise RuntimeError(f"Full-FTBC cache checksum failed for T={time_steps}")
        return schedule, cached["metadata"]
    schedule, metadata = fit_full_ftbc(
        ann,
        snn_template,
        fit_batches,
        time_steps,
        args,
        device,
    )
    atomic_torch_save(
        {
            "signature": signature,
            "time_steps": int(time_steps),
            "schedule": schedule,
            "schedule_sha256": metadata["schedule_sha256"],
            "metadata": metadata,
        },
        cache_path,
    )
    payload["calibration"][str(time_steps)] = metadata
    return schedule, metadata


def initial_payload(args, protocol, signature, checkpoint, dataset):
    return {
        "status": "initialized",
        "signature": signature,
        "protocol": protocol,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "platform": platform.platform(),
            "device_argument": args.device,
            "gpu": None,
            "active_elapsed_seconds": 0.0,
        },
        "checkpoint": checkpoint,
        "dataset": dataset,
        "ann": None,
        "calibration": OrderedDict(),
        "compression": OrderedDict(),
        "results": OrderedDict((name, OrderedDict()) for name in CONFIGS),
        "layers": OrderedDict((name, OrderedDict()) for name in CONFIGS),
        "fallback_checks": [],
        "acceptance_checks": [],
    }


def load_or_initialize_progress(args, paths, protocol, signature, checkpoint, dataset):
    if args.resume:
        payload = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise RuntimeError("Resume protocol signature mismatch")
        return payload
    payload = initial_payload(args, protocol, signature, checkpoint, dataset)
    atomic_json_save(payload, paths["progress"])
    return payload


def metric_table(lines, title, payload, key, formatter):
    times = payload["protocol"]["time_steps"]
    lines.extend(
        [
            f"## {title}",
            "",
            "| Config | " + " | ".join(f"T={value}" for value in times) + " |",
            "|---|" + "---:|" * len(times),
        ]
    )
    for name in CONFIGS:
        cells = []
        for value in times:
            result = payload["results"].get(name, {}).get(str(value))
            cells.append("-" if result is None else formatter(result[key]))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")


def write_report(path, payload):
    protocol = payload["protocol"]
    times = protocol["time_steps"]
    lines = [
        "# ImageNet QCFS + Full/PA-FTBC + Standard/HA-SNM Ablation",
        "",
        f"Status: {payload['status']}",
        "",
        f"- Architecture: `{protocol['architecture']}`",
        f"- QCFS L: {protocol['qcfs_L']}",
        f"- Checkpoint: `{payload['checkpoint']['filename']}`",
        f"- Checkpoint SHA256: `{payload['checkpoint']['sha256']}`",
        f"- Calibration tensor SHA256: `{protocol['fit_sha256']}`",
        f"- ImageNet validation samples: {payload['dataset']['validation_samples']:,}",
        f"- Evaluation temporal batch budget: {protocol['eval_temporal_batch_budget']}",
        f"- Protocol version: `{protocol['protocol_version']}`",
        f"- Implementation SHA256: `{protocol['implementation']['sha256']}`",
        f"- GPU: `{payload['runtime'].get('gpu')}`",
        f"- Total active elapsed: {payload['runtime'].get('active_elapsed_seconds', 0.0):.3f}s",
        f"- Published ANN reference: [ANN2SNN_SRP]({protocol['ann_reference_url']})",
        "- All configurations use R0; HA-SNM is frozen at start=1.25, end=0.5, reference=8.",
        "- Full-FTBC uses two fixed training images, 50 iterations, and alpha=0.5 in formal runs.",
        "- PA-FTBC is constructed from the same Full-FTBC teacher and falls back to Full at T<=4.",
        "",
        "## ANN reference",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    ann = payload.get("ann")
    if ann:
        lines.extend(
            [
                f"| Top-1 | {ann['top1']:.2f}% |",
                f"| Top-5 | {ann['top5']:.2f}% |",
                f"| Samples | {int(ann['samples']):,} |",
                f"| Elapsed | {ann['elapsed']:.3f}s |",
            ]
        )
    else:
        lines.append("| Status | not evaluated |")
    lines.append("")

    lines.extend(
        [
            "## Primary accuracy",
            "",
            "| Config | "
            + " | ".join(f"T={value}" for value in times)
            + " | Mean |",
            "|---|" + "---:|" * (len(times) + 1),
        ]
    )
    for name in CONFIGS:
        values = [
            payload["results"].get(name, {}).get(str(value)) for value in times
        ]
        cells = ["-" if item is None else f"{item['top1']:.2f}%" for item in values]
        complete = [item["top1"] for item in values if item is not None]
        mean = "-" if len(complete) != len(times) else f"{sum(complete) / len(complete):.2f}%"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {mean} |")
    lines.append("")

    metric_table(lines, "Top-5 accuracy", payload, "top5", lambda x: f"{x:.2f}%")
    metric_table(lines, "ANN-SNN logit MSE", payload, "logit_mse", lambda x: f"{x:.8f}")
    metric_table(lines, "Positive spike rate", payload, "positive_rate", lambda x: f"{100*x:.6f}%")
    metric_table(lines, "Negative spike rate", payload, "negative_rate", lambda x: f"{100*x:.6f}%")
    metric_table(lines, "Overall spike sparsity", payload, "sparsity", lambda x: f"{100*x:.6f}%")
    metric_table(lines, "Input-driven SOPs", payload, "sops", lambda x: f"{int(x):,}")
    metric_table(lines, "FTBC parameters", payload, "ftbc_parameters", lambda x: f"{int(x):,}")
    metric_table(lines, "FTBC storage bytes", payload, "ftbc_bytes", lambda x: f"{int(x):,}")
    metric_table(lines, "FTBC synthesis MACs", payload, "ftbc_synthesis_macs", lambda x: f"{int(x):,}")
    metric_table(lines, "Peak CUDA memory", payload, "peak_memory_bytes", lambda x: f"{x / 1024**3:.3f} GiB")
    metric_table(lines, "Evaluation elapsed", payload, "evaluation_elapsed", lambda x: f"{x:.3f}s")
    metric_table(lines, "Pure inference seconds/image", payload, "seconds_per_image", lambda x: f"{x:.8f}")

    lines.extend(
        [
            "## Full-FTBC calibration",
            "",
            "| T | Elapsed | Peak CUDA memory | Parameters | Bytes | Schedule SHA256 |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for value in times:
        item = payload["calibration"].get(str(value))
        if item is None:
            lines.append(f"| {value} | - | - | - | - | - |")
        else:
            lines.append(
                f"| {value} | {item['elapsed']:.3f}s | "
                f"{item.get('peak_memory_bytes', 0) / 1024**3:.3f} GiB | "
                f"{item['parameters']:,} | "
                f"{item['bytes']:,} | `{item['schedule_sha256']}` |"
            )
    lines.append("")

    lines.extend(
        [
            "## PA-FTBC compression",
            "",
            "| T | Representation | Full params | PA params | Storage saving | Explained energy | Compression elapsed |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for value in times:
        item = payload["compression"].get(str(value))
        if item is None:
            lines.append(f"| {value} | - | - | - | - | - | - |")
        else:
            saving = 1.0 - item["pa_parameters"] / max(item["full_parameters"], 1)
            lines.append(
                f"| {value} | {item['representation']} | "
                f"{item['full_parameters']:,} | {item['pa_parameters']:,} | "
                f"{100*saving:.2f}% | {item['explained_energy']:.8f} | "
                f"{item.get('compression_elapsed', 0.0):.3f}s |"
            )
    lines.append("")

    lines.extend(
        [
            "## Exact fallback checks",
            "",
            "| T | Mode | Full config | PA config | Exact |",
            "|---:|---|---|---|---|",
        ]
    )
    for item in payload["fallback_checks"]:
        lines.append(
            f"| {item['time_steps']} | {item['mode']} | {item['full']} | "
            f"{item['pa']} | {'yes' if item['exact'] else 'no'} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Completion checks",
            "",
            "| Check | T | Expected | Passed |",
            "|---|---:|---:|---|",
        ]
    )
    for item in payload.get("acceptance_checks", []):
        lines.append(
            f"| {item['check']} | {item.get('time_steps', '-')} | "
            f"{item.get('expected', '-')} | {'yes' if item['passed'] else 'no'} |"
        )
    lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_full_fallback(payload, time_steps, mode):
    full_name = CONFIG_BY_FAMILY_MODE[("full", mode)]
    pa_name = CONFIG_BY_FAMILY_MODE[("pa", mode)]
    key = str(time_steps)
    if key not in payload["results"][full_name]:
        raise RuntimeError(f"Full fallback source is missing for T={time_steps}, {mode}")
    payload["results"][pa_name][key] = copy.deepcopy(
        payload["results"][full_name][key]
    )
    payload["layers"][pa_name][key] = copy.deepcopy(
        payload["layers"][full_name][key]
    )
    payload["fallback_checks"] = [
        item
        for item in payload["fallback_checks"]
        if not (item["time_steps"] == time_steps and item["mode"] == mode)
    ]
    payload["fallback_checks"].append(
        {
            "time_steps": int(time_steps),
            "mode": mode,
            "full": full_name,
            "pa": pa_name,
            "exact": payload["results"][pa_name][key]
            == payload["results"][full_name][key],
        }
    )


def update_compression_metadata(payload, time_steps, schedule, compression):
    key = str(time_steps)
    full_parameters = sum(
        tensor.numel() for values in schedule.values() for tensor in values
    )
    if int(time_steps) <= 4:
        payload["compression"][key] = {
            "representation": "Full-FTBC fallback",
            "full_parameters": int(full_parameters),
            "pa_parameters": int(full_parameters),
            "explained_energy": 1.0,
            "compression_elapsed": 0.0,
            "layers": OrderedDict(
                (
                    name,
                    {
                        "representation": "full",
                        "channels": int(values[0].numel()),
                        "mse": 0.0,
                        "nrmse": 0.0,
                        "max_abs_error": 0.0,
                    },
                )
                for name, values in schedule.items()
            ),
        }
        return
    if compression is None:
        return
    payload["compression"][key] = {
        "representation": compression["structure"],
        "full_parameters": int(full_parameters),
        "pa_parameters": int(compression["ftbc_parameters"]),
        "pa_bytes": int(compression["ftbc_bytes"]),
        "explained_energy": float(compression["explained_energy"]),
        "compression_elapsed": float(compression["compression_elapsed"]),
        "layers": compression["layers"],
    }


def validate_completed_payload(payload, args):
    """Enforce invariants that must hold before a run can become complete."""
    checks = []
    expected_dataset_samples = int(payload["dataset"]["validation_samples"])
    scientific_t1_keys = (
        "top1",
        "top5",
        "logit_mse",
        "positive_spikes",
        "negative_spikes",
        "positive_rate",
        "negative_rate",
        "sparsity",
        "sops",
        "evaluated_samples",
    )
    expected_ann_samples = (
        min(
            expected_dataset_samples,
            int(args.eval_temporal_batch_budget) * args.validation_batches,
        )
        if args.validation_batches
        else expected_dataset_samples
    )
    if int(payload["ann"]["samples"]) != expected_ann_samples:
        raise RuntimeError(
            f"ANN evaluated {payload['ann']['samples']} samples, "
            f"expected {expected_ann_samples}"
        )
    checks.append(
        {
            "check": "ann_sample_count",
            "expected": expected_ann_samples,
            "passed": True,
        }
    )
    for time_steps in args.time_steps:
        key = str(time_steps)
        expected_samples = (
            min(
                expected_dataset_samples,
                evaluation_batch_size(
                    time_steps,
                    args.eval_temporal_batch_budget,
                )
                * args.validation_batches,
            )
            if args.validation_batches
            else expected_dataset_samples
        )
        for name in CONFIGS:
            if key not in payload["results"][name]:
                raise RuntimeError(f"Completion is missing {name} at T={time_steps}")
            result = payload["results"][name][key]
            if int(result["evaluated_samples"]) != expected_samples:
                raise RuntimeError(
                    f"{name} T={time_steps} evaluated "
                    f"{result['evaluated_samples']} samples, expected {expected_samples}"
                )
            if int(result["ftbc_bytes"]) != 4 * int(result["ftbc_parameters"]):
                raise RuntimeError(f"{name} T={time_steps} is not FP32 FTBC storage")
        checks.append(
            {
                "check": "evaluation_sample_count",
                "time_steps": int(time_steps),
                "expected": expected_samples,
                "passed": True,
            }
        )

        compression = payload["compression"].get(key)
        if compression is None:
            raise RuntimeError(f"Missing PA compression metadata at T={time_steps}")
        if time_steps <= 4:
            if int(compression["pa_parameters"]) != int(
                compression["full_parameters"]
            ):
                raise RuntimeError(f"PA must fall back to Full at T={time_steps}")
            for mode in MODES:
                full_name = CONFIG_BY_FAMILY_MODE[("full", mode)]
                pa_name = CONFIG_BY_FAMILY_MODE[("pa", mode)]
                if payload["results"][full_name][key] != payload["results"][pa_name][key]:
                    raise RuntimeError(
                        f"PA fallback differs from Full at T={time_steps}, {mode}"
                    )
        else:
            if int(compression["pa_parameters"]) >= int(
                compression["full_parameters"]
            ):
                raise RuntimeError(f"PA does not reduce parameters at T={time_steps}")
        checks.append(
            {
                "check": "pa_storage_and_fallback",
                "time_steps": int(time_steps),
                "passed": True,
            }
        )

    if 1 in args.time_steps:
        for family in FAMILIES:
            names = [CONFIG_BY_FAMILY_MODE[(family, mode)] for mode in MODES]
            reference = payload["results"][names[0]]["1"]
            for name in names:
                result = payload["results"][name]["1"]
                if int(result["negative_spikes"]) != 0:
                    raise RuntimeError(f"{name} emitted negative spikes at T=1")
                if any(result[key] != reference[key] for key in scientific_t1_keys):
                    raise RuntimeError(
                        f"T=1 SNM modes differ scientifically in family {family}"
                    )
        checks.append({"check": "t1_snm_equivalence", "passed": True})

    expected_fallback_checks = 3 * sum(
        int(time_steps <= 4) for time_steps in args.time_steps
    )
    exact_fallback_checks = sum(
        int(item.get("exact", False)) for item in payload["fallback_checks"]
    )
    if exact_fallback_checks != expected_fallback_checks:
        raise RuntimeError(
            f"Expected {expected_fallback_checks} exact fallback checks, "
            f"got {exact_fallback_checks}"
        )
    checks.append(
        {
            "check": "fallback_cache_count",
            "expected": expected_fallback_checks,
            "passed": True,
        }
    )
    return checks


def run_experiment(
    args,
    train_data,
    validation_data,
    dataset_signature,
    fit_batches,
    ann,
    snn_template,
    checkpoint,
    device,
):
    session_started = time.perf_counter()
    paths = check_output_collisions(args)
    protocol = protocol_payload(
        args,
        checkpoint,
        dataset_signature,
        batches_sha256(fit_batches),
    )
    signature = canonical_sha256(protocol)
    payload = load_or_initialize_progress(
        args,
        paths,
        protocol,
        signature,
        checkpoint,
        dataset_signature,
    )
    previous_active_elapsed = float(
        payload["runtime"].get("active_elapsed_seconds", 0.0)
    )

    def persist():
        payload["runtime"]["active_elapsed_seconds"] = (
            previous_active_elapsed + time.perf_counter() - session_started
        )
        atomic_json_save(payload, paths["progress"])

    payload["runtime"]["gpu"] = torch.cuda.get_device_name(device)
    payload["status"] = "ann_reference"
    persist()

    ann_cache = load_or_compute_ann_cache(
        ann,
        validation_data,
        args,
        device,
        paths["cache_dir"] / "ann_logits.pt",
        signature,
    )
    payload["ann"] = {
        key: value for key, value in ann_cache.items() if key not in {"logits", "labels"}
    }
    if not args.smoke:
        expected = PROTOCOLS[args.architecture]["expected_ann_top1"]
        if abs(float(ann_cache["top1"]) - expected) > EXPECTED_ANN_TOLERANCE_PP:
            payload["status"] = "ann_gate_failed"
            persist()
            write_report(paths["output"], payload)
            raise RuntimeError(
                f"ANN Top-1 {ann_cache['top1']:.2f}% differs from expected "
                f"{expected:.2f}% by more than {EXPECTED_ANN_TOLERANCE_PP:.2f}pp"
            )
    payload["status"] = "running"
    persist()

    for time_steps in args.time_steps:
        key = str(time_steps)
        schedule, calibration = load_or_fit_full_schedule(
            ann,
            snn_template,
            fit_batches,
            time_steps,
            args,
            device,
            signature,
            payload,
        )
        payload["calibration"][key] = calibration
        update_compression_metadata(payload, time_steps, schedule, None)
        persist()

        for family in FAMILIES:
            for mode in MODES:
                config_name = CONFIG_BY_FAMILY_MODE[(family, mode)]
                if key in payload["results"][config_name]:
                    continue
                if family == "pa" and time_steps <= 4:
                    copy_full_fallback(payload, time_steps, mode)
                    persist()
                    write_report(paths["output"], payload)
                    continue
                print(
                    f"[{args.architecture}] T={time_steps} {config_name}",
                    flush=True,
                )
                model, compression = build_variant(
                    snn_template,
                    schedule,
                    family,
                    mode,
                    time_steps,
                    device,
                )
                if family == "pa":
                    update_compression_metadata(
                        payload,
                        time_steps,
                        schedule,
                        compression,
                    )
                metrics, layers = evaluate_snn(
                    model,
                    validation_data,
                    ann_cache,
                    time_steps,
                    args.architecture,
                    args,
                    device,
                )
                payload["results"][config_name][key] = metrics
                payload["layers"][config_name][key] = layers
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                persist()
                write_report(paths["output"], payload)

    payload["acceptance_checks"] = validate_completed_payload(payload, args)
    payload["status"] = "complete"
    persist()
    write_report(paths["output"], payload)
    return paths["output"]


def print_preflight(args, checkpoint, dataset_signature, device, fit_sha256):
    paths = run_paths(args)
    payload = {
        "status": "preflight_passed",
        "architecture": args.architecture,
        "qcfs_L": PROTOCOLS[args.architecture]["L"],
        "checkpoint": checkpoint,
        "dataset": dataset_signature,
        "fit_sha256": fit_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "implementation": implementation_signature(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "output": str(paths["output"]),
        "progress": str(paths["progress"]),
        "cache_dir": str(paths["cache_dir"]),
        "collisions": [
            str(path)
            for path in (paths["output"], paths["progress"], paths["cache_dir"])
            if path.exists()
        ],
        "batch_sizes": {
            str(value): evaluation_batch_size(
                value, args.eval_temporal_batch_budget
            )
            for value in args.time_steps
        },
    }
    if payload["collisions"]:
        raise FileExistsError(
            "Preflight found existing output artifacts: "
            + ", ".join(payload["collisions"])
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU ImageNet QCFS + Full/PA-FTBC + standard/HA-SNM "
            "nine-way ablation"
        )
    )
    parser.add_argument(
        "--architecture",
        choices=tuple(PROTOCOLS),
        required=True,
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device", default="0")
    parser.add_argument("--time-steps", nargs="+", type=int)
    parser.add_argument("--fit-batch-size", type=int, default=2)
    parser.add_argument("--calibration-iterations", type=int)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eval-temporal-batch-budget", type=int, default=32)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--timing-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(cli_args=None):
    args = resolve_args(build_parser().parse_args(cli_args))
    validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet smoke and formal experiments require CUDA")
    device = torch.device("cuda")

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    actual_sha256 = checkpoint_sha256(checkpoint_path)
    if actual_sha256.lower() != args.checkpoint_sha256.lower():
        raise RuntimeError(
            f"Unexpected checkpoint SHA256 {actual_sha256}; "
            f"expected {args.checkpoint_sha256}"
        )
    if not (args.data_root / "train").is_dir() or not (
        args.data_root / "val"
    ).is_dir():
        raise FileNotFoundError(
            f"ImageNet root must contain train/ and val/: {args.data_root}"
        )

    train_data, validation_data = GetImageNetDatasets(root=args.data_root)
    dataset_signature = inspect_imagenet_datasets(
        train_data,
        validation_data,
        require_official=True,
    )
    fit_batches = materialize_fit_batch(
        train_data,
        args.fit_batch_size,
        args.seed,
    )
    fit_sha256 = batches_sha256(fit_batches)
    ann, snn_template, checkpoint = load_qcfs_pair(
        checkpoint_path,
        "imagenet",
        args.architecture,
        device,
    )
    protocol = PROTOCOLS[args.architecture]
    ann.set_L(protocol["L"])
    ann.set_T(0)
    if len(named_signed_layers(snn_template)) != protocol["signed_layers"]:
        raise RuntimeError("Unexpected SignedIF layer count")

    if args.preflight_only:
        print_preflight(
            args,
            checkpoint,
            dataset_signature,
            device,
            fit_sha256,
        )
        return

    output = run_experiment(
        args,
        train_data,
        validation_data,
        dataset_signature,
        fit_batches,
        ann,
        snn_template,
        checkpoint,
        device,
    )
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
