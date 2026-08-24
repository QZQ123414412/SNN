"""QCFS + Full-FTBC + Accuracy-Gated SNM ablation on CIFAR-10/100."""

import argparse
import copy
import json
import math
import os
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

from calibration import bias_corr_model
from a_snm import DEFAULT_TIME_STEPS, a_snm_enabled, select_a_snm_modes
from models import SignedIF
from preprocess import datapool
from scripts.experiments.qcfs_checkpoint import load_qcfs_pair
from scripts.experiments.run_state_ftbc_ablation import materialize_calibration_batches
from scripts.experiments.run_stats_ablation import summarize_layer_stats
from scripts.experiments.run_temporal_lr_gated_snm import (
    architecture_output,
    batches_sha256,
    collect_architecture_stats,
    synchronize,
)
from spike_stats import (
    reset_signed_spike_stats,
    set_signed_spike_stats_enabled,
    summarize_ftbc_storage,
)
from utils import seed_all, val


DATASET_PROTOCOLS = {
    "cifar100": {
        "label": "CIFAR-100",
        "default_checkpoints": {
            "resnet20": Path(
                "cifar100-checkpoints/"
                "resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth"
            ),
            "vgg16": Path("cifar100-checkpoints/cifar100-vgg16-l8-example.pth"),
        },
        "expected_sha256": {
            "resnet20": "1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2",
            "vgg16": "8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339",
        },
        "resnet20_eval_profile": "paper_era",
        "normalization": "CIFAR-100 normalization",
        "default_output": Path(
            "docs/results/comparative_ablation/cifar100/"
            "FULL_FTBC_ASNM_CIFAR100.md"
        ),
    },
    "cifar10": {
        "label": "CIFAR-10",
        "default_checkpoints": {
            "resnet20": Path(
                "cifar10-checkpoints/"
                "resnet20_L[8]_bs128_fixed_repo_seed42_testbest.pth"
            ),
            "vgg16": Path("cifar10-checkpoints/cifar10-vgg16-example.pth"),
        },
        "expected_sha256": {
            "resnet20": "eb8301ebda8ae91e52f2f273306befa5d349931c05b829a9440dafa05df70631",
            "vgg16": "093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84",
        },
        "resnet20_eval_profile": "fixed_repo",
        "normalization": "CIFAR-10 normalization",
        "default_output": Path(
            "docs/results/comparative_ablation/cifar10/"
            "FULL_FTBC_ASNM_CIFAR10.md"
        ),
    },
}
CIFAR10_RESNET20_L4_PROTOCOL = {
    "checkpoint": Path(
        "cifar10-checkpoints/"
        "resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth"
    ),
    "expected_sha256": (
        "851e5475413440193a9e26aa6b6400cd"
        "23dcd8ef4794c60bd0e08728d2f409c3"
    ),
    "evaluation_profile": "paper_era",
    "default_output": Path(
        "docs/results/comparative_ablation/cifar10/"
        "FULL_FTBC_ASNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.md"
    ),
}
# Backward-compatible CIFAR-100 aliases used by the temporal-LR experiment.
DEFAULT_CHECKPOINTS = DATASET_PROTOCOLS["cifar100"]["default_checkpoints"]
EXPECTED_CHECKPOINT_SHA256 = DATASET_PROTOCOLS["cifar100"]["expected_sha256"]
CONFIGS = OrderedDict(
    [
        ("A_QCFS_R0", {"family": "qcfs", "mode": "off"}),
        (
            "B_QCFS_STANDARD_SNM_R0",
            {"family": "qcfs", "mode": "on"},
        ),
        ("C_QCFS_ASNM_R0", {"family": "qcfs", "mode": "a_snm"}),
        (
            "D_QCFS_FULL_FTBC_R0",
            {"family": "full", "mode": "off"},
        ),
        (
            "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
            {"family": "full", "mode": "on"},
        ),
        (
            "F_QCFS_FULL_FTBC_ASNM_R0",
            {"family": "full", "mode": "a_snm"},
        ),
    ]
)
BASE_CONFIGS = {
    "qcfs": {"off": "A_QCFS_R0", "on": "B_QCFS_STANDARD_SNM_R0"},
    "full": {
        "off": "D_QCFS_FULL_FTBC_R0",
        "on": "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
    },
}
A_SNM_CONFIGS = {
    "qcfs": "C_QCFS_ASNM_R0",
    "full": "F_QCFS_FULL_FTBC_ASNM_R0",
}


def configure_snn(model, time_steps, signed, ftbc_mode):
    """Apply every fixed SNN control from the experiment protocol."""
    model.set_T(int(time_steps))
    model.set_coding_mode("rate", schedule="rate", ratio=1.0)
    model.set_signed(bool(signed))
    model.set_r0(True)
    model.reset_all_bias()
    model.set_ftbc_mode(ftbc_mode)
    model.set_snm_negative_margin(0.0)
    return model


def snapshot_full_ftbc(model, time_steps):
    schedule = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, SignedIF):
            continue
        if module.ftbc_mode != "full" or module.time_based_bias is None:
            raise RuntimeError(f"Signed layer {name!r} has no calibrated Full-FTBC")
        if len(module.time_based_bias) != int(time_steps):
            raise RuntimeError(
                f"Signed layer {name!r} has {len(module.time_based_bias)} biases, "
                f"expected {time_steps}"
            )
        schedule[name] = [item.detach().cpu().clone() for item in module.time_based_bias]
    if not schedule:
        raise RuntimeError("No SignedIF layers were captured")
    return schedule


def restore_full_ftbc(model, schedule, device):
    modules = dict(model.named_modules())
    signed_names = {
        name for name, module in model.named_modules() if isinstance(module, SignedIF)
    }
    if set(schedule) != signed_names:
        raise RuntimeError(
            "Full-FTBC topology mismatch: "
            f"missing={sorted(signed_names - set(schedule))}, "
            f"unexpected={sorted(set(schedule) - signed_names)}"
        )
    for name, values in schedule.items():
        modules[name].time_based_bias = [value.to(device) for value in values]
    return model


def build_plain_model(template, time_steps, signed, device):
    model = copy.deepcopy(template).to(device)
    return configure_snn(model, time_steps, signed=signed, ftbc_mode="none")


def build_full_model(template, schedule, time_steps, signed, device):
    model = copy.deepcopy(template).to(device)
    configure_snn(model, time_steps, signed=signed, ftbc_mode="full")
    return restore_full_ftbc(model, schedule, device)


@torch.no_grad()
def evaluate_validation(model, ann, batches, device, time_steps, architecture):
    ann.eval()
    model.eval()
    set_signed_spike_stats_enabled(model, SignedIF, True)
    reset_signed_spike_stats(model, SignedIF)
    correct = 0
    total = 0
    squared_error = 0.0
    logit_values = 0
    synchronize(device)
    started = time.perf_counter()
    for inputs, targets in batches:
        inputs = inputs.to(device)
        targets = targets.to(device)
        ann_logits = ann(inputs)
        snn_logits = model(inputs).mean(0)
        squared_error += float(
            torch.nn.functional.mse_loss(
                snn_logits,
                ann_logits,
                reduction="sum",
            ).item()
        )
        logit_values += snn_logits.numel()
        correct += int(snn_logits.argmax(dim=1).eq(targets).sum().item())
        total += int(targets.numel())
    synchronize(device)
    elapsed = time.perf_counter() - started
    layer_stats = collect_architecture_stats(model, architecture)
    summary = summarize_layer_stats(layer_stats)
    set_signed_spike_stats_enabled(model, SignedIF, False)
    return {
        "acc": 100.0 * correct / max(total, 1),
        "logit_mse": squared_error / max(logit_values, 1),
        "sops": int(summary["sops"]),
        "positive_spikes": int(summary["positive_spikes"]),
        "negative_spikes": int(summary["negative_spikes"]),
        "elapsed": elapsed,
    }


@torch.no_grad()
def evaluate_test(model, ann, loader, device, time_steps, architecture):
    """Collect statistics/MSE, then run a separate statistics-free timer."""
    ann.eval()
    model.eval()
    storage = summarize_ftbc_storage(model, SignedIF)
    set_signed_spike_stats_enabled(model, SignedIF, True)
    reset_signed_spike_stats(model, SignedIF)
    correct = 0
    total = 0
    squared_error = 0.0
    logit_values = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        ann_logits = ann(inputs)
        snn_logits = model(inputs).mean(0)
        squared_error += float(
            torch.nn.functional.mse_loss(
                snn_logits,
                ann_logits,
                reduction="sum",
            ).item()
        )
        logit_values += snn_logits.numel()
        correct += int(snn_logits.argmax(dim=1).eq(targets).sum().item())
        total += int(targets.numel())

    layer_stats = collect_architecture_stats(model, architecture)
    summary = summarize_layer_stats(layer_stats)
    statistics_accuracy = 100.0 * correct / max(total, 1)
    set_signed_spike_stats_enabled(model, SignedIF, False)
    synchronize(device)
    started = time.perf_counter()
    timed_accuracy = val(model, loader, device, time_steps)
    synchronize(device)
    inference_elapsed = time.perf_counter() - started
    if not math.isclose(timed_accuracy, statistics_accuracy, abs_tol=1e-12):
        raise RuntimeError(
            "Non-deterministic test accuracy: "
            f"{statistics_accuracy} vs {timed_accuracy}"
        )

    summary.update(
        {
            "acc": statistics_accuracy,
            "logit_mse": squared_error / max(logit_values, 1),
            "ftbc_parameters": storage["parameters"],
            "ftbc_bytes": storage["bytes"],
            "calibration_elapsed": 0.0,
            "inference_elapsed": inference_elapsed,
            "evaluated_samples": total,
        }
    )
    return summary, [asdict(item) for item in layer_stats]


def validate_t1_special_case(off_metrics, on_metrics, family):
    if 1 not in off_metrics or 1 not in on_metrics:
        return
    if int(on_metrics[1]["negative_spikes"]) != 0:
        raise RuntimeError(f"{family} standard SNM emitted negative spikes at T=1")
    for key in ("acc", "logit_mse", "sops"):
        if not math.isclose(
            float(off_metrics[1][key]),
            float(on_metrics[1][key]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{family} T=1 SNM-off/on mismatch for {key}: "
                f"{off_metrics[1][key]} vs {on_metrics[1][key]}"
            )


def save_progress(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def metric_table(lines, title, payload, key, formatter):
    time_steps = payload["protocol"]["time_steps"]
    lines.extend(
        [
            f"## {title}",
            "",
            "| Config | " + " | ".join(f"T={value}" for value in time_steps) + " |",
            "|---|" + "---:|" * len(time_steps),
        ]
    )
    for name in CONFIGS:
        cells = []
        for time_steps_value in time_steps:
            item = payload["results"].get(name, {}).get(str(time_steps_value))
            cells.append("-" if item is None else formatter(item[key]))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")


def selected_time_label(name, gates, time_steps):
    if name in {"A_QCFS_R0", "D_QCFS_FULL_FTBC_R0"}:
        return "none"
    if name in {"B_QCFS_STANDARD_SNM_R0", "E_QCFS_FULL_FTBC_STANDARD_SNM_R0"}:
        return ", ".join(str(value) for value in time_steps)
    family = "qcfs" if name == "C_QCFS_ASNM_R0" else "full"
    selected = [
        str(value) for value in time_steps if gates[family][str(value)]
    ]
    return ", ".join(selected) if selected else "none"


def mean_accuracy(payload, config_name):
    time_steps = payload["protocol"]["time_steps"]
    return sum(
        float(payload["results"][config_name][str(value)]["acc"])
        for value in time_steps
    ) / len(time_steps)


def append_accuracy_comparisons(lines, payload):
    time_steps = payload["protocol"]["time_steps"]
    comparisons = (
        ("C_QCFS_ASNM_R0", "A_QCFS_R0", "C-A"),
        ("C_QCFS_ASNM_R0", "B_QCFS_STANDARD_SNM_R0", "C-B"),
        ("F_QCFS_FULL_FTBC_ASNM_R0", "D_QCFS_FULL_FTBC_R0", "F-D"),
        (
            "F_QCFS_FULL_FTBC_ASNM_R0",
            "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
            "F-E",
        ),
    )
    lines.extend(
        [
            "## Accuracy comparisons",
            "",
            "| Comparison | "
            + " | ".join(f"T={value}" for value in time_steps)
            + " | Mean |",
            "|---|" + "---:|" * (len(time_steps) + 1),
        ]
    )
    for selected_name, baseline_name, label in comparisons:
        deltas = [
            float(payload["results"][selected_name][str(value)]["acc"])
            - float(payload["results"][baseline_name][str(value)]["acc"])
            for value in time_steps
        ]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{value:+.2f}pp" for value in deltas)
            + f" | {sum(deltas) / len(deltas):+.2f}pp |"
        )
    lines.append("")


def append_gate_test_diagnostics(lines, payload):
    time_steps = payload["protocol"]["time_steps"]
    lines.extend(
        [
            "## Validation-gate versus test-oracle diagnostic",
            "",
            "This table is post-hoc only. Test accuracy never changes the frozen A-SNM gate.",
            "",
            "| Family | T | Validation selection | Test off | Test on | Test oracle | A-SNM oracle gap |",
            "|---|---:|---|---:|---:|---|---:|",
        ]
    )
    for family, selected_name in A_SNM_CONFIGS.items():
        off_name = BASE_CONFIGS[family]["off"]
        on_name = BASE_CONFIGS[family]["on"]
        for value in time_steps:
            key = str(value)
            off_accuracy = float(payload["results"][off_name][key]["acc"])
            on_accuracy = float(payload["results"][on_name][key]["acc"])
            selected_accuracy = float(payload["results"][selected_name][key]["acc"])
            oracle_mode = "on" if on_accuracy > off_accuracy else "off"
            oracle_accuracy = max(off_accuracy, on_accuracy)
            selected_mode = "on" if payload["gates"][family][key] else "off"
            lines.append(
                f"| {family} | {value} | {selected_mode} | {off_accuracy:.2f}% | "
                f"{on_accuracy:.2f}% | {oracle_mode} | "
                f"{selected_accuracy - oracle_accuracy:+.2f}pp |"
            )
    lines.append("")


def write_report(path, payload):
    protocol = payload["protocol"]
    time_steps = protocol["time_steps"]
    lines = [
        f"# QCFS + Full-FTBC + Accuracy-Gated SNM {protocol['dataset_label']} Ablation",
        "",
        "- Status: complete",
        f"- Dataset: {protocol['dataset_label']}",
        f"- Architecture: {protocol['architecture']}",
        f"- Checkpoint: `{protocol['checkpoint']['filename']}`",
        f"- Checkpoint SHA256: `{protocol['checkpoint']['sha256']}`",
        f"- ANN accuracy on the {protocol['test_samples']:,}-image test set: {protocol['ann_accuracy']:.2f}%",
        f"- Time steps: {time_steps}",
        f"- Full-FTBC fit: {protocol['fit_batches']} x {protocol['batch_size']}, alpha={protocol['alpha']}",
        f"- A-SNM validation: {protocol['validation_batches']} x {protocol['batch_size']}",
        f"- Fit batch SHA256: `{protocol['fit_sha256']}`",
        f"- Validation batch SHA256: `{protocol['validation_sha256']}`",
        "- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, "
        f"ToTensor, {protocol['normalization']}, and Cutout(1,16).",
        "- The test loader uses only ToTensor and normalization, with shuffle=False.",
        f"- Every SNN uses QCFS L={protocol['qcfs_L']}, rate coding, rate schedule, ratio=1.0, R0=True, FP32.",
        f"- ResNet20 evaluation profile: {protocol['resnet20_eval_profile']}.",
        "- Full-FTBC is independently fitted for every T with SNM off and frozen before validation/test.",
        "- A-SNM independently enables SNM at each T only when SNM-on has strictly higher validation accuracy; ties select off.",
        "- A-SNM changes only the standard SNM on/off state, uses margin=0, and stores one frozen Boolean per evaluated T.",
        "- During ablation, test images are first accessed after both families' A-SNM decisions are frozen.",
        f"- Checkpoint-selection note: {protocol['checkpoint_selection_note']}",
        f"- Checkpoint-interpretation note: {protocol['checkpoint_interpretation_note']}",
        "",
        "## Primary accuracy table",
        "",
        "| Config | " + " | ".join(f"T={value}" for value in time_steps) + " | SNM-on T |",
        "|---|" + "---:|" * len(time_steps) + "---|",
    ]
    for name in CONFIGS:
        values = [
            f"{payload['results'][name][str(value)]['acc']:.2f}%"
            for value in time_steps
        ]
        lines.append(
            f"| {name} | "
            + " | ".join(values)
            + f" | {selected_time_label(name, payload['gates'], time_steps)} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Mean accuracy over evaluated time steps",
            "",
            "| Config | Mean accuracy |",
            "|---|---:|",
        ]
    )
    for name in CONFIGS:
        lines.append(f"| {name} | {mean_accuracy(payload, name):.2f}% |")
    lines.append("")

    append_accuracy_comparisons(lines, payload)

    metric_table(lines, "ANN-SNN logit MSE", payload, "logit_mse", lambda x: f"{x:.6f}")
    metric_table(lines, "Positive spike rate", payload, "positive_rate", lambda x: f"{100*x:.4f}%")
    metric_table(lines, "Negative spike rate", payload, "negative_rate", lambda x: f"{100*x:.4f}%")
    metric_table(lines, "Overall spike sparsity", payload, "sparsity", lambda x: f"{100*x:.4f}%")
    metric_table(lines, "Input-driven SOPs", payload, "sops", lambda x: f"{int(x):,}")
    metric_table(lines, "Full-FTBC parameters", payload, "ftbc_parameters", lambda x: f"{int(x):,}")
    metric_table(lines, "Full-FTBC storage bytes", payload, "ftbc_bytes", lambda x: f"{int(x):,}")
    metric_table(lines, "Full-FTBC calibration elapsed", payload, "calibration_elapsed", lambda x: f"{x:.3f}s")
    metric_table(lines, "Inference elapsed (statistics disabled)", payload, "inference_elapsed", lambda x: f"{x:.3f}s")

    lines.extend(
        [
            "## A-SNM selection",
            "",
            "| Family | SNM-on T | Validation inference + selection |",
            "|---|---|---:|",
            f"| QCFS | {selected_time_label('C_QCFS_ASNM_R0', payload['gates'], time_steps)} | {payload['selection_elapsed']['qcfs']:.3f}s |",
            f"| Full-FTBC | {selected_time_label('F_QCFS_FULL_FTBC_ASNM_R0', payload['gates'], time_steps)} | {payload['selection_elapsed']['full']:.3f}s |",
            "",
        ]
    )
    for family, title in (("qcfs", "QCFS"), ("full", "Full-FTBC")):
        lines.extend(
            [
                f"### {title} accuracy-gate trace",
                "",
                "| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for time_steps_value, item in payload["selection_trace"][family].items():
            lines.append(
                f"| {time_steps_value} | {item['off_accuracy']:.4f}% | "
                f"{item['on_accuracy']:.4f}% | {item['accuracy_gain']:+.4f}pp | "
                f"{item['selected_mode']} |"
            )
        lines.append("")

    append_gate_test_diagnostics(lines, payload)

    lines.extend(
        [
            "## Deployment equivalence checks",
            "",
            "| Config | T | Expected source | Exact cached result |",
            "|---|---:|---|---|",
        ]
    )
    for item in payload["equivalence_checks"]:
        lines.append(
            f"| {item['config']} | {item['time_steps']} | {item['source']} | "
            f"{'yes' if item['exact'] else 'no'} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_architecture(
    args,
    architecture,
    checkpoint_path,
    expected_sha256,
    fit_batches,
    validation_batches,
    test_loader,
    device,
):
    output = architecture_output(
        args.output,
        architecture,
        multiple=len(args.architectures) > 1,
    )
    progress_path = output.with_suffix(".progress.json")
    if not args.overwrite and (output.exists() or progress_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite an existing result: {output} or {progress_path}"
        )

    ann_template, snn_template, checkpoint = load_qcfs_pair(
        checkpoint_path,
        args.dataset,
        architecture,
        device,
    )
    if checkpoint["sha256"].lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Unexpected {args.dataset}/{architecture} checkpoint SHA256: "
            f"{checkpoint['sha256']} (expected {expected_sha256})"
        )
    ann_template.set_T(0)
    if hasattr(ann_template, "set_L"):
        ann_template.set_L(args.L)
    if hasattr(snn_template, "set_L"):
        snn_template.set_L(args.L)
    evaluation_profile = "not-applicable"
    if architecture == "resnet20" and hasattr(
        ann_template, "set_qcfs_training_profile"
    ):
        evaluation_profile = args.resnet20_eval_profile
        ann_template.set_qcfs_training_profile(evaluation_profile)
        if hasattr(snn_template, "set_qcfs_training_profile"):
            snn_template.set_qcfs_training_profile(evaluation_profile)
    ann_template.eval()
    set_signed_spike_stats_enabled(snn_template, SignedIF, False)

    if args.dataset == "cifar10" and architecture == "resnet20":
        checkpoint_selection_note = (
            "the checkpoint is selected by the highest accuracy observed on "
            "the 10,000-image CIFAR-10 test set during 300 training epochs; "
            "this creates model-selection bias"
        )
        checkpoint_interpretation_note = (
            f"the ResNet20 checkpoint is evaluated with QCFS L={args.L} and "
            f"the {evaluation_profile} profile; the checkpoint training "
            "provenance is recorded separately"
        )
    elif args.dataset == "cifar10" and architecture == "vgg16":
        checkpoint_selection_note = (
            "the existing VGG16 checkpoint selection procedure is not recorded"
        )
        checkpoint_interpretation_note = (
            "the legacy VGG16 checkpoint was probably trained with L=4 and is "
            "evaluated post-hoc with L=8; it is not an L=8-trained model"
        )
    else:
        checkpoint_selection_note = "unchanged from the frozen CIFAR-100 protocol"
        checkpoint_interpretation_note = (
            "the checkpoint is evaluated under its frozen CIFAR-100 protocol"
        )

    payload = {
        "status": "selecting_a_snm_modes",
        "protocol": {
            "dataset": args.dataset,
            "dataset_label": DATASET_PROTOCOLS[args.dataset]["label"],
            "architecture": architecture,
            "checkpoint": checkpoint,
            "ann_accuracy": None,
            "time_steps": list(args.time_steps),
            "batch_size": args.batch_size,
            "fit_batches": args.fit_batches,
            "validation_batches": args.validation_batches,
            "fit_sha256": batches_sha256(fit_batches),
            "validation_sha256": batches_sha256(validation_batches),
            "alpha": args.alpha,
            "qcfs_L": args.L,
            "normalization": DATASET_PROTOCOLS[args.dataset]["normalization"],
            "resnet20_eval_profile": evaluation_profile,
            "checkpoint_selection_note": checkpoint_selection_note,
            "checkpoint_interpretation_note": checkpoint_interpretation_note,
            "a_snm_rule": "SNM-on iff on validation accuracy > off validation accuracy; ties select off",
            "seed": args.seed,
            "coding_mode": "rate",
            "schedule": "rate",
            "ratio": 1.0,
            "r0": True,
            "snm_margin": 0.0,
            "ftbc_mode": "full",
            "test_batches": args.test_batches,
            "test_samples": None,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": "float32",
            "validation_inferences": 4 * len(args.time_steps),
        },
        "calibration": OrderedDict(),
        "validation": {
            "qcfs": {"off": OrderedDict(), "on": OrderedDict()},
            "full": {"off": OrderedDict(), "on": OrderedDict()},
        },
        "gates": {},
        "selection_elapsed": {"qcfs": 0.0, "full": 0.0},
        "selection_trace": {},
        "results": OrderedDict((name, OrderedDict()) for name in CONFIGS),
        "layers": OrderedDict((name, OrderedDict()) for name in CONFIGS),
        "equivalence_checks": [],
    }
    schedules = {}

    for time_steps in args.time_steps:
        key = str(time_steps)
        print(f"[{architecture}] T={time_steps}: QCFS SNM-off/on validation", flush=True)
        for mode, signed in (("off", False), ("on", True)):
            model = build_plain_model(snn_template, time_steps, signed, device)
            metrics = evaluate_validation(
                model,
                ann_template,
                validation_batches,
                device,
                time_steps,
                architecture,
            )
            payload["validation"]["qcfs"][mode][key] = metrics
            payload["selection_elapsed"]["qcfs"] += metrics["elapsed"]
            del model

        print(f"[{architecture}] T={time_steps}: unsigned Full-FTBC fit", flush=True)
        teacher = copy.deepcopy(snn_template).to(device)
        configure_snn(teacher, time_steps, signed=False, ftbc_mode="full")
        ann = copy.deepcopy(ann_template).to(device)
        ann.set_T(0)
        synchronize(device)
        calibration_started = time.perf_counter()
        bias_corr_model(
            ann=ann,
            snn=teacher,
            T=time_steps,
            train_loader=fit_batches,
            curr_t_alpha=args.alpha,
            num_cali_sample_batches=args.fit_batches,
            ftbc_mode="full",
        )
        synchronize(device)
        calibration_elapsed = time.perf_counter() - calibration_started
        schedules[time_steps] = snapshot_full_ftbc(teacher, time_steps)
        storage = summarize_ftbc_storage(teacher, SignedIF)
        payload["calibration"][key] = {
            "elapsed": calibration_elapsed,
            "ftbc_parameters": storage["parameters"],
            "ftbc_bytes": storage["bytes"],
        }

        print(f"[{architecture}] T={time_steps}: Full-FTBC SNM-off/on validation", flush=True)
        for mode, signed in (("off", False), ("on", True)):
            model = copy.deepcopy(teacher).to(device)
            model.set_signed(signed)
            metrics = evaluate_validation(
                model,
                ann,
                validation_batches,
                device,
                time_steps,
                architecture,
            )
            payload["validation"]["full"][mode][key] = metrics
            payload["selection_elapsed"]["full"] += metrics["elapsed"]
            del model
        del ann, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_progress(progress_path, payload)

    frozen_modes = {}
    for family in ("qcfs", "full"):
        off_metrics = {
            int(key): value for key, value in payload["validation"][family]["off"].items()
        }
        on_metrics = {
            int(key): value for key, value in payload["validation"][family]["on"].items()
        }
        validate_t1_special_case(off_metrics, on_metrics, family)
        synchronize(device)
        selection_started = time.perf_counter()
        selected_modes, trace = select_a_snm_modes(
            off_metrics,
            on_metrics,
            time_steps=args.time_steps,
        )
        synchronize(device)
        payload["selection_elapsed"][family] += time.perf_counter() - selection_started
        frozen_modes[family] = selected_modes
        payload["gates"][family] = {
            str(time_steps_value): enabled
            for time_steps_value, enabled in selected_modes.items()
        }
        payload["selection_trace"][family] = trace
        enabled_times = [
            str(time_steps_value)
            for time_steps_value, enabled in selected_modes.items()
            if enabled
        ]
        print(
            f"[{architecture}] selected {family} A-SNM on T="
            + (",".join(enabled_times) if enabled_times else "none"),
            flush=True,
        )

    payload["status"] = "a_snm_modes_frozen_testing"
    save_progress(progress_path, payload)

    evaluation_loader = (
        materialize_calibration_batches(test_loader, args.test_batches)
        if args.test_batches > 0
        else test_loader
    )
    payload["protocol"]["ann_accuracy"] = val(
        ann_template, evaluation_loader, device, 0
    )
    if args.test_batches > 0:
        payload["protocol"]["test_samples"] = sum(
            int(targets.numel()) for _, targets in evaluation_loader
        )
    else:
        payload["protocol"]["test_samples"] = len(evaluation_loader.dataset)
        if payload["protocol"]["test_samples"] != 10000:
            raise RuntimeError(
                f"Formal {DATASET_PROTOCOLS[args.dataset]['label']} test set "
                "must contain 10,000 images"
            )
    if args.dataset == "cifar10" and args.test_batches == 0:
        ann_accuracy = float(payload["protocol"]["ann_accuracy"])
        if architecture == "resnet20" and ann_accuracy < 90.0:
            raise RuntimeError(
                f"CIFAR-10/ResNet20 ANN accuracy {ann_accuracy:.2f}% is below "
                "the fixed 90% sanity floor"
            )
        if architecture == "vgg16" and not math.isclose(
            ann_accuracy, 95.51, abs_tol=0.1
        ):
            raise RuntimeError(
                f"CIFAR-10/VGG16 ANN accuracy {ann_accuracy:.2f}% does not "
                "reproduce the expected 95.51% within 0.1pp"
            )

    for time_steps in args.time_steps:
        key = str(time_steps)
        base_results = {}
        base_layers = {}
        for family in ("qcfs", "full"):
            for mode, signed in (("off", False), ("on", True)):
                config_name = BASE_CONFIGS[family][mode]
                print(f"[{architecture}] T={time_steps}: test {config_name}", flush=True)
                if family == "qcfs":
                    model = build_plain_model(snn_template, time_steps, signed, device)
                else:
                    model = build_full_model(
                        snn_template,
                        schedules[time_steps],
                        time_steps,
                        signed,
                        device,
                    )
                result, layers = evaluate_test(
                    model,
                    ann_template,
                    evaluation_loader,
                    device,
                    time_steps,
                    architecture,
                )
                if family == "full":
                    result["calibration_elapsed"] = payload["calibration"][key]["elapsed"]
                result.update(
                    {
                        "snm_mode": mode,
                        "snm_enabled": signed,
                        "a_snm_enabled": None,
                        "source_config": None,
                    }
                )
                payload["results"][config_name][key] = result
                payload["layers"][config_name][key] = layers
                base_results[(family, mode)] = result
                base_layers[(family, mode)] = layers
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        for family in ("qcfs", "full"):
            enabled = a_snm_enabled(frozen_modes[family], time_steps)
            mode = "on" if enabled else "off"
            source_name = BASE_CONFIGS[family][mode]
            config_name = A_SNM_CONFIGS[family]
            result = copy.deepcopy(base_results[(family, mode)])
            layers = copy.deepcopy(base_layers[(family, mode)])
            result.update(
                {
                    "snm_mode": "a_snm",
                    "snm_enabled": enabled,
                    "a_snm_enabled": enabled,
                    "a_snm_selection_elapsed": payload["selection_elapsed"][family],
                    "source_config": source_name,
                }
            )
            payload["results"][config_name][key] = result
            payload["layers"][config_name][key] = layers
            payload["equivalence_checks"].append(
                {
                    "config": config_name,
                    "time_steps": time_steps,
                    "source": source_name,
                    "exact": result["acc"] == base_results[(family, mode)]["acc"]
                    and result["logit_mse"] == base_results[(family, mode)]["logit_mse"]
                    and result["sops"] == base_results[(family, mode)]["sops"]
                    and result["negative_spikes"]
                    == base_results[(family, mode)]["negative_spikes"],
                }
            )
        save_progress(progress_path, payload)

    if not all(item["exact"] for item in payload["equivalence_checks"]):
        raise RuntimeError("A-SNM deployment equivalence check failed")
    payload["status"] = "complete"
    save_progress(progress_path, payload)
    write_report(output, payload)
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="QCFS + Full-FTBC + Accuracy-Gated SNM six-way ablation"
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_PROTOCOLS),
        default="cifar100",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("resnet20", "vgg16"),
        default=("resnet20", "vgg16"),
    )
    parser.add_argument(
        "--resnet20_checkpoint", type=Path
    )
    parser.add_argument(
        "--vgg16_checkpoint", type=Path
    )
    parser.add_argument("--resnet20_checkpoint_sha256")
    parser.add_argument("--vgg16_checkpoint_sha256")
    parser.add_argument(
        "--resnet20_eval_profile",
        choices=("fixed_repo", "paper_era"),
    )
    parser.add_argument("-L", "--L", type=int, default=8)
    parser.add_argument(
        "--time_steps", nargs="+", type=int, default=DEFAULT_TIME_STEPS
    )
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--fit_batches", type=int, default=5)
    parser.add_argument("--validation_batches", type=int, default=5)
    parser.add_argument("--test_batches", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
    )
    return parser


def resolve_protocol_args(args):
    protocol = DATASET_PROTOCOLS[args.dataset]
    l4_resnet20 = (
        args.dataset == "cifar10"
        and args.L == 4
        and tuple(args.architectures) == ("resnet20",)
    )
    for architecture in ("resnet20", "vgg16"):
        checkpoint_name = f"{architecture}_checkpoint"
        if getattr(args, checkpoint_name) is None:
            if architecture == "resnet20" and l4_resnet20:
                checkpoint = CIFAR10_RESNET20_L4_PROTOCOL["checkpoint"]
            else:
                checkpoint = protocol["default_checkpoints"][architecture]
            setattr(
                args,
                checkpoint_name,
                checkpoint,
            )
        sha_name = f"{architecture}_checkpoint_sha256"
        if getattr(args, sha_name) is None:
            if architecture == "resnet20" and l4_resnet20:
                expected_sha256 = CIFAR10_RESNET20_L4_PROTOCOL[
                    "expected_sha256"
                ]
            else:
                expected_sha256 = protocol["expected_sha256"][architecture]
            setattr(args, sha_name, expected_sha256)
    if args.resnet20_eval_profile is None:
        if l4_resnet20:
            args.resnet20_eval_profile = CIFAR10_RESNET20_L4_PROTOCOL[
                "evaluation_profile"
            ]
        else:
            args.resnet20_eval_profile = protocol["resnet20_eval_profile"]
    if args.output is None:
        if l4_resnet20:
            args.output = CIFAR10_RESNET20_L4_PROTOCOL["default_output"]
        else:
            args.output = protocol["default_output"]
    return args


def validate_args(args):
    if args.batch_size <= 0 or args.fit_batches <= 0 or args.validation_batches <= 0:
        raise ValueError("Batch size and calibration batch counts must be positive")
    if len(set(args.time_steps)) != len(args.time_steps):
        raise ValueError("Time steps must be unique")
    if tuple(sorted(args.time_steps)) != tuple(args.time_steps):
        raise ValueError("Time steps must be sorted")
    if args.test_batches < 0:
        raise ValueError("test_batches must be non-negative")
    if args.alpha != 0.4:
        raise ValueError("The experiment plan fixes Full-FTBC alpha at 0.4")
    if args.L not in (4, 8):
        raise ValueError("The A-SNM protocol supports QCFS L=4 or L=8")
    if args.L == 4:
        if args.dataset != "cifar10" or tuple(args.architectures) != ("resnet20",):
            raise ValueError(
                "QCFS L=4 is locked to the CIFAR-10/ResNet20 paper-aligned protocol"
            )
        if args.resnet20_eval_profile != "paper_era":
            raise ValueError(
                "The CIFAR-10/ResNet20 L=4 protocol requires paper_era evaluation"
            )
    for architecture in args.architectures:
        expected_sha256 = getattr(args, f"{architecture}_checkpoint_sha256")
        if not expected_sha256:
            raise ValueError(
                f"{args.dataset}/{architecture} requires an expected checkpoint "
                "SHA256 before any experiment run"
            )
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in expected_sha256
        ):
            raise ValueError(
                f"Invalid {architecture} checkpoint SHA256: {expected_sha256!r}"
            )
    formal = (
        tuple(args.time_steps) == DEFAULT_TIME_STEPS
        and args.batch_size == 200
        and args.fit_batches == 5
        and args.validation_batches == 5
        and args.test_batches == 0
    )
    if not formal and "archive" not in {part.lower() for part in args.output.parts}:
        raise ValueError("Non-formal smoke runs must write under docs/archive")


def main(cli_args=None):
    args = resolve_protocol_args(build_parser().parse_args(cli_args))
    validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.test_batches == 0:
        raise RuntimeError("The formal experiment requires CUDA")
    seed_all(args.seed)
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    all_calibration = materialize_calibration_batches(
        train_loader,
        args.fit_batches + args.validation_batches,
    )
    fit_batches = all_calibration[: args.fit_batches]
    validation_batches = all_calibration[args.fit_batches :]
    checkpoints = {
        "resnet20": args.resnet20_checkpoint,
        "vgg16": args.vgg16_checkpoint,
    }
    checkpoint_sha256 = {
        "resnet20": args.resnet20_checkpoint_sha256,
        "vgg16": args.vgg16_checkpoint_sha256,
    }
    outputs = []
    for architecture in args.architectures:
        outputs.append(
            run_architecture(
                args,
                architecture,
                checkpoints[architecture],
                checkpoint_sha256[architecture],
                fit_batches,
                validation_batches,
                test_loader,
                device,
            )
        )
    for output in outputs:
        print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
