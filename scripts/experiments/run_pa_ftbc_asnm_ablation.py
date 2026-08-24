"""QCFS + Full/Temporal-LR/Parity-Anchor FTBC + A-SNM twelve-way ablation."""

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch

from a_snm import DEFAULT_TIME_STEPS, a_snm_enabled, select_a_snm_modes
from calibration import bias_corr_model
from models import SignedIF
from parity_anchor_ftbc import (
    compress_full_ftbc_teacher as compress_parity_anchor,
    named_signed_layers,
)
from preprocess import datapool
from scripts.experiments.qcfs_checkpoint import load_qcfs_pair
from scripts.experiments.run_full_ftbc_asnm_ablation import (
    CIFAR10_RESNET20_L4_PROTOCOL,
    DATASET_PROTOCOLS,
    build_full_model,
    build_plain_model,
    configure_snn,
    evaluate_test,
    evaluate_validation,
    snapshot_full_ftbc,
    validate_t1_special_case,
)
from scripts.experiments.run_state_ftbc_ablation import (
    materialize_calibration_batches,
)
from scripts.experiments.run_temporal_lr_asnm_ablation import (
    TEST_EQUIVALENCE_KEYS,
    VALIDATION_EQUIVALENCE_KEYS,
    build_temporal_model,
    exact_metrics,
)
from scripts.experiments.run_temporal_lr_gated_snm import (
    batches_sha256,
    synchronize,
)
from spike_stats import (
    set_signed_spike_stats_enabled,
    summarize_ftbc_storage,
)
from utils import seed_all, val


FAMILIES = ("qcfs", "full", "temporal", "pa")
CONFIGS = OrderedDict(
    [
        ("A_QCFS_R0", {"family": "qcfs", "mode": "off"}),
        ("B_QCFS_STANDARD_SNM_R0", {"family": "qcfs", "mode": "on"}),
        ("C_QCFS_ASNM_R0", {"family": "qcfs", "mode": "a_snm"}),
        ("D_QCFS_FULL_FTBC_R0", {"family": "full", "mode": "off"}),
        ("E_QCFS_FULL_FTBC_STANDARD_SNM_R0", {"family": "full", "mode": "on"}),
        ("F_QCFS_FULL_FTBC_ASNM_R0", {"family": "full", "mode": "a_snm"}),
        ("G_QCFS_TEMPORAL_LR_FTBC_R0", {"family": "temporal", "mode": "off"}),
        ("H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0", {"family": "temporal", "mode": "on"}),
        ("I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0", {"family": "temporal", "mode": "a_snm"}),
        ("J_QCFS_PA_FTBC_R0", {"family": "pa", "mode": "off"}),
        ("K_QCFS_PA_FTBC_STANDARD_SNM_R0", {"family": "pa", "mode": "on"}),
        ("L_QCFS_PA_FTBC_ASNM_R0", {"family": "pa", "mode": "a_snm"}),
    ]
)
BASE_CONFIGS = {
    family: {
        mode: next(
            name
            for name, config in CONFIGS.items()
            if config["family"] == family and config["mode"] == mode
        )
        for mode in ("off", "on")
    }
    for family in FAMILIES
}
A_SNM_CONFIGS = {
    family: next(
        name
        for name, config in CONFIGS.items()
        if config["family"] == family and config["mode"] == "a_snm"
    )
    for family in FAMILIES
}
FAMILY_TITLES = {
    "qcfs": "QCFS",
    "full": "Full-FTBC",
    "temporal": "Temporal-LR FTBC",
    "pa": "Parity-Anchor FTBC",
}
EXPECTED_ANN_ACCURACY = {
    ("cifar10", "resnet20", 4): 90.72,
    ("cifar10", "vgg16", 8): 95.51,
    ("cifar100", "resnet20", 8): 68.68,
    ("cifar100", "vgg16", 8): 77.35,
}
FORMAL_BATCH_HASHES = {
    "cifar10": {
        "fit": "053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df",
        "validation": "237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c",
    },
    "cifar100": {
        "fit": "9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a",
        "validation": "d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3",
    },
}
DEFAULT_OUTPUTS = {
    ("cifar10", "resnet20", 4): Path(
        "docs/results/comparative_ablation/cifar10/"
        "PA_FTBC_ASNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.md"
    ),
    ("cifar10", "vgg16", 8): Path(
        "docs/results/comparative_ablation/cifar10/PA_FTBC_ASNM_CIFAR10_VGG16_L8.md"
    ),
    ("cifar100", "resnet20", 8): Path(
        "docs/results/comparative_ablation/cifar100/PA_FTBC_ASNM_CIFAR100_RESNET20_L8.md"
    ),
    ("cifar100", "vgg16", 8): Path(
        "docs/results/comparative_ablation/cifar100/PA_FTBC_ASNM_CIFAR100_VGG16_L8.md"
    ),
}


def protocol_key(args):
    return args.dataset, args.architecture, int(args.L)


def is_formal_protocol(args):
    return (
        tuple(args.time_steps) == tuple(DEFAULT_TIME_STEPS)
        and args.batch_size == 200
        and args.fit_batches == 5
        and args.validation_batches == 5
        and args.test_batches == 0
        and args.alpha == 0.4
        and args.temporal_rank == 4
        and args.pa_coefficients == 4
    )


def resolve_protocol_args(args):
    key = protocol_key(args)
    if key not in DEFAULT_OUTPUTS:
        raise ValueError(
            "Supported protocols are CIFAR-10/ResNet20 L=4 and all other "
            "requested dataset/architecture combinations at L=8"
        )
    base = DATASET_PROTOCOLS[args.dataset]
    if key == ("cifar10", "resnet20", 4):
        default_checkpoint = CIFAR10_RESNET20_L4_PROTOCOL["checkpoint"]
        default_sha256 = CIFAR10_RESNET20_L4_PROTOCOL["expected_sha256"]
        default_profile = CIFAR10_RESNET20_L4_PROTOCOL["evaluation_profile"]
    else:
        default_checkpoint = base["default_checkpoints"][args.architecture]
        default_sha256 = base["expected_sha256"][args.architecture]
        default_profile = (
            base["resnet20_eval_profile"]
            if args.architecture == "resnet20"
            else "not-applicable"
        )
    args.checkpoint = args.checkpoint or default_checkpoint
    args.checkpoint_sha256 = args.checkpoint_sha256 or default_sha256
    args.resnet20_eval_profile = args.resnet20_eval_profile or default_profile
    args.output = args.output or DEFAULT_OUTPUTS[key]
    return args


def validate_args(args):
    if args.batch_size <= 0 or args.fit_batches <= 0 or args.validation_batches <= 0:
        raise ValueError("Batch size and calibration batch counts must be positive")
    if tuple(sorted(set(args.time_steps))) != tuple(args.time_steps):
        raise ValueError("Time steps must be unique and sorted")
    if args.test_batches < 0:
        raise ValueError("test_batches must be non-negative")
    if args.alpha != 0.4:
        raise ValueError("The protocol fixes Full-FTBC alpha at 0.4")
    if args.temporal_rank != 4 or args.pa_coefficients != 4:
        raise ValueError("Temporal rank and PA coefficient count are fixed at four")
    if len(args.checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in args.checkpoint_sha256
    ):
        raise ValueError("A valid expected checkpoint SHA256 is required")
    if not is_formal_protocol(args) and "archive" not in {
        part.lower() for part in args.output.parts
    }:
        raise ValueError("Non-formal smoke runs must write under docs/archive")


def save_progress(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_pa_model(template, schedule, time_steps, signed, device):
    model = build_full_model(template, schedule, time_steps, signed, device)
    full_storage = summarize_ftbc_storage(model, SignedIF)
    if int(time_steps) <= 4:
        layers = OrderedDict(
            (
                name,
                {
                    "representation": "full",
                    "channels": int(module.time_based_bias[0].numel()),
                    "mse": 0.0,
                    "nrmse": 0.0,
                    "max_abs_error": 0.0,
                },
            )
            for name, module in named_signed_layers(model).items()
        )
        return model, {
            "coefficient_count": int(time_steps),
            "time_steps": int(time_steps),
            "compressed_channels": sum(x["channels"] for x in layers.values()),
            "basis_stored": False,
            "threshold_normalize": False,
            "structure": "Full-FTBC fallback",
            "explained_energy": 1.0,
            "layers": layers,
            "fallback_to_full": True,
            "compression_elapsed": 0.0,
            "ftbc_parameters": int(full_storage["parameters"]),
            "ftbc_bytes": int(full_storage["bytes"]),
            "ftbc_synthesis_macs": 0,
        }
    synchronize(device)
    started = time.perf_counter()
    report = compress_parity_anchor(model)
    synchronize(device)
    report["compression_elapsed"] = time.perf_counter() - started
    report["fallback_to_full"] = False
    storage = summarize_ftbc_storage(model, SignedIF)
    report["ftbc_parameters"] = int(storage["parameters"])
    report["ftbc_bytes"] = int(storage["bytes"])
    report["ftbc_synthesis_macs"] = int(storage["synthesis_macs"])
    return model, report


def metric_table(lines, title, payload, key, formatter):
    times = payload["protocol"]["time_steps"]
    lines.extend(
        [
            f"## {title}",
            "",
            "| Config | " + " | ".join(f"T={t}" for t in times) + " |",
            "|---|" + "---:|" * len(times),
        ]
    )
    for name in CONFIGS:
        cells = [formatter(payload["results"][name][str(t)][key]) for t in times]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")


def selected_time_label(name, gates, time_steps):
    config = CONFIGS[name]
    if config["mode"] == "off":
        return "none"
    if config["mode"] == "on":
        return ", ".join(str(t) for t in time_steps)
    selected = [str(t) for t in time_steps if gates[config["family"]][str(t)]]
    return ", ".join(selected) if selected else "none"


def mean_accuracy(payload, config_name):
    values = [
        payload["results"][config_name][str(t)]["acc"]
        for t in payload["protocol"]["time_steps"]
    ]
    return sum(values) / len(values)


def write_report(path, payload):
    protocol = payload["protocol"]
    times = protocol["time_steps"]
    lines = [
        "# QCFS + Full-FTBC + Temporal-LR FTBC + Parity-Anchor FTBC + A-SNM Ablation",
        "",
        f"Status: {payload['status']}",
        "",
        f"- Dataset/architecture: {protocol['dataset_label']}/{protocol['architecture']}",
        f"- QCFS L: {protocol['qcfs_L']}",
        f"- ANN accuracy: {protocol['ann_accuracy']:.2f}%",
        f"- Checkpoint: `{protocol['checkpoint']['filename']}`",
        f"- Checkpoint SHA256: `{protocol['checkpoint']['sha256']}`",
        f"- Fit/validation SHA256: `{protocol['fit_sha256']}` / `{protocol['validation_sha256']}`",
        f"- Test samples: {protocol['test_samples']:,}",
        f"- Evaluation profile: `{protocol['resnet20_eval_profile']}`",
        "- Full-FTBC is fitted independently at every T with SNM off.",
        "- Temporal-LR uses a shared learned rank-4 SVD basis with threshold normalization.",
        "- PA-FTBC uses no SVD or stored basis: t=0/t=1 anchors plus tail mean and tail parity.",
        "- Both compressed methods fall back exactly to Full-FTBC at T<=4.",
        "- Every family freezes its own strict accuracy-gated A-SNM decisions before test inference.",
        f"- Checkpoint note: {protocol['checkpoint_interpretation_note']}.",
        "",
        "## Primary accuracy table",
        "",
        "| Config | " + " | ".join(f"T={t}" for t in times) + " | SNM-on T |",
        "|---|" + "---:|" * len(times) + "---|",
    ]
    for name in CONFIGS:
        cells = [f"{payload['results'][name][str(t)]['acc']:.2f}%" for t in times]
        lines.append(
            f"| {name} | " + " | ".join(cells) +
            f" | {selected_time_label(name, payload['gates'], times)} |"
        )
    lines.extend(
        [
            "",
            "## Mean accuracy over evaluated time steps",
            "",
            "| Config | Mean accuracy |",
            "|---|---:|",
        ]
    )
    for name in CONFIGS:
        lines.append(f"| {name} | {mean_accuracy(payload, name):.2f}% |")
    lines.extend(
        [
            "",
            "## PA-FTBC accuracy comparisons",
            "",
            "| T | PA off - Temporal off | PA standard - Temporal standard | PA A-SNM - Temporal A-SNM |",
            "|---:|---:|---:|---:|",
        ]
    )
    comparisons = (
        ("J_QCFS_PA_FTBC_R0", "G_QCFS_TEMPORAL_LR_FTBC_R0"),
        ("K_QCFS_PA_FTBC_STANDARD_SNM_R0", "H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0"),
        ("L_QCFS_PA_FTBC_ASNM_R0", "I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0"),
    )
    for t in times:
        differences = [
            payload["results"][left][str(t)]["acc"]
            - payload["results"][right][str(t)]["acc"]
            for left, right in comparisons
        ]
        lines.append(f"| {t} | " + " | ".join(f"{x:+.2f}pp" for x in differences) + " |")
    lines.append(
        "| Mean | "
        + " | ".join(
            f"{mean_accuracy(payload, left) - mean_accuracy(payload, right):+.2f}pp"
            for left, right in comparisons
        )
        + " |"
    )
    lines.append("")

    metric_table(lines, "ANN-SNN logit MSE", payload, "logit_mse", lambda x: f"{x:.8f}")
    metric_table(lines, "Positive spike rate", payload, "positive_rate", lambda x: f"{100*x:.6f}%")
    metric_table(lines, "Negative spike rate", payload, "negative_rate", lambda x: f"{100*x:.6f}%")
    metric_table(lines, "Overall spike sparsity", payload, "sparsity", lambda x: f"{100*x:.6f}%")
    metric_table(lines, "Input-driven SOPs", payload, "sops", lambda x: f"{x:,}")
    metric_table(lines, "FTBC parameters", payload, "ftbc_parameters", lambda x: f"{x:,}")
    metric_table(lines, "FTBC storage bytes", payload, "ftbc_bytes", lambda x: f"{x:,}")
    metric_table(lines, "Bias synthesis MACs", payload, "ftbc_synthesis_macs", lambda x: f"{x:,}")
    metric_table(lines, "Full-teacher calibration elapsed", payload, "calibration_elapsed", lambda x: f"{x:.6f}")
    metric_table(lines, "Compression elapsed", payload, "compression_elapsed", lambda x: f"{x:.6f}")
    metric_table(lines, "Inference elapsed (statistics disabled)", payload, "inference_elapsed", lambda x: f"{x:.6f}")

    lines.extend(
        [
            "## Compression summary",
            "",
            "| T | Full params | Temporal params | PA params | Temporal saving | PA saving | Temporal MACs | PA MACs | Temporal energy | PA energy |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for t in times:
        temporal = payload["compression"]["temporal"][str(t)]
        pa = payload["compression"]["pa"][str(t)]
        lines.append(
            f"| {t} | {temporal['full_parameters']:,} | {temporal['ftbc_parameters']:,} | "
            f"{pa['ftbc_parameters']:,} | {100*temporal['storage_reduction']:.2f}% | "
            f"{100*pa['storage_reduction']:.2f}% | {temporal['ftbc_synthesis_macs']:,} | "
            f"{pa['ftbc_synthesis_macs']:,} | {temporal['explained_energy']:.6f} | "
            f"{pa['explained_energy']:.6f} |"
        )
    lines.append("")

    lines.extend(["## A-SNM selection", ""])
    for family in FAMILIES:
        enabled = [str(t) for t in times if payload["gates"][family][str(t)]]
        lines.append(
            f"- {FAMILY_TITLES[family]} SNM-on T: "
            + (", ".join(enabled) if enabled else "none")
            + f"; selection elapsed: {payload['selection_elapsed'][family]:.6f}s."
        )
        lines.extend(
            [
                "",
                f"### {FAMILY_TITLES[family]} accuracy-gate trace",
                "",
                "| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |",
                "|---:|---:|---:|---|",
            ]
        )
        for time_steps_value, item in payload["selection_trace"][family].items():
            lines.append(
                f"| {time_steps_value} | {item['off_accuracy']:.2f}% | "
                f"{item['on_accuracy']:.2f}% | {item['selected_mode']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Validation-selection generalization audit",
            "",
            "| Family | T | Selected | Test off | Test on | Test-best | Match |",
            "|---|---:|---|---:|---:|---|---|",
        ]
    )
    for item in payload["generalization_audit"]:
        lines.append(
            f"| {FAMILY_TITLES[item['family']]} | {item['time_steps']} | "
            f"{item['selected_mode']} | {item['test_off_accuracy']:.2f}% | "
            f"{item['test_on_accuracy']:.2f}% | {item['test_best_mode']} | "
            f"{'yes' if item['matches_test_best'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Equivalence checks",
            "",
            "| Kind | Name | T | Source | Exact |",
            "|---|---|---:|---|---|",
        ]
    )
    for item in payload["equivalence_checks"]:
        lines.append(
            f"| {item['kind']} | {item['name']} | {item['time_steps']} | "
            f"{item['source']} | {'yes' if item['exact'] else 'no'} |"
        )
    lines.append("")

    for family in ("temporal", "pa"):
        lines.extend(
            [
                f"## Per-layer {FAMILY_TITLES[family]} reconstruction",
                "",
            ]
        )
        for t in times:
            lines.extend(
                [
                    f"### T={t}",
                    "",
                    "| Layer | Representation | Channels | MSE | NRMSE | Max abs error |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for layer_name, item in payload["compression"][family][str(t)]["layers"].items():
                lines.append(
                    f"| `{layer_name}` | {item['representation']} | {item['channels']} | "
                    f"{item['mse']:.8f} | {item['nrmse']:.8f} | {item['max_abs_error']:.8f} |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args, fit_batches, validation_batches, test_loader, device):
    output = args.output
    progress = output.with_suffix(".progress.json")
    if not args.overwrite and (output.exists() or progress.exists()):
        raise FileExistsError(f"Refusing to overwrite {output} or {progress}")

    ann_template, snn_template, checkpoint = load_qcfs_pair(
        args.checkpoint,
        args.dataset,
        args.architecture,
        device,
    )
    if checkpoint["sha256"].lower() != args.checkpoint_sha256.lower():
        raise RuntimeError(
            f"Unexpected checkpoint SHA256 {checkpoint['sha256']} "
            f"(expected {args.checkpoint_sha256})"
        )
    ann_template.set_T(0)
    if hasattr(ann_template, "set_L"):
        ann_template.set_L(args.L)
    if hasattr(snn_template, "set_L"):
        snn_template.set_L(args.L)
    evaluation_profile = "not-applicable"
    if args.architecture == "resnet20":
        evaluation_profile = args.resnet20_eval_profile
        ann_template.set_qcfs_training_profile(evaluation_profile)
        snn_template.set_qcfs_training_profile(evaluation_profile)
    ann_template.eval()
    set_signed_spike_stats_enabled(snn_template, SignedIF, False)

    if protocol_key(args) == ("cifar10", "resnet20", 4):
        checkpoint_note = (
            "CIFAR-10/ResNet20 QCFS-L4 paper-aligned retrained checkpoint; "
            "selected by peak test accuracy during training and therefore subject "
            "to test-set model-selection bias; not a strict paper reproduction"
        )
    else:
        checkpoint_note = "existing frozen repository checkpoint and evaluation protocol"

    payload = {
        "status": "selecting_a_snm_modes",
        "protocol": {
            "dataset": args.dataset,
            "dataset_label": DATASET_PROTOCOLS[args.dataset]["label"],
            "architecture": args.architecture,
            "qcfs_L": args.L,
            "resnet20_eval_profile": evaluation_profile,
            "normalization": DATASET_PROTOCOLS[args.dataset]["normalization"],
            "checkpoint": checkpoint,
            "checkpoint_interpretation_note": checkpoint_note,
            "ann_accuracy": None,
            "time_steps": list(args.time_steps),
            "batch_size": args.batch_size,
            "fit_batches": args.fit_batches,
            "validation_batches": args.validation_batches,
            "fit_sha256": batches_sha256(fit_batches),
            "validation_sha256": batches_sha256(validation_batches),
            "alpha": args.alpha,
            "temporal_rank": args.temporal_rank,
            "pa_coefficients": args.pa_coefficients,
            "pa_structure": "t0 anchor + t1 anchor + tail mean + tail parity",
            "compressed_fallback": "Full-FTBC at T<=4",
            "a_snm_rule": "SNM-on iff validation on accuracy > off accuracy; ties off",
            "seed": args.seed,
            "coding_mode": "rate",
            "schedule": "rate",
            "ratio": 1.0,
            "r0": True,
            "snm_margin": 0.0,
            "test_batches": args.test_batches,
            "test_samples": None,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": "float32",
        },
        "calibration": OrderedDict(),
        "compression": {"temporal": OrderedDict(), "pa": OrderedDict()},
        "validation": {
            family: {"off": OrderedDict(), "on": OrderedDict()}
            for family in FAMILIES
        },
        "gates": {},
        "selection_elapsed": {family: 0.0 for family in FAMILIES},
        "selection_trace": {},
        "results": OrderedDict((name, OrderedDict()) for name in CONFIGS),
        "layers": OrderedDict((name, OrderedDict()) for name in CONFIGS),
        "generalization_audit": [],
        "equivalence_checks": [],
    }
    schedules = {}

    for time_steps in args.time_steps:
        key = str(time_steps)
        print(f"[{args.dataset}/{args.architecture}] T={time_steps}: QCFS validation", flush=True)
        for mode, signed in (("off", False), ("on", True)):
            model = build_plain_model(snn_template, time_steps, signed, device)
            metrics = evaluate_validation(
                model, ann_template, validation_batches, device, time_steps, args.architecture
            )
            payload["validation"]["qcfs"][mode][key] = metrics
            payload["selection_elapsed"]["qcfs"] += metrics["elapsed"]
            del model

        print(f"[{args.dataset}/{args.architecture}] T={time_steps}: Full fit", flush=True)
        teacher = copy.deepcopy(snn_template).to(device)
        configure_snn(teacher, time_steps, signed=False, ftbc_mode="full")
        ann = copy.deepcopy(ann_template).to(device)
        ann.set_T(0)
        synchronize(device)
        started = time.perf_counter()
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
        calibration_elapsed = time.perf_counter() - started
        schedules[time_steps] = snapshot_full_ftbc(teacher, time_steps)
        full_storage = summarize_ftbc_storage(teacher, SignedIF)
        payload["calibration"][key] = {
            "elapsed": calibration_elapsed,
            "ftbc_parameters": int(full_storage["parameters"]),
            "ftbc_bytes": int(full_storage["bytes"]),
        }

        for family in ("full", "temporal", "pa"):
            print(f"[{args.dataset}/{args.architecture}] T={time_steps}: {family} validation", flush=True)
            report_for_time = None
            for mode, signed in (("off", False), ("on", True)):
                if family == "full":
                    model = build_full_model(
                        snn_template, schedules[time_steps], time_steps, signed, device
                    )
                elif family == "temporal":
                    model, report = build_temporal_model(
                        snn_template,
                        schedules[time_steps],
                        time_steps,
                        signed,
                        device,
                        args.architecture,
                        rank=args.temporal_rank,
                    )
                    report_for_time = report_for_time or report
                else:
                    model, report = build_pa_model(
                        snn_template,
                        schedules[time_steps],
                        time_steps,
                        signed,
                        device,
                    )
                    report_for_time = report_for_time or report
                metrics = evaluate_validation(
                    model, ann, validation_batches, device, time_steps, args.architecture
                )
                payload["validation"][family][mode][key] = metrics
                payload["selection_elapsed"][family] += metrics["elapsed"]
                del model
            if family in ("temporal", "pa"):
                report_for_time["full_parameters"] = int(full_storage["parameters"])
                report_for_time["full_bytes"] = int(full_storage["bytes"])
                report_for_time["storage_ratio"] = (
                    report_for_time["ftbc_parameters"] / max(int(full_storage["parameters"]), 1)
                )
                report_for_time["storage_reduction"] = 1.0 - report_for_time["storage_ratio"]
                payload["compression"][family][key] = report_for_time

        if time_steps <= 4:
            for compressed in ("temporal", "pa"):
                for mode in ("off", "on"):
                    exact = exact_metrics(
                        payload["validation"]["full"][mode][key],
                        payload["validation"][compressed][mode][key],
                        VALIDATION_EQUIVALENCE_KEYS,
                    )
                    payload["equivalence_checks"].append(
                        {
                            "kind": "validation fallback",
                            "name": f"{mode}:full={compressed}",
                            "time_steps": time_steps,
                            "source": f"Full-FTBC {mode}",
                            "exact": exact,
                        }
                    )
                    if not exact:
                        raise RuntimeError(f"T={time_steps} {compressed} validation fallback mismatch")
        del ann, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_progress(progress, payload)

    frozen_modes = {}
    for family in FAMILIES:
        off = {int(k): v for k, v in payload["validation"][family]["off"].items()}
        on = {int(k): v for k, v in payload["validation"][family]["on"].items()}
        validate_t1_special_case(off, on, family)
        synchronize(device)
        started = time.perf_counter()
        selected, trace = select_a_snm_modes(off, on, time_steps=args.time_steps)
        synchronize(device)
        payload["selection_elapsed"][family] += time.perf_counter() - started
        frozen_modes[family] = selected
        payload["gates"][family] = {str(t): enabled for t, enabled in selected.items()}
        payload["selection_trace"][family] = trace
        enabled = [str(t) for t, value in selected.items() if value]
        print(f"Selected {family} A-SNM T=" + (",".join(enabled) if enabled else "none"), flush=True)

    for time_steps in args.time_steps:
        if time_steps <= 4:
            for compressed in ("temporal", "pa"):
                exact = frozen_modes["full"][time_steps] == frozen_modes[compressed][time_steps]
                payload["equivalence_checks"].append(
                    {
                        "kind": "gate fallback",
                        "name": f"full={compressed}",
                        "time_steps": time_steps,
                        "source": "identical validation metrics",
                        "exact": exact,
                    }
                )
                if not exact:
                    raise RuntimeError(f"T={time_steps} {compressed} gate fallback mismatch")

    payload["status"] = "a_snm_modes_frozen_testing"
    save_progress(progress, payload)
    evaluation_loader = (
        materialize_calibration_batches(test_loader, args.test_batches)
        if args.test_batches > 0
        else test_loader
    )
    payload["protocol"]["ann_accuracy"] = val(ann_template, evaluation_loader, device, 0)
    if args.test_batches > 0:
        payload["protocol"]["test_samples"] = sum(
            int(targets.numel()) for _, targets in evaluation_loader
        )
    else:
        payload["protocol"]["test_samples"] = len(evaluation_loader.dataset)
        if payload["protocol"]["test_samples"] != 10000:
            raise RuntimeError("Formal CIFAR test set must contain 10,000 images")
    expected_ann = EXPECTED_ANN_ACCURACY[protocol_key(args)]
    if is_formal_protocol(args) and not math.isclose(
        payload["protocol"]["ann_accuracy"], expected_ann, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"Unexpected ANN accuracy {payload['protocol']['ann_accuracy']} != {expected_ann}"
        )

    for time_steps in args.time_steps:
        key = str(time_steps)
        base_results = {}
        base_layers = {}
        for family in FAMILIES:
            for mode, signed in (("off", False), ("on", True)):
                name = BASE_CONFIGS[family][mode]
                print(f"[{args.dataset}/{args.architecture}] T={time_steps}: test {name}", flush=True)
                if family == "qcfs":
                    model = build_plain_model(snn_template, time_steps, signed, device)
                elif family == "full":
                    model = build_full_model(
                        snn_template, schedules[time_steps], time_steps, signed, device
                    )
                elif family == "temporal":
                    model, _ = build_temporal_model(
                        snn_template,
                        schedules[time_steps],
                        time_steps,
                        signed,
                        device,
                        args.architecture,
                        rank=args.temporal_rank,
                    )
                else:
                    model, _ = build_pa_model(
                        snn_template,
                        schedules[time_steps],
                        time_steps,
                        signed,
                        device,
                    )
                result, layers = evaluate_test(
                    model, ann_template, evaluation_loader, device, time_steps, args.architecture
                )
                storage = summarize_ftbc_storage(model, SignedIF)
                result["ftbc_synthesis_macs"] = int(storage["synthesis_macs"])
                result["calibration_elapsed"] = (
                    0.0 if family == "qcfs" else payload["calibration"][key]["elapsed"]
                )
                result["compression_elapsed"] = (
                    payload["compression"][family][key]["compression_elapsed"]
                    if family in ("temporal", "pa")
                    else 0.0
                )
                result["effective_ftbc_modes"] = sorted(
                    {module.ftbc_mode for module in named_signed_layers(model).values()}
                )
                result.update(
                    {
                        "snm_mode": mode,
                        "snm_enabled": signed,
                        "a_snm_enabled": None,
                        "source_config": None,
                    }
                )
                payload["results"][name][key] = result
                payload["layers"][name][key] = layers
                base_results[(family, mode)] = result
                base_layers[(family, mode)] = layers
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if time_steps <= 4:
            for compressed in ("temporal", "pa"):
                for mode in ("off", "on"):
                    exact = exact_metrics(
                        base_results[("full", mode)],
                        base_results[(compressed, mode)],
                        TEST_EQUIVALENCE_KEYS,
                    )
                    payload["equivalence_checks"].append(
                        {
                            "kind": "test fallback",
                            "name": f"{mode}:full={compressed}",
                            "time_steps": time_steps,
                            "source": BASE_CONFIGS["full"][mode],
                            "exact": exact,
                        }
                    )
                    if not exact:
                        raise RuntimeError(f"T={time_steps} {compressed} test fallback mismatch")

        for family in FAMILIES:
            enabled = a_snm_enabled(frozen_modes[family], time_steps)
            mode = "on" if enabled else "off"
            source = BASE_CONFIGS[family][mode]
            name = A_SNM_CONFIGS[family]
            result = copy.deepcopy(base_results[(family, mode)])
            layers = copy.deepcopy(base_layers[(family, mode)])
            result.update(
                {
                    "snm_mode": "a_snm",
                    "snm_enabled": enabled,
                    "a_snm_enabled": enabled,
                    "a_snm_selection_elapsed": payload["selection_elapsed"][family],
                    "source_config": source,
                }
            )
            payload["results"][name][key] = result
            payload["layers"][name][key] = layers
            exact = exact_metrics(result, base_results[(family, mode)], TEST_EQUIVALENCE_KEYS)
            payload["equivalence_checks"].append(
                {
                    "kind": "A-SNM cache",
                    "name": name,
                    "time_steps": time_steps,
                    "source": source,
                    "exact": exact,
                }
            )
            if not exact:
                raise RuntimeError(f"T={time_steps} {name} cache mismatch")
            off_acc = base_results[(family, "off")]["acc"]
            on_acc = base_results[(family, "on")]["acc"]
            best = "on" if on_acc > off_acc else "off"
            payload["generalization_audit"].append(
                {
                    "family": family,
                    "time_steps": time_steps,
                    "selected_mode": mode,
                    "test_off_accuracy": off_acc,
                    "test_on_accuracy": on_acc,
                    "test_best_mode": best,
                    "matches_test_best": mode == best,
                }
            )
        save_progress(progress, payload)

    if not all(item["exact"] for item in payload["equivalence_checks"]):
        raise RuntimeError("One or more equivalence checks failed")
    payload["status"] = "complete"
    save_progress(progress, payload)
    write_report(output, payload)
    return output, payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="QCFS + Full/Temporal-LR/PA-FTBC + A-SNM twelve-way ablation"
    )
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--architecture", choices=("resnet20", "vgg16"), required=True)
    parser.add_argument("-L", type=int, choices=(4, 8), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint_sha256")
    parser.add_argument(
        "--resnet20_eval_profile",
        choices=("fixed_repo", "paper_era", "not-applicable"),
    )
    parser.add_argument("--time_steps", nargs="+", type=int, default=DEFAULT_TIME_STEPS)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--fit_batches", type=int, default=5)
    parser.add_argument("--validation_batches", type=int, default=5)
    parser.add_argument("--test_batches", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--temporal_rank", type=int, default=4)
    parser.add_argument("--pa_coefficients", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(cli_args=None):
    args = resolve_protocol_args(build_parser().parse_args(cli_args))
    validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.test_batches == 0:
        raise RuntimeError("Formal experiment requires CUDA")
    seed_all(args.seed)
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    materialized = materialize_calibration_batches(
        train_loader,
        args.fit_batches + args.validation_batches,
    )
    fit_batches = materialized[: args.fit_batches]
    validation_batches = materialized[args.fit_batches :]
    if is_formal_protocol(args):
        expected = FORMAL_BATCH_HASHES[args.dataset]
        actual_fit = batches_sha256(fit_batches)
        actual_validation = batches_sha256(validation_batches)
        if actual_fit != expected["fit"] or actual_validation != expected["validation"]:
            raise RuntimeError(
                "Formal calibration tensor hash mismatch: "
                f"fit={actual_fit}, validation={actual_validation}"
            )
    output, _ = run_experiment(
        args,
        fit_batches,
        validation_batches,
        test_loader,
        device,
    )
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
