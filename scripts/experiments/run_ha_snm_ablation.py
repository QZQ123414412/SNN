"""Four-model Full/Temporal-LR/PA-FTBC ablation for HA-SNM.

HA-SNM (Horizon-Annealed SNM) keeps the original one-spike credit rule while
annealing only the *decision* threshold.  Early cancellation requires stronger
evidence; the threshold is relaxed near the horizon to correct residual
over-firing after more temporal evidence has arrived.
"""

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

from a_snm import DEFAULT_TIME_STEPS
from calibration import bias_corr_model
from models import SignedIF
from parity_anchor_ftbc import named_signed_layers
from preprocess import datapool
from scripts.experiments.qcfs_checkpoint import load_qcfs_pair
from scripts.experiments.run_full_ftbc_asnm_ablation import (
    CIFAR10_RESNET20_L4_PROTOCOL,
    DATASET_PROTOCOLS,
    build_full_model,
    configure_snn,
    evaluate_test,
    snapshot_full_ftbc,
)
from scripts.experiments.run_pa_ftbc_asnm_ablation import (
    EXPECTED_ANN_ACCURACY,
    FORMAL_BATCH_HASHES,
    build_pa_model,
    build_temporal_model,
    is_formal_protocol,
    protocol_key,
    validate_args,
)
from scripts.experiments.run_state_ftbc_ablation import (
    materialize_calibration_batches,
)
from scripts.experiments.run_temporal_lr_asnm_ablation import (
    TEST_EQUIVALENCE_KEYS,
    exact_metrics,
)
from scripts.experiments.run_temporal_lr_gated_snm import (
    batches_sha256,
    synchronize,
)
from spike_stats import set_signed_spike_stats_enabled, summarize_ftbc_storage
from utils import seed_all, val


FAMILIES = ("full", "temporal", "pa")
MODES = ("off", "standard", "ha")
FAMILY_LABELS = {
    "full": "Full-FTBC",
    "temporal": "Temporal-LR FTBC",
    "pa": "PA-FTBC",
}
MODE_LABELS = {
    "off": "SNM-off",
    "standard": "standard SNM",
    "ha": "HA-SNM",
}
CONFIGS = OrderedDict(
    (
        f"{letter}_QCFS_{family.upper()}_FTBC_{mode.upper()}_R0",
        {"family": family, "mode": mode},
    )
    for letter, (family, mode) in zip(
        "ABCDEFGHI",
        ((family, mode) for family in FAMILIES for mode in MODES),
    )
)
CONFIG_BY_FAMILY_MODE = {
    (item["family"], item["mode"]): name for name, item in CONFIGS.items()
}

DEFAULT_OUTPUTS = {
    ("cifar10", "resnet20", 4): Path(
        "docs/results/comparative_ablation/cifar10/"
        "HA_SNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.md"
    ),
    ("cifar10", "vgg16", 8): Path(
        "docs/results/comparative_ablation/cifar10/HA_SNM_CIFAR10_VGG16_L8.md"
    ),
    ("cifar100", "resnet20", 8): Path(
        "docs/results/comparative_ablation/cifar100/HA_SNM_CIFAR100_RESNET20_L8.md"
    ),
    ("cifar100", "vgg16", 8): Path(
        "docs/results/comparative_ablation/cifar100/HA_SNM_CIFAR100_VGG16_L8.md"
    ),
}


def resolve_args(args):
    key = protocol_key(args)
    if key not in DEFAULT_OUTPUTS:
        raise ValueError(f"Unsupported formal protocol: {key}")
    base = DATASET_PROTOCOLS[args.dataset]
    if key == ("cifar10", "resnet20", 4):
        checkpoint = CIFAR10_RESNET20_L4_PROTOCOL["checkpoint"]
        sha256 = CIFAR10_RESNET20_L4_PROTOCOL["expected_sha256"]
        profile = CIFAR10_RESNET20_L4_PROTOCOL["evaluation_profile"]
    else:
        checkpoint = base["default_checkpoints"][args.architecture]
        sha256 = base["expected_sha256"][args.architecture]
        profile = (
            base["resnet20_eval_profile"]
            if args.architecture == "resnet20"
            else "not-applicable"
        )
    args.checkpoint = args.checkpoint or checkpoint
    args.checkpoint_sha256 = args.checkpoint_sha256 or sha256
    args.resnet20_eval_profile = args.resnet20_eval_profile or profile
    args.output = args.output or DEFAULT_OUTPUTS[key]
    return args


def validate_ha_args(args):
    validate_args(args)
    if args.ha_start <= 0 or args.ha_end <= 0:
        raise ValueError("HA-SNM threshold multipliers must be positive")
    if args.ha_start < args.ha_end:
        raise ValueError("HA-SNM start must not be below end")
    if args.ha_reference <= 0:
        raise ValueError("HA-SNM reference horizon must be positive")


def save_progress(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def configure_snm(model, mode, start, end, reference):
    if mode == "off":
        model.set_signed(False)
        model.set_snm_mode("standard")
    elif mode == "standard":
        model.set_signed(True)
        model.set_snm_mode("standard")
    elif mode == "ha":
        model.set_signed(True)
        model.set_snm_mode(
            "horizon_annealed", start=start, end=end, reference=reference
        )
    else:
        raise ValueError(mode)
    model.set_snm_negative_margin(0.0)
    return model


def build_variant(template, schedule, family, mode, time_steps, args, device):
    signed = mode != "off"
    if family == "full":
        model = build_full_model(template, schedule, time_steps, signed, device)
        compression = None
    elif family == "temporal":
        model, compression = build_temporal_model(
            template,
            schedule,
            time_steps,
            signed,
            device,
            args.architecture,
            rank=args.temporal_rank,
        )
    elif family == "pa":
        model, compression = build_pa_model(
            template,
            schedule,
            time_steps,
            signed,
            device,
        )
    else:
        raise ValueError(family)
    configure_snm(
        model, mode, args.ha_start, args.ha_end, args.ha_reference
    )
    return model, compression


def mean_metric(payload, name, key):
    times = payload["protocol"]["time_steps"]
    return sum(float(payload["results"][name][str(t)][key]) for t in times) / len(times)


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


def write_report(path, payload):
    p = payload["protocol"]
    times = p["time_steps"]
    lines = [
        "# QCFS + Full/Temporal-LR/PA-FTBC + HA-SNM Ablation",
        "",
        f"Status: {payload['status']}",
        "",
        f"- Dataset/architecture: {p['dataset_label']}/{p['architecture']}",
        f"- QCFS L: {p['qcfs_L']}",
        f"- ANN accuracy: {p['ann_accuracy']:.2f}%",
        f"- Checkpoint: `{p['checkpoint']['filename']}`",
        f"- Checkpoint SHA256: `{p['checkpoint']['sha256']}`",
        f"- Fit/validation SHA256: `{p['fit_sha256']}` / `{p['validation_sha256']}`",
        f"- Test samples: {p['test_samples']:,}",
        f"- Evaluation profile: `{p['resnet20_eval_profile']}`",
        f"- HA-SNM threshold schedule: start={p['ha_snm']['start']}, end={p['ha_snm']['end']}, reference horizon={p['ha_snm']['reference']}, linear.",
        "- HA-SNM keeps the original transmitted-credit/R0 rule and changes only the negative-spike decision threshold.",
        "- It uses the original -theta event amplitude, adds no dense neuron state, and has two global FP32 deployment constants plus one fixed reference horizon (12 bytes if stored).",
        "- Full-FTBC is fitted independently at every T with SNM off; Temporal-LR and PA are compressed from that same teacher.",
        "- Temporal-LR and PA fall back exactly to Full-FTBC at T<=4.",
        f"- Checkpoint note: {p['checkpoint_note']}.",
        "",
        "## Primary accuracy",
        "",
        "| Config | " + " | ".join(f"T={t}" for t in times) + " | Mean |",
        "|---|" + "---:|" * (len(times) + 1),
    ]
    for name in CONFIGS:
        cells = [f"{payload['results'][name][str(t)]['acc']:.2f}%" for t in times]
        lines.append(
            f"| {name} | " + " | ".join(cells) + f" | {mean_metric(payload, name, 'acc'):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## HA-SNM accuracy gain",
            "",
            "| Family | " + " | ".join(f"T={t}" for t in times) + " | Mean |",
            "|---|" + "---:|" * (len(times) + 1),
        ]
    )
    for family in FAMILIES:
        ha = CONFIG_BY_FAMILY_MODE[(family, "ha")]
        standard = CONFIG_BY_FAMILY_MODE[(family, "standard")]
        deltas = [
            payload["results"][ha][str(t)]["acc"]
            - payload["results"][standard][str(t)]["acc"]
            for t in times
        ]
        lines.append(
            f"| {FAMILY_LABELS[family]}: HA - standard | "
            + " | ".join(f"{x:+.2f}pp" for x in deltas)
            + f" | {sum(deltas) / len(deltas):+.3f}pp |"
        )
        off = CONFIG_BY_FAMILY_MODE[(family, "off")]
        off_deltas = [
            payload["results"][ha][str(t)]["acc"]
            - payload["results"][off][str(t)]["acc"]
            for t in times
        ]
        lines.append(
            f"| {FAMILY_LABELS[family]}: HA - off | "
            + " | ".join(f"{x:+.2f}pp" for x in off_deltas)
            + f" | {sum(off_deltas) / len(off_deltas):+.3f}pp |"
        )
    lines.append("")

    metric_table(lines, "ANN-SNN logit MSE", payload, "logit_mse", lambda x: f"{x:.8f}")
    metric_table(lines, "Positive spike rate", payload, "positive_rate", lambda x: f"{100*x:.6f}%")
    metric_table(lines, "Negative spike rate", payload, "negative_rate", lambda x: f"{100*x:.6f}%")
    metric_table(lines, "Overall spike sparsity", payload, "sparsity", lambda x: f"{100*x:.6f}%")
    metric_table(lines, "Input-driven SOPs", payload, "sops", lambda x: f"{int(x):,}")
    metric_table(lines, "FTBC parameters", payload, "ftbc_parameters", lambda x: f"{int(x):,}")
    metric_table(lines, "FTBC storage bytes", payload, "ftbc_bytes", lambda x: f"{int(x):,}")
    metric_table(lines, "Bias synthesis MACs", payload, "ftbc_synthesis_macs", lambda x: f"{int(x):,}")
    metric_table(lines, "Inference elapsed", payload, "inference_elapsed", lambda x: f"{x:.6f}")

    lines.extend(
        [
            "## HA-SNM overhead",
            "",
            "| Item | Value |",
            "|---|---:|",
            "| Additional dense per-neuron state | 0 bytes |",
            "| Global constants | 3 (12 bytes if all stored as FP32) |",
            f"| SignedIF layers | {p['signed_if_layers']} |",
            "| Per layer/time decision overhead | one scalar threshold interpolation and the existing comparison |",
            "",
            "## Exact fallback checks",
            "",
            "| T | Mode | Full=Temporal | Full=PA |",
            "|---:|---|---|---|",
        ]
    )
    checks = {(x["time_steps"], x["mode"]): x for x in payload["equivalence_checks"]}
    for t in times:
        if t > 4:
            continue
        for mode in MODES:
            item = checks[(t, mode)]
            lines.append(
                f"| {t} | {MODE_LABELS[mode]} | "
                f"{'yes' if item['full_temporal_exact'] else 'no'} | "
                f"{'yes' if item['full_pa_exact'] else 'no'} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args, fit_batches, validation_batches, test_loader, device):
    output = Path(args.output)
    progress = output.with_suffix(".progress.json")
    if not args.overwrite and (output.exists() or progress.exists()):
        raise FileExistsError(f"Refusing to overwrite {output} or {progress}")

    ann_template, snn_template, checkpoint = load_qcfs_pair(
        args.checkpoint, args.dataset, args.architecture, device
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
    if args.architecture == "resnet20":
        ann_template.set_qcfs_training_profile(args.resnet20_eval_profile)
        snn_template.set_qcfs_training_profile(args.resnet20_eval_profile)
    set_signed_spike_stats_enabled(snn_template, SignedIF, False)

    checkpoint_note = (
        "paper-aligned retrained checkpoint selected by peak test accuracy; not a strict paper reproduction"
        if protocol_key(args) == ("cifar10", "resnet20", 4)
        else "existing frozen repository checkpoint and evaluation protocol"
    )
    payload = {
        "status": "calibrating",
        "protocol": {
            "dataset": args.dataset,
            "dataset_label": DATASET_PROTOCOLS[args.dataset]["label"],
            "architecture": args.architecture,
            "qcfs_L": args.L,
            "resnet20_eval_profile": args.resnet20_eval_profile,
            "checkpoint": checkpoint,
            "checkpoint_note": checkpoint_note,
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
            "ha_snm": {
                "start": args.ha_start,
                "end": args.ha_end,
                "reference": args.ha_reference,
                "schedule": "linear with inverse-horizon normalization",
            },
            "test_batches": args.test_batches,
            "test_samples": None,
            "signed_if_layers": len(named_signed_layers(snn_template)),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": "float32",
        },
        "calibration": OrderedDict(),
        "compression": {"temporal": OrderedDict(), "pa": OrderedDict()},
        "results": OrderedDict((name, OrderedDict()) for name in CONFIGS),
        "layers": OrderedDict((name, OrderedDict()) for name in CONFIGS),
        "equivalence_checks": [],
    }
    schedules = {}

    for time_steps in args.time_steps:
        key = str(time_steps)
        print(f"[{args.dataset}/{args.architecture}] T={time_steps}: fit Full-FTBC", flush=True)
        teacher = copy.deepcopy(snn_template).to(device)
        configure_snn(teacher, time_steps, signed=False, ftbc_mode="full")
        teacher.set_snm_mode("standard")
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
        elapsed = time.perf_counter() - started
        schedules[time_steps] = snapshot_full_ftbc(teacher, time_steps)
        storage = summarize_ftbc_storage(teacher, SignedIF)
        payload["calibration"][key] = {
            "elapsed": elapsed,
            "ftbc_parameters": int(storage["parameters"]),
            "ftbc_bytes": int(storage["bytes"]),
        }
        for family in ("temporal", "pa"):
            model, report = build_variant(
                snn_template, schedules[time_steps], family, "off", time_steps, args, device
            )
            report["full_parameters"] = int(storage["parameters"])
            report["full_bytes"] = int(storage["bytes"])
            report["storage_ratio"] = report["ftbc_parameters"] / max(int(storage["parameters"]), 1)
            report["storage_reduction"] = 1.0 - report["storage_ratio"]
            payload["compression"][family][key] = report
            del model
        del ann, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_progress(progress, payload)

    evaluation_loader = (
        materialize_calibration_batches(test_loader, args.test_batches)
        if args.test_batches > 0
        else test_loader
    )
    payload["protocol"]["ann_accuracy"] = val(ann_template, evaluation_loader, device, 0)
    payload["protocol"]["test_samples"] = (
        sum(int(targets.numel()) for _, targets in evaluation_loader)
        if args.test_batches > 0
        else len(evaluation_loader.dataset)
    )
    if args.test_batches == 0 and payload["protocol"]["test_samples"] != 10000:
        raise RuntimeError("Formal CIFAR test set must contain 10,000 images")
    expected_ann = EXPECTED_ANN_ACCURACY[protocol_key(args)]
    if is_formal_protocol(args) and not math.isclose(
        payload["protocol"]["ann_accuracy"], expected_ann, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"Unexpected ANN accuracy {payload['protocol']['ann_accuracy']} != {expected_ann}"
        )

    payload["status"] = "testing"
    save_progress(progress, payload)
    for time_steps in args.time_steps:
        key = str(time_steps)
        time_results = {}
        for family in FAMILIES:
            for mode in MODES:
                name = CONFIG_BY_FAMILY_MODE[(family, mode)]
                print(
                    f"[{args.dataset}/{args.architecture}] T={time_steps}: {family}/{mode}",
                    flush=True,
                )
                model, _ = build_variant(
                    snn_template,
                    schedules[time_steps],
                    family,
                    mode,
                    time_steps,
                    args,
                    device,
                )
                result, layers = evaluate_test(
                    model,
                    ann_template,
                    evaluation_loader,
                    device,
                    time_steps,
                    args.architecture,
                )
                storage = summarize_ftbc_storage(model, SignedIF)
                result.update(
                    {
                        "family": family,
                        "snm_mode": mode,
                        "snm_enabled": mode != "off",
                        "ha_start": args.ha_start if mode == "ha" else None,
                        "ha_end": args.ha_end if mode == "ha" else None,
                        "ftbc_synthesis_macs": int(storage["synthesis_macs"]),
                        "calibration_elapsed": payload["calibration"][key]["elapsed"],
                        "compression_elapsed": (
                            payload["compression"][family][key]["compression_elapsed"]
                            if family in ("temporal", "pa")
                            else 0.0
                        ),
                        "effective_ftbc_modes": sorted(
                            {module.ftbc_mode for module in named_signed_layers(model).values()}
                        ),
                    }
                )
                payload["results"][name][key] = result
                payload["layers"][name][key] = layers
                time_results[(family, mode)] = result
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                save_progress(progress, payload)

        if time_steps <= 4:
            for mode in MODES:
                full = time_results[("full", mode)]
                temporal = time_results[("temporal", mode)]
                pa = time_results[("pa", mode)]
                check = {
                    "time_steps": time_steps,
                    "mode": mode,
                    "full_temporal_exact": exact_metrics(
                        full, temporal, TEST_EQUIVALENCE_KEYS
                    ),
                    "full_pa_exact": exact_metrics(full, pa, TEST_EQUIVALENCE_KEYS),
                }
                payload["equivalence_checks"].append(check)
                if not check["full_temporal_exact"] or not check["full_pa_exact"]:
                    raise RuntimeError(f"T={time_steps} {mode} compressed fallback mismatch")

    payload["status"] = "complete"
    save_progress(progress, payload)
    write_report(output, payload)
    return output, payload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full/Temporal-LR/PA-FTBC standard-SNM versus HA-SNM ablation"
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
    parser.add_argument("--ha_start", type=float, default=1.25)
    parser.add_argument("--ha_end", type=float, default=0.5)
    parser.add_argument("--ha_reference", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(cli_args=None):
    args = resolve_args(build_parser().parse_args(cli_args))
    validate_ha_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.test_batches == 0:
        raise RuntimeError("Formal experiment requires CUDA")
    seed_all(args.seed)
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    materialized = materialize_calibration_batches(
        train_loader, args.fit_batches + args.validation_batches
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
        args, fit_batches, validation_batches, test_loader, device
    )
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
