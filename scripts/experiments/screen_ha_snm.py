"""Validation-only screen for one global HA-SNM threshold schedule."""

import argparse
import copy
import json
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

from calibration import bias_corr_model
from models import SignedIF
from preprocess import datapool
from scripts.experiments.run_full_ftbc_asnm_ablation import (
    build_full_model,
    configure_snn,
    evaluate_validation,
    snapshot_full_ftbc,
)
from scripts.experiments.run_ha_snm_ablation import (
    FAMILIES,
    FORMAL_BATCH_HASHES,
    build_parser as build_ablation_parser,
    build_variant,
    resolve_args,
    validate_ha_args,
)
from scripts.experiments.run_state_ftbc_ablation import materialize_calibration_batches
from scripts.experiments.run_temporal_lr_gated_snm import batches_sha256, synchronize
from scripts.experiments.qcfs_checkpoint import load_qcfs_pair
from spike_stats import set_signed_spike_stats_enabled
from utils import seed_all


DEFAULT_CANDIDATES = (
    (1.0, 1.0),
    (1.5, 1.0),
    (2.0, 1.0),
    (1.25, 0.75),
    (1.5, 0.75),
    (2.0, 0.75),
    (1.25, 0.5),
    (1.5, 0.5),
    (2.0, 0.5),
)


def parse_candidate(value):
    try:
        start, end = (float(item) for item in value.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("candidate must use START:END") from error
    if start <= 0 or end <= 0 or start < end:
        raise argparse.ArgumentTypeError("candidate requires START>=END>0")
    return start, end


def candidate_key(candidate):
    return f"{candidate[0]:g}:{candidate[1]:g}"


def write_report(path, payload):
    times = payload["protocol"]["time_steps"]
    lines = [
        "# HA-SNM Validation-Only Schedule Screen",
        "",
        "Status: complete",
        "",
        f"- Dataset/architecture: {payload['protocol']['dataset']}/{payload['protocol']['architecture']}",
        f"- QCFS L: {payload['protocol']['qcfs_L']}",
        f"- Checkpoint SHA256: `{payload['protocol']['checkpoint']['sha256']}`",
        f"- Fit/validation hashes: `{payload['protocol']['fit_sha256']}` / `{payload['protocol']['validation_sha256']}`",
        "- This screen never evaluates test images.",
        "- Accuracy is the primary objective; ties use lower logit MSE, then fewer SOPs.",
        "",
        "## Macro validation ranking",
        "",
        "| Rank | Start:end | Mean accuracy | Mean logit MSE | Negative-spike ratio vs standard | Mean SOP ratio vs standard |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(payload["ranking"], 1):
        lines.append(
            f"| {rank} | {item['candidate']} | {item['mean_accuracy']:.4f}% | "
            f"{item['mean_logit_mse']:.8f} | {item['mean_negative_ratio']:.6f} | "
            f"{item['mean_sop_ratio']:.6f} |"
        )
    lines.extend(["", "## Validation accuracy detail", ""])
    for family in FAMILIES:
        lines.extend(
            [
                f"### {family}",
                "",
                "| Start:end | " + " | ".join(f"T={t}" for t in times) + " | Mean |",
                "|---|" + "---:|" * (len(times) + 1),
            ]
        )
        for candidate in payload["candidates"]:
            values = [payload["results"][candidate][family][str(t)]["acc"] for t in times]
            lines.append(
                f"| {candidate} | "
                + " | ".join(f"{value:.2f}%" for value in values)
                + f" | {sum(values) / len(values):.4f}% |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_screen(args, fit_batches, validation_batches, device):
    output = Path(args.output)
    progress = output.with_suffix(".json")
    if output.exists() or progress.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {progress}")
    ann_template, snn_template, checkpoint = load_qcfs_pair(
        args.checkpoint, args.dataset, args.architecture, device
    )
    if checkpoint["sha256"].lower() != args.checkpoint_sha256.lower():
        raise RuntimeError("Checkpoint SHA256 mismatch")
    ann_template.set_T(0)
    if hasattr(ann_template, "set_L"):
        ann_template.set_L(args.L)
    if hasattr(snn_template, "set_L"):
        snn_template.set_L(args.L)
    if args.architecture == "resnet20":
        ann_template.set_qcfs_training_profile(args.resnet20_eval_profile)
        snn_template.set_qcfs_training_profile(args.resnet20_eval_profile)
    set_signed_spike_stats_enabled(snn_template, SignedIF, False)

    candidates = tuple(args.candidates)
    payload = {
        "status": "running",
        "protocol": {
            "dataset": args.dataset,
            "architecture": args.architecture,
            "qcfs_L": args.L,
            "checkpoint": checkpoint,
            "time_steps": list(args.time_steps),
            "fit_batches": args.fit_batches,
            "validation_batches": args.validation_batches,
            "batch_size": args.batch_size,
            "fit_sha256": batches_sha256(fit_batches),
            "validation_sha256": batches_sha256(validation_batches),
            "test_images_evaluated": 0,
        },
        "candidates": [candidate_key(x) for x in candidates],
        "results": OrderedDict(
            (
                candidate_key(candidate),
                OrderedDict((family, OrderedDict()) for family in FAMILIES),
            )
            for candidate in candidates
        ),
        "ranking": [],
    }

    for time_steps in args.time_steps:
        print(f"T={time_steps}: fit Full-FTBC", flush=True)
        teacher = copy.deepcopy(snn_template).to(device)
        configure_snn(teacher, time_steps, signed=False, ftbc_mode="full")
        teacher.set_snm_mode("standard")
        ann = copy.deepcopy(ann_template).to(device)
        ann.set_T(0)
        bias_corr_model(
            ann=ann,
            snn=teacher,
            T=time_steps,
            train_loader=fit_batches,
            curr_t_alpha=args.alpha,
            num_cali_sample_batches=args.fit_batches,
            ftbc_mode="full",
        )
        schedule = snapshot_full_ftbc(teacher, time_steps)
        for family in FAMILIES:
            for candidate in candidates:
                args.ha_start, args.ha_end = candidate
                model, _ = build_variant(
                    snn_template,
                    schedule,
                    family,
                    "standard" if candidate == (1.0, 1.0) else "ha",
                    time_steps,
                    args,
                    device,
                )
                metrics = evaluate_validation(
                    model,
                    ann,
                    validation_batches,
                    device,
                    time_steps,
                    args.architecture,
                )
                payload["results"][candidate_key(candidate)][family][str(time_steps)] = metrics
                del model
        del ann, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    standard = payload["results"]["1:1"]
    cells = len(FAMILIES) * len(args.time_steps)
    for candidate in payload["candidates"]:
        accuracy = 0.0
        mse = 0.0
        negative_ratio = 0.0
        sop_ratio = 0.0
        for family in FAMILIES:
            for time_steps in args.time_steps:
                key = str(time_steps)
                item = payload["results"][candidate][family][key]
                baseline = standard[family][key]
                accuracy += item["acc"]
                mse += item["logit_mse"]
                negative_ratio += item["negative_spikes"] / max(
                    baseline["negative_spikes"], 1
                )
                sop_ratio += item["sops"] / max(baseline["sops"], 1)
        payload["ranking"].append(
            {
                "candidate": candidate,
                "mean_accuracy": accuracy / cells,
                "mean_logit_mse": mse / cells,
                "mean_negative_ratio": negative_ratio / cells,
                "mean_sop_ratio": sop_ratio / cells,
            }
        )
    payload["ranking"].sort(
        key=lambda item: (-item["mean_accuracy"], item["mean_logit_mse"], item["mean_sop_ratio"])
    )
    payload["status"] = "complete"
    progress.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(output, payload)
    return output


def build_parser():
    parser = build_ablation_parser()
    parser.description = "Validation-only HA-SNM global threshold screen"
    parser.set_defaults(time_steps=[2, 4, 8, 16, 32], test_batches=1)
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=parse_candidate,
        default=DEFAULT_CANDIDATES,
    )
    return parser


def main(cli_args=None):
    args = resolve_args(build_parser().parse_args(cli_args))
    if args.output is None or "archive" not in {part.lower() for part in args.output.parts}:
        raise ValueError("Schedule screens must write under docs/archive")
    validate_ha_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    train_loader, _ = datapool(args.dataset, args.batch_size)
    materialized = materialize_calibration_batches(
        train_loader, args.fit_batches + args.validation_batches
    )
    fit_batches = materialized[: args.fit_batches]
    validation_batches = materialized[args.fit_batches :]
    output = run_screen(args, fit_batches, validation_batches, device)
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
