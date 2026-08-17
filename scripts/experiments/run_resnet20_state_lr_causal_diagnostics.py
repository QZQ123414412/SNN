"""Causal diagnostics for the ResNet20 SNM x state-low-rank failure."""

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
import torch.nn as nn

from calibration import bias_corr_model
from models import SignedIF
from preprocess import datapool
from scripts.experiments.qcfs_checkpoint import load_qcfs_pair
from scripts.experiments.run_resnet20_qcfs_ablation import (
    FORMAL_ALPHA,
    FORMAL_BATCH_SIZE,
    FORMAL_CALI_BATCHES,
    FORMAL_CHECKPOINT_SHA256,
    FORMAL_COEFFICIENT_CLIP,
    FORMAL_MAX_T32_CONVERSION_GAP,
    FORMAL_MIN_ANN_ACCURACY,
    FORMAL_OVER_WEIGHT,
    FORMAL_QCFS_L,
    FORMAL_QCFS_TRAINING_PROFILE,
    FORMAL_SEED,
    FORMAL_UNDER_WEIGHT,
    calibration_batches_sha256,
    configure_snn,
    format_pct,
    load_progress,
    progress_path_for,
    save_progress,
    synchronize,
    validate_t32_conversion,
)
from scripts.experiments.run_state_ftbc_ablation import (
    materialize_calibration_batches,
)
from scripts.experiments.run_stats_ablation import summarize_layer_stats
from spike_stats import (
    collect_resnet20_spike_stats,
    reset_signed_spike_stats,
    set_signed_spike_stats_enabled,
    summarize_ftbc_storage,
)
from utils import seed_all, val


TIME_STEPS = (4, 8, 16, 32)
DEFAULT_CHECKPOINT = (
    "cifar100-checkpoints/"
    "resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth"
)
DEFAULT_OUTPUT = (
    "docs/results/comparative_ablation/cifar100/"
    "RESNET20_STATE_LR_CAUSAL_DIAGNOSTICS.md"
)

VARIANTS = OrderedDict(
    [
        (
            "E_REFERENCE_STATE_LR",
            dict(
                calibration_signed=False,
                inference_signed=False,
                state_bias=True,
                global_clip=False,
            ),
        ),
        (
            "F_REFERENCE_SNM_STATE_LR",
            dict(
                calibration_signed=True,
                inference_signed=True,
                state_bias=True,
                global_clip=False,
            ),
        ),
        (
            "G_E_COEFFICIENTS_SNM_ON",
            dict(
                calibration_signed=False,
                inference_signed=True,
                state_bias=True,
                global_clip=False,
            ),
        ),
        (
            "H_F_BIAS_STATE_OFF",
            dict(
                calibration_signed=True,
                inference_signed=True,
                state_bias=False,
                global_clip=False,
            ),
        ),
        (
            "I_F_FINAL_GLOBAL_CLIP",
            dict(
                calibration_signed=True,
                inference_signed=True,
                state_bias=True,
                global_clip=True,
            ),
        ),
    ]
)


def clamp_final_state_lr_coefficients(model, fraction=0.25):
    """Clamp accumulated low-rank coefficients after all calibration batches."""
    changed = 0
    total = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, SignedIF):
                continue
            limit = float(fraction) * float(module.thresh.detach().abs().item())
            for tensor in (module.bias_base, module.bias_slope, module.bias_state):
                if tensor is None:
                    continue
                changed += int((tensor.abs() > limit).sum().item())
                total += tensor.numel()
                tensor.clamp_(min=-limit, max=limit)
    return {"changed": changed, "total": total}


def coefficient_summary(model, fraction=0.25):
    """Summarize active state-low-rank coefficients relative to thresholds."""
    ratios = []
    for module in model.modules():
        if not isinstance(module, SignedIF):
            continue
        tensors = [module.bias_base, module.bias_slope]
        if module.enable_state_bias:
            tensors.append(module.bias_state)
        threshold = float(module.thresh.detach().abs().item())
        for tensor in tensors:
            if tensor is not None:
                ratios.append(tensor.detach().abs().reshape(-1).cpu() / threshold)
    if not ratios:
        return {
            "max_coefficient_ratio": 0.0,
            "fraction_over_global_limit": 0.0,
        }
    ratios = torch.cat(ratios)
    return {
        "max_coefficient_ratio": float(ratios.max().item()),
        "fraction_over_global_limit": float(
            (ratios > float(fraction)).float().mean().item()
        ),
    }


def calibrate_state_lr(
    snn_template,
    ann_template,
    time_steps,
    signed,
    calibration_batches,
    device,
):
    model = copy.deepcopy(snn_template).to(device)
    configure_snn(
        model,
        {"signed": bool(signed), "ftbc_mode": "state_low_rank"},
        time_steps,
    )
    set_signed_spike_stats_enabled(model, SignedIF, False)
    ann = copy.deepcopy(ann_template).to(device)
    ann.set_T(0)
    ann.set_L(FORMAL_QCFS_L)
    synchronize(device)
    start = time.perf_counter()
    bias_corr_model(
        ann=ann,
        snn=model,
        T=time_steps,
        train_loader=calibration_batches,
        curr_t_alpha=FORMAL_ALPHA,
        num_cali_sample_batches=FORMAL_CALI_BATCHES,
        ftbc_mode="state_low_rank",
        ridge=1e-3,
        over_weight=FORMAL_OVER_WEIGHT,
        under_weight=FORMAL_UNDER_WEIGHT,
        coefficient_clip=FORMAL_COEFFICIENT_CLIP,
    )
    synchronize(device)
    elapsed = time.perf_counter() - start
    del ann
    return model, elapsed


def build_variants(unsigned_model, signed_model):
    """Create the five causal variants from two shared calibrations."""
    variants = OrderedDict()
    variants["E_REFERENCE_STATE_LR"] = copy.deepcopy(unsigned_model)
    variants["F_REFERENCE_SNM_STATE_LR"] = copy.deepcopy(signed_model)

    e_coefficients_snm = copy.deepcopy(unsigned_model)
    e_coefficients_snm.set_signed(True)
    variants["G_E_COEFFICIENTS_SNM_ON"] = e_coefficients_snm

    no_state = copy.deepcopy(signed_model)
    no_state.set_state_bias_enabled(False)
    variants["H_F_BIAS_STATE_OFF"] = no_state

    final_clip = copy.deepcopy(signed_model)
    clip_stats = clamp_final_state_lr_coefficients(
        final_clip, FORMAL_COEFFICIENT_CLIP
    )
    variants["I_F_FINAL_GLOBAL_CLIP"] = final_clip
    return variants, clip_stats


def evaluate_precalibrated(
    model_template,
    test_loader,
    device,
    time_steps,
    calibration_elapsed,
    clip_changed=0,
):
    model = copy.deepcopy(model_template).to(device)
    storage = summarize_ftbc_storage(model, SignedIF)
    coefficients = coefficient_summary(model, FORMAL_COEFFICIENT_CLIP)

    set_signed_spike_stats_enabled(model, SignedIF, True)
    reset_signed_spike_stats(model, SignedIF)
    accuracy = val(model, test_loader, device, time_steps)
    layer_stats = collect_resnet20_spike_stats(model, SignedIF, nn.Conv2d)
    summary = summarize_layer_stats(layer_stats)

    set_signed_spike_stats_enabled(model, SignedIF, False)
    synchronize(device)
    start = time.perf_counter()
    timed_accuracy = val(model, test_loader, device, time_steps)
    synchronize(device)
    inference_elapsed = time.perf_counter() - start
    if timed_accuracy != accuracy:
        raise RuntimeError(
            f"Non-deterministic evaluation: {accuracy} vs {timed_accuracy}"
        )

    summary.update(
        {
            "acc": accuracy,
            "effective_ftbc_mode": "state_low_rank",
            "ftbc_parameters": storage["parameters"],
            "ftbc_bytes": storage["bytes"],
            "calibration_elapsed": calibration_elapsed,
            "inference_elapsed": inference_elapsed,
            "max_coefficient_ratio": coefficients["max_coefficient_ratio"],
            "fraction_over_global_limit": coefficients[
                "fraction_over_global_limit"
            ],
            "final_clip_changed": int(clip_changed),
        }
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, layer_stats


def diagnostic_signature(checkpoint_metadata, calibration_sha256):
    return {
        "checkpoint_sha256": checkpoint_metadata["sha256"],
        "configs": list(VARIANTS),
        "configuration_matrix": {
            name: dict(config) for name, config in VARIANTS.items()
        },
        "time_steps": list(TIME_STEPS),
        "batch_size": FORMAL_BATCH_SIZE,
        "cali_batches": FORMAL_CALI_BATCHES,
        "calibration_sha256": calibration_sha256,
        "seed": FORMAL_SEED,
        "qcfs_L": FORMAL_QCFS_L,
        "alpha": FORMAL_ALPHA,
        "ridge": 1e-3,
        "coefficient_clip": FORMAL_COEFFICIENT_CLIP,
        "over_weight": FORMAL_OVER_WEIGHT,
        "under_weight": FORMAL_UNDER_WEIGHT,
    }


def _complete(results):
    return all(t in results[name] for name in VARIANTS for t in TIME_STEPS)


def write_report(
    path,
    results,
    layer_results,
    checkpoint_metadata,
    ann_accuracy,
    t32_accuracy,
    t32_gap,
    calibration_sha256,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "COMPLETE" if _complete(results) else "INCOMPLETE"
    with path.open("w", encoding="utf-8") as report:
        report.write("# ResNet20 State-LR x SNM Causal Diagnostics\n\n")
        report.write(f"- Status: **{status}**\n")
        report.write("- Dataset / architecture: CIFAR-100 / ResNet20\n")
        report.write("- CSRR: disabled in every configuration\n")
        report.write(f"- QCFS L: {FORMAL_QCFS_L}\n")
        report.write(f"- Training-log checkpoint accuracy: 68.78%\n")
        report.write(f"- Re-evaluated ANN accuracy: {ann_accuracy:.2f}%\n")
        report.write(
            f"- Pre-run A_QCFS_R0 T=32: {t32_accuracy:.2f}% "
            f"(gap={t32_gap:.2f}pp)\n"
        )
        report.write(f"- Checkpoint: {checkpoint_metadata['filename']}\n")
        report.write(f"- Checkpoint SHA256: `{checkpoint_metadata['sha256']}`\n")
        report.write(f"- Calibration data SHA256: `{calibration_sha256}`\n")
        report.write(
            f"- Calibration: {FORMAL_CALI_BATCHES} x {FORMAL_BATCH_SIZE}, "
            f"alpha={FORMAL_ALPHA}, ridge=0.001, "
            f"per-update clip={FORMAL_COEFFICIENT_CLIP}, "
            f"w_under={FORMAL_UNDER_WEIGHT}, "
            f"w_over={FORMAL_OVER_WEIGHT}\n"
        )
        report.write(f"- Time steps: {list(TIME_STEPS)}\n\n")

        report.write("## Causal Switch Matrix\n\n")
        report.write(
            "| Variant | Calibration SNM | Inference SNM | State term | "
            "Post-calibration global clip |\n"
        )
        report.write("|---|---|---|---|---|\n")
        for name, config in VARIANTS.items():
            report.write(
                f"| {name} | {config['calibration_signed']} | "
                f"{config['inference_signed']} | {config['state_bias']} | "
                f"{config['global_clip']} |\n"
            )
        report.write(
            "\nE/G share one unsigned calibration for each T. F/H/I share one "
            "signed calibration for each T. H disables the jointly fitted "
            "state term without refitting base/slope.\n\n"
        )

        metrics = [
            ("Accuracy", "acc", lambda value: f"{value:.2f}%"),
            ("Input-driven SOPs", "sops", lambda value: f"{int(value):,}"),
            ("Time-scale Operations", "scale_operations", lambda value: f"{int(value):,}"),
            ("Positive Spike Rate", "positive_rate", format_pct),
            ("Negative Spike Rate", "negative_rate", format_pct),
            ("Overall Spike Sparsity", "sparsity", format_pct),
            ("FTBC Parameters", "ftbc_parameters", lambda value: f"{int(value):,}"),
            ("FTBC Storage Bytes", "ftbc_bytes", lambda value: f"{int(value):,}"),
            ("Calibration Time", "calibration_elapsed", lambda value: f"{value:.1f}s"),
            ("Inference Time", "inference_elapsed", lambda value: f"{value:.1f}s"),
            ("Max |Coefficient| / Threshold", "max_coefficient_ratio", lambda value: f"{value:.4f}"),
            ("Coefficient Fraction > 0.25", "fraction_over_global_limit", format_pct),
            ("Coefficients Changed by Final Clip", "final_clip_changed", lambda value: f"{int(value):,}"),
        ]
        for title, key, formatter in metrics:
            report.write(f"## {title}\n\n")
            report.write(
                "| Variant | "
                + " | ".join(f"T={t}" for t in TIME_STEPS)
                + " |\n"
            )
            report.write("|" + "---|" * (len(TIME_STEPS) + 1) + "\n")
            for name in VARIANTS:
                cells = [
                    formatter(results[name][t][key]) if t in results[name] else "-"
                    for t in TIME_STEPS
                ]
                report.write(f"| {name} | " + " | ".join(cells) + " |\n")
            report.write("\n")

        if _complete(results):
            report.write("## Accuracy Deltas versus F Reference\n\n")
            report.write("| Intervention | " + " | ".join(f"T={t}" for t in TIME_STEPS) + " |\n")
            report.write("|" + "---|" * (len(TIME_STEPS) + 1) + "\n")
            for name in (
                "G_E_COEFFICIENTS_SNM_ON",
                "H_F_BIAS_STATE_OFF",
                "I_F_FINAL_GLOBAL_CLIP",
            ):
                deltas = [
                    results[name][t]["acc"]
                    - results["F_REFERENCE_SNM_STATE_LR"][t]["acc"]
                    for t in TIME_STEPS
                ]
                report.write(
                    f"| {name} - F | "
                    + " | ".join(f"{value:+.2f}pp" for value in deltas)
                    + " |\n"
                )
            report.write("\n")

        report.write("## Per-layer Detail\n\n")
        for name in VARIANTS:
            for t in TIME_STEPS:
                if t not in layer_results[name]:
                    continue
                report.write(f"### {name}, T={t}\n\n")
                report.write(
                    "| Layer | PosRate | NegRate | Sparsity | InputSpikes | "
                    "SOPs | ScaleOps |\n"
                )
                report.write("|---|---:|---:|---:|---:|---:|---:|\n")
                for item in layer_results[name][t]:
                    positive = format_pct(item.positive_spike_rate) if item.has_spike_output else "-"
                    negative = format_pct(item.negative_spike_rate) if item.has_spike_output else "-"
                    sparsity = format_pct(item.spike_sparsity) if item.has_spike_output else "-"
                    report.write(
                        f"| {item.name} | {positive} | {negative} | "
                        f"{sparsity} | {item.total_input_spikes:,} | "
                        f"{item.sops:,} | {item.scale_operations:,} |\n"
                    )
                report.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    output_path = Path(args.output)
    progress_path = progress_path_for(output_path)
    if (output_path.exists() or progress_path.exists()) and not (
        args.resume or args.overwrite
    ):
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --resume or --overwrite"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(FORMAL_SEED)
    train_loader, test_loader = datapool("cifar100", FORMAL_BATCH_SIZE)
    calibration_batches = materialize_calibration_batches(
        train_loader, FORMAL_CALI_BATCHES
    )
    calibration_sha256 = calibration_batches_sha256(calibration_batches)

    ann_template, snn_template, checkpoint_metadata = load_qcfs_pair(
        args.checkpoint, "cifar100", "resnet20", device
    )
    if checkpoint_metadata["sha256"] != FORMAL_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint is not the selected 68.78% weight file")
    ann_template.set_L(FORMAL_QCFS_L)
    ann_template.set_T(0)
    ann_template.set_qcfs_training_profile(FORMAL_QCFS_TRAINING_PROFILE)
    ann_accuracy = val(ann_template, test_loader, device, 0)
    if ann_accuracy < FORMAL_MIN_ANN_ACCURACY:
        raise RuntimeError(
            f"QCFS ANN gate failed: {ann_accuracy:.2f}% < "
            f"{FORMAL_MIN_ANN_ACCURACY:.2f}%"
        )
    t32_accuracy, t32_gap = validate_t32_conversion(
        snn_template,
        test_loader,
        device,
        ann_accuracy,
        FORMAL_MAX_T32_CONVERSION_GAP,
    )

    signature = diagnostic_signature(checkpoint_metadata, calibration_sha256)
    if args.resume:
        results, layer_results = load_progress(
            progress_path, signature, VARIANTS
        )
    else:
        results = {name: {} for name in VARIANTS}
        layer_results = {name: {} for name in VARIANTS}
        save_progress(progress_path, signature, results, layer_results)

    write_report(
        output_path,
        results,
        layer_results,
        checkpoint_metadata,
        ann_accuracy,
        t32_accuracy,
        t32_gap,
        calibration_sha256,
    )

    for time_steps in TIME_STEPS:
        if all(time_steps in results[name] for name in VARIANTS):
            print(f"Skipping completed T={time_steps}")
            continue
        print(f"\n{'=' * 72}\nCausal diagnostics T={time_steps}\n{'=' * 72}")
        unsigned_model, unsigned_calibration = calibrate_state_lr(
            snn_template,
            ann_template,
            time_steps,
            False,
            calibration_batches,
            device,
        )
        signed_model, signed_calibration = calibrate_state_lr(
            snn_template,
            ann_template,
            time_steps,
            True,
            calibration_batches,
            device,
        )
        variants, clip_stats = build_variants(unsigned_model, signed_model)
        for name, model in variants.items():
            if time_steps in results[name]:
                continue
            calibration_elapsed = (
                unsigned_calibration
                if VARIANTS[name]["calibration_signed"] is False
                else signed_calibration
            )
            summary, per_layer = evaluate_precalibrated(
                model,
                test_loader,
                device,
                time_steps,
                calibration_elapsed,
                clip_changed=(
                    clip_stats["changed"]
                    if name == "I_F_FINAL_GLOBAL_CLIP"
                    else 0
                ),
            )
            results[name][time_steps] = summary
            layer_results[name][time_steps] = per_layer
            save_progress(progress_path, signature, results, layer_results)
            write_report(
                output_path,
                results,
                layer_results,
                checkpoint_metadata,
                ann_accuracy,
                t32_accuracy,
                t32_gap,
                calibration_sha256,
            )
            print(
                f"{name} T={time_steps}: acc={summary['acc']:.2f}% "
                f"neg={format_pct(summary['negative_rate'])} "
                f"max_coeff={summary['max_coefficient_ratio']:.4f} "
                f"changed={summary['final_clip_changed']}"
            )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
