"""Six-way CIFAR-100/ResNet20 QCFS ablation without CSRR."""

import argparse
import copy
import hashlib
import json
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
from models import SignedIF
from preprocess import datapool
from scripts.experiments.qcfs_checkpoint import load_qcfs_pair
from scripts.experiments.run_state_ftbc_ablation import (
    materialize_calibration_batches,
)
from scripts.experiments.run_stats_ablation import summarize_layer_stats
from spike_stats import (
    SpikeLayerStats,
    collect_resnet20_spike_stats,
    reset_signed_spike_stats,
    set_signed_spike_stats_enabled,
    summarize_ftbc_storage,
)
from utils import seed_all, val


FORMAL_TIME_STEPS = (1, 2, 4, 8, 16, 32)
FORMAL_BATCH_SIZE = 200
FORMAL_SEED = 42
FORMAL_QCFS_L = 8
FORMAL_ALPHA = 0.4
FORMAL_RIDGE = 1e-3
FORMAL_COEFFICIENT_CLIP = 0.25
FORMAL_OVER_WEIGHT = 2.5
FORMAL_UNDER_WEIGHT = 1.0
FORMAL_CALI_BATCHES = 5
# The selected retrained checkpoint reached 68.78% in its training log. The
# exact file re-evaluates at 68.65--68.68% on the current CPU/GPU stack, so the
# locked gate allows at most 0.15 percentage points of numerical drift.
FORMAL_REPORTED_CHECKPOINT_ACCURACY = 68.78
FORMAL_MIN_ANN_ACCURACY = 68.63
FORMAL_MAX_T32_CONVERSION_GAP = 2.0
FORMAL_CHECKPOINT_SHA256 = (
    "1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2"
)
FORMAL_QCFS_TRAINING_PROFILE = "paper_era"
PAPER_QCFS_ACCURACY = {
    0: 69.94,
    2: 19.96,
    4: 34.14,
    8: 55.37,
    16: 67.33,
    32: 69.82,
}
OFFICIAL_REPOSITORY = "https://github.com/putshua/ANN_SNN_QCFS"
OFFICIAL_CHECKPOINT_FOLDER = (
    "https://drive.google.com/drive/folders/"
    "1P-2egAraWtsQYNzp8lcJvZVEG_KLVV5Q"
)


CONFIGS = OrderedDict(
    [
        ("A_QCFS_R0", dict(signed=False, ftbc_mode="none")),
        ("B_QCFS_SNM_R0", dict(signed=True, ftbc_mode="none")),
        ("C_QCFS_R0_FULL_FTBC", dict(signed=False, ftbc_mode="full")),
        ("D_QCFS_SNM_R0_FULL_FTBC", dict(signed=True, ftbc_mode="full")),
        ("E_QCFS_R0_STATE_LR", dict(signed=False, ftbc_mode="state_low_rank")),
        ("F_QCFS_SNM_R0_STATE_LR", dict(signed=True, ftbc_mode="state_low_rank")),
    ]
)


def resolve_time_steps(run_kind, time_steps):
    if time_steps is not None:
        return list(time_steps)
    return list(FORMAL_TIME_STEPS) if run_kind == "formal" else [2, 4]


def resolve_calibration_batches(run_kind, cali_batches):
    if cali_batches is not None:
        return int(cali_batches)
    return FORMAL_CALI_BATCHES if run_kind == "formal" else 1


def effective_ftbc_mode(requested_mode, time_steps):
    if requested_mode == "state_low_rank" and int(time_steps) < 3:
        return "full"
    return requested_mode


def validate_formal_protocol(args):
    if args.run_kind != "formal":
        return
    errors = []
    if tuple(args.time_steps) != FORMAL_TIME_STEPS:
        errors.append(f"time_steps must be {list(FORMAL_TIME_STEPS)}")
    if list(args.configs) != list(CONFIGS):
        errors.append("all six configurations must be run in the fixed order")
    fixed_values = [
        ("batch_size", args.batch_size, FORMAL_BATCH_SIZE),
        ("cali_batches", args.cali_batches, FORMAL_CALI_BATCHES),
        ("seed", args.seed, FORMAL_SEED),
        ("QCFS L", args.qcfs_L, FORMAL_QCFS_L),
        ("alpha", args.alpha, FORMAL_ALPHA),
        ("ridge", args.ridge, FORMAL_RIDGE),
        (
            "coefficient_clip",
            args.coefficient_clip,
            FORMAL_COEFFICIENT_CLIP,
        ),
        ("over_weight", args.over_weight, FORMAL_OVER_WEIGHT),
        ("under_weight", args.under_weight, FORMAL_UNDER_WEIGHT),
        (
            "min_ann_accuracy",
            args.min_ann_accuracy,
            FORMAL_MIN_ANN_ACCURACY,
        ),
        (
            "max_t32_conversion_gap",
            args.max_t32_conversion_gap,
            FORMAL_MAX_T32_CONVERSION_GAP,
        ),
    ]
    for label, actual, expected in fixed_values:
        if actual != expected:
            errors.append(f"{label} must be {expected}")
    if errors:
        raise ValueError("Invalid formal protocol: " + "; ".join(errors))


def validate_formal_checkpoint(args, checkpoint_metadata):
    """Lock a formal run to the user-selected 68.78% checkpoint file."""
    if args.run_kind != "formal":
        return
    actual_sha256 = checkpoint_metadata["sha256"].lower()
    if actual_sha256 != FORMAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Formal checkpoint mismatch: expected the selected 68.78% "
            f"checkpoint SHA256 {FORMAL_CHECKPOINT_SHA256}, got "
            f"{actual_sha256}"
        )


def resolve_output(args):
    if args.output:
        return Path(args.output)
    if args.run_kind == "formal":
        return Path(
            "docs/results/comparative_ablation/cifar100/"
            "RESNET20_SIX_WAY_ABLATION.md"
        )
    return Path(
        "docs/archive/experiments/resnet20/"
        "RESNET20_SIX_WAY_SMOKE.md"
    )


def progress_path_for(output_path):
    return output_path.with_suffix(".progress.json")


def calibration_batches_sha256(calibration_batches):
    digest = hashlib.sha256()
    for inputs, targets in calibration_batches:
        for tensor in (inputs, targets):
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(str(tuple(contiguous.shape)).encode("ascii"))
            digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def protocol_signature(
    args,
    configs,
    checkpoint_metadata,
    calibration_sha256,
):
    return {
        "checkpoint_sha256": checkpoint_metadata["sha256"],
        "weight_origin": args.weight_origin,
        "official_commit": args.official_commit,
        "run_kind": args.run_kind,
        "configs": list(configs),
        "configuration_matrix": {
            name: dict(config) for name, config in configs.items()
        },
        "time_steps": list(args.time_steps),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "qcfs_L": args.qcfs_L,
        "qcfs_training_profile": FORMAL_QCFS_TRAINING_PROFILE,
        "alpha": args.alpha,
        "ridge": args.ridge,
        "coefficient_clip": args.coefficient_clip,
        "over_weight": args.over_weight,
        "under_weight": args.under_weight,
        "cali_batches": args.cali_batches,
        "calibration_sha256": calibration_sha256,
        "min_ann_accuracy": args.min_ann_accuracy,
        "max_t32_conversion_gap": args.max_t32_conversion_gap,
    }


def save_progress(path, signature, results, layer_results):
    payload = {
        "signature": signature,
        "results": {
            name: {str(t): summary for t, summary in values.items()}
            for name, values in results.items()
        },
        "layer_results": {
            name: {
                str(t): [asdict(item) for item in layer_stats]
                for t, layer_stats in values.items()
            }
            for name, values in layer_results.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as progress_file:
        json.dump(payload, progress_file, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def load_progress(path, expected_signature, configs):
    with path.open("r", encoding="utf-8") as progress_file:
        payload = json.load(progress_file)
    if payload.get("signature") != expected_signature:
        raise RuntimeError(
            "Progress file does not match the checkpoint or experiment protocol"
        )
    results = {name: {} for name in configs}
    layer_results = {name: {} for name in configs}
    for name, values in payload.get("results", {}).items():
        if name not in results:
            raise RuntimeError(f"Unexpected config in progress file: {name}")
        results[name] = {int(t): summary for t, summary in values.items()}
    for name, values in payload.get("layer_results", {}).items():
        if name not in layer_results:
            raise RuntimeError(f"Unexpected layer config in progress file: {name}")
        layer_results[name] = {
            int(t): [SpikeLayerStats(**item) for item in items]
            for t, items in values.items()
        }
    return results, layer_results


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_snn(model, config, time_steps):
    model.set_T(time_steps)
    model.set_coding_mode(
        "rate",
        schedule="rate",
        ratio=1.0,
        positive_margin=0.5,
        negative_margin=0.5,
        r0_mode="legacy_clamp",
    )
    model.set_signed(config["signed"])
    model.set_r0(True)
    model.reset_all_bias()
    mode = effective_ftbc_mode(config["ftbc_mode"], time_steps)
    model.set_ftbc_mode(mode)
    return mode


def validate_t32_conversion(
    snn_template,
    test_loader,
    device,
    ann_accuracy,
    max_gap,
):
    """Reject a non-converting QCFS checkpoint before creating a report."""
    model = copy.deepcopy(snn_template).to(device)
    configure_snn(model, CONFIGS["A_QCFS_R0"], time_steps=32)
    set_signed_spike_stats_enabled(model, SignedIF, False)
    accuracy = val(model, test_loader, device, 32)
    conversion_gap = ann_accuracy - accuracy
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if conversion_gap > max_gap:
        raise RuntimeError(
            f"T=32 conversion gate failed: gap={conversion_gap:.2f}pp > "
            f"{max_gap:.2f}pp (ANN={ann_accuracy:.2f}%, "
            f"A_QCFS_R0={accuracy:.2f}%)"
        )
    return accuracy, conversion_gap


def run_one(
    config,
    snn_template,
    ann_template,
    time_steps,
    calibration_batches,
    test_loader,
    device,
    args,
):
    model = copy.deepcopy(snn_template).to(device)
    mode = configure_snn(model, config, time_steps)
    set_signed_spike_stats_enabled(model, SignedIF, False)

    calibration_elapsed = 0.0
    if mode != "none":
        ann = copy.deepcopy(ann_template).to(device)
        ann.set_T(0)
        ann.set_L(args.qcfs_L)
        synchronize(device)
        calibration_start = time.perf_counter()
        bias_corr_model(
            ann=ann,
            snn=model,
            T=time_steps,
            train_loader=calibration_batches,
            curr_t_alpha=args.alpha,
            num_cali_sample_batches=args.cali_batches,
            ftbc_mode=mode,
            ridge=args.ridge,
            over_weight=args.over_weight,
            under_weight=args.under_weight,
            coefficient_clip=args.coefficient_clip,
        )
        synchronize(device)
        calibration_elapsed = time.perf_counter() - calibration_start
        del ann

    storage = summarize_ftbc_storage(model, SignedIF)
    set_signed_spike_stats_enabled(model, SignedIF, True)
    reset_signed_spike_stats(model, SignedIF)
    accuracy = val(model, test_loader, device, time_steps)
    layer_stats = collect_resnet20_spike_stats(model, SignedIF, nn.Conv2d)
    summary = summarize_layer_stats(layer_stats)

    set_signed_spike_stats_enabled(model, SignedIF, False)
    synchronize(device)
    inference_start = time.perf_counter()
    timed_accuracy = val(model, test_loader, device, time_steps)
    synchronize(device)
    inference_elapsed = time.perf_counter() - inference_start
    if timed_accuracy != accuracy:
        raise RuntimeError(
            f"Non-deterministic evaluation accuracy: {accuracy} vs "
            f"{timed_accuracy}"
        )

    summary.update(
        {
            "acc": accuracy,
            "effective_ftbc_mode": mode,
            "ftbc_parameters": storage["parameters"],
            "ftbc_bytes": storage["bytes"],
            "calibration_elapsed": calibration_elapsed,
            "inference_elapsed": inference_elapsed,
        }
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, layer_stats


def format_pct(value):
    return f"{float(value) * 100:.4f}%"


def _result_is_complete(configs, time_steps, results):
    return all(
        time_steps_value in results[name]
        for name in configs
        for time_steps_value in time_steps
    )


def write_report(
    path,
    args,
    configs,
    results,
    layer_results,
    checkpoint_metadata,
    ann_accuracy,
    t32_gate_accuracy,
    t32_gate_gap,
    calibration_sha256,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = _result_is_complete(configs, args.time_steps, results)
    status = "COMPLETE" if complete else "INCOMPLETE"
    with path.open("w", encoding="utf-8") as report:
        report.write("# CIFAR-100 / ResNet20 QCFS Six-way Ablation\n\n")
        report.write(f"- Status: **{status}**\n")
        report.write(f"- Run kind: {args.run_kind}\n")
        report.write("- CSRR: disabled in every configuration\n")
        report.write("- Dataset / architecture: CIFAR-100 / ResNet20\n")
        report.write(f"- QCFS activation levels: L={args.qcfs_L}\n")
        report.write(
            f"- Selected checkpoint training-log accuracy: "
            f"{FORMAL_REPORTED_CHECKPOINT_ACCURACY:.2f}%\n"
        )
        report.write(f"- QCFS ANN accuracy re-evaluated here: {ann_accuracy:.2f}%\n")
        report.write(
            f"- QCFS training profile: {FORMAL_QCFS_TRAINING_PROFILE}\n"
        )
        report.write(
            f"- Pre-report A_QCFS_R0 T=32 gate: "
            f"{t32_gate_accuracy:.2f}% (gap={t32_gate_gap:.2f}pp)\n"
        )
        report.write(f"- Weight origin: {args.weight_origin}\n")
        report.write(f"- Checkpoint: {checkpoint_metadata['filename']}\n")
        report.write(
            f"- Checkpoint size: {checkpoint_metadata['size_bytes']:,} bytes\n"
        )
        report.write(f"- Checkpoint SHA256: `{checkpoint_metadata['sha256']}`\n")
        report.write(
            f"- Official implementation: {OFFICIAL_REPOSITORY} "
            f"(commit `{args.official_commit}`)\n"
        )
        report.write(
            f"- Official checkpoint folder checked: {OFFICIAL_CHECKPOINT_FOLDER}\n"
        )
        report.write(f"- Time steps: {args.time_steps}\n")
        report.write(
            f"- Calibration: batches={args.cali_batches}, batch_size="
            f"{args.batch_size}, alpha={args.alpha}, ridge={args.ridge}, "
            f"coefficient_clip={args.coefficient_clip}, "
            f"w_under={args.under_weight}, w_over={args.over_weight}\n"
        )
        report.write(f"- Calibration data SHA256: `{calibration_sha256}`\n")
        report.write(f"- Seed: {args.seed}\n")
        report.write(
            "- Coding is rate, ratio=1, R0=legacy_clamp in every group; "
            "ScaleOps are therefore expected to be zero.\n"
        )
        report.write(
            "- The over/under weights apply only to state-low-rank "
            "regression. Full FTBC retains the preceding per-timestep "
            "mean-bias solver.\n"
        )
        report.write(
            "- SOPs are input-driven; positive and negative spikes both count "
            "as events. Raw image input is not counted as a spike source.\n\n"
        )

        report.write("## Configuration Matrix\n\n")
        report.write("| Config | QCFS | SNM | R0 | Full FTBC | State-LR FTBC | CSRR |\n")
        report.write("|---|---|---|---|---|---|---|\n")
        for name, config in configs.items():
            report.write(
                f"| {name} | Yes | {'Yes' if config['signed'] else 'No'} | "
                f"Yes | {'Yes' if config['ftbc_mode'] == 'full' else 'No'} | "
                f"{'Yes' if config['ftbc_mode'] == 'state_low_rank' else 'No'} | No |\n"
            )
        report.write("\n")

        metrics = [
            ("Accuracy", "acc", lambda value: f"{value:.2f}%"),
            ("Input-driven SOPs", "sops", lambda value: f"{value:,}"),
            ("Time-scale operations", "scale_operations", lambda value: f"{value:,}"),
            ("Positive spike rate", "positive_rate", format_pct),
            ("Negative spike rate", "negative_rate", format_pct),
            ("Overall spike sparsity", "sparsity", format_pct),
            ("FTBC parameters", "ftbc_parameters", lambda value: f"{value:,}"),
            ("FTBC storage bytes", "ftbc_bytes", lambda value: f"{value:,}"),
            ("Calibration elapsed", "calibration_elapsed", lambda value: f"{value:.1f}s"),
            ("Inference elapsed (statistics disabled)", "inference_elapsed", lambda value: f"{value:.1f}s"),
        ]
        for title, key, formatter in metrics:
            report.write(f"## {title}\n\n")
            report.write(
                "| Config | "
                + " | ".join(f"T={value}" for value in args.time_steps)
                + " |\n"
            )
            report.write("|" + "---|" * (len(args.time_steps) + 1) + "\n")
            for name in configs:
                cells = [
                    formatter(results[name][time_steps][key])
                    if time_steps in results[name]
                    else "-"
                    for time_steps in args.time_steps
                ]
                report.write(f"| {name} | " + " | ".join(cells) + " |\n")
            report.write("\n")

        report.write("## Effective FTBC Mode\n\n")
        report.write(
            "| Config | "
            + " | ".join(f"T={value}" for value in args.time_steps)
            + " |\n"
        )
        report.write("|" + "---|" * (len(args.time_steps) + 1) + "\n")
        for name in configs:
            cells = [
                results[name][time_steps]["effective_ftbc_mode"]
                if time_steps in results[name]
                else "-"
                for time_steps in args.time_steps
            ]
            report.write(f"| {name} | " + " | ".join(cells) + " |\n")
        report.write(
            "\nState-LR has three channel-wise coefficients. At T=1 and T=2 "
            "it falls back to full FTBC; full-vs-low-rank comparisons are "
            "interpretable from T>=4.\n\n"
        )

        report.write("## Published QCFS Reference (CIFAR-100 / ResNet20)\n\n")
        report.write("| ANN | T=2 | T=4 | T=8 | T=16 | T=32 |\n")
        report.write("|---:|---:|---:|---:|---:|---:|\n")
        report.write(
            "| 69.94% | 19.96% | 34.14% | 55.37% | 67.33% | 69.82% |\n\n"
        )

        report.write("## Per-layer Detail\n\n")
        for name in configs:
            for time_steps in args.time_steps:
                if time_steps not in layer_results[name]:
                    continue
                report.write(f"### {name}, T={time_steps}\n\n")
                report.write(
                    "| Layer | PosRate | NegRate | Sparsity | InputSpikes | "
                    "SOPs | ScaleOps |\n"
                )
                report.write("|---|---:|---:|---:|---:|---:|---:|\n")
                for item in layer_results[name][time_steps]:
                    positive_rate = (
                        format_pct(item.positive_spike_rate)
                        if item.has_spike_output
                        else "-"
                    )
                    negative_rate = (
                        format_pct(item.negative_spike_rate)
                        if item.has_spike_output
                        else "-"
                    )
                    sparsity = (
                        format_pct(item.spike_sparsity)
                        if item.has_spike_output
                        else "-"
                    )
                    report.write(
                        f"| {item.name} | {positive_rate} | "
                        f"{negative_rate} | {sparsity} | "
                        f"{item.total_input_spikes:,} | {item.sops:,} | "
                        f"{item.scale_operations:,} |\n"
                    )
                report.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(
        description="CIFAR-100/ResNet20 QCFS six-way ablation without CSRR"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weight_origin", choices=("author_pretrained", "official_implementation_retrained"), default="official_implementation_retrained")
    parser.add_argument("--official_commit", default="eca136bd085087567013240ee14fb6159a2b6da7")
    parser.add_argument("--run_kind", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("-dev", "--device", default="0")
    parser.add_argument("-b", "--batch_size", default=FORMAL_BATCH_SIZE, type=int)
    parser.add_argument("--seed", default=FORMAL_SEED, type=int)
    parser.add_argument("--qcfs_L", default=FORMAL_QCFS_L, type=int)
    parser.add_argument("--alpha", default=FORMAL_ALPHA, type=float)
    parser.add_argument("--ridge", default=FORMAL_RIDGE, type=float)
    parser.add_argument(
        "--coefficient_clip", default=FORMAL_COEFFICIENT_CLIP, type=float
    )
    parser.add_argument("--over_weight", default=FORMAL_OVER_WEIGHT, type=float)
    parser.add_argument("--under_weight", default=FORMAL_UNDER_WEIGHT, type=float)
    parser.add_argument("--cali_batches", type=int)
    parser.add_argument("--time_steps", nargs="+", type=int)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument(
        "--min_ann_accuracy", default=FORMAL_MIN_ANN_ACCURACY, type=float
    )
    parser.add_argument(
        "--max_t32_conversion_gap",
        default=FORMAL_MAX_T32_CONVERSION_GAP,
        type=float,
    )
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.time_steps = resolve_time_steps(args.run_kind, args.time_steps)
    args.cali_batches = resolve_calibration_batches(
        args.run_kind, args.cali_batches
    )
    validate_formal_protocol(args)
    output_path = resolve_output(args)
    progress_path = progress_path_for(output_path)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if (output_path.exists() or progress_path.exists()) and not (
        args.overwrite or args.resume
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing result/progress for {output_path}. "
            "Choose a new --output, pass --resume, or explicitly pass "
            "--overwrite."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    train_loader, test_loader = datapool("cifar100", args.batch_size)
    calibration_batches = materialize_calibration_batches(
        train_loader, args.cali_batches
    )
    calibration_sha256 = calibration_batches_sha256(calibration_batches)
    ann_template, snn_template, checkpoint_metadata = load_qcfs_pair(
        args.checkpoint, "cifar100", "resnet20", device
    )
    validate_formal_checkpoint(args, checkpoint_metadata)
    ann_template.set_L(args.qcfs_L)
    ann_template.set_T(0)
    ann_template.set_qcfs_training_profile(FORMAL_QCFS_TRAINING_PROFILE)
    ann_accuracy = val(ann_template, test_loader, device, 0)
    if ann_accuracy < args.min_ann_accuracy:
        raise RuntimeError(
            f"QCFS ANN gate failed: {ann_accuracy:.2f}% < "
            f"{args.min_ann_accuracy:.2f}%"
        )
    print(
        f"QCFS checkpoint accepted: ANN={ann_accuracy:.2f}% "
        f"SHA256={checkpoint_metadata['sha256']}"
    )
    t32_gate_accuracy, t32_gate_gap = validate_t32_conversion(
        snn_template,
        test_loader,
        device,
        ann_accuracy,
        args.max_t32_conversion_gap,
    )
    print(
        f"T=32 conversion accepted: A_QCFS_R0={t32_gate_accuracy:.2f}% "
        f"gap={t32_gate_gap:.2f}pp"
    )

    configs = OrderedDict((name, CONFIGS[name]) for name in args.configs)
    signature = protocol_signature(
        args,
        configs,
        checkpoint_metadata,
        calibration_sha256,
    )
    if args.resume:
        if not progress_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without progress file: {progress_path}"
            )
        results, layer_results = load_progress(
            progress_path, signature, configs
        )
    else:
        results = {name: {} for name in configs}
        layer_results = {name: {} for name in configs}
        save_progress(progress_path, signature, results, layer_results)
        write_report(
            output_path,
            args,
            configs,
            results,
            layer_results,
            checkpoint_metadata,
            ann_accuracy,
            t32_gate_accuracy,
            t32_gate_gap,
            calibration_sha256,
        )
    for name, config in configs.items():
        for time_steps in args.time_steps:
            if time_steps in results[name]:
                print(f"Skipping completed point: {name} T={time_steps}")
                continue
            print(f"\n{'=' * 72}\n{name} T={time_steps}\n{'=' * 72}")
            summary, per_layer = run_one(
                config,
                snn_template,
                ann_template,
                time_steps,
                calibration_batches,
                test_loader,
                device,
                args,
            )
            results[name][time_steps] = summary
            layer_results[name][time_steps] = per_layer
            save_progress(progress_path, signature, results, layer_results)
            print(
                f"{name} T={time_steps}: acc={summary['acc']:.2f}% "
                f"sops={summary['sops']:,} "
                f"sparsity={format_pct(summary['sparsity'])} "
                f"bias={summary['ftbc_bytes']:,}B "
                f"cal={summary['calibration_elapsed']:.1f}s "
                f"infer={summary['inference_elapsed']:.1f}s"
            )
            write_report(
                output_path,
                args,
                configs,
                results,
                layer_results,
                checkpoint_metadata,
                ann_accuracy,
                t32_gate_accuracy,
                t32_gate_gap,
                calibration_sha256,
            )
    write_report(
        output_path,
        args,
        configs,
        results,
        layer_results,
        checkpoint_metadata,
        ann_accuracy,
        t32_gate_accuracy,
        t32_gate_gap,
        calibration_sha256,
    )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
