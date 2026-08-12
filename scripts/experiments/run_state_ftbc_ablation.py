# 对比完整FTBC与状态条件低秩FTBC
import argparse
import copy
import os
import sys
import time
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
from scripts.experiments.run_stats_ablation import (
    load_signed_model,
    summarize_layer_stats,
)
from spike_stats import (
    collect_signed_spike_stats,
    reset_signed_spike_stats,
    set_signed_spike_stats_enabled,
    summarize_ftbc_storage,
)
from utils import seed_all, val


CONFIGS = {
    "F_FULL_FTBC": {
        "ftbc_mode": "full",
        "over_weight": 1.0,
        "under_weight": 1.0,
    },
    "G_STATE_LR": {
        "ftbc_mode": "state_low_rank",
        "over_weight": 1.0,
        "under_weight": 1.0,
    },
    "H_STATE_LR_SOPS": {
        "ftbc_mode": "state_low_rank",
        "over_weight": 2.0,
        "under_weight": 1.0,
    },
}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def materialize_calibration_batches(train_loader, count):
    batches = []
    for index, (inputs, targets) in enumerate(train_loader):
        if index >= count:
            break
        batches.append((inputs.clone(), targets.clone()))
    if len(batches) != count:
        raise RuntimeError(
            f"Requested {count} calibration batches, but only found {len(batches)}"
        )
    return batches


def effective_ftbc_mode(requested_mode, T):
    if requested_mode == "state_low_rank" and T < 3:
        return "full"
    return requested_mode


def run_one(
    cfg,
    model_template,
    ann_template,
    T,
    calibration_batches,
    test_loader,
    device,
    args,
):
    model = copy.deepcopy(model_template).to(device)
    model.set_signed(True)
    model.set_r0(True)
    model.reset_all_bias()
    model.set_T(T)

    mode = effective_ftbc_mode(cfg["ftbc_mode"], T)
    model.set_ftbc_mode(mode)

    ann = copy.deepcopy(ann_template).to(device)
    ann.set_signed(False)
    ann.set_r0(False)

    synchronize(device)
    calibration_start = time.perf_counter()
    bias_corr_model(
        ann=ann,
        snn=model,
        T=T,
        train_loader=calibration_batches,
        curr_t_alpha=args.alpha,
        num_cali_sample_batches=args.cali_batches,
        ftbc_mode=mode,
        ridge=args.ridge,
        over_weight=cfg["over_weight"],
        under_weight=cfg["under_weight"],
        coefficient_clip=args.coefficient_clip,
    )
    synchronize(device)
    calibration_elapsed = time.perf_counter() - calibration_start
    storage = summarize_ftbc_storage(model, SignedIF)

    set_signed_spike_stats_enabled(model, SignedIF, True)
    reset_signed_spike_stats(model, SignedIF)
    accuracy = val(model, test_loader, device, T)
    layer_stats = collect_signed_spike_stats(model, SignedIF, nn.Conv2d, nn.Linear)
    summary = summarize_layer_stats(layer_stats)

    # Pure latency excludes spike-stat reduction kernels and host synchronization.
    set_signed_spike_stats_enabled(model, SignedIF, False)
    synchronize(device)
    inference_start = time.perf_counter()
    timed_accuracy = val(model, test_loader, device, T)
    synchronize(device)
    inference_elapsed = time.perf_counter() - inference_start
    if timed_accuracy != accuracy:
        raise RuntimeError(
            f"Non-deterministic evaluation accuracy: {accuracy} vs {timed_accuracy}"
        )
    summary.update(
        {
            "acc": accuracy,
            "effective_mode": mode,
            "ftbc_parameters": storage["parameters"],
            "ftbc_bytes": storage["bytes"],
            "calibration_elapsed": calibration_elapsed,
            "inference_elapsed": inference_elapsed,
            "total_elapsed": calibration_elapsed + inference_elapsed,
        }
    )

    del ann
    del model
    torch.cuda.empty_cache()
    return summary, layer_stats


def format_pct(value):
    return f"{value * 100:.4f}%"


def write_report(path, args, selected_configs, results, layer_results):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as report:
        report.write("# State-conditioned Low-rank FTBC Ablation\n\n")
        report.write(f"- Dataset: {args.dataset}\n")
        report.write("- Model: VGG16\n")
        report.write(f"- Time steps: {args.time_steps}\n")
        report.write(
            f"- Calibration: batches={args.cali_batches}, alpha={args.alpha}, "
            f"ridge={args.ridge}, coefficient_clip={args.coefficient_clip}\n"
        )
        report.write("- All configurations reuse the same materialized calibration batches.\n\n")

        metrics = [
            ("Accuracy", "acc", lambda value: f"{value:.2f}%"),
            ("Input-driven SOPs", "sops", lambda value: f"{value:,}"),
            (
                "Time-scale operations",
                "scale_operations",
                lambda value: f"{value:,}",
            ),
            ("Positive spike rate", "positive_rate", format_pct),
            ("Negative spike rate", "negative_rate", format_pct),
            ("Overall spike sparsity", "sparsity", format_pct),
            ("FTBC parameters", "ftbc_parameters", lambda value: f"{value:,}"),
            ("FTBC storage bytes", "ftbc_bytes", lambda value: f"{value:,}"),
            ("Calibration elapsed", "calibration_elapsed", lambda value: f"{value:.1f}s"),
            (
                "Inference elapsed (statistics disabled)",
                "inference_elapsed",
                lambda value: f"{value:.1f}s",
            ),
        ]

        for title, key, formatter in metrics:
            report.write(f"## {title}\n\n")
            report.write(
                "| Config | "
                + " | ".join(f"T={T}" for T in args.time_steps)
                + " |\n"
            )
            report.write("|" + "---|" * (len(args.time_steps) + 1) + "\n")
            for name in selected_configs:
                values = [
                    formatter(results[name][T][key]) if T in results[name] else "-"
                    for T in args.time_steps
                ]
                report.write(f"| {name} | " + " | ".join(values) + " |\n")
            report.write("\n")

        report.write("## Effective FTBC Mode\n\n")
        report.write("| Config | " + " | ".join(f"T={T}" for T in args.time_steps) + " |\n")
        report.write("|" + "---|" * (len(args.time_steps) + 1) + "\n")
        for name in selected_configs:
            modes = [
                results[name][T]["effective_mode"] if T in results[name] else "-"
                for T in args.time_steps
            ]
            report.write(f"| {name} | " + " | ".join(modes) + " |\n")
        report.write("\n")

        report.write("## Per-layer Detail\n\n")
        for name in selected_configs:
            for T in args.time_steps:
                if T not in layer_results[name]:
                    continue
                report.write(f"### {name}, T={T}\n\n")
                report.write(
                    "| Layer | PosRate | NegRate | Sparsity | "
                    "InputSpikes | SOPs | ScaleOps |\n"
                )
                report.write("|---|---:|---:|---:|---:|---:|---:|\n")
                for item in layer_results[name][T]:
                    report.write(
                        f"| {item.name} | {format_pct(item.positive_spike_rate)} | "
                        f"{format_pct(item.negative_spike_rate)} | "
                        f"{format_pct(item.spike_sparsity)} | "
                        f"{item.total_input_spikes:,} | {item.sops:,} | "
                        f"{item.scale_operations:,} |\n"
                    )
                report.write("\n")


def main():
    parser = argparse.ArgumentParser(description="State-conditioned low-rank FTBC ablation")
    parser.add_argument("-data", "--dataset", default="cifar100")
    parser.add_argument("-id", "--identifier", required=True)
    parser.add_argument("-dev", "--device", default="0")
    parser.add_argument("-b", "--batch_size", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("--ridge", default=1e-3, type=float)
    parser.add_argument("--coefficient_clip", default=0.25, type=float)
    parser.add_argument("--cali_batches", default=5, type=int)
    parser.add_argument("--time_steps", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--configs", nargs="+", choices=CONFIGS.keys(), default=list(CONFIGS.keys()))
    parser.add_argument(
        "--output",
        default=(
            "docs/results/state_low_rank_ftbc/final/cifar100/"
            "STATE_LOW_RANK_FTBC_cifar100.md"
        ),
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    calibration_batches = materialize_calibration_batches(
        train_loader, args.cali_batches
    )
    model_template = load_signed_model(args, device)
    ann_template = copy.deepcopy(model_template)

    results = {name: {} for name in args.configs}
    layer_results = {name: {} for name in args.configs}
    for name in args.configs:
        cfg = CONFIGS[name]
        for T in args.time_steps:
            print(f"\n{'=' * 72}\n{name} T={T}\n{'=' * 72}")
            summary, per_layer = run_one(
                cfg,
                model_template,
                ann_template,
                T,
                calibration_batches,
                test_loader,
                device,
                args,
            )
            results[name][T] = summary
            layer_results[name][T] = per_layer
            print(
                f"{name} T={T}: acc={summary['acc']:.2f}% "
                f"sops={summary['sops']:,} sparsity={format_pct(summary['sparsity'])} "
                f"bias={summary['ftbc_bytes']:,}B "
                f"cal={summary['calibration_elapsed']:.1f}s "
                f"infer={summary['inference_elapsed']:.1f}s"
            )
            write_report(args.output, args, args.configs, results, layer_results)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
