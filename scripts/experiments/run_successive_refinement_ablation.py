# 评估逐次精化编码及其与状态低秩FTBC的组合
import argparse
import copy
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
from scripts.experiments.run_state_ftbc_ablation import (
    materialize_calibration_batches,
)
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


FINAL_TIME_SCALE_RATIO = 1.0
FINAL_POSITIVE_MARGIN = 0.55
FINAL_NEGATIVE_MARGIN = 1.30
FINAL_OVER_WEIGHT = 2.5


BASE_CONFIGS = OrderedDict(
    [
        (
            "C_RATE_SNM_R0",
            dict(
                coding_mode="rate",
                schedule="rate",
                ratio=1.0,
                signed=True,
                r0=True,
                ftbc_mode="none",
                expand_ratios=False,
                positive_margin=0.5,
                negative_margin=0.5,
                r0_mode="legacy_clamp",
                over_weight=None,
            ),
        ),
        (
            "H_RATE_STATE_LR",
            dict(
                coding_mode="rate",
                schedule="rate",
                ratio=1.0,
                signed=True,
                r0=True,
                ftbc_mode="state_low_rank",
                expand_ratios=False,
                positive_margin=0.5,
                negative_margin=0.5,
                r0_mode="legacy_clamp",
                over_weight=2.0,
            ),
        ),
        (
            "I_SR_UNIFORM_SNM_R0",
            dict(
                coding_mode="successive_refinement",
                schedule="geometric",
                ratio=1.0,
                signed=True,
                r0=True,
                ftbc_mode="none",
                expand_ratios=False,
                positive_margin=0.5,
                negative_margin=0.5,
                r0_mode="credit_only",
                over_weight=None,
            ),
        ),
        (
            "J_SR_GEOM_UNSIGNED",
            dict(
                coding_mode="successive_refinement",
                schedule="geometric",
                ratio=None,
                signed=False,
                r0=False,
                ftbc_mode="none",
                expand_ratios=True,
                positive_margin=0.5,
                negative_margin=0.5,
                r0_mode="credit_only",
                over_weight=None,
            ),
        ),
        (
            "K_SR_GEOM_SNM",
            dict(
                coding_mode="successive_refinement",
                schedule="geometric",
                ratio=None,
                signed=True,
                r0=False,
                ftbc_mode="none",
                expand_ratios=True,
                positive_margin=0.5,
                negative_margin=0.5,
                r0_mode="credit_only",
                over_weight=None,
            ),
        ),
        (
            "L_SR_GEOM_SNM_R0",
            dict(
                coding_mode="successive_refinement",
                schedule="geometric",
                ratio=None,
                signed=True,
                r0=True,
                ftbc_mode="none",
                expand_ratios=True,
                positive_margin=0.5,
                negative_margin=0.5,
                r0_mode="credit_only",
                over_weight=None,
            ),
        ),
        (
            "M_SR_GEOM_STATE_LR",
            dict(
                coding_mode="successive_refinement",
                schedule="geometric",
                ratio=None,
                signed=True,
                r0=True,
                ftbc_mode="state_low_rank",
                expand_ratios=True,
                positive_margin=0.5,
                negative_margin=0.5,
                r0_mode="credit_only",
                over_weight=None,
            ),
        ),
    ]
)


def _format_ratio(ratio):
    return f"{float(ratio):g}"


def expand_configurations(
    selected,
    ratios,
    negative_margins=(0.5,),
    positive_margins=(0.5,),
):
    expanded = OrderedDict()
    for name in selected:
        base = BASE_CONFIGS[name]
        if not base["expand_ratios"]:
            expanded[name] = dict(base)
            continue
        for ratio in ratios:
            for positive_margin in positive_margins:
                for negative_margin in negative_margins:
                    ratio = float(ratio)
                    positive_margin = float(positive_margin)
                    negative_margin = float(negative_margin)
                    config = dict(base)
                    config["ratio"] = ratio
                    config["positive_margin"] = positive_margin
                    config["negative_margin"] = negative_margin
                    suffix = f"_R{_format_ratio(ratio)}"
                    if abs(positive_margin - 0.5) > 1e-9:
                        suffix += f"_P{_format_ratio(positive_margin)}"
                    if abs(negative_margin - 0.5) > 1e-9:
                        suffix += f"_N{_format_ratio(negative_margin)}"
                    expanded[f"{name}{suffix}"] = config
    return expanded


def effective_ftbc_mode(requested_mode, time_steps):
    if requested_mode == "state_low_rank" and int(time_steps) < 3:
        return "full"
    return requested_mode


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one(
    config,
    model_template,
    ann_template,
    time_steps,
    calibration_batches,
    test_loader,
    device,
    args,
):
    model = copy.deepcopy(model_template).to(device)
    model.set_T(time_steps)
    model.set_coding_mode(
        config["coding_mode"],
        schedule=config["schedule"],
        ratio=config["ratio"],
        positive_margin=config["positive_margin"],
        negative_margin=config["negative_margin"],
        r0_mode=config["r0_mode"],
    )
    model.set_signed(config["signed"])
    model.set_r0(config["r0"])
    model.reset_all_bias()
    set_signed_spike_stats_enabled(model, SignedIF, False)

    ftbc_mode = effective_ftbc_mode(
        config["ftbc_mode"],
        time_steps,
    )
    model.set_ftbc_mode(ftbc_mode)

    calibration_elapsed = 0.0
    calibration_over_weight = (
        args.over_weight
        if config["over_weight"] is None
        else float(config["over_weight"])
    )
    if ftbc_mode != "none":
        ann = copy.deepcopy(ann_template).to(device)
        ann.set_T(0)
        ann.set_coding_mode("rate", schedule="rate", ratio=1.0)
        ann.set_signed(False)
        ann.set_r0(False)
        ann.set_ftbc_mode("none")

        synchronize(device)
        calibration_start = time.perf_counter()
        bias_corr_model(
            ann=ann,
            snn=model,
            T=time_steps,
            train_loader=calibration_batches,
            curr_t_alpha=args.alpha,
            num_cali_sample_batches=args.cali_batches,
            ftbc_mode=ftbc_mode,
            ridge=args.ridge,
            over_weight=calibration_over_weight,
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
    layer_stats = collect_signed_spike_stats(
        model,
        SignedIF,
        nn.Conv2d,
        nn.Linear,
    )
    summary = summarize_layer_stats(layer_stats)

    set_signed_spike_stats_enabled(model, SignedIF, False)
    synchronize(device)
    inference_start = time.perf_counter()
    timed_accuracy = val(model, test_loader, device, time_steps)
    synchronize(device)
    inference_elapsed = time.perf_counter() - inference_start
    if timed_accuracy != accuracy:
        raise RuntimeError(
            f"Non-deterministic evaluation accuracy: "
            f"{accuracy} vs {timed_accuracy}"
        )

    summary.update(
        {
            "acc": accuracy,
            "coding_mode": config["coding_mode"],
            "schedule": config["schedule"],
            "ratio": float(config["ratio"]),
            "effective_ftbc_mode": ftbc_mode,
            "over_weight": calibration_over_weight,
            "ftbc_parameters": storage["parameters"],
            "ftbc_bytes": storage["bytes"],
            "calibration_elapsed": calibration_elapsed,
            "inference_elapsed": inference_elapsed,
            "total_elapsed": calibration_elapsed + inference_elapsed,
        }
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, layer_stats


def format_pct(value):
    return f"{float(value) * 100:.4f}%"


def write_report(path, args, configs, results, layer_results):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report:
        report.write("# Signed Successive-Refinement Ablation\n\n")
        report.write(f"- Dataset: {args.dataset}\n")
        report.write("- Model: VGG16\n")
        report.write(f"- Checkpoint: {args.identifier}\n")
        report.write(f"- Time steps: {args.time_steps}\n")
        report.write(f"- Global geometric ratios: {args.ratios}\n")
        report.write(
            f"- Positive hysteresis margins: {args.positive_margins}\n"
        )
        report.write(
            f"- Negative hysteresis margins: {args.negative_margins}\n"
        )
        report.write(
            f"- Calibration: batches={args.cali_batches}, "
            f"alpha={args.alpha}, ridge={args.ridge}, "
            f"coefficient_clip={args.coefficient_clip}\n"
        )
        report.write(
            "- SOPs are input-driven event operations. ScaleOps are reported "
            "separately and are not added to SOPs.\n\n"
        )

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
            (
                "Calibration elapsed",
                "calibration_elapsed",
                lambda value: f"{value:.1f}s",
            ),
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

        report.write("## Configuration Detail\n\n")
        report.write(
            "| Config | Coding | Schedule | Ratio | PosMargin | "
            "NegMargin | R0 mode | w_over | Effective FTBC by T |\n"
        )
        report.write("|---|---|---|---:|---:|---:|---|---:|---|\n")
        for name, config in configs.items():
            modes = ", ".join(
                f"T={time_steps}:{results[name][time_steps]['effective_ftbc_mode']}"
                for time_steps in args.time_steps
                if time_steps in results[name]
            )
            report.write(
                f"| {name} | {config['coding_mode']} | "
                f"{config['schedule']} | {config['ratio']:.4g} | "
                f"{config['positive_margin']:.4g} | "
                f"{config['negative_margin']:.4g} | "
                f"{config['r0_mode']} | "
                f"{(config['over_weight'] if config['over_weight'] is not None else args.over_weight):.4g} | "
                f"{modes} |\n"
            )
        report.write("\n")

        report.write("## Per-layer Detail\n\n")
        for name in configs:
            for time_steps in args.time_steps:
                if time_steps not in layer_results[name]:
                    continue
                report.write(f"### {name}, T={time_steps}\n\n")
                report.write(
                    "| Layer | PosRate | NegRate | Sparsity | "
                    "InputSpikes | SOPs | ScaleOps |\n"
                )
                report.write("|---|---:|---:|---:|---:|---:|---:|\n")
                for item in layer_results[name][time_steps]:
                    report.write(
                        f"| {item.name} | "
                        f"{format_pct(item.positive_spike_rate)} | "
                        f"{format_pct(item.negative_spike_rate)} | "
                        f"{format_pct(item.spike_sparsity)} | "
                        f"{item.total_input_spikes:,} | {item.sops:,} | "
                        f"{item.scale_operations:,} |\n"
                    )
                report.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Signed successive-refinement ablation"
    )
    parser.add_argument("-data", "--dataset", default="cifar100")
    parser.add_argument("-id", "--identifier", required=True)
    parser.add_argument("-dev", "--device", default="0")
    parser.add_argument("-b", "--batch_size", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("--ridge", default=1e-3, type=float)
    parser.add_argument("--coefficient_clip", default=0.25, type=float)
    parser.add_argument(
        "--over_weight",
        default=FINAL_OVER_WEIGHT,
        type=float,
    )
    parser.add_argument("--under_weight", default=1.0, type=float)
    parser.add_argument("--cali_batches", default=5, type=int)
    parser.add_argument(
        "--time_steps",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(BASE_CONFIGS),
        default=list(BASE_CONFIGS),
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[FINAL_TIME_SCALE_RATIO],
    )
    parser.add_argument(
        "--negative_margins",
        nargs="+",
        type=float,
        default=[FINAL_NEGATIVE_MARGIN],
    )
    parser.add_argument(
        "--positive_margins",
        nargs="+",
        type=float,
        default=[FINAL_POSITIVE_MARGIN],
    )
    parser.add_argument(
        "--output",
        default=(
            "docs/results/successive_refinement/"
            "SR_ABLATION_cifar100.md"
        ),
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    calibration_batches = materialize_calibration_batches(
        train_loader,
        args.cali_batches,
    )
    model_template = load_signed_model(args, device)
    ann_template = copy.deepcopy(model_template)
    configs = expand_configurations(
        args.configs,
        args.ratios,
        negative_margins=args.negative_margins,
        positive_margins=args.positive_margins,
    )

    results = {name: {} for name in configs}
    layer_results = {name: {} for name in configs}
    for name, config in configs.items():
        for time_steps in args.time_steps:
            print(f"\n{'=' * 72}\n{name} T={time_steps}\n{'=' * 72}")
            summary, per_layer = run_one(
                config,
                model_template,
                ann_template,
                time_steps,
                calibration_batches,
                test_loader,
                device,
                args,
            )
            results[name][time_steps] = summary
            layer_results[name][time_steps] = per_layer
            print(
                f"{name} T={time_steps}: acc={summary['acc']:.2f}% "
                f"sops={summary['sops']:,} "
                f"scale_ops={summary['scale_operations']:,} "
                f"sparsity={format_pct(summary['sparsity'])} "
                f"bias={summary['ftbc_bytes']:,}B "
                f"cal={summary['calibration_elapsed']:.1f}s "
                f"infer={summary['inference_elapsed']:.1f}s"
            )
            write_report(
                args.output,
                args,
                configs,
                results,
                layer_results,
            )

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
