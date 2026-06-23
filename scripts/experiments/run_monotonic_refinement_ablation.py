# 运行单调有符号逐次精化编码的完整消融实验
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
from scripts.experiments.run_stats_ablation import load_signed_model
from spike_stats import (
    collect_signed_spike_stats,
    reset_signed_spike_stats,
    set_signed_spike_stats_enabled,
    summarize_ftbc_storage,
)
from utils import seed_all


CONFIGS = {
    "A_RATE_QCFS": {
        "signed": False,
        "r0": False,
        "coding": "rate",
        "schedule": "uniform",
        "readout": "event_mean",
        "ftbc": "none",
    },
    "B_RATE_SNM_R0": {
        "signed": True,
        "r0": True,
        "coding": "rate",
        "schedule": "uniform",
        "readout": "event_mean",
        "ftbc": "none",
    },
    "C_UNIFORM_REFINEMENT": {
        "signed": True,
        "r0": True,
        "coding": "monotonic_refinement",
        "schedule": "uniform",
        "readout": "event_mean",
        "ftbc": "none",
    },
    "D_READOUT_BINARY": {
        "signed": True,
        "r0": True,
        "coding": "rate",
        "schedule": "uniform",
        "readout": "weighted",
        "readout_schedule": "binary",
        "ftbc": "none",
    },
    "E_BINARY_REFINEMENT": {
        "signed": True,
        "r0": True,
        "coding": "monotonic_refinement",
        "schedule": "binary",
        "readout": "event_mean",
        "ftbc": "none",
    },
    "F_CALIBRATED_REFINEMENT": {
        "signed": True,
        "r0": True,
        "coding": "monotonic_refinement",
        "schedule": "calibrated",
        "readout": "event_mean",
        "ftbc": "none",
    },
    "G_CALIBRATED_FULL_FTBC": {
        "signed": True,
        "r0": True,
        "coding": "monotonic_refinement",
        "schedule": "calibrated",
        "readout": "event_mean",
        "ftbc": "full",
    },
    "H_CALIBRATED_STATE_LR": {
        "signed": True,
        "r0": True,
        "coding": "monotonic_refinement",
        "schedule": "calibrated",
        "readout": "event_mean",
        "ftbc": "state_low_rank",
    },
}


def effective_ftbc_mode(requested_mode, time_steps):
    if requested_mode == "state_low_rank" and int(time_steps) < 3:
        return "full"
    return requested_mode


def select_calibrated_parameters(candidate_metrics, accuracy_tolerance=0.05):
    """Choose the lowest-SOP candidate within tolerance of best accuracy."""
    if not candidate_metrics:
        raise ValueError("candidate_metrics must not be empty")
    best_accuracy = max(item["acc"] for item in candidate_metrics.values())
    eligible = [
        (ratio, item)
        for ratio, item in candidate_metrics.items()
        if item["acc"] >= best_accuracy - float(accuracy_tolerance)
    ]
    return min(
        eligible,
        key=lambda pair: (pair[1]["sops"], -pair[1]["acc"], pair[0]),
    )[0]


def select_calibrated_ratio(candidate_metrics, accuracy_tolerance=0.05):
    return select_calibrated_parameters(
        candidate_metrics,
        accuracy_tolerance=accuracy_tolerance,
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def materialize_batches(loader, count):
    batches = []
    for index, (inputs, targets) in enumerate(loader):
        if index >= count:
            break
        batches.append((inputs.clone(), targets.clone()))
    if len(batches) != count:
        raise RuntimeError(f"Requested {count} batches, found {len(batches)}")
    return batches


def use_deterministic_calibration_transform(train_loader, test_loader):
    """Use training samples with the deterministic evaluation transform."""
    train_loader.dataset.transform = test_loader.dataset.transform


def summarize_layer_stats(layer_stats):
    total_positive = sum(item.positive_spikes for item in layer_stats)
    total_negative = sum(item.negative_spikes for item in layer_stats)
    total_observations = sum(item.total_observations for item in layer_stats)
    total_spikes = total_positive + total_negative
    return {
        "positive_spikes": total_positive,
        "negative_spikes": total_negative,
        "positive_rate": total_positive / max(total_observations, 1),
        "negative_rate": total_negative / max(total_observations, 1),
        "sparsity": 1.0 - total_spikes / max(total_observations, 1),
        "sops": sum(item.sops for item in layer_stats),
        "scale_ops": sum(item.scale_operations for item in layer_stats),
    }


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    correct = 0
    total = 0
    model.eval()
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        outputs = model(inputs.to(device))
        if model.T > 0:
            outputs = model.aggregate_temporal_output(outputs)
        predicted = outputs.argmax(dim=1).cpu()
        total += int(targets.numel())
        correct += int(predicted.eq(targets).sum().item())
    if total == 0:
        raise RuntimeError("evaluation loader produced no samples")
    return 100.0 * correct / total


def configure_model(
    model,
    cfg,
    time_steps,
    ratio,
    positive_margin=0.5,
    negative_margin=0.5,
):
    model.set_signed(cfg["signed"])
    model.set_r0(cfg["r0"])
    model.reset_all_bias()
    model.set_T(time_steps)

    schedule = cfg["schedule"]
    if schedule == "calibrated":
        schedule = "geometric"
    model.set_coding_mode(
        cfg["coding"],
        schedule=schedule,
        ratio=ratio,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
    )
    model.set_readout_mode(
        cfg["readout"],
        schedule=cfg.get("readout_schedule", "uniform"),
        ratio=ratio,
    )
    ftbc_mode = effective_ftbc_mode(cfg["ftbc"], time_steps)
    model.set_ftbc_mode(ftbc_mode)
    return ftbc_mode


def evaluate_with_statistics(model, loader, device, max_batches=0):
    set_signed_spike_stats_enabled(model, SignedIF, True)
    reset_signed_spike_stats(model, SignedIF)
    accuracy = evaluate(model, loader, device, max_batches=max_batches)
    layers = collect_signed_spike_stats(model, SignedIF, nn.Conv2d, nn.Linear)
    return accuracy, summarize_layer_stats(layers), layers


def calibrate_temporal_parameters(
    model_template,
    time_steps,
    ratios,
    positive_margins,
    negative_margins,
    calibration_batches,
    device,
    accuracy_tolerance,
):
    candidate_metrics = {}
    cfg = CONFIGS["F_CALIBRATED_REFINEMENT"]
    for ratio in ratios:
        for positive_margin in positive_margins:
            for negative_margin in negative_margins:
                model = copy.deepcopy(model_template).to(device)
                configure_model(
                    model,
                    cfg,
                    time_steps,
                    ratio,
                    positive_margin=positive_margin,
                    negative_margin=negative_margin,
                )
                accuracy, summary, _ = evaluate_with_statistics(
                    model,
                    calibration_batches,
                    device,
                )
                key = (
                    float(ratio),
                    float(positive_margin),
                    float(negative_margin),
                )
                candidate_metrics[key] = {
                    "acc": accuracy,
                    "sops": summary["sops"],
                    "scale_ops": summary["scale_ops"],
                }
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    selected = select_calibrated_parameters(
        candidate_metrics,
        accuracy_tolerance=accuracy_tolerance,
    )
    return selected, candidate_metrics


def run_one(
    cfg,
    ratio,
    positive_margin,
    negative_margin,
    model_template,
    ann_template,
    time_steps,
    calibration_batches,
    test_loader,
    device,
    args,
):
    model = copy.deepcopy(model_template).to(device)
    ftbc_mode = configure_model(
        model,
        cfg,
        time_steps,
        ratio,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
    )

    calibration_elapsed = 0.0
    if cfg["ftbc"] != "none":
        ann = copy.deepcopy(ann_template).to(device)
        ann.set_T(0)
        ann.set_signed(False)
        ann.set_r0(False)
        ann.set_coding_mode("rate")
        synchronize(device)
        started = time.perf_counter()
        bias_corr_model(
            ann=ann,
            snn=model,
            T=time_steps,
            train_loader=calibration_batches,
            curr_t_alpha=args.alpha,
            num_cali_sample_batches=len(calibration_batches),
            ftbc_mode=ftbc_mode,
            ridge=args.ridge,
            over_weight=args.over_weight,
            under_weight=args.under_weight,
            coefficient_clip=args.coefficient_clip,
        )
        synchronize(device)
        calibration_elapsed = time.perf_counter() - started
        del ann

    accuracy, summary, layers = evaluate_with_statistics(
        model,
        test_loader,
        device,
        max_batches=args.max_test_batches,
    )
    storage = summarize_ftbc_storage(model, SignedIF)

    set_signed_spike_stats_enabled(model, SignedIF, False)
    synchronize(device)
    started = time.perf_counter()
    timed_accuracy = evaluate(
        model,
        test_loader,
        device,
        max_batches=args.max_test_batches,
    )
    synchronize(device)
    inference_elapsed = time.perf_counter() - started
    if timed_accuracy != accuracy:
        raise RuntimeError(
            f"Evaluation changed with statistics disabled: {accuracy} vs {timed_accuracy}"
        )

    summary.update(
        {
            "acc": accuracy,
            "ratio": float(ratio),
            "positive_margin": float(positive_margin),
            "negative_margin": float(negative_margin),
            "ftbc_mode": ftbc_mode,
            "ftbc_parameters": storage["parameters"],
            "ftbc_bytes": storage["bytes"],
            "calibration_elapsed": calibration_elapsed,
            "inference_elapsed": inference_elapsed,
        }
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, layers


def format_pct(value):
    return f"{value * 100:.4f}%"


def write_report(
    path,
    args,
    selected_configs,
    results,
    layer_results,
    selected_parameters,
    parameter_screen,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as report:
        report.write("# Monotonic Signed Successive Refinement Ablation\n\n")
        report.write(f"- Dataset: {args.dataset}\n")
        report.write("- Model: VGG16\n")
        report.write(f"- Time steps: {args.time_steps}\n")
        report.write(f"- Configs: {', '.join(selected_configs)}\n")
        report.write(
            "- Ratio selection uses calibration batches only; the test set is never used "
            "to choose temporal weights.\n"
        )
        report.write(
            "- SOPs count non-zero input events. ScaleOps are reported separately and "
            "are not added to SOPs.\n\n"
        )

        metrics = [
            ("Accuracy", "acc", lambda value: f"{value:.2f}%"),
            ("Input-driven SOPs", "sops", lambda value: f"{value:,}"),
            ("ScaleOps", "scale_ops", lambda value: f"{value:,}"),
            ("Positive spike rate", "positive_rate", format_pct),
            ("Negative spike rate", "negative_rate", format_pct),
            ("Overall spike sparsity", "sparsity", format_pct),
            ("FTBC parameters", "ftbc_parameters", lambda value: f"{value:,}"),
            ("FTBC storage bytes", "ftbc_bytes", lambda value: f"{value:,}"),
            ("Calibration elapsed", "calibration_elapsed", lambda value: f"{value:.2f}s"),
            ("Pure inference elapsed", "inference_elapsed", lambda value: f"{value:.2f}s"),
            ("Selected ratio", "ratio", lambda value: f"{value:g}"),
            ("Positive margin", "positive_margin", lambda value: f"{value:g}"),
            ("Negative margin", "negative_margin", lambda value: f"{value:g}"),
        ]
        for title, key, formatter in metrics:
            report.write(f"## {title}\n\n")
            report.write(
                "| Config | "
                + " | ".join(f"T={value}" for value in args.time_steps)
                + " |\n"
            )
            report.write("|" + "---|" * (len(args.time_steps) + 1) + "\n")
            for name in selected_configs:
                cells = [
                    formatter(results[name][time_steps][key])
                    if time_steps in results[name]
                    else "-"
                    for time_steps in args.time_steps
                ]
                report.write(f"| {name} | " + " | ".join(cells) + " |\n")
            report.write("\n")

        report.write("## Calibration Parameter Screen\n\n")
        for time_steps in args.time_steps:
            if time_steps not in parameter_screen:
                continue
            selected = selected_parameters[time_steps]
            report.write(
                f"### T={time_steps}, selected=(ratio={selected[0]:g}, "
                f"positive_margin={selected[1]:g}, negative_margin={selected[2]:g})\n\n"
            )
            report.write(
                "| Ratio | Positive margin | Negative margin | "
                "Calibration Accuracy | SOPs | ScaleOps |\n"
            )
            report.write("|---:|---:|---:|---:|---:|---:|\n")
            for params, item in sorted(parameter_screen[time_steps].items()):
                report.write(
                    f"| {params[0]:g} | {params[1]:g} | {params[2]:g} | "
                    f"{item['acc']:.2f}% | "
                    f"{item['sops']:,} | {item['scale_ops']:,} |\n"
                )
            report.write("\n")

        report.write("## Per-layer Detail\n\n")
        for name in selected_configs:
            for time_steps in args.time_steps:
                if time_steps not in layer_results[name]:
                    continue
                report.write(f"### {name}, T={time_steps}\n\n")
                report.write(
                    "| Layer | PosRate | NegRate | Sparsity | InputSpikes | "
                    "SOPs | ScaleOps | PosByTime | NegByTime |\n"
                )
                report.write("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
                for item in layer_results[name][time_steps]:
                    report.write(
                        f"| {item.name} | {format_pct(item.positive_spike_rate)} | "
                        f"{format_pct(item.negative_spike_rate)} | "
                        f"{format_pct(item.spike_sparsity)} | "
                        f"{item.total_input_spikes:,} | {item.sops:,} | "
                        f"{item.scale_operations:,} | "
                        f"{list(item.positive_spikes_by_time)} | "
                        f"{list(item.negative_spikes_by_time)} |\n"
                    )
                report.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Monotonic signed successive refinement ablation"
    )
    parser.add_argument("-data", "--dataset", default="cifar100")
    parser.add_argument("-id", "--identifier", required=True)
    parser.add_argument("-dev", "--device", default="0")
    parser.add_argument("-b", "--batch_size", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--time_steps", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--configs", nargs="+", choices=CONFIGS.keys(), default=list(CONFIGS))
    parser.add_argument("--ratio_candidates", nargs="+", type=float, default=[1.0, 1.05, 1.1, 1.25, 1.5, 2.0])
    parser.add_argument("--positive_margin_candidates", nargs="+", type=float, default=[0.5, 0.55])
    parser.add_argument("--negative_margin_candidates", nargs="+", type=float, default=[0.5, 1.0, 1.3])
    parser.add_argument("--accuracy_tolerance", default=0.5, type=float)
    parser.add_argument("--cali_batches", default=5, type=int)
    parser.add_argument("--max_test_batches", default=0, type=int)
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("--ridge", default=1e-3, type=float)
    parser.add_argument("--over_weight", default=1.0, type=float)
    parser.add_argument("--under_weight", default=1.0, type=float)
    parser.add_argument("--coefficient_clip", default=0.25, type=float)
    parser.add_argument(
        "--output",
        default="docs/results/monotonic_refinement/MSSR_ABLATION_cifar100.md",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    use_deterministic_calibration_transform(train_loader, test_loader)
    calibration_batches = materialize_batches(train_loader, args.cali_batches)
    model_template = load_signed_model(args, device)
    ann_template = copy.deepcopy(model_template)

    needs_calibrated_ratio = any(
        CONFIGS[name]["schedule"] == "calibrated" for name in args.configs
    )
    selected_parameters = {}
    parameter_screen = {}
    if needs_calibrated_ratio:
        for time_steps in args.time_steps:
            if time_steps == 1:
                selected_parameters[time_steps] = (1.0, 0.5, 0.5)
                parameter_screen[time_steps] = {
                    (1.0, 0.5, 0.5): {"acc": 0.0, "sops": 0, "scale_ops": 0}
                }
                continue
            selected, candidates = calibrate_temporal_parameters(
                model_template=model_template,
                time_steps=time_steps,
                ratios=args.ratio_candidates,
                positive_margins=args.positive_margin_candidates,
                negative_margins=args.negative_margin_candidates,
                calibration_batches=calibration_batches,
                device=device,
                accuracy_tolerance=args.accuracy_tolerance,
            )
            selected_parameters[time_steps] = selected
            parameter_screen[time_steps] = candidates

    results = {name: {} for name in args.configs}
    layer_results = {name: {} for name in args.configs}
    for name in args.configs:
        cfg = CONFIGS[name]
        for time_steps in args.time_steps:
            if cfg["schedule"] == "calibrated":
                ratio, positive_margin, negative_margin = selected_parameters[time_steps]
            elif cfg["schedule"] == "binary" or cfg.get("readout_schedule") == "binary":
                ratio = 2.0
                positive_margin = 0.5
                negative_margin = 0.5
            else:
                ratio = 1.0
                positive_margin = 0.5
                negative_margin = 0.5
            print(
                f"\n{'=' * 72}\n{name} T={time_steps} ratio={ratio:g} "
                f"p={positive_margin:g} n={negative_margin:g}\n{'=' * 72}"
            )
            summary, layers = run_one(
                cfg=cfg,
                ratio=ratio,
                positive_margin=positive_margin,
                negative_margin=negative_margin,
                model_template=model_template,
                ann_template=ann_template,
                time_steps=time_steps,
                calibration_batches=calibration_batches,
                test_loader=test_loader,
                device=device,
                args=args,
            )
            results[name][time_steps] = summary
            layer_results[name][time_steps] = layers
            print(
                f"{name} T={time_steps}: acc={summary['acc']:.2f}% "
                f"sops={summary['sops']:,} scale_ops={summary['scale_ops']:,} "
                f"sparsity={format_pct(summary['sparsity'])} "
                f"infer={summary['inference_elapsed']:.2f}s"
            )
            write_report(
                args.output,
                args,
                args.configs,
                results,
                layer_results,
                selected_parameters,
                parameter_screen,
            )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
