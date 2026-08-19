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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from calibration import bias_corr_model
from horizon_gate import aggregate_subset_metrics, select_robust_candidate
from models.layer import SignedIF
from preprocess import datapool
from scripts.experiments.run_state_ftbc_ablation import (
    materialize_calibration_batches,
)
from scripts.experiments.run_temporal_lr_gated_snm import (
    DEFAULT_CHECKPOINTS,
    RESNET20_CHECKPOINT_SHA256,
    architecture_output,
    batches_sha256,
    configure_base_snn,
    evaluate_calibration_validation,
    evaluate_test,
    load_qcfs_pair,
    make_deployment_compressed,
    synchronize,
)
from spike_stats import set_signed_spike_stats_enabled
from temporal_lr import gate_groups, set_group_margins
from utils import seed_all, val


MODES = ("off", "standard", "stage_gated")


def fixed_disjoint_protocol_batches(train_loader, test_loader, args):
    fit_count = args.fit_batches * args.batch_size
    validation_count = args.validation_batches * args.batch_size
    required = fit_count + args.validation_subsets * validation_count
    if required > len(train_loader.dataset):
        raise ValueError("Requested fixed subsets exceed the CIFAR-100 training set")

    generator = torch.Generator().manual_seed(args.protocol_seed)
    permutation = torch.randperm(len(train_loader.dataset), generator=generator).tolist()
    fit_indices = permutation[:fit_count]
    validation_indices = []
    cursor = fit_count
    for _ in range(args.validation_subsets):
        validation_indices.append(permutation[cursor : cursor + validation_count])
        cursor += validation_count

    fit_loader = DataLoader(
        Subset(train_loader.dataset, fit_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    fit_batches = materialize_calibration_batches(fit_loader, args.fit_batches)

    evaluation_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            *test_loader.dataset.transform.transforms,
        ]
    )
    evaluation_train = datasets.CIFAR100(
        root=train_loader.dataset.root,
        train=True,
        transform=evaluation_transform,
        download=False,
    )
    validation_subsets = []
    for indices in validation_indices:
        loader = DataLoader(
            Subset(evaluation_train, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        validation_subsets.append(
            materialize_calibration_batches(loader, args.validation_batches)
        )
    return fit_batches, validation_subsets, {
        "fit_indices": fit_indices,
        "validation_indices": validation_indices,
    }


def evaluate_validation_subsets(
    model,
    ann,
    validation_subsets,
    device,
    time_steps,
    architecture,
    std_weight,
):
    subset_metrics = [
        evaluate_calibration_validation(
            model,
            ann,
            batches,
            device,
            time_steps,
            architecture,
        )
        for batches in validation_subsets
    ]
    return aggregate_subset_metrics(subset_metrics, std_weight=std_weight)


def zero_margins(model, architecture):
    return OrderedDict((name, 0.0) for name in gate_groups(model, architecture))


def configure_horizon_mode(model, architecture, mode, stage_margins):
    if mode not in MODES:
        raise ValueError(mode)
    margins = zero_margins(model, architecture)
    if mode == "stage_gated":
        margins.update((name, float(value)) for name, value in stage_margins.items())
    set_group_margins(model, architecture, margins)
    model.set_signed(mode != "off")
    return margins


def select_stage_margins_multi(
    model,
    ann,
    validation_subsets,
    device,
    time_steps,
    architecture,
    args,
):
    selected = zero_margins(model, architecture)
    model.set_signed(True)
    set_group_margins(model, architecture, selected)
    trace = OrderedDict()
    for group_name in reversed(tuple(selected)):
        trials = OrderedDict()
        for candidate in args.margin_candidates:
            proposed = OrderedDict(selected)
            proposed[group_name] = float(candidate)
            set_group_margins(model, architecture, proposed)
            trials[str(float(candidate))] = evaluate_validation_subsets(
                model,
                ann,
                validation_subsets,
                device,
                time_steps,
                architecture,
                args.validation_std_weight,
            )
        winner, scored = select_robust_candidate(
            trials,
            accuracy_tolerance=args.accuracy_tolerance,
            sop_weight=args.sop_weight,
            negative_weight=args.negative_weight,
        )
        selected[group_name] = float(winner)
        set_group_margins(model, architecture, selected)
        trace[group_name] = scored
    final = evaluate_validation_subsets(
        model,
        ann,
        validation_subsets,
        device,
        time_steps,
        architecture,
        args.validation_std_weight,
    )
    return selected, final, trace


def evaluate_family(
    base_model,
    ann,
    validation_subsets,
    evaluation_loader,
    device,
    time_steps,
    architecture,
    args,
):
    stage_model = copy.deepcopy(base_model)
    stage_margins, _, stage_trace = select_stage_margins_multi(
        stage_model,
        ann,
        validation_subsets,
        device,
        time_steps,
        architecture,
        args,
    )
    del stage_model

    validation = OrderedDict()
    for mode in MODES:
        model = copy.deepcopy(base_model)
        configure_horizon_mode(model, architecture, mode, stage_margins)
        validation[mode] = evaluate_validation_subsets(
            model,
            ann,
            validation_subsets,
            device,
            time_steps,
            architecture,
            args.validation_std_weight,
        )
        del model
    selected_mode, horizon_trace = select_robust_candidate(
        validation,
        accuracy_tolerance=args.accuracy_tolerance,
        sop_weight=args.sop_weight,
        negative_weight=args.negative_weight,
    )
    if selected_mode == "stage_gated" and all(
        float(value) == 0.0 for value in stage_margins.values()
    ):
        selected_mode = "standard"

    test_results = OrderedDict()
    layers = OrderedDict()
    for mode in MODES:
        model = copy.deepcopy(base_model)
        applied_margins = configure_horizon_mode(
            model,
            architecture,
            mode,
            stage_margins,
        )
        metrics, per_layer = evaluate_test(
            model,
            evaluation_loader,
            device,
            time_steps,
            architecture,
        )
        metrics.update(
            {
                "selected_by_validation": mode == selected_mode,
                "horizon_mode": mode,
                "stage_margins": applied_margins,
                "horizon_gate_parameters": 1,
                "stage_gate_parameters": len(applied_margins)
                if mode == "stage_gated"
                else 0,
            }
        )
        test_results[mode] = metrics
        layers[mode] = per_layer
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "selected_mode": selected_mode,
        "stage_margins": stage_margins,
        "validation": validation,
        "horizon_trace": horizon_trace,
        "stage_trace": stage_trace,
        "test": test_results,
        "layers": layers,
    }


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_metric_table(lines, title, payload, key, formatter):
    times = payload["protocol"]["time_steps"]
    families = payload["protocol"]["families"]
    lines.extend(
        [
            f"## {title}",
            "",
            "| Family / mode | " + " | ".join(f"T={t}" for t in times) + " |",
            "|---|" + "---:|" * len(times),
        ]
    )
    for family in families:
        for mode in MODES:
            values = []
            for time_steps in times:
                item = payload["families"].get(family, {}).get(str(time_steps))
                values.append("-" if item is None else formatter(item["test"][mode][key]))
            lines.append(f"| {family}/{mode} | " + " | ".join(values) + " |")
    lines.append("")


def write_report(path, payload):
    protocol = payload["protocol"]
    lines = [
        "# Robust Horizon-Gated SNM Validation",
        "",
        f"- Architecture: {protocol['architecture']}",
        f"- Checkpoint: {protocol['checkpoint']['path']}",
        f"- Checkpoint SHA256: `{protocol['checkpoint']['sha256']}`",
        f"- ANN accuracy: {protocol['ann_accuracy']:.2f}%",
        f"- Time steps: {protocol['time_steps']}",
        f"- Fit set: {protocol['fit_batches']} x {protocol['batch_size']}, original calibration augmentation.",
        f"- Validation: {protocol['validation_subsets']} disjoint subsets x "
        f"{protocol['validation_batches']} x {protocol['batch_size']}, mild crop/flip without AutoAugment or Cutout.",
        f"- Fit SHA256: `{protocol['fit_sha256']}`",
        f"- Validation SHA256: {protocol['validation_sha256']}",
        "- Test set is evaluated only after the horizon mode is selected.",
        "- Robust accuracy = validation mean - "
        f"{protocol['validation_std_weight']} x subset standard deviation.",
        "- Within the accuracy tolerance, ANN-SNN logit MSE is minimized before event overhead.",
        "",
        "## Validation-selected horizon mode",
        "",
        "| Family | T | Selected | Off val. | Standard val. | Stage val. | Stage margins |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for family in protocol["families"]:
        for time_steps in protocol["time_steps"]:
            item = payload["families"].get(family, {}).get(str(time_steps))
            if item is None:
                continue
            val_items = item["validation"]
            fmt = lambda name: (
                f"{val_items[name]['mean_acc']:.2f}+/-{val_items[name]['std_acc']:.2f}"
            )
            margins = ", ".join(
                f"{name}={value:g}" for name, value in item["stage_margins"].items()
            )
            lines.append(
                f"| {family} | {time_steps} | {item['selected_mode']} | "
                f"{fmt('off')} | {fmt('standard')} | {fmt('stage_gated')} | {margins} |"
            )
    lines.append("")
    format_metric_table(lines, "Test accuracy", payload, "acc", lambda x: f"{x:.2f}%")
    format_metric_table(
        lines,
        "Negative spike rate",
        payload,
        "negative_rate",
        lambda x: f"{100*x:.4f}%",
    )
    format_metric_table(lines, "Input-driven SOPs", payload, "sops", lambda x: f"{int(x):,}")
    format_metric_table(lines, "FTBC storage bytes", payload, "ftbc_bytes", lambda x: f"{int(x):,}")
    format_metric_table(
        lines,
        "Inference elapsed",
        payload,
        "inference_elapsed",
        lambda x: f"{x:.1f}s",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_architecture(
    args,
    architecture,
    checkpoint_path,
    fit_batches,
    validation_subsets,
    evaluation_loader,
    device,
):
    ann_template, snn_template, checkpoint = load_qcfs_pair(
        checkpoint_path,
        "cifar100",
        architecture,
        device,
    )
    if architecture == "resnet20" and checkpoint["sha256"] != RESNET20_CHECKPOINT_SHA256:
        raise RuntimeError("ResNet20 experiment must use the selected 68.78% checkpoint")
    ann_template.set_T(0)
    if hasattr(ann_template, "set_L"):
        ann_template.set_L(8)
    if architecture == "resnet20" and hasattr(ann_template, "set_qcfs_training_profile"):
        ann_template.set_qcfs_training_profile("paper_era")
    set_signed_spike_stats_enabled(snn_template, SignedIF, False)
    ann_accuracy = val(ann_template, evaluation_loader, device, 0)

    families = ["temporal"]
    if architecture == "resnet20":
        families.append("hybrid")
    output = architecture_output(
        args.output,
        architecture,
        multiple=len(args.architectures) > 1,
    )
    progress_path = output.with_suffix(".progress.json")
    payload = {
        "protocol": {
            "architecture": architecture,
            "checkpoint": checkpoint,
            "ann_accuracy": ann_accuracy,
            "time_steps": list(args.time_steps),
            "families": families,
            "batch_size": args.batch_size,
            "fit_batches": args.fit_batches,
            "validation_subsets": args.validation_subsets,
            "validation_batches": args.validation_batches,
            "validation_std_weight": args.validation_std_weight,
            "fit_sha256": batches_sha256(fit_batches),
            "validation_sha256": [batches_sha256(item) for item in validation_subsets],
            "margin_candidates": list(args.margin_candidates),
            "accuracy_tolerance": args.accuracy_tolerance,
            "protocol_seed": args.protocol_seed,
        },
        "families": OrderedDict((name, OrderedDict()) for name in families),
    }

    for time_steps in args.time_steps:
        print(f"[{architecture}] T={time_steps}: unsigned Full teacher", flush=True)
        teacher = copy.deepcopy(snn_template).to(device)
        configure_base_snn(teacher, time_steps, signed=False, ftbc_mode="full")
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

        for family in families:
            print(f"[{architecture}] T={time_steps}: robust {family} horizon gate", flush=True)
            base_model, compression = make_deployment_compressed(
                teacher,
                rank=4,
                architecture=architecture,
                time_steps=time_steps,
                hybrid=family == "hybrid",
            )
            family_result = evaluate_family(
                base_model,
                ann,
                validation_subsets,
                evaluation_loader,
                device,
                time_steps,
                architecture,
                args,
            )
            family_result["calibration_elapsed"] = calibration_elapsed
            family_result["compression"] = compression
            payload["families"][family][str(time_steps)] = family_result
            save_json(progress_path, payload)
            write_report(output, payload)
            del base_model
        del ann, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_json(progress_path, payload)
    write_report(output, payload)
    return output


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("resnet20", "vgg16"),
        default=("resnet20", "vgg16"),
    )
    parser.add_argument("--resnet20_checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["resnet20"])
    parser.add_argument("--vgg16_checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["vgg16"])
    parser.add_argument("--time_steps", nargs="+", type=int, default=(4, 8, 16, 32))
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--fit_batches", type=int, default=5)
    parser.add_argument("--validation_subsets", type=int, default=3)
    parser.add_argument("--validation_batches", type=int, default=5)
    parser.add_argument("--test_batches", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument(
        "--margin_candidates",
        nargs="+",
        type=float,
        default=(0.0, 0.25, 0.5, 1.0, 2.0),
    )
    parser.add_argument("--validation_std_weight", type=float, default=0.5)
    parser.add_argument("--accuracy_tolerance", type=float, default=0.1)
    parser.add_argument("--sop_weight", type=float, default=0.05)
    parser.add_argument("--negative_weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol_seed", type=int, default=20260817)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/results/comparative_ablation/cifar100/"
            "ROBUST_HORIZON_GATE_CIFAR100.md"
        ),
    )
    return parser


def main(cli_args=None):
    args = build_parser().parse_args(cli_args)
    if args.fit_batches <= 0 or args.validation_batches <= 0:
        raise ValueError("Fit and validation batches must be positive")
    if args.validation_subsets < 2:
        raise ValueError("At least two disjoint validation subsets are required")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    train_loader, test_loader = datapool("cifar100", args.batch_size)
    fit_batches, validation_subsets, _ = fixed_disjoint_protocol_batches(
        train_loader,
        test_loader,
        args,
    )
    if args.test_batches > 0:
        evaluation_loader = materialize_calibration_batches(
            test_loader,
            args.test_batches,
        )
    else:
        evaluation_loader = test_loader
    checkpoints = {
        "resnet20": args.resnet20_checkpoint,
        "vgg16": args.vgg16_checkpoint,
    }
    outputs = []
    for architecture in args.architectures:
        outputs.append(
            run_architecture(
                args,
                architecture,
                checkpoints[architecture],
                fit_batches,
                validation_subsets,
                evaluation_loader,
                device,
            )
        )
    for output in outputs:
        print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
