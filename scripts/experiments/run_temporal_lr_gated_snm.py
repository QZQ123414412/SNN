"""CIFAR-100 evaluation of teacher-compressed Temporal-LR plus gated SNM."""

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
    collect_resnet20_spike_stats,
    collect_signed_spike_stats,
    reset_signed_spike_stats,
    set_signed_spike_stats_enabled,
    summarize_ftbc_storage,
)
from temporal_lr import (
    compress_full_ftbc_teacher,
    gate_groups,
    set_group_margins,
    snm_runtime_state_bytes_per_sample,
)
from utils import seed_all, val


DEFAULT_CHECKPOINTS = {
    "resnet20": Path(
        "cifar100-checkpoints/"
        "resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth"
    ),
    "vgg16": Path("cifar100-checkpoints/cifar100-vgg16-l8-example.pth"),
}
RESNET20_CHECKPOINT_SHA256 = (
    "1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2"
)
CONFIGS = OrderedDict(
    [
        ("A_QCFS_R0", "plain_unsigned"),
        ("B_QCFS_STANDARD_SNM_R0", "plain_signed"),
        ("C_FULL_UNSIGNED_TEACHER", "full_unsigned"),
        ("D_FULL_STANDARD_SNM_AFTER_UNSIGNED", "full_signed"),
        ("E_TEMPORAL_R4_UNSIGNED", "temporal_unsigned"),
        ("F_TEMPORAL_R4_STANDARD_SNM", "temporal_signed"),
        ("G_TEMPORAL_R4_GATED_SNM", "temporal_gated"),
        ("H_HYBRID_R4_UNSIGNED", "hybrid_unsigned"),
        ("I_HYBRID_R4_GATED_SNM", "hybrid_gated"),
    ]
)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batches_sha256(batches):
    digest = hashlib.sha256()
    for inputs, targets in batches:
        for tensor in (inputs, targets):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def configure_base_snn(model, time_steps, signed=False, ftbc_mode="none"):
    model.set_T(time_steps)
    if hasattr(model, "set_coding_mode"):
        model.set_coding_mode("rate", schedule="rate", ratio=1.0)
    model.set_signed(signed)
    model.set_r0(True)
    model.reset_all_bias()
    model.set_ftbc_mode(ftbc_mode)
    if hasattr(model, "set_snm_negative_margin"):
        model.set_snm_negative_margin(0.0)
    return model


def collect_architecture_stats(model, architecture):
    if architecture == "resnet20":
        return collect_resnet20_spike_stats(model, SignedIF, nn.Conv2d)
    return collect_signed_spike_stats(model, SignedIF, nn.Conv2d, nn.Linear)


def evaluate_test(model, loader, device, time_steps, architecture):
    storage = summarize_ftbc_storage(model, SignedIF)
    set_signed_spike_stats_enabled(model, SignedIF, True)
    reset_signed_spike_stats(model, SignedIF)
    accuracy = val(model, loader, device, time_steps)
    layer_stats = collect_architecture_stats(model, architecture)
    summary = summarize_layer_stats(layer_stats)
    state_bytes = snm_runtime_state_bytes_per_sample(model)

    set_signed_spike_stats_enabled(model, SignedIF, False)
    synchronize(device)
    started = time.perf_counter()
    timed_accuracy = val(model, loader, device, time_steps)
    synchronize(device)
    inference_elapsed = time.perf_counter() - started
    if timed_accuracy != accuracy:
        raise RuntimeError(
            f"Non-deterministic test accuracy: {accuracy} vs {timed_accuracy}"
        )

    summary.update(
        {
            "acc": accuracy,
            "ftbc_parameters": storage["parameters"],
            "ftbc_bytes": storage["bytes"],
            "ftbc_synthesis_macs": storage["synthesis_macs"],
            "snm_runtime_state_bytes_per_sample": state_bytes,
            "inference_elapsed": inference_elapsed,
        }
    )
    return summary, [asdict(item) for item in layer_stats]


@torch.no_grad()
def evaluate_calibration_validation(
    model,
    ann,
    batches,
    device,
    time_steps,
    architecture,
):
    ann.eval()
    model.eval()
    set_signed_spike_stats_enabled(model, SignedIF, True)
    reset_signed_spike_stats(model, SignedIF)
    correct = 0
    total = 0
    squared_error = 0.0
    logit_values = 0
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
    layer_stats = collect_architecture_stats(model, architecture)
    summary = summarize_layer_stats(layer_stats)
    set_signed_spike_stats_enabled(model, SignedIF, False)
    return {
        "acc": 100.0 * correct / max(total, 1),
        "logit_mse": squared_error / max(logit_values, 1),
        "sops": int(summary["sops"]),
        "positive_spikes": int(summary["positive_spikes"]),
        "negative_spikes": int(summary["negative_spikes"]),
    }


def gate_score(metrics, baseline, sop_weight, negative_weight):
    epsilon = 1e-12
    mse_ratio = metrics["logit_mse"] / max(baseline["logit_mse"], epsilon)
    sop_ratio = metrics["sops"] / max(baseline["sops"], 1)
    if baseline["negative_spikes"] > 0:
        negative_ratio = (
            metrics["negative_spikes"] / baseline["negative_spikes"]
        )
    else:
        negative_ratio = 0.0 if metrics["negative_spikes"] == 0 else 1.0
    return (
        mse_ratio
        + float(sop_weight) * sop_ratio
        + float(negative_weight) * negative_ratio
    )


def select_gate_margins(
    model,
    ann,
    validation_batches,
    device,
    time_steps,
    architecture,
    candidates,
    sop_weight,
    negative_weight,
    accuracy_tolerance,
):
    groups = gate_groups(model, architecture)
    selected = OrderedDict((name, 0.0) for name in groups)
    model.set_signed(True)
    set_group_margins(model, architecture, selected)
    baseline = evaluate_calibration_validation(
        model,
        ann,
        validation_batches,
        device,
        time_steps,
        architecture,
    )
    trace = OrderedDict()
    for group_name in reversed(tuple(groups)):
        trials = []
        for candidate in candidates:
            proposed = OrderedDict(selected)
            proposed[group_name] = float(candidate)
            set_group_margins(model, architecture, proposed)
            metrics = evaluate_calibration_validation(
                model,
                ann,
                validation_batches,
                device,
                time_steps,
                architecture,
            )
            score = gate_score(
                metrics,
                baseline,
                sop_weight=sop_weight,
                negative_weight=negative_weight,
            )
            trials.append(
                {
                    "candidate": float(candidate),
                    "score": float(score),
                    **metrics,
                }
            )
        best_accuracy = max(item["acc"] for item in trials)
        accuracy_eligible = [
            item
            for item in trials
            if item["acc"] >= best_accuracy - float(accuracy_tolerance)
        ]
        winner = min(
            accuracy_eligible,
            key=lambda item: (item["score"], -item["acc"], item["sops"]),
        )
        selected[group_name] = winner["candidate"]
        set_group_margins(model, architecture, selected)
        trace[group_name] = trials
    final = evaluate_calibration_validation(
        model,
        ann,
        validation_batches,
        device,
        time_steps,
        architecture,
    )
    return selected, baseline, final, trace


def hybrid_full_layers(architecture):
    if architecture == "resnet20":
        return ("conv4_x.2.act",)
    if architecture == "vgg16":
        return ("classifier.5",)
    raise ValueError(architecture)


def make_compressed(teacher, rank, architecture, hybrid=False):
    model = copy.deepcopy(teacher)
    full_names = hybrid_full_layers(architecture) if hybrid else ()
    compression = compress_full_ftbc_teacher(
        model,
        rank=rank,
        full_layer_names=full_names,
    )
    return model, compression


def make_deployment_compressed(
    teacher,
    rank,
    architecture,
    time_steps,
    hybrid=False,
):
    """Use Full-FTBC at T<=4, where rank-4 has no storage advantage."""
    if int(time_steps) <= 4:
        return copy.deepcopy(teacher), {
            "fallback_to_full": True,
            "reason": "T<=4 has no rank-4 storage advantage",
            "effective_rank": int(time_steps),
            "full_layer_names": [],
            "compression_elapsed": 0.0,
        }
    started = time.perf_counter()
    model, compression = make_compressed(
        teacher,
        rank=rank,
        architecture=architecture,
        hybrid=hybrid,
    )
    compression["fallback_to_full"] = False
    compression["compression_elapsed"] = time.perf_counter() - started
    return model, compression


def validation_rank_screen(
    teacher,
    ann,
    validation_batches,
    device,
    time_steps,
    architecture,
    ranks,
):
    screen = OrderedDict()
    for rank in ranks:
        model, compression = make_compressed(
            teacher,
            rank=rank,
            architecture=architecture,
            hybrid=False,
        )
        model.set_signed(False)
        metrics = evaluate_calibration_validation(
            model,
            ann,
            validation_batches,
            device,
            time_steps,
            architecture,
        )
        storage = summarize_ftbc_storage(model, SignedIF)
        screen[str(rank)] = {
            **metrics,
            "effective_rank": compression["effective_rank"],
            "explained_energy": compression["explained_energy"],
            "ftbc_parameters": storage["parameters"],
            "ftbc_bytes": storage["bytes"],
            "ftbc_synthesis_macs": storage["synthesis_macs"],
        }
        del model
    return screen


def build_variant(
    kind,
    snn_template,
    teacher,
    time_steps,
    architecture,
    rank,
    selected_margins,
):
    compression = None
    if kind.startswith("plain"):
        model = copy.deepcopy(snn_template)
        configure_base_snn(
            model,
            time_steps=time_steps,
            signed=kind == "plain_signed",
            ftbc_mode="none",
        )
    elif kind.startswith("full"):
        model = copy.deepcopy(teacher)
        model.set_signed(kind == "full_signed")
    elif kind.startswith("temporal"):
        model, compression = make_deployment_compressed(
            teacher,
            rank=rank,
            architecture=architecture,
            time_steps=time_steps,
            hybrid=False,
        )
        model.set_signed(kind != "temporal_unsigned")
        if kind == "temporal_gated":
            set_group_margins(model, architecture, selected_margins)
    elif kind.startswith("hybrid"):
        model, compression = make_deployment_compressed(
            teacher,
            rank=rank,
            architecture=architecture,
            time_steps=time_steps,
            hybrid=True,
        )
        model.set_signed(kind == "hybrid_gated")
        if kind == "hybrid_gated":
            set_group_margins(model, architecture, selected_margins)
    else:
        raise ValueError(kind)
    return model, compression


def save_progress(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def format_table(title, configs, time_steps, results, key, formatter):
    lines = [f"## {title}", ""]
    lines.append("| Config | " + " | ".join(f"T={t}" for t in time_steps) + " |")
    lines.append("|---|" + "---:|" * len(time_steps))
    for config in configs:
        values = []
        for t in time_steps:
            value = results.get(config, {}).get(str(t), {}).get(key)
            values.append("-" if value is None else formatter(value))
        lines.append(f"| {config} | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def write_report(path, payload):
    protocol = payload["protocol"]
    time_steps = protocol["time_steps"]
    configs = list(CONFIGS)
    results = payload["results"]
    lines = ["# QCFS Temporal-LR + Gated-SNM CIFAR-100 Experiment", ""]
    lines.extend(
        [
            f"- Architecture: {protocol['architecture']}",
            f"- Checkpoint: {protocol['checkpoint']['filename']}",
            f"- Checkpoint SHA256: `{protocol['checkpoint']['sha256']}`",
            f"- ANN accuracy: {protocol['ann_accuracy']:.2f}%",
            f"- Time steps: {time_steps}",
            f"- Calibration: {protocol['fit_batches']} x {protocol['batch_size']}",
            f"- Gate validation: {protocol['validation_batches']} x {protocol['batch_size']}",
            f"- Fit data SHA256: `{protocol['fit_sha256']}`",
            f"- Validation data SHA256: `{protocol['validation_sha256']}`",
            "- SNM is disabled for Full-FTBC teacher calibration.",
            "- Rank and gate margins use calibration-validation data only.",
            "- Runtime state is the existing R0 membrane plus transmitted-credit state; "
            "Gated-SNM adds only four FP32 margins (16 bytes), not another dense state.",
            "",
        ]
    )
    lines += format_table("Accuracy", configs, time_steps, results, "acc", lambda x: f"{x:.2f}%")
    lines += format_table("Input-driven SOPs", configs, time_steps, results, "sops", lambda x: f"{int(x):,}")
    lines += format_table("Positive spike rate", configs, time_steps, results, "positive_rate", lambda x: f"{100*x:.4f}%")
    lines += format_table("Negative spike rate", configs, time_steps, results, "negative_rate", lambda x: f"{100*x:.4f}%")
    lines += format_table("Overall sparsity", configs, time_steps, results, "sparsity", lambda x: f"{100*x:.4f}%")
    lines += format_table("FTBC parameters", configs, time_steps, results, "ftbc_parameters", lambda x: f"{int(x):,}")
    lines += format_table("FTBC storage bytes", configs, time_steps, results, "ftbc_bytes", lambda x: f"{int(x):,}")
    lines += format_table("Temporal bias synthesis MACs", configs, time_steps, results, "ftbc_synthesis_macs", lambda x: f"{int(x):,}")
    lines += format_table("SNM gate parameters", configs, time_steps, results, "snm_gate_parameters", lambda x: f"{int(x):,}")
    lines += format_table("R0 neuron runtime state bytes per sample", configs, time_steps, results, "snm_runtime_state_bytes_per_sample", lambda x: f"{int(x):,}")
    lines += format_table("Full-teacher calibration elapsed", configs, time_steps, results, "calibration_elapsed", lambda x: f"{x:.1f}s")
    lines += format_table("Temporal compression elapsed", configs, time_steps, results, "compression_elapsed", lambda x: f"{x:.3f}s")
    lines += format_table("Inference elapsed", configs, time_steps, results, "inference_elapsed", lambda x: f"{x:.1f}s")

    lines.extend(["## Rank screen on calibration validation", ""])
    for t in time_steps:
        lines.extend(
            [
                f"### T={t}",
                "",
                "| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, item in payload.get("rank_screens", {}).get(str(t), {}).items():
            lines.append(
                f"| {rank} | {item['effective_rank']} | {item['acc']:.2f}% | "
                f"{item['logit_mse']:.6f} | {item['explained_energy']:.6f} | "
                f"{item['ftbc_bytes']:,} |"
            )
        lines.append("")

    lines.extend(["## Selected SNM margins", ""])
    lines.append("| T | Early | Middle | Late | Final | Baseline val acc. | Gated val acc. |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for t in time_steps:
        gate = payload.get("gates", {}).get(str(t))
        if gate is None:
            continue
        margins = gate["selected_margins"]
        lines.append(
            f"| {t} | {margins['early']} | {margins['middle']} | "
            f"{margins['late']} | {margins['final']} | "
            f"{gate['baseline']['acc']:.2f}% | {gate['final']['acc']:.2f}% |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def architecture_output(base_output, architecture, multiple):
    if not multiple:
        return Path(base_output)
    path = Path(base_output)
    return path.with_name(f"{path.stem}_{architecture}{path.suffix}")


def run_architecture(
    args,
    architecture,
    checkpoint_path,
    fit_batches,
    validation_batches,
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
            "batch_size": args.batch_size,
            "fit_batches": args.fit_batches,
            "validation_batches": args.validation_batches,
            "fit_sha256": batches_sha256(fit_batches),
            "validation_sha256": batches_sha256(validation_batches),
            "main_rank": args.main_rank,
            "rank_candidates": list(args.rank_candidates),
            "margin_candidates": list(args.margin_candidates),
            "gate_sop_weight": args.gate_sop_weight,
            "gate_negative_weight": args.gate_negative_weight,
            "gate_accuracy_tolerance": args.gate_accuracy_tolerance,
            "test_batches": args.test_batches,
        },
        "results": OrderedDict((name, OrderedDict()) for name in CONFIGS),
        "rank_screens": OrderedDict(),
        "gates": OrderedDict(),
        "compression": OrderedDict(),
        "layers": OrderedDict(),
    }

    for time_steps in args.time_steps:
        print(f"[{architecture}] T={time_steps}: unsigned Full-FTBC teacher", flush=True)
        teacher = copy.deepcopy(snn_template).to(device)
        configure_base_snn(
            teacher,
            time_steps=time_steps,
            signed=False,
            ftbc_mode="full",
        )
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

        payload["rank_screens"][str(time_steps)] = validation_rank_screen(
            teacher,
            ann,
            validation_batches,
            device,
            time_steps,
            architecture,
            args.rank_candidates,
        )

        gated_model, gate_compression = make_deployment_compressed(
            teacher,
            rank=args.main_rank,
            architecture=architecture,
            time_steps=time_steps,
            hybrid=False,
        )
        selected, baseline, final, trace = select_gate_margins(
            gated_model,
            ann,
            validation_batches,
            device,
            time_steps,
            architecture,
            args.margin_candidates,
            args.gate_sop_weight,
            args.gate_negative_weight,
            args.gate_accuracy_tolerance,
        )
        payload["gates"][str(time_steps)] = {
            "selected_margins": selected,
            "baseline": baseline,
            "final": final,
            "trace": trace,
        }
        del gated_model

        for config_name, kind in CONFIGS.items():
            print(f"[{architecture}] T={time_steps}: {config_name}", flush=True)
            model, compression = build_variant(
                kind,
                snn_template=snn_template,
                teacher=teacher,
                time_steps=time_steps,
                architecture=architecture,
                rank=args.main_rank,
                selected_margins=selected,
            )
            result, layers = evaluate_test(
                model,
                evaluation_loader,
                device,
                time_steps,
                architecture,
            )
            result["calibration_elapsed"] = (
                0.0 if kind.startswith("plain") else calibration_elapsed
            )
            result["compression_elapsed"] = (
                0.0 if compression is None else compression.get("compression_elapsed", 0.0)
            )
            result["effective_ftbc_modes"] = sorted(
                {
                    module.ftbc_mode
                    for module in model.modules()
                    if isinstance(module, SignedIF)
                }
            )
            result["snm_gate_parameters"] = (
                len(gate_groups(model, architecture)) if "gated" in kind else 0
            )
            result["selected_margins"] = selected if "gated" in kind else None
            payload["results"][config_name][str(time_steps)] = result
            payload["layers"].setdefault(config_name, OrderedDict())[str(time_steps)] = layers
            if compression is not None:
                payload["compression"].setdefault(config_name, OrderedDict())[
                    str(time_steps)
                ] = compression
            save_progress(progress_path, payload)
            write_report(output, payload)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del ann, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_progress(progress_path, payload)
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
    parser.add_argument("--validation_batches", type=int, default=5)
    parser.add_argument("--test_batches", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--main_rank", type=int, default=4)
    parser.add_argument("--rank_candidates", nargs="+", type=int, default=(2, 4, 6))
    parser.add_argument(
        "--margin_candidates",
        nargs="+",
        type=float,
        default=(0.0, 0.25, 0.5, 1.0, 2.0),
    )
    parser.add_argument("--gate_sop_weight", type=float, default=0.05)
    parser.add_argument("--gate_negative_weight", type=float, default=0.02)
    parser.add_argument("--gate_accuracy_tolerance", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/results/comparative_ablation/cifar100/"
            "TEMPORAL_LR_GATED_SNM_CIFAR100.md"
        ),
    )
    return parser


def main(cli_args=None):
    args = build_parser().parse_args(cli_args)
    if args.fit_batches <= 0 or args.validation_batches <= 0:
        raise ValueError("Fit and validation batches must be positive")
    if set(args.rank_candidates) != {2, 4, 6}:
        raise ValueError("The fixed rank screen must contain 2, 4, and 6")
    if args.main_rank != 4:
        raise ValueError("The primary Temporal-LR configuration is fixed at rank 4")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    train_loader, test_loader = datapool("cifar100", args.batch_size)
    all_calibration = materialize_calibration_batches(
        train_loader,
        args.fit_batches + args.validation_batches,
    )
    fit_batches = all_calibration[: args.fit_batches]
    validation_batches = all_calibration[args.fit_batches :]
    if args.test_batches > 0:
        evaluation_loader = materialize_calibration_batches(
            test_loader,
            args.test_batches,
        )
    else:
        evaluation_loader = test_loader
    outputs = []
    checkpoints = {
        "resnet20": args.resnet20_checkpoint,
        "vgg16": args.vgg16_checkpoint,
    }
    for architecture in args.architectures:
        outputs.append(
            run_architecture(
                args,
                architecture,
                checkpoints[architecture],
                fit_batches,
                validation_batches,
                evaluation_loader,
                device,
            )
        )
    for output in outputs:
        print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
