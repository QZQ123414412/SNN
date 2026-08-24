"""Nine-way QCFS, Full/Temporal-LR FTBC, and A-SNM ablation."""

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

from a_snm import DEFAULT_TIME_STEPS, a_snm_enabled, select_a_snm_modes
from calibration import bias_corr_model
from models import SignedIF
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
    save_progress,
    snapshot_full_ftbc,
    validate_t1_special_case,
)
from scripts.experiments.run_state_ftbc_ablation import (
    materialize_calibration_batches,
)
from scripts.experiments.run_temporal_lr_gated_snm import (
    batches_sha256,
    make_deployment_compressed,
    synchronize,
)
from spike_stats import (
    set_signed_spike_stats_enabled,
    summarize_ftbc_storage,
)
from temporal_lr import named_signed_layers
from utils import seed_all, val


FORMAL_BATCH_HASHES = {
    ("cifar100", 8): {
        "fit": "9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a",
        "validation": "d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3",
    },
    ("cifar10", 4): {
        "fit": "053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df",
        "validation": "237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c",
    },
}
CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL = {
    **CIFAR10_RESNET20_L4_PROTOCOL,
    "ann_accuracy": 90.72,
    "default_output": Path(
        "docs/results/comparative_ablation/cifar10/"
        "TEMPORAL_LR_ASNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.md"
    ),
}
CIFAR100_DEFAULT_OUTPUT = Path(
    "docs/results/comparative_ablation/cifar100/"
    "TEMPORAL_LR_ASNM_CIFAR100.md"
)
TEMPORAL_RANK = 4
FAMILIES = ("qcfs", "full", "temporal")
CONFIGS = OrderedDict(
    [
        ("A_QCFS_R0", {"family": "qcfs", "mode": "off"}),
        (
            "B_QCFS_STANDARD_SNM_R0",
            {"family": "qcfs", "mode": "on"},
        ),
        ("C_QCFS_ASNM_R0", {"family": "qcfs", "mode": "a_snm"}),
        (
            "D_QCFS_FULL_FTBC_R0",
            {"family": "full", "mode": "off"},
        ),
        (
            "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
            {"family": "full", "mode": "on"},
        ),
        (
            "F_QCFS_FULL_FTBC_ASNM_R0",
            {"family": "full", "mode": "a_snm"},
        ),
        (
            "G_QCFS_TEMPORAL_LR_FTBC_R0",
            {"family": "temporal", "mode": "off"},
        ),
        (
            "H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0",
            {"family": "temporal", "mode": "on"},
        ),
        (
            "I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0",
            {"family": "temporal", "mode": "a_snm"},
        ),
    ]
)
BASE_CONFIGS = {
    "qcfs": {"off": "A_QCFS_R0", "on": "B_QCFS_STANDARD_SNM_R0"},
    "full": {
        "off": "D_QCFS_FULL_FTBC_R0",
        "on": "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
    },
    "temporal": {
        "off": "G_QCFS_TEMPORAL_LR_FTBC_R0",
        "on": "H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0",
    },
}
A_SNM_CONFIGS = {
    "qcfs": "C_QCFS_ASNM_R0",
    "full": "F_QCFS_FULL_FTBC_ASNM_R0",
    "temporal": "I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0",
}
FAMILY_TITLES = {
    "qcfs": "QCFS",
    "full": "Full-FTBC",
    "temporal": "Temporal-LR FTBC",
}
VALIDATION_EQUIVALENCE_KEYS = (
    "acc",
    "logit_mse",
    "sops",
    "positive_spikes",
    "negative_spikes",
)
TEST_EQUIVALENCE_KEYS = VALIDATION_EQUIVALENCE_KEYS + (
    "positive_rate",
    "negative_rate",
    "sparsity",
    "ftbc_parameters",
    "ftbc_bytes",
)


def is_formal_protocol(args):
    return (
        tuple(args.time_steps) == DEFAULT_TIME_STEPS
        and args.batch_size == 200
        and args.fit_batches == 5
        and args.validation_batches == 5
        and args.test_batches == 0
        and args.seed == 42
        and args.rank == TEMPORAL_RANK
    )


def architecture_output(base_output, architecture, multiple):
    path = Path(base_output)
    if not multiple:
        return path
    return path.with_name(f"{path.stem}_{architecture.upper()}{path.suffix}")


def build_temporal_model(
    template,
    schedule,
    time_steps,
    signed,
    device,
    architecture,
    rank=TEMPORAL_RANK,
):
    teacher = build_full_model(
        template,
        schedule,
        time_steps,
        signed=False,
        device=device,
    )
    model, compression = make_deployment_compressed(
        teacher,
        rank=rank,
        architecture=architecture,
        time_steps=time_steps,
        hybrid=False,
    )
    model.set_signed(bool(signed))
    model.set_snm_negative_margin(0.0)
    return model, normalize_compression_report(model, compression, time_steps, rank)


def normalize_compression_report(model, compression, time_steps, rank):
    report = copy.deepcopy(compression)
    fallback = bool(report.get("fallback_to_full", False))
    if fallback:
        report.update(
            {
                "requested_rank": int(rank),
                "effective_rank": int(time_steps),
                "time_steps": int(time_steps),
                "compressed_channels": sum(
                    int(module.time_based_bias[0].numel())
                    for module in named_signed_layers(model).values()
                ),
                "full_layer_names": list(named_signed_layers(model)),
                "threshold_normalize": True,
                "explained_energy": 1.0,
                "layers": OrderedDict(
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
                ),
            }
        )
    storage = summarize_ftbc_storage(model, SignedIF)
    report["ftbc_parameters"] = int(storage["parameters"])
    report["ftbc_bytes"] = int(storage["bytes"])
    report["ftbc_synthesis_macs"] = int(storage["synthesis_macs"])
    return report


def exact_metrics(left, right, keys):
    return all(left[key] == right[key] for key in keys)


def selected_time_label(config_name, gates, time_steps):
    meta = CONFIGS[config_name]
    if meta["mode"] == "off":
        return "none"
    if meta["mode"] == "on":
        return ", ".join(str(value) for value in time_steps)
    selected = [
        str(value)
        for value in time_steps
        if gates[meta["family"]][str(value)]
    ]
    return ", ".join(selected) if selected else "none"


def metric_table(lines, title, payload, key, formatter):
    time_steps = payload["protocol"]["time_steps"]
    lines.extend(
        [
            f"## {title}",
            "",
            "| Config | " + " | ".join(f"T={value}" for value in time_steps) + " |",
            "|---|" + "---:|" * len(time_steps),
        ]
    )
    for config_name in CONFIGS:
        cells = []
        for time_steps_value in time_steps:
            result = payload["results"].get(config_name, {}).get(
                str(time_steps_value)
            )
            cells.append("-" if result is None else formatter(result[key]))
        lines.append(f"| {config_name} | " + " | ".join(cells) + " |")
    lines.append("")


def write_report(path, payload):
    protocol = payload["protocol"]
    time_steps = protocol["time_steps"]
    lines = [
        "# QCFS + Full-FTBC + Temporal-LR FTBC + A-SNM "
        f"{protocol['dataset_label']} Ablation",
        "",
        f"- Status: {payload['status']}",
        f"- Dataset: {protocol['dataset_label']}",
        f"- Architecture: {protocol['architecture']}",
        f"- QCFS levels: L={protocol['qcfs_L']}",
        f"- ResNet20 evaluation profile: {protocol['resnet20_eval_profile']}",
        f"- Checkpoint: `{protocol['checkpoint']['filename']}`",
        f"- Checkpoint SHA256: `{protocol['checkpoint']['sha256']}`",
        f"- ANN accuracy on the {protocol['test_samples']:,}-image test set: {protocol['ann_accuracy']:.2f}%",
        f"- Time steps: {time_steps}",
        f"- Full-FTBC fit: {protocol['fit_batches']} x {protocol['batch_size']}, alpha={protocol['alpha']}",
        f"- A-SNM validation: {protocol['validation_batches']} x {protocol['batch_size']}",
        f"- Temporal-LR: shared rank-{protocol['rank']} basis, threshold-normalized, no exempt layer.",
        "- Temporal-LR falls back to Full-FTBC at T<=4 and is active at T>4.",
        f"- Fit batch SHA256: `{protocol['fit_sha256']}`",
        f"- Validation batch SHA256: `{protocol['validation_sha256']}`",
        "- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, "
        f"ToTensor, {protocol['normalization']}, and Cutout(1,16).",
        "- Test uses only ToTensor and normalization with shuffle=False.",
        f"- Every SNN uses QCFS L={protocol['qcfs_L']}, rate coding/schedule, "
        "ratio=1.0, R0=True, SNM margin=0, FP32.",
        "- Full-FTBC is independently fitted at every T with SNM off; Temporal-LR is compressed from that frozen teacher.",
        "- Each family enables SNM only when SNM-on has strictly higher validation accuracy; ties select off.",
        "- Test data is first accessed after all three A-SNM families are frozen.",
        "- A-SNM guarantees validation-set selection only; test-set reversals are reported diagnostically and never retuned.",
        f"- Checkpoint-selection note: {protocol['checkpoint_selection_note']}.",
        f"- Checkpoint-interpretation note: {protocol['checkpoint_interpretation_note']}.",
        "",
        "## Primary accuracy table",
        "",
        "| Config | "
        + " | ".join(f"T={value}" for value in time_steps)
        + " | SNM-on T |",
        "|---|" + "---:|" * len(time_steps) + "---|",
    ]
    for config_name in CONFIGS:
        values = [
            f"{payload['results'][config_name][str(value)]['acc']:.2f}%"
            for value in time_steps
        ]
        lines.append(
            f"| {config_name} | "
            + " | ".join(values)
            + f" | {selected_time_label(config_name, payload['gates'], time_steps)} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Mean accuracy over evaluated time steps",
            "",
            "| Config | Mean accuracy |",
            "|---|---:|",
        ]
    )
    for config_name in CONFIGS:
        mean_accuracy = sum(
            payload["results"][config_name][str(value)]["acc"]
            for value in time_steps
        ) / len(time_steps)
        lines.append(f"| {config_name} | {mean_accuracy:.2f}% |")
    lines.append("")

    lines.extend(
        [
            "## Temporal-LR accuracy comparisons",
            "",
            "| Comparison | "
            + " | ".join(f"T={value}" for value in time_steps)
            + " | Mean |",
            "|---|" + "---:|" * (len(time_steps) + 1),
        ]
    )
    for label, left_name, right_name in (
        ("G-D", "G_QCFS_TEMPORAL_LR_FTBC_R0", "D_QCFS_FULL_FTBC_R0"),
        (
            "H-E",
            "H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0",
            "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
        ),
        (
            "I-F",
            "I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0",
            "F_QCFS_FULL_FTBC_ASNM_R0",
        ),
    ):
        differences = [
            payload["results"][left_name][str(value)]["acc"]
            - payload["results"][right_name][str(value)]["acc"]
            for value in time_steps
        ]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{value:+.2f}pp" for value in differences)
            + f" | {sum(differences) / len(differences):+.2f}pp |"
        )
    lines.append("")

    metric_table(
        lines,
        "ANN-SNN logit MSE",
        payload,
        "logit_mse",
        lambda value: f"{value:.6f}",
    )
    metric_table(
        lines,
        "Positive spike rate",
        payload,
        "positive_rate",
        lambda value: f"{100 * value:.4f}%",
    )
    metric_table(
        lines,
        "Negative spike rate",
        payload,
        "negative_rate",
        lambda value: f"{100 * value:.4f}%",
    )
    metric_table(
        lines,
        "Overall spike sparsity",
        payload,
        "sparsity",
        lambda value: f"{100 * value:.4f}%",
    )
    metric_table(
        lines,
        "Input-driven SOPs",
        payload,
        "sops",
        lambda value: f"{int(value):,}",
    )
    metric_table(
        lines,
        "FTBC parameters",
        payload,
        "ftbc_parameters",
        lambda value: f"{int(value):,}",
    )
    metric_table(
        lines,
        "FTBC storage bytes",
        payload,
        "ftbc_bytes",
        lambda value: f"{int(value):,}",
    )
    metric_table(
        lines,
        "Temporal bias synthesis MACs",
        payload,
        "ftbc_synthesis_macs",
        lambda value: f"{int(value):,}",
    )
    metric_table(
        lines,
        "Full-teacher calibration elapsed",
        payload,
        "calibration_elapsed",
        lambda value: f"{value:.3f}s",
    )
    metric_table(
        lines,
        "Temporal compression elapsed",
        payload,
        "compression_elapsed",
        lambda value: f"{value:.3f}s",
    )
    metric_table(
        lines,
        "Inference elapsed (statistics disabled)",
        payload,
        "inference_elapsed",
        lambda value: f"{value:.3f}s",
    )

    lines.extend(
        [
            "## Temporal-LR compression",
            "",
            "| T | Effective mode | Effective rank | Explained energy | Full parameters | Temporal parameters | Storage ratio | Storage reduction | Synthesis MACs |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for time_steps_value in time_steps:
        item = payload["compression"][str(time_steps_value)]
        mode = "full fallback" if item["fallback_to_full"] else "temporal_low_rank"
        lines.append(
            f"| {time_steps_value} | {mode} | {item['effective_rank']} | "
            f"{item['explained_energy']:.6f} | {item['full_parameters']:,} | "
            f"{item['ftbc_parameters']:,} | {item['storage_ratio']:.6f} | "
            f"{100 * item['storage_reduction']:.2f}% | "
            f"{item['ftbc_synthesis_macs']:,} |"
        )
    lines.append("")

    lines.extend(
        [
            "## A-SNM selection",
            "",
            "| Family | SNM-on T | Validation inference + selection |",
            "|---|---|---:|",
        ]
    )
    for family in FAMILIES:
        config_name = A_SNM_CONFIGS[family]
        lines.append(
            f"| {FAMILY_TITLES[family]} | "
            f"{selected_time_label(config_name, payload['gates'], time_steps)} | "
            f"{payload['selection_elapsed'][family]:.3f}s |"
        )
    lines.append("")

    for family in FAMILIES:
        lines.extend(
            [
                f"### {FAMILY_TITLES[family]} accuracy-gate trace",
                "",
                "| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for time_steps_value, item in payload["selection_trace"][family].items():
            lines.append(
                f"| {time_steps_value} | {item['off_accuracy']:.4f}% | "
                f"{item['on_accuracy']:.4f}% | {item['accuracy_gain']:+.4f}pp | "
                f"{item['selected_mode']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Validation-selection generalization audit",
            "",
            "This table is diagnostic only. Test accuracy never changes a frozen A-SNM decision.",
            "",
            "| Family | T | Selected from validation | Test off | Test on | Test-best mode | Match |",
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
    lines.append("")

    lines.extend(
        [
            "## Equivalence checks",
            "",
            "| Kind | Config/family | T | Expected source | Exact |",
            "|---|---|---:|---|---|",
        ]
    )
    for item in payload["equivalence_checks"]:
        lines.append(
            f"| {item['kind']} | {item['name']} | {item['time_steps']} | "
            f"{item['source']} | {'yes' if item['exact'] else 'no'} |"
        )
    lines.append("")

    lines.extend(["## Per-layer Temporal-LR reconstruction", ""])
    for time_steps_value in time_steps:
        compression = payload["compression"][str(time_steps_value)]
        lines.extend(
            [
                f"### T={time_steps_value}",
                "",
                "| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for layer_name, item in compression["layers"].items():
            lines.append(
                f"| `{layer_name}` | {item['representation']} | {item['channels']} | "
                f"{item['mse']:.8f} | {item['nrmse']:.8f} | "
                f"{item['max_abs_error']:.8f} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_architecture(
    args,
    architecture,
    checkpoint_path,
    expected_checkpoint_sha256,
    fit_batches,
    validation_batches,
    test_loader,
    device,
):
    output = architecture_output(
        args.output,
        architecture,
        multiple=len(args.architectures) > 1,
    )
    progress_path = output.with_suffix(".progress.json")
    if not args.overwrite and (output.exists() or progress_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite an existing result: {output} or {progress_path}"
        )

    ann_template, snn_template, checkpoint = load_qcfs_pair(
        checkpoint_path,
        args.dataset,
        architecture,
        device,
    )
    if checkpoint["sha256"].lower() != expected_checkpoint_sha256.lower():
        raise RuntimeError(
            f"Unexpected {args.dataset}/{architecture} checkpoint SHA256: "
            f"{checkpoint['sha256']} (expected {expected_checkpoint_sha256})"
        )
    ann_template.set_T(0)
    if hasattr(ann_template, "set_L"):
        ann_template.set_L(args.L)
    if hasattr(snn_template, "set_L"):
        snn_template.set_L(args.L)
    evaluation_profile = "not-applicable"
    if architecture == "resnet20" and hasattr(
        ann_template, "set_qcfs_training_profile"
    ):
        evaluation_profile = args.resnet20_eval_profile
        ann_template.set_qcfs_training_profile(evaluation_profile)
        if hasattr(snn_template, "set_qcfs_training_profile"):
            snn_template.set_qcfs_training_profile(evaluation_profile)
    ann_template.eval()
    set_signed_spike_stats_enabled(snn_template, SignedIF, False)

    if args.dataset == "cifar10" and architecture == "resnet20":
        checkpoint_selection_note = (
            "the checkpoint is selected by the highest accuracy observed on "
            "the 10,000-image CIFAR-10 test set during 300 training epochs; "
            "this creates model-selection bias"
        )
        checkpoint_interpretation_note = (
            f"the checkpoint is the CIFAR-10/ResNet20 QCFS-L{args.L} "
            f"paper-aligned retrained model evaluated with the "
            f"{evaluation_profile} profile; it is not a strict reproduction "
            "of the paper's reported accuracy"
        )
    else:
        checkpoint_selection_note = "unchanged from the frozen CIFAR-100 protocol"
        checkpoint_interpretation_note = (
            "the checkpoint is evaluated under its frozen CIFAR-100 protocol"
        )

    payload = {
        "status": "selecting_a_snm_modes",
        "protocol": {
            "dataset": args.dataset,
            "dataset_label": DATASET_PROTOCOLS[args.dataset]["label"],
            "architecture": architecture,
            "qcfs_L": args.L,
            "resnet20_eval_profile": evaluation_profile,
            "normalization": DATASET_PROTOCOLS[args.dataset]["normalization"],
            "checkpoint": checkpoint,
            "checkpoint_selection_note": checkpoint_selection_note,
            "checkpoint_interpretation_note": checkpoint_interpretation_note,
            "ann_accuracy": None,
            "time_steps": list(args.time_steps),
            "batch_size": args.batch_size,
            "fit_batches": args.fit_batches,
            "validation_batches": args.validation_batches,
            "fit_sha256": batches_sha256(fit_batches),
            "validation_sha256": batches_sha256(validation_batches),
            "alpha": args.alpha,
            "rank": args.rank,
            "threshold_normalize": True,
            "full_layer_names": [],
            "temporal_fallback": "Full-FTBC at T<=4",
            "a_snm_rule": "SNM-on iff on validation accuracy > off validation accuracy; ties select off",
            "seed": args.seed,
            "coding_mode": "rate",
            "schedule": "rate",
            "ratio": 1.0,
            "r0": True,
            "snm_margin": 0.0,
            "test_batches": args.test_batches,
            "test_samples": None,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "dtype": "float32",
        },
        "calibration": OrderedDict(),
        "compression": OrderedDict(),
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
        print(f"[{architecture}] T={time_steps}: QCFS validation", flush=True)
        for mode, signed in (("off", False), ("on", True)):
            model = build_plain_model(snn_template, time_steps, signed, device)
            metrics = evaluate_validation(
                model,
                ann_template,
                validation_batches,
                device,
                time_steps,
                architecture,
            )
            payload["validation"]["qcfs"][mode][key] = metrics
            payload["selection_elapsed"]["qcfs"] += metrics["elapsed"]
            del model

        print(f"[{architecture}] T={time_steps}: Full-FTBC fit", flush=True)
        teacher = copy.deepcopy(snn_template).to(device)
        configure_snn(teacher, time_steps, signed=False, ftbc_mode="full")
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
        schedules[time_steps] = snapshot_full_ftbc(teacher, time_steps)
        full_storage = summarize_ftbc_storage(teacher, SignedIF)
        payload["calibration"][key] = {
            "elapsed": calibration_elapsed,
            "ftbc_parameters": int(full_storage["parameters"]),
            "ftbc_bytes": int(full_storage["bytes"]),
        }

        print(f"[{architecture}] T={time_steps}: Full-FTBC validation", flush=True)
        for mode, signed in (("off", False), ("on", True)):
            model = copy.deepcopy(teacher).to(device)
            model.set_signed(signed)
            metrics = evaluate_validation(
                model,
                ann,
                validation_batches,
                device,
                time_steps,
                architecture,
            )
            payload["validation"]["full"][mode][key] = metrics
            payload["selection_elapsed"]["full"] += metrics["elapsed"]
            del model

        print(f"[{architecture}] T={time_steps}: Temporal-LR validation", flush=True)
        compression_for_time = None
        for mode, signed in (("off", False), ("on", True)):
            model, compression = build_temporal_model(
                snn_template,
                schedules[time_steps],
                time_steps,
                signed,
                device,
                architecture,
                rank=args.rank,
            )
            if compression_for_time is None:
                compression_for_time = compression
            metrics = evaluate_validation(
                model,
                ann,
                validation_batches,
                device,
                time_steps,
                architecture,
            )
            payload["validation"]["temporal"][mode][key] = metrics
            payload["selection_elapsed"]["temporal"] += metrics["elapsed"]
            del model

        compression_for_time["full_parameters"] = int(full_storage["parameters"])
        compression_for_time["full_bytes"] = int(full_storage["bytes"])
        compression_for_time["storage_ratio"] = (
            compression_for_time["ftbc_parameters"]
            / max(int(full_storage["parameters"]), 1)
        )
        compression_for_time["storage_reduction"] = (
            1.0 - compression_for_time["storage_ratio"]
        )
        payload["compression"][key] = compression_for_time

        if time_steps <= 4:
            for mode in ("off", "on"):
                exact = exact_metrics(
                    payload["validation"]["full"][mode][key],
                    payload["validation"]["temporal"][mode][key],
                    VALIDATION_EQUIVALENCE_KEYS,
                )
                payload["equivalence_checks"].append(
                    {
                        "kind": "validation fallback",
                        "name": f"{mode}:full=temporal",
                        "time_steps": time_steps,
                        "source": f"Full-FTBC {mode}",
                        "exact": exact,
                    }
                )
                if not exact:
                    raise RuntimeError(
                        f"T={time_steps} {mode} Temporal-LR fallback validation mismatch"
                    )

        del ann, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_progress(progress_path, payload)

    frozen_modes = {}
    for family in FAMILIES:
        off_metrics = {
            int(key): value
            for key, value in payload["validation"][family]["off"].items()
        }
        on_metrics = {
            int(key): value
            for key, value in payload["validation"][family]["on"].items()
        }
        validate_t1_special_case(off_metrics, on_metrics, family)
        synchronize(device)
        selection_started = time.perf_counter()
        selected_modes, trace = select_a_snm_modes(
            off_metrics,
            on_metrics,
            time_steps=args.time_steps,
        )
        synchronize(device)
        payload["selection_elapsed"][family] += (
            time.perf_counter() - selection_started
        )
        frozen_modes[family] = selected_modes
        payload["gates"][family] = {
            str(time_steps_value): enabled
            for time_steps_value, enabled in selected_modes.items()
        }
        payload["selection_trace"][family] = trace
        enabled_times = [
            str(time_steps_value)
            for time_steps_value, enabled in selected_modes.items()
            if enabled
        ]
        print(
            f"[{architecture}] selected {family} A-SNM on T="
            + (",".join(enabled_times) if enabled_times else "none"),
            flush=True,
        )

    for time_steps in args.time_steps:
        if time_steps <= 4:
            exact = frozen_modes["full"][time_steps] == frozen_modes["temporal"][
                time_steps
            ]
            payload["equivalence_checks"].append(
                {
                    "kind": "gate fallback",
                    "name": "full=temporal",
                    "time_steps": time_steps,
                    "source": "identical validation metrics",
                    "exact": exact,
                }
            )
            if not exact:
                raise RuntimeError(f"T={time_steps} fallback A-SNM gate mismatch")

    payload["status"] = "a_snm_modes_frozen_testing"
    save_progress(progress_path, payload)

    evaluation_loader = (
        materialize_calibration_batches(test_loader, args.test_batches)
        if args.test_batches > 0
        else test_loader
    )
    payload["protocol"]["ann_accuracy"] = val(
        ann_template,
        evaluation_loader,
        device,
        0,
    )
    if (
        is_formal_protocol(args)
        and args.dataset == "cifar10"
        and args.L == 4
        and abs(
            payload["protocol"]["ann_accuracy"]
            - CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL["ann_accuracy"]
        )
        > 1e-12
    ):
        raise RuntimeError(
            "Unexpected CIFAR-10/ResNet20 L=4 ANN accuracy: "
            f"{payload['protocol']['ann_accuracy']}"
        )
    if args.test_batches > 0:
        payload["protocol"]["test_samples"] = sum(
            int(targets.numel()) for _, targets in evaluation_loader
        )
    else:
        payload["protocol"]["test_samples"] = len(evaluation_loader.dataset)
        if payload["protocol"]["test_samples"] != 10000:
            raise RuntimeError(
                f"Formal {payload['protocol']['dataset_label']} test set must "
                "contain 10,000 images"
            )

    for time_steps in args.time_steps:
        key = str(time_steps)
        base_results = {}
        base_layers = {}
        for family in FAMILIES:
            for mode, signed in (("off", False), ("on", True)):
                config_name = BASE_CONFIGS[family][mode]
                print(f"[{architecture}] T={time_steps}: test {config_name}", flush=True)
                compression = None
                if family == "qcfs":
                    model = build_plain_model(
                        snn_template,
                        time_steps,
                        signed,
                        device,
                    )
                elif family == "full":
                    model = build_full_model(
                        snn_template,
                        schedules[time_steps],
                        time_steps,
                        signed,
                        device,
                    )
                else:
                    model, compression = build_temporal_model(
                        snn_template,
                        schedules[time_steps],
                        time_steps,
                        signed,
                        device,
                        architecture,
                        rank=args.rank,
                    )
                result, layers = evaluate_test(
                    model,
                    ann_template,
                    evaluation_loader,
                    device,
                    time_steps,
                    architecture,
                )
                storage = summarize_ftbc_storage(model, SignedIF)
                result["ftbc_synthesis_macs"] = int(storage["synthesis_macs"])
                result["calibration_elapsed"] = (
                    0.0
                    if family == "qcfs"
                    else payload["calibration"][key]["elapsed"]
                )
                result["compression_elapsed"] = (
                    payload["compression"][key]["compression_elapsed"]
                    if family == "temporal"
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
                payload["results"][config_name][key] = result
                payload["layers"][config_name][key] = layers
                base_results[(family, mode)] = result
                base_layers[(family, mode)] = layers
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if time_steps <= 4:
            for mode in ("off", "on"):
                exact = exact_metrics(
                    base_results[("full", mode)],
                    base_results[("temporal", mode)],
                    TEST_EQUIVALENCE_KEYS,
                )
                payload["equivalence_checks"].append(
                    {
                        "kind": "test fallback",
                        "name": f"{mode}:full=temporal",
                        "time_steps": time_steps,
                        "source": BASE_CONFIGS["full"][mode],
                        "exact": exact,
                    }
                )
                if not exact:
                    raise RuntimeError(
                        f"T={time_steps} {mode} Temporal-LR fallback test mismatch"
                    )

        for family in FAMILIES:
            enabled = a_snm_enabled(frozen_modes[family], time_steps)
            mode = "on" if enabled else "off"
            source_name = BASE_CONFIGS[family][mode]
            config_name = A_SNM_CONFIGS[family]
            result = copy.deepcopy(base_results[(family, mode)])
            layers = copy.deepcopy(base_layers[(family, mode)])
            result.update(
                {
                    "snm_mode": "a_snm",
                    "snm_enabled": enabled,
                    "a_snm_enabled": enabled,
                    "a_snm_selection_elapsed": payload["selection_elapsed"][family],
                    "source_config": source_name,
                }
            )
            payload["results"][config_name][key] = result
            payload["layers"][config_name][key] = layers
            exact = exact_metrics(
                result,
                base_results[(family, mode)],
                TEST_EQUIVALENCE_KEYS,
            )
            payload["equivalence_checks"].append(
                {
                    "kind": "A-SNM cache",
                    "name": config_name,
                    "time_steps": time_steps,
                    "source": source_name,
                    "exact": exact,
                }
            )
            if not exact:
                raise RuntimeError(
                    f"T={time_steps} {config_name} cache equivalence failed"
                )

            off_accuracy = base_results[(family, "off")]["acc"]
            on_accuracy = base_results[(family, "on")]["acc"]
            test_best_mode = "on" if on_accuracy > off_accuracy else "off"
            payload["generalization_audit"].append(
                {
                    "family": family,
                    "time_steps": time_steps,
                    "selected_mode": mode,
                    "test_off_accuracy": off_accuracy,
                    "test_on_accuracy": on_accuracy,
                    "test_best_mode": test_best_mode,
                    "matches_test_best": mode == test_best_mode,
                }
            )
        save_progress(progress_path, payload)

    if not all(item["exact"] for item in payload["equivalence_checks"]):
        raise RuntimeError("One or more deployment equivalence checks failed")
    payload["status"] = "complete"
    save_progress(progress_path, payload)
    write_report(output, payload)
    return output, payload


def write_summary(path, architecture_payloads):
    time_steps = next(iter(architecture_payloads.values()))["protocol"]["time_steps"]
    lines = [
        "# QCFS + Temporal-LR FTBC + A-SNM CIFAR-100 Summary",
        "",
        "- Status: complete",
        "- The final method is `I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0`.",
        "- Rank-4 Temporal-LR falls back to Full-FTBC at T<=4.",
        "- A-SNM is selected independently on the fixed 1,000-image augmented validation set.",
        "",
        "## Final-method accuracy",
        "",
        "| Architecture | "
        + " | ".join(f"T={value}" for value in time_steps)
        + " | SNM-on T |",
        "|---|" + "---:|" * len(time_steps) + "---|",
    ]
    for architecture, payload in architecture_payloads.items():
        values = [
            f"{payload['results'][A_SNM_CONFIGS['temporal']][str(value)]['acc']:.2f}%"
            for value in time_steps
        ]
        selected = selected_time_label(
            A_SNM_CONFIGS["temporal"],
            payload["gates"],
            time_steps,
        )
        lines.append(
            f"| {architecture} | " + " | ".join(values) + f" | {selected} |"
        )
    lines.extend(
        [
            "",
            "## Temporal-LR storage reduction versus Full-FTBC",
            "",
            "| Architecture | "
            + " | ".join(f"T={value}" for value in time_steps)
            + " |",
            "|---|" + "---:|" * len(time_steps),
        ]
    )
    for architecture, payload in architecture_payloads.items():
        values = [
            f"{100 * payload['compression'][str(value)]['storage_reduction']:.2f}%"
            for value in time_steps
        ]
        lines.append(f"| {architecture} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Source reports",
            "",
        ]
    )
    for architecture in architecture_payloads:
        lines.append(f"- `{path.stem.replace('_SUMMARY', '')}_{architecture.upper()}.md`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(
        description="QCFS + Full/Temporal-LR FTBC + A-SNM nine-way ablation"
    )
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "cifar100"),
        default="cifar100",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("resnet20", "vgg16"),
        default=("resnet20", "vgg16"),
    )
    parser.add_argument(
        "--resnet20_checkpoint",
        type=Path,
    )
    parser.add_argument(
        "--vgg16_checkpoint",
        type=Path,
    )
    parser.add_argument("--resnet20_checkpoint_sha256")
    parser.add_argument("--vgg16_checkpoint_sha256")
    parser.add_argument(
        "--resnet20_eval_profile",
        choices=("fixed_repo", "paper_era"),
    )
    parser.add_argument("-L", "--L", type=int, default=8)
    parser.add_argument(
        "--time_steps",
        nargs="+",
        type=int,
        default=DEFAULT_TIME_STEPS,
    )
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--fit_batches", type=int, default=5)
    parser.add_argument("--validation_batches", type=int, default=5)
    parser.add_argument("--test_batches", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--rank", type=int, default=TEMPORAL_RANK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
    )
    return parser


def resolve_protocol_args(args):
    protocol = DATASET_PROTOCOLS[args.dataset]
    cifar10_l4 = (
        args.dataset == "cifar10"
        and args.L == 4
        and tuple(args.architectures) == ("resnet20",)
    )
    for architecture in ("resnet20", "vgg16"):
        checkpoint_name = f"{architecture}_checkpoint"
        if getattr(args, checkpoint_name) is None:
            checkpoint = (
                CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL["checkpoint"]
                if architecture == "resnet20" and cifar10_l4
                else protocol["default_checkpoints"][architecture]
            )
            setattr(args, checkpoint_name, checkpoint)
        sha_name = f"{architecture}_checkpoint_sha256"
        if getattr(args, sha_name) is None:
            expected_sha256 = (
                CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL["expected_sha256"]
                if architecture == "resnet20" and cifar10_l4
                else protocol["expected_sha256"][architecture]
            )
            setattr(args, sha_name, expected_sha256)
    if args.resnet20_eval_profile is None:
        args.resnet20_eval_profile = (
            CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL["evaluation_profile"]
            if cifar10_l4
            else protocol["resnet20_eval_profile"]
        )
    if args.output is None:
        args.output = (
            CIFAR10_RESNET20_L4_TEMPORAL_PROTOCOL["default_output"]
            if cifar10_l4
            else CIFAR100_DEFAULT_OUTPUT
        )
    return args


def validate_args(args):
    if args.batch_size <= 0 or args.fit_batches <= 0 or args.validation_batches <= 0:
        raise ValueError("Batch size and calibration batch counts must be positive")
    if len(set(args.time_steps)) != len(args.time_steps):
        raise ValueError("Time steps must be unique")
    if tuple(sorted(args.time_steps)) != tuple(args.time_steps):
        raise ValueError("Time steps must be sorted")
    if args.test_batches < 0:
        raise ValueError("test_batches must be non-negative")
    if args.alpha != 0.4:
        raise ValueError("The experiment plan fixes Full-FTBC alpha at 0.4")
    if args.rank != TEMPORAL_RANK:
        raise ValueError("The experiment plan fixes Temporal-LR rank at 4")
    if args.dataset == "cifar100":
        if args.L != 8:
            raise ValueError("The frozen CIFAR-100 Temporal-LR protocol requires L=8")
    elif not (
        args.L == 4 and tuple(args.architectures) == ("resnet20",)
    ):
        raise ValueError(
            "The CIFAR-10 Temporal-LR protocol is locked to ResNet20 L=4"
        )
    if (
        args.dataset == "cifar10"
        and args.resnet20_eval_profile != "paper_era"
    ):
        raise ValueError(
            "The CIFAR-10/ResNet20 L=4 Temporal-LR protocol requires "
            "paper_era evaluation"
        )
    for architecture in args.architectures:
        expected_sha256 = getattr(args, f"{architecture}_checkpoint_sha256")
        if not expected_sha256:
            raise ValueError(
                f"{args.dataset}/{architecture} requires an expected checkpoint "
                "SHA256 before any experiment run"
            )
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in expected_sha256
        ):
            raise ValueError(
                f"Invalid {architecture} checkpoint SHA256: {expected_sha256!r}"
            )
    if not is_formal_protocol(args) and "archive" not in {
        part.lower() for part in args.output.parts
    }:
        raise ValueError("Non-formal smoke runs must write under docs/archive")


def main(cli_args=None):
    args = resolve_protocol_args(build_parser().parse_args(cli_args))
    validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.test_batches == 0:
        raise RuntimeError("The formal experiment requires CUDA")
    seed_all(args.seed)
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    all_calibration = materialize_calibration_batches(
        train_loader,
        args.fit_batches + args.validation_batches,
    )
    fit_batches = all_calibration[: args.fit_batches]
    validation_batches = all_calibration[args.fit_batches :]
    fit_hash = batches_sha256(fit_batches)
    validation_hash = batches_sha256(validation_batches)
    if is_formal_protocol(args):
        expected_hashes = FORMAL_BATCH_HASHES[(args.dataset, args.L)]
        if fit_hash != expected_hashes["fit"]:
            raise RuntimeError(f"Unexpected formal fit batch SHA256: {fit_hash}")
        if validation_hash != expected_hashes["validation"]:
            raise RuntimeError(
                f"Unexpected formal validation batch SHA256: {validation_hash}"
            )

    summary_path = args.output.with_name(f"{args.output.stem}_SUMMARY.md")
    if (
        is_formal_protocol(args)
        and args.dataset == "cifar100"
        and len(args.architectures) == 2
        and not args.overwrite
        and summary_path.exists()
    ):
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_path}")

    checkpoints = {
        "resnet20": args.resnet20_checkpoint,
        "vgg16": args.vgg16_checkpoint,
    }
    checkpoint_sha256 = {
        "resnet20": args.resnet20_checkpoint_sha256,
        "vgg16": args.vgg16_checkpoint_sha256,
    }
    outputs = []
    architecture_payloads = OrderedDict()
    for architecture in args.architectures:
        output, payload = run_architecture(
            args,
            architecture,
            checkpoints[architecture],
            checkpoint_sha256[architecture],
            fit_batches,
            validation_batches,
            test_loader,
            device,
        )
        outputs.append(output)
        architecture_payloads[architecture] = payload

    if (
        is_formal_protocol(args)
        and args.dataset == "cifar100"
        and len(architecture_payloads) == 2
    ):
        write_summary(summary_path, architecture_payloads)
        outputs.append(summary_path)
    for output in outputs:
        print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
