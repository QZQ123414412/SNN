"""Generate traceable CIFAR-10 and cross-dataset A-SNM summaries."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.experiments.run_full_ftbc_asnm_ablation import (
    A_SNM_CONFIGS,
    BASE_CONFIGS,
    CONFIGS,
    selected_time_label,
)


def load_complete_progress(path, dataset, architecture):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Incomplete experiment progress: {path}")
    protocol = payload.get("protocol", {})
    if protocol.get("dataset") != dataset:
        raise RuntimeError(
            f"Dataset mismatch in {path}: {protocol.get('dataset')} != {dataset}"
        )
    if protocol.get("architecture") != architecture:
        raise RuntimeError(
            f"Architecture mismatch in {path}: "
            f"{protocol.get('architecture')} != {architecture}"
        )
    if int(protocol.get("test_samples", 0)) != 10000:
        raise RuntimeError(f"Formal result must evaluate 10,000 images: {path}")
    if not all(item.get("exact") for item in payload.get("equivalence_checks", [])):
        raise RuntimeError(f"Cached A-SNM equivalence check failed: {path}")
    return payload


def mean_accuracy(payload, config_name):
    time_steps = payload["protocol"]["time_steps"]
    return sum(
        float(payload["results"][config_name][str(value)]["acc"])
        for value in time_steps
    ) / len(time_steps)


def append_architecture_summary(lines, architecture, payload):
    time_steps = payload["protocol"]["time_steps"]
    lines.extend(
        [
            f"## {architecture}",
            "",
            f"- Checkpoint: `{payload['protocol']['checkpoint']['filename']}`",
            f"- SHA256: `{payload['protocol']['checkpoint']['sha256']}`",
            f"- ANN accuracy: {payload['protocol']['ann_accuracy']:.2f}%",
            f"- QCFS A-SNM SNM-on T: {selected_time_label('C_QCFS_ASNM_R0', payload['gates'], time_steps)}",
            f"- Full-FTBC A-SNM SNM-on T: {selected_time_label('F_QCFS_FULL_FTBC_ASNM_R0', payload['gates'], time_steps)}",
            "",
            "| Config | "
            + " | ".join(f"T={value}" for value in time_steps)
            + " | Mean |",
            "|---|" + "---:|" * (len(time_steps) + 1),
        ]
    )
    for name in CONFIGS:
        accuracies = [
            float(payload["results"][name][str(value)]["acc"])
            for value in time_steps
        ]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{value:.2f}%" for value in accuracies)
            + f" | {sum(accuracies) / len(accuracies):.2f}% |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Mean accuracy change |",
            "|---|---:|",
        ]
    )
    for selected_name, baseline_name, label in (
        ("C_QCFS_ASNM_R0", "A_QCFS_R0", "C-A"),
        ("C_QCFS_ASNM_R0", "B_QCFS_STANDARD_SNM_R0", "C-B"),
        ("F_QCFS_FULL_FTBC_ASNM_R0", "D_QCFS_FULL_FTBC_R0", "F-D"),
        (
            "F_QCFS_FULL_FTBC_ASNM_R0",
            "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
            "F-E",
        ),
    ):
        delta = mean_accuracy(payload, selected_name) - mean_accuracy(
            payload, baseline_name
        )
        lines.append(f"| {label} | {delta:+.2f}pp |")
    lines.append("")


def write_cifar10_summary(path, payloads, source_paths):
    lines = [
        "# CIFAR-10 QCFS + Full-FTBC + A-SNM Summary",
        "",
        "- Status: complete",
        "- Source: script-generated from the two formal progress JSON files.",
        "- A-SNM uses validation accuracy only; test-oracle results are diagnostic.",
        "- ResNet20 uses a test-best checkpoint, so its reported accuracy includes model-selection bias.",
        "- VGG16 is a legacy checkpoint probably trained with L=4 and evaluated post-hoc with L=8.",
        "",
        "## Sources",
        "",
    ]
    for architecture, source in source_paths.items():
        lines.append(f"- {architecture}: `{source.as_posix()}`")
    lines.append("")
    for architecture in ("resnet20", "vgg16"):
        append_architecture_summary(lines, architecture, payloads[architecture])
    atomic_write(path, "\n".join(lines))


def write_cross_dataset_summary(path, cifar10_payloads, cifar100_payloads):
    lines = [
        "# CIFAR-10 versus CIFAR-100 Full-FTBC + A-SNM Comparison",
        "",
        "- Status: complete",
        "- Source: script-generated from four formal progress JSON files.",
        "- Raw accuracy levels across datasets are descriptive, not a controlled measure of dataset difficulty.",
        "- ResNet20 is the cleaner protocol comparison; CIFAR-10/VGG16 has uncertain training L provenance.",
        "",
    ]
    for architecture in ("resnet20", "vgg16"):
        c10 = cifar10_payloads[architecture]
        c100 = cifar100_payloads[architecture]
        lines.extend(
            [
                f"## {architecture}",
                "",
                "| Config | CIFAR-10 mean | CIFAR-100 mean | C10-C100 |",
                "|---|---:|---:|---:|",
            ]
        )
        for name in CONFIGS:
            c10_mean = mean_accuracy(c10, name)
            c100_mean = mean_accuracy(c100, name)
            lines.append(
                f"| {name} | {c10_mean:.2f}% | {c100_mean:.2f}% | "
                f"{c10_mean - c100_mean:+.2f}pp |"
            )
        time_steps = c10["protocol"]["time_steps"]
        lines.extend(
            [
                "",
                f"- CIFAR-10 QCFS gate: {selected_time_label('C_QCFS_ASNM_R0', c10['gates'], time_steps)}",
                f"- CIFAR-100 QCFS gate: {selected_time_label('C_QCFS_ASNM_R0', c100['gates'], time_steps)}",
                f"- CIFAR-10 Full-FTBC gate: {selected_time_label('F_QCFS_FULL_FTBC_ASNM_R0', c10['gates'], time_steps)}",
                f"- CIFAR-100 Full-FTBC gate: {selected_time_label('F_QCFS_FULL_FTBC_ASNM_R0', c100['gates'], time_steps)}",
                "",
            ]
        )
    atomic_write(path, "\n".join(lines))


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_parser():
    root = Path("docs/results/comparative_ablation")
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in ("cifar10", "cifar100"):
        for architecture in ("resnet20", "vgg16"):
            parser.add_argument(
                f"--{dataset}_{architecture}",
                type=Path,
                default=root
                / dataset
                / f"FULL_FTBC_ASNM_{dataset.upper()}_{architecture}.progress.json",
            )
    parser.add_argument(
        "--cifar10_output",
        type=Path,
        default=root / "cifar10" / "FULL_FTBC_ASNM_CIFAR10_SUMMARY.md",
    )
    parser.add_argument(
        "--cross_dataset_output",
        type=Path,
        default=root
        / "cifar10"
        / "FULL_FTBC_ASNM_CIFAR10_CIFAR100_COMPARISON.md",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(cli_args=None):
    args = build_parser().parse_args(cli_args)
    outputs = (args.cifar10_output, args.cross_dataset_output)
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite summary output: "
                + ", ".join(str(path) for path in existing)
            )
    source_paths = {
        dataset: {
            architecture: getattr(args, f"{dataset}_{architecture}")
            for architecture in ("resnet20", "vgg16")
        }
        for dataset in ("cifar10", "cifar100")
    }
    payloads = {
        dataset: {
            architecture: load_complete_progress(
                source_paths[dataset][architecture], dataset, architecture
            )
            for architecture in ("resnet20", "vgg16")
        }
        for dataset in ("cifar10", "cifar100")
    }
    write_cifar10_summary(
        args.cifar10_output,
        payloads["cifar10"],
        source_paths["cifar10"],
    )
    write_cross_dataset_summary(
        args.cross_dataset_output,
        payloads["cifar10"],
        payloads["cifar100"],
    )
    for output in outputs:
        print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
