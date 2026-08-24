"""Combine the two completed ImageNet Full/PA-FTBC + HA-SNM runs."""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.experiments.run_imagenet_ftbc_pa_ha_ablation import (
    CONFIGS,
    DEFAULT_TIME_STEPS,
    PROTOCOLS,
)


RESULT_DIR = Path("docs/results/comparative_ablation/imagenet")
DEFAULT_PROGRESS = {
    architecture: RESULT_DIR
    / (
        f"IMAGENET_{architecture.upper()}_L{protocol['L']}_"
        "FULL_PA_HA_SNM.progress.json"
    )
    for architecture, protocol in PROTOCOLS.items()
}
DEFAULT_OUTPUT = RESULT_DIR / "IMAGENET_FULL_PA_HA_SNM_TWO_MODEL_SUMMARY.md"


def load_completed_progress(path, architecture):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Progress file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Progress is not complete: {path}")
    protocol = payload.get("protocol", {})
    if protocol.get("architecture") != architecture:
        raise RuntimeError(
            f"Expected {architecture} progress, got {protocol.get('architecture')}"
        )
    if tuple(protocol.get("time_steps", ())) != DEFAULT_TIME_STEPS:
        raise RuntimeError(f"Unexpected time-step protocol in {path}")
    for name in CONFIGS:
        results = payload.get("results", {}).get(name, {})
        missing = [value for value in DEFAULT_TIME_STEPS if str(value) not in results]
        if missing:
            raise RuntimeError(f"{path} is missing {name} at T={missing}")
    return payload


def mean_metric(payload, config_name, key):
    return sum(
        float(payload["results"][config_name][str(value)][key])
        for value in DEFAULT_TIME_STEPS
    ) / len(DEFAULT_TIME_STEPS)


def accuracy_section(lines, architecture, payload):
    lines.extend(
        [
            f"## {architecture} accuracy",
            "",
            "| Config | "
            + " | ".join(f"T={value}" for value in DEFAULT_TIME_STEPS)
            + " | Mean Top-1 | Mean Top-5 |",
            "|---|" + "---:|" * (len(DEFAULT_TIME_STEPS) + 2),
        ]
    )
    for name in CONFIGS:
        values = payload["results"][name]
        top1_cells = [f"{values[str(value)]['top1']:.2f}%" for value in DEFAULT_TIME_STEPS]
        lines.append(
            f"| {name} | "
            + " | ".join(top1_cells)
            + f" | {mean_metric(payload, name, 'top1'):.2f}%"
            + f" | {mean_metric(payload, name, 'top5'):.2f}% |"
        )
    lines.append("")


def effect_section(lines, architecture, payload):
    comparisons = (
        ("Full-FTBC (SNM off)", "A_QCFS_R0", "D_QCFS_FULL_FTBC_R0"),
        ("PA-FTBC (SNM off)", "A_QCFS_R0", "G_QCFS_PA_FTBC_R0"),
        ("Standard SNM on QCFS", "A_QCFS_R0", "B_QCFS_STANDARD_SNM_R0"),
        ("HA-SNM on QCFS", "A_QCFS_R0", "C_QCFS_HA_SNM_R0"),
        (
            "Standard SNM on Full",
            "D_QCFS_FULL_FTBC_R0",
            "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
        ),
        (
            "HA-SNM on Full",
            "D_QCFS_FULL_FTBC_R0",
            "F_QCFS_FULL_FTBC_HA_SNM_R0",
        ),
        (
            "Standard SNM on PA",
            "G_QCFS_PA_FTBC_R0",
            "H_QCFS_PA_FTBC_STANDARD_SNM_R0",
        ),
        (
            "HA-SNM on PA",
            "G_QCFS_PA_FTBC_R0",
            "I_QCFS_PA_FTBC_HA_SNM_R0",
        ),
    )
    lines.extend(
        [
            f"## {architecture} Top-1 effects",
            "",
            "Positive values mean the named method improves its matched baseline.",
            "",
            "| Effect | "
            + " | ".join(f"T={value}" for value in DEFAULT_TIME_STEPS)
            + " | Mean |",
            "|---|" + "---:|" * (len(DEFAULT_TIME_STEPS) + 1),
        ]
    )
    for title, baseline, method in comparisons:
        differences = [
            float(payload["results"][method][str(value)]["top1"])
            - float(payload["results"][baseline][str(value)]["top1"])
            for value in DEFAULT_TIME_STEPS
        ]
        lines.append(
            f"| {title} | "
            + " | ".join(f"{value:+.2f}pp" for value in differences)
            + f" | {sum(differences) / len(differences):+.2f}pp |"
        )
    lines.append("")


def storage_section(lines, payloads):
    lines.extend(
        [
            "## PA-FTBC storage",
            "",
            "| Architecture | T | Full params | PA params | PA bytes | Saving |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for architecture, payload in payloads.items():
        for value in DEFAULT_TIME_STEPS:
            item = payload["compression"][str(value)]
            full_parameters = int(item["full_parameters"])
            pa_parameters = int(item["pa_parameters"])
            pa_bytes = int(
                item.get(
                    "pa_bytes",
                    payload["results"]["G_QCFS_PA_FTBC_R0"][str(value)][
                        "ftbc_bytes"
                    ],
                )
            )
            saving = 1.0 - pa_parameters / max(full_parameters, 1)
            lines.append(
                f"| {architecture} | {value} | {full_parameters:,} | "
                f"{pa_parameters:,} | {pa_bytes:,} | {100*saving:.2f}% |"
            )
    lines.append("")


def write_summary(path, payloads):
    lines = [
        "# ImageNet Full/PA-FTBC + Standard/HA-SNM Two-Model Summary",
        "",
        "Status: complete",
        "",
        "This summary is generated only from two completed 50,000-image formal runs.",
        "All Top-1 differences are percentage-point differences between matched configurations.",
        "",
        "| Architecture | QCFS L | ANN Top-1 | ANN Top-5 | GPU | Active elapsed |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for architecture, payload in payloads.items():
        lines.append(
            f"| {architecture} | {payload['protocol']['qcfs_L']} | "
            f"{payload['ann']['top1']:.2f}% | {payload['ann']['top5']:.2f}% | "
            f"{payload['runtime'].get('gpu')} | "
            f"{payload['runtime'].get('active_elapsed_seconds', 0.0):.3f}s |"
        )
    lines.append("")
    for architecture, payload in payloads.items():
        accuracy_section(lines, architecture, payload)
        effect_section(lines, architecture, payload)
    storage_section(lines, payloads)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize completed ResNet34 and VGG16 ImageNet ablations"
    )
    parser.add_argument(
        "--resnet34-progress",
        type=Path,
        default=DEFAULT_PROGRESS["resnet34"],
    )
    parser.add_argument(
        "--vgg16-progress",
        type=Path,
        default=DEFAULT_PROGRESS["vgg16"],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(cli_args=None):
    args = build_parser().parse_args(cli_args)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite summary: {args.output}")
    payloads = {
        "ResNet34": load_completed_progress(args.resnet34_progress, "resnet34"),
        "VGG16": load_completed_progress(args.vgg16_progress, "vgg16"),
    }
    write_summary(args.output, payloads)
    print(f"Summary: {args.output}")


if __name__ == "__main__":
    main()
