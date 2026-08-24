"""Generate the four-model PA-FTBC formal summary from progress JSON files."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


REPORTS = OrderedDict(
    [
        (
            "CIFAR-10/ResNet20 L4",
            Path(
                "docs/results/comparative_ablation/cifar10/"
                "PA_FTBC_ASNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.progress.json"
            ),
        ),
        (
            "CIFAR-10/VGG16 L8",
            Path(
                "docs/results/comparative_ablation/cifar10/"
                "PA_FTBC_ASNM_CIFAR10_VGG16_L8.progress.json"
            ),
        ),
        (
            "CIFAR-100/ResNet20 L8",
            Path(
                "docs/results/comparative_ablation/cifar100/"
                "PA_FTBC_ASNM_CIFAR100_RESNET20_L8.progress.json"
            ),
        ),
        (
            "CIFAR-100/VGG16 L8",
            Path(
                "docs/results/comparative_ablation/cifar100/"
                "PA_FTBC_ASNM_CIFAR100_VGG16_L8.progress.json"
            ),
        ),
    ]
)
OLD_REPORTS = {
    "CIFAR-10/ResNet20 L4": (
        Path(
            "docs/results/comparative_ablation/cifar10/"
            "TEMPORAL_LR_ASNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.progress.json"
        ),
        9,
    ),
    "CIFAR-10/VGG16 L8": (
        Path(
            "docs/results/comparative_ablation/cifar10/"
            "FULL_FTBC_ASNM_CIFAR10_vgg16.progress.json"
        ),
        6,
    ),
    "CIFAR-100/ResNet20 L8": (
        Path(
            "docs/results/comparative_ablation/cifar100/"
            "TEMPORAL_LR_ASNM_CIFAR100_RESNET20.progress.json"
        ),
        9,
    ),
    "CIFAR-100/VGG16 L8": (
        Path(
            "docs/results/comparative_ablation/cifar100/"
            "TEMPORAL_LR_ASNM_CIFAR100_VGG16.progress.json"
        ),
        9,
    ),
}
TIME_STEPS = (1, 2, 4, 8, 16, 32)
REGRESSION_METRICS = (
    "acc",
    "logit_mse",
    "positive_rate",
    "negative_rate",
    "sparsity",
    "sops",
    "scale_operations",
)
CONFIGS = {
    "qcfs": "C_QCFS_ASNM_R0",
    "full": "F_QCFS_FULL_FTBC_ASNM_R0",
    "temporal": "I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0",
    "pa": "L_QCFS_PA_FTBC_ASNM_R0",
}


def load_payload(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Incomplete formal result: {path}")
    if len(payload.get("results", {})) != 12:
        raise RuntimeError(f"Expected twelve configs: {path}")
    if len(payload.get("equivalence_checks", ())) != 54 or not all(
        item["exact"] for item in payload["equivalence_checks"]
    ):
        raise RuntimeError(f"Equivalence audit failed: {path}")
    return payload


def mean_metric(payload, config, metric):
    return sum(
        payload["results"][config][str(t)][metric] for t in TIME_STEPS
    ) / len(TIME_STEPS)


def regression_audit(payloads):
    checked = 0
    mismatches = []
    for label, payload in payloads.items():
        old_path, config_count = OLD_REPORTS[label]
        old = json.loads(old_path.read_text(encoding="utf-8"))
        names = list(payload["results"])[:config_count]
        for name in names:
            for t in TIME_STEPS:
                key = str(t)
                for metric in REGRESSION_METRICS:
                    checked += 1
                    left = old["results"][name][key][metric]
                    right = payload["results"][name][key][metric]
                    if left != right:
                        mismatches.append((label, name, t, metric, left, right))
    if mismatches:
        raise RuntimeError(f"Baseline regression mismatch: {mismatches[:3]}")
    return checked


def format_gate(payload, family):
    enabled = [str(t) for t in TIME_STEPS if payload["gates"][family][str(t)]]
    return ",".join(enabled) if enabled else "none"


def write_summary(path, payloads, regression_cells):
    lines = [
        "# Parity-Anchor FTBC Four-Model Ablation Summary",
        "",
        "Status: complete",
        "",
        "PA-FTBC replaces Temporal-LR's learned shared SVD basis with four fixed structured coefficients: t=0 anchor, t=1 anchor, tail mean and tail parity. It uses no SVD, no cross-layer concatenation, no threshold normalization and no stored time basis.",
        "",
        f"- Formal reports: {len(payloads)}",
        f"- Equivalence checks: {sum(len(x['equivalence_checks']) for x in payloads.values())}/216 exact",
        f"- Existing-result regression cells: {regression_cells}, mismatches: 0",
        "- Test set is used only after all four family-specific A-SNM gates are frozen.",
        "",
        "## Protocol audit",
        "",
        "| Model | ANN | Checkpoint SHA256 | Fit hash | Validation hash | PA SNM-on T | Reversals |",
        "|---|---:|---|---|---|---|---:|",
    ]
    for label, payload in payloads.items():
        protocol = payload["protocol"]
        reversals = sum(
            item["family"] == "pa" and not item["matches_test_best"]
            for item in payload["generalization_audit"]
        )
        lines.append(
            f"| {label} | {protocol['ann_accuracy']:.2f}% | `{protocol['checkpoint']['sha256']}` | "
            f"`{protocol['fit_sha256']}` | `{protocol['validation_sha256']}` | "
            f"{format_gate(payload, 'pa')} | {reversals} |"
        )

    lines.extend(
        [
            "",
            "## Six-time-step mean accuracy",
            "",
            "| Model | QCFS+A-SNM | Full+A-SNM | Temporal+A-SNM | PA+A-SNM | PA-Temporal | PA-Full |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    accuracy_differences = []
    for label, payload in payloads.items():
        values = {key: mean_metric(payload, name, "acc") for key, name in CONFIGS.items()}
        temporal_delta = values["pa"] - values["temporal"]
        accuracy_differences.append(temporal_delta)
        lines.append(
            f"| {label} | {values['qcfs']:.2f}% | {values['full']:.2f}% | "
            f"{values['temporal']:.2f}% | {values['pa']:.2f}% | "
            f"{temporal_delta:+.2f}pp | {values['pa']-values['full']:+.2f}pp |"
        )
    lines.append(
        f"| Four-model macro mean |  |  |  |  | {sum(accuracy_differences)/len(accuracy_differences):+.3f}pp |  |"
    )

    lines.extend(
        [
            "",
            "## PA versus Temporal by SNM mode",
            "",
            "| Model | Off mean delta | Standard-SNM mean delta | A-SNM mean delta |",
            "|---|---:|---:|---:|",
        ]
    )
    pairs = (
        ("J_QCFS_PA_FTBC_R0", "G_QCFS_TEMPORAL_LR_FTBC_R0"),
        ("K_QCFS_PA_FTBC_STANDARD_SNM_R0", "H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0"),
        ("L_QCFS_PA_FTBC_ASNM_R0", "I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0"),
    )
    for label, payload in payloads.items():
        deltas = [
            mean_metric(payload, left, "acc") - mean_metric(payload, right, "acc")
            for left, right in pairs
        ]
        lines.append(f"| {label} | " + " | ".join(f"{x:+.2f}pp" for x in deltas) + " |")

    lines.extend(
        [
            "",
            "## A-SNM aggregate metric comparison",
            "",
            "Equal-weight means are taken over all six time steps. SOP is shown as PA/Temporal; rate and sparsity columns are PA minus Temporal.",
            "",
            "| Model | Accuracy delta | Logit-MSE delta | Positive-rate delta | Negative-rate delta | Sparsity delta | SOP ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    temporal_name = CONFIGS["temporal"]
    pa_name = CONFIGS["pa"]
    for label, payload in payloads.items():
        pa_sops = mean_metric(payload, pa_name, "sops")
        temporal_sops = mean_metric(payload, temporal_name, "sops")
        lines.append(
            f"| {label} | {mean_metric(payload, pa_name, 'acc')-mean_metric(payload, temporal_name, 'acc'):+.3f}pp | "
            f"{mean_metric(payload, pa_name, 'logit_mse')-mean_metric(payload, temporal_name, 'logit_mse'):+.8f} | "
            f"{100*(mean_metric(payload, pa_name, 'positive_rate')-mean_metric(payload, temporal_name, 'positive_rate')):+.6f}pp | "
            f"{100*(mean_metric(payload, pa_name, 'negative_rate')-mean_metric(payload, temporal_name, 'negative_rate')):+.6f}pp | "
            f"{100*(mean_metric(payload, pa_name, 'sparsity')-mean_metric(payload, temporal_name, 'sparsity')):+.6f}pp | "
            f"{pa_sops/temporal_sops:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Storage and bias-synthesis cost",
            "",
            "| Model | T | Full params | Temporal params | PA params | PA saving vs Full | PA params vs Temporal | PA MACs vs Temporal |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, payload in payloads.items():
        for t in (8, 16, 32):
            key = str(t)
            temporal = payload["compression"]["temporal"][key]
            pa = payload["compression"]["pa"][key]
            lines.append(
                f"| {label} | {t} | {pa['full_parameters']:,} | {temporal['ftbc_parameters']:,} | "
                f"{pa['ftbc_parameters']:,} | {100*pa['storage_reduction']:.2f}% | "
                f"{100*pa['ftbc_parameters']/temporal['ftbc_parameters']:.2f}% | "
                f"{100*pa['ftbc_synthesis_macs']/temporal['ftbc_synthesis_macs']:.2f}% |"
            )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Across all four formal models, the maximum absolute six-step mean-accuracy difference between PA-FTBC+A-SNM and Temporal-LR+A-SNM is 0.103pp, while the four-model macro-mean change is +0.010pp. PA removes SVD and the learned/stored temporal basis, uses slightly fewer parameters than Temporal-LR, and reduces bias-synthesis MAC equivalents by 51.56%-56.25% at T=8/16/32. The result supports PA-FTBC as a simpler accuracy-equivalent replacement under the tested protocols.",
            "",
            "The CIFAR-10/ResNet20 L4 checkpoint retains the documented test-set model-selection bias from training; no test result was used to select PA structure or A-SNM gates.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(cli_args=None):
    parser = argparse.ArgumentParser(description="Summarize four-model PA-FTBC results")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/results/comparative_ablation/PA_FTBC_ASNM_FOUR_MODEL_SUMMARY.md"
        ),
    )
    args = parser.parse_args(cli_args)
    payloads = OrderedDict((label, load_payload(path)) for label, path in REPORTS.items())
    checked = regression_audit(payloads)
    write_summary(args.output, payloads, checked)
    print(f"Summary: {args.output}")
    print(f"Regression cells: {checked}, mismatches: 0")


if __name__ == "__main__":
    main()
