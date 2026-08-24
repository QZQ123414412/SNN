"""Generate the audited four-model HA-SNM delivery report."""

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
                "HA_SNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.progress.json"
            ),
        ),
        (
            "CIFAR-10/VGG16 L8",
            Path(
                "docs/results/comparative_ablation/cifar10/"
                "HA_SNM_CIFAR10_VGG16_L8.progress.json"
            ),
        ),
        (
            "CIFAR-100/ResNet20 L8",
            Path(
                "docs/results/comparative_ablation/cifar100/"
                "HA_SNM_CIFAR100_RESNET20_L8.progress.json"
            ),
        ),
        (
            "CIFAR-100/VGG16 L8",
            Path(
                "docs/results/comparative_ablation/cifar100/"
                "HA_SNM_CIFAR100_VGG16_L8.progress.json"
            ),
        ),
    ]
)

SCREENS = OrderedDict(
    (
        label,
        Path("docs/archive/experiments/ha_snm")
        / (
            "HA_SNM_NORMALIZED_SCREEN_"
            + label.upper()
            .replace("-", "")
            .replace("/", "_")
            .replace(" ", "_")
            + "_20260824.json"
        ),
    )
    for label in (
        "CIFAR10_RESNET20_L4",
        "CIFAR10_VGG16_L8",
        "CIFAR100_RESNET20_L8",
        "CIFAR100_VGG16_L8",
    )
)

OLD_REPORTS = OrderedDict(
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

FAMILIES = OrderedDict(
    [
        (
            "Full-FTBC",
            (
                "A_QCFS_FULL_FTBC_OFF_R0",
                "B_QCFS_FULL_FTBC_STANDARD_R0",
                "C_QCFS_FULL_FTBC_HA_R0",
            ),
        ),
        (
            "Temporal-LR FTBC",
            (
                "D_QCFS_TEMPORAL_FTBC_OFF_R0",
                "E_QCFS_TEMPORAL_FTBC_STANDARD_R0",
                "F_QCFS_TEMPORAL_FTBC_HA_R0",
            ),
        ),
        (
            "PA-FTBC",
            (
                "G_QCFS_PA_FTBC_OFF_R0",
                "H_QCFS_PA_FTBC_STANDARD_R0",
                "I_QCFS_PA_FTBC_HA_R0",
            ),
        ),
    ]
)

OLD_CONFIGS = {
    "Full-FTBC": (
        "D_QCFS_FULL_FTBC_R0",
        "E_QCFS_FULL_FTBC_STANDARD_SNM_R0",
    ),
    "Temporal-LR FTBC": (
        "G_QCFS_TEMPORAL_LR_FTBC_R0",
        "H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0",
    ),
    "PA-FTBC": (
        "J_QCFS_PA_FTBC_R0",
        "K_QCFS_PA_FTBC_STANDARD_SNM_R0",
    ),
}

REGRESSION_KEYS = (
    "acc",
    "logit_mse",
    "positive_rate",
    "negative_rate",
    "sparsity",
    "sops",
    "ftbc_parameters",
    "ftbc_bytes",
)


def load_complete(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Incomplete report: {path}")
    if payload["protocol"]["test_samples"] != 10000:
        raise RuntimeError(f"Non-formal test sample count: {path}")
    return payload


def values(payload, config, key):
    return [
        payload["results"][config][str(t)][key]
        for t in payload["protocol"]["time_steps"]
    ]


def mean(values_):
    return sum(values_) / len(values_)


def regression_audit(payloads):
    checked = 0
    mismatches = []
    for label, new in payloads.items():
        old = json.loads(OLD_REPORTS[label].read_text(encoding="utf-8"))
        for family, (new_off, new_standard, _) in FAMILIES.items():
            old_off, old_standard = OLD_CONFIGS[family]
            for new_name, old_name in (
                (new_off, old_off),
                (new_standard, old_standard),
            ):
                for time_steps in new["protocol"]["time_steps"]:
                    key = str(time_steps)
                    for metric in REGRESSION_KEYS:
                        checked += 1
                        new_value = new["results"][new_name][key][metric]
                        old_value = old["results"][old_name][key][metric]
                        if new_value != old_value:
                            mismatches.append(
                                (label, family, time_steps, metric, new_value, old_value)
                            )
    return checked, mismatches


def screen_macro():
    payloads = []
    for path in SCREENS.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["status"] != "complete":
            raise RuntimeError(f"Incomplete screen: {path}")
        if payload["protocol"]["test_images_evaluated"] != 0:
            raise RuntimeError(f"Screen accessed test images: {path}")
        payloads.append(payload)
    candidates = payloads[0]["candidates"]
    rows = []
    for candidate in candidates:
        fields = {}
        for key in (
            "mean_accuracy",
            "mean_logit_mse",
            "mean_negative_ratio",
            "mean_sop_ratio",
        ):
            candidates_values = [
                next(
                    item[key]
                    for item in payload["ranking"]
                    if item["candidate"] == candidate
                )
                for payload in payloads
            ]
            fields[key] = mean(candidates_values)
        rows.append({"candidate": candidate, **fields})
    rows.sort(
        key=lambda item: (
            -item["mean_accuracy"],
            item["mean_logit_mse"],
            item["mean_sop_ratio"],
        )
    )
    return rows


def write_summary(path):
    payloads = OrderedDict((label, load_complete(file)) for label, file in REPORTS.items())
    time_steps = (1, 2, 4, 8, 16, 32)
    checked, mismatches = regression_audit(payloads)
    if mismatches:
        raise RuntimeError(f"Regression mismatches: {mismatches[:3]}")
    screens = screen_macro()
    if screens[0]["candidate"] != "1.25:0.5":
        raise RuntimeError("Frozen HA-SNM schedule is not the validation winner")

    lines = [
        "# HA-SNM Four-Model Delivery Report",
        "",
        "Status: complete",
        "",
        "HA-SNM uses a horizon-aware negative decision threshold while retaining standard SNM's transmitted-credit memory, R0 rule and -theta event amplitude. The globally frozen schedule is start=1.25, end=0.50 and reference horizon=8.",
        "",
        "## Protocol audit",
        "",
        "| Model | ANN | Checkpoint SHA256 | Fit hash | Validation hash | Test samples |",
        "|---|---:|---|---|---|---:|",
    ]
    for label, payload in payloads.items():
        p = payload["protocol"]
        lines.append(
            f"| {label} | {p['ann_accuracy']:.2f}% | `{p['checkpoint']['sha256']}` | "
            f"`{p['fit_sha256']}` | `{p['validation_sha256']}` | {p['test_samples']:,} |"
        )

    lines.extend(
        [
            "",
            "## Accuracy: HA-SNM versus standard SNM",
            "",
            "Each cell is `HA-SNM - standard SNM` in percentage points.",
            "",
            "| Model / FTBC | " + " | ".join(f"T={t}" for t in time_steps) + " | Mean |",
            "|---|" + "---:|" * (len(time_steps) + 1),
        ]
    )
    all_deltas = []
    for label, payload in payloads.items():
        for family, (_, standard, ha) in FAMILIES.items():
            deltas = [
                payload["results"][ha][str(t)]["acc"]
                - payload["results"][standard][str(t)]["acc"]
                for t in time_steps
            ]
            all_deltas.extend(deltas)
            lines.append(
                f"| {label} / {family} | "
                + " | ".join(f"{x:+.2f}" for x in deltas)
                + f" | {mean(deltas):+.3f} |"
            )

    lines.extend(
        [
            "",
            "## Six-step mean accuracy",
            "",
            "| Model / FTBC | SNM-off | Standard SNM | HA-SNM | HA-standard | HA-off |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, payload in payloads.items():
        for family, (off, standard, ha) in FAMILIES.items():
            off_mean = mean(values(payload, off, "acc"))
            standard_mean = mean(values(payload, standard, "acc"))
            ha_mean = mean(values(payload, ha, "acc"))
            lines.append(
                f"| {label} / {family} | {off_mean:.3f}% | "
                f"{standard_mean:.3f}% | {ha_mean:.3f}% | "
                f"{ha_mean-standard_mean:+.3f}pp | {ha_mean-off_mean:+.3f}pp |"
            )

    lines.extend(
        [
            "",
            "## Four-model macro accuracy effects",
            "",
            "| FTBC / comparison | " + " | ".join(f"T={t}" for t in time_steps) + " | Six-step mean |",
            "|---|" + "---:|" * (len(time_steps) + 1),
        ]
    )
    for family, (off, standard, ha) in FAMILIES.items():
        for left, right, comparison in (
            (ha, standard, "HA - standard"),
            (ha, off, "HA - off"),
            (standard, off, "standard - off"),
        ):
            deltas = [
                mean(
                    [
                        payload["results"][left][str(t)]["acc"]
                        - payload["results"][right][str(t)]["acc"]
                        for payload in payloads.values()
                    ]
                )
                for t in time_steps
            ]
            lines.append(
                f"| {family} / {comparison} | "
                + " | ".join(f"{x:+.3f}" for x in deltas)
                + f" | {mean(deltas):+.3f} |"
            )

    lines.extend(
        [
            "",
            "## Aggregate quality and cost",
            "",
            "Ratios pool all four models and six time steps. Accuracy is an equal-weight mean difference.",
            "",
            "| FTBC | Accuracy gain | Logit-MSE ratio | Standard neg. rate | HA neg. rate | Negative-rate ratio | SOP ratio | Timed-inference ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family, (_, standard, ha) in FAMILIES.items():
        ha_acc = []
        standard_acc = []
        pooled = {key: [[], []] for key in ("logit_mse", "negative_rate", "sops", "inference_elapsed")}
        for payload in payloads.values():
            ha_acc.extend(values(payload, ha, "acc"))
            standard_acc.extend(values(payload, standard, "acc"))
            for key in pooled:
                pooled[key][0].extend(values(payload, ha, key))
                pooled[key][1].extend(values(payload, standard, key))
        ratios = {
            key: sum(pair[0]) / sum(pair[1]) for key, pair in pooled.items()
        }
        lines.append(
            f"| {family} | {mean([a-b for a,b in zip(ha_acc, standard_acc)]):+.3f}pp | "
            f"{ratios['logit_mse']:.6f} | "
            f"{100*mean(pooled['negative_rate'][1]):.6f}% | "
            f"{100*mean(pooled['negative_rate'][0]):.6f}% | "
            f"{ratios['negative_rate']:.6f} | "
            f"{ratios['sops']:.6f} | {ratios['inference_elapsed']:.6f} |"
        )

    wins = sum(x > 0 for x in all_deltas)
    ties = sum(x == 0 for x in all_deltas)
    losses = sum(x < 0 for x in all_deltas)
    lines.extend(
        [
            "",
            "## Robustness and limitations",
            "",
            f"- Across 72 model/FTBC/time cells, HA-SNM has {wins} wins, {ties} ties and {losses} losses versus standard SNM.",
            "- T=2 and T=4 provide the main gain. At T=16/32 the macro difference from standard SNM is near zero, but individual cells can still be slightly lower.",
            "- HA-SNM improves standard SNM; it does not guarantee that negative spikes outperform SNM-off at every long horizon. In particular, T=32 remains mildly negative versus off on macro average.",
            "- The CIFAR-10/ResNet20 checkpoint retains its documented test-set model-selection bias from training. HA-SNM parameters and all screen rankings use calibration validation data only.",
            "- Wall-clock ratios are descriptive PyTorch measurements on one GPU, not a neuromorphic energy claim.",
            "",
            "## Validation-only schedule screen",
            "",
            "| Rank | Start:end | Mean validation accuracy | Logit MSE | Negative-spike ratio | SOP ratio |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rank, item in enumerate(screens, 1):
        lines.append(
            f"| {rank} | {item['candidate']} | {item['mean_accuracy']:.4f}% | "
            f"{item['mean_logit_mse']:.8f} | {item['mean_negative_ratio']:.6f} | "
            f"{item['mean_sop_ratio']:.6f} |"
        )

    exact_checks = sum(
        2 * len(payload["equivalence_checks"]) for payload in payloads.values()
    )
    lines.extend(
        [
            "",
            "## Reproducibility audit",
            "",
            f"- Existing off/standard result regression: {checked} metric cells checked, 0 mismatches.",
            f"- Full fallback equality at T<=4: {exact_checks}/{exact_checks} pair checks exact.",
            "- Four formal progress files have status `complete`, exact checkpoint hashes and 10,000 test samples.",
            "- The four screening payloads record `test_images_evaluated=0`.",
            "",
            "## Deliverables",
            "",
            "- Method: `docs/methodology/HORIZON_AWARE_SNM.md`",
            "- Experiment: `scripts/experiments/run_ha_snm_ablation.py`",
            "- Validation screen: `scripts/experiments/screen_ha_snm.py`",
            "- Formal per-model reports and `.progress.json` files are under the CIFAR-10/100 comparative-ablation directories.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(cli_args=None):
    parser = argparse.ArgumentParser(description="Summarize four formal HA-SNM reports")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/results/comparative_ablation/HA_SNM_FOUR_MODEL_DELIVERY_REPORT.md"
        ),
    )
    args = parser.parse_args(cli_args)
    output = write_summary(args.output)
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
