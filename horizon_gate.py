from collections import OrderedDict
import math


def aggregate_subset_metrics(subset_metrics, std_weight=0.5):
    """Aggregate disjoint validation subsets without pooling away instability."""
    if not subset_metrics:
        raise ValueError("At least one validation subset is required")
    required = {"acc", "logit_mse", "sops", "negative_spikes"}
    for metrics in subset_metrics:
        missing = required.difference(metrics)
        if missing:
            raise ValueError(f"Missing validation metrics: {sorted(missing)}")

    def mean(key):
        return sum(float(item[key]) for item in subset_metrics) / len(subset_metrics)

    mean_acc = mean("acc")
    variance = sum(
        (float(item["acc"]) - mean_acc) ** 2 for item in subset_metrics
    ) / len(subset_metrics)
    std_acc = math.sqrt(variance)
    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "robust_acc": mean_acc - float(std_weight) * std_acc,
        "mean_logit_mse": mean("logit_mse"),
        "mean_sops": mean("sops"),
        "mean_negative_spikes": mean("negative_spikes"),
        "subsets": list(subset_metrics),
    }


def select_robust_candidate(
    candidates,
    accuracy_tolerance=0.1,
    sop_weight=0.05,
    negative_weight=0.02,
):
    """Select accuracy-stable candidate, then minimize event overhead.

    ``candidates`` maps a stable name to an aggregate returned by
    :func:`aggregate_subset_metrics`. Accuracy is measured in percentage
    points, so the default tolerance is 0.1pp.
    """
    if not candidates:
        raise ValueError("At least one candidate is required")
    best_robust = max(float(item["robust_acc"]) for item in candidates.values())
    eligible = OrderedDict(
        (name, item)
        for name, item in candidates.items()
        if float(item["robust_acc"]) >= best_robust - float(accuracy_tolerance)
    )
    min_sops = max(min(float(item["mean_sops"]) for item in eligible.values()), 1.0)
    positive_negative_counts = [
        float(item["mean_negative_spikes"])
        for item in eligible.values()
        if float(item["mean_negative_spikes"]) > 0
    ]
    min_negative = min(positive_negative_counts) if positive_negative_counts else 1.0

    def overhead(item):
        negative = float(item["mean_negative_spikes"])
        negative_ratio = 0.0 if negative == 0 else negative / min_negative
        return (
            float(sop_weight) * float(item["mean_sops"]) / min_sops
            + float(negative_weight) * negative_ratio
        )

    winner = min(
        eligible,
        key=lambda name: (
            float(eligible[name]["mean_logit_mse"]),
            overhead(eligible[name]),
            -float(eligible[name]["robust_acc"]),
            name,
        ),
    )
    trace = OrderedDict()
    for name, item in candidates.items():
        trace[name] = {
            **item,
            "accuracy_eligible": name in eligible,
            "overhead_score": overhead(item) if name in eligible else None,
        }
    return winner, trace
