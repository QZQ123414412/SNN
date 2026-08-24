"""Validation-accuracy gating for the Signed Neuron with Memory (A-SNM)."""

from collections import OrderedDict


DEFAULT_TIME_STEPS = (1, 2, 4, 8, 16, 32)


def _validate_accuracy_metrics(metrics_by_time, time_steps, label):
    for time_steps_value in time_steps:
        if time_steps_value not in metrics_by_time:
            raise ValueError(f"{label} is missing T={time_steps_value}")
        if "acc" not in metrics_by_time[time_steps_value]:
            raise ValueError(f"{label} T={time_steps_value} is missing acc")


def select_a_snm_modes(
    off_metrics,
    on_metrics,
    time_steps=DEFAULT_TIME_STEPS,
):
    """Select SNM independently at every T using validation accuracy only.

    SNM is enabled exactly when its validation accuracy is strictly greater
    than SNM-off. Ties select SNM-off, which preserves accuracy while avoiding
    unnecessary negative spikes and their downstream event cost.
    """
    time_steps = tuple(int(value) for value in time_steps)
    if not time_steps:
        raise ValueError("At least one time step is required")
    _validate_accuracy_metrics(off_metrics, time_steps, "SNM-off metrics")
    _validate_accuracy_metrics(on_metrics, time_steps, "SNM-on metrics")

    selected = OrderedDict()
    trace = OrderedDict()
    for time_steps_value in time_steps:
        off_accuracy = float(off_metrics[time_steps_value]["acc"])
        on_accuracy = float(on_metrics[time_steps_value]["acc"])
        enabled = on_accuracy > off_accuracy
        selected[time_steps_value] = enabled
        trace[str(time_steps_value)] = {
            "off_accuracy": off_accuracy,
            "on_accuracy": on_accuracy,
            "accuracy_gain": on_accuracy - off_accuracy,
            "selected_mode": "on" if enabled else "off",
        }
    return selected, trace


def a_snm_enabled(selected_modes, time_steps):
    """Return the frozen A-SNM state for one evaluated time horizon."""
    time_steps = int(time_steps)
    if time_steps not in selected_modes:
        raise ValueError(f"A-SNM has no frozen decision for T={time_steps}")
    return bool(selected_modes[time_steps])
