# 统计SOPs、脉冲率和稀疏性
from dataclasses import dataclass

from models.temporal_coding import make_time_scales


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def estimate_conv2d_fanout(in_channels, out_channels, kernel_size, groups=1):
    del in_channels
    kh, kw = _pair(kernel_size)
    return int((out_channels // groups) * kh * kw)


def estimate_linear_fanout(in_features, out_features):
    del in_features
    return int(out_features)


@dataclass
class SpikeLayerStats:
    name: str
    kind: str
    time_steps: int
    output_neurons_per_step: int
    positive_spikes: int
    negative_spikes: int
    input_positive_spikes: int = 0
    input_negative_spikes: int = 0
    synaptic_ops_per_input_spike: int = 0
    input_spikes_by_time: tuple = ()
    time_scales: tuple = ()

    @property
    def total_observations(self):
        return max(int(self.time_steps) * int(self.output_neurons_per_step), 1)

    @property
    def total_spikes(self):
        return int(self.positive_spikes) + int(self.negative_spikes)

    @property
    def total_input_spikes(self):
        return int(self.input_positive_spikes) + int(self.input_negative_spikes)

    @property
    def positive_spike_rate(self):
        return self.positive_spikes / self.total_observations

    @property
    def negative_spike_rate(self):
        return self.negative_spikes / self.total_observations

    @property
    def total_spike_rate(self):
        return self.total_spikes / self.total_observations

    @property
    def spike_sparsity(self):
        return 1.0 - self.total_spike_rate

    @property
    def sops(self):
        return int(self.total_input_spikes * self.synaptic_ops_per_input_spike)

    @property
    def scale_operations(self):
        if not self.input_spikes_by_time or not self.time_scales:
            return 0
        scaled_events = sum(
            int(spikes)
            for spikes, scale in zip(
                self.input_spikes_by_time,
                self.time_scales,
            )
            if abs(float(scale) - 1.0) > 1e-6
        )
        return int(scaled_events * self.synaptic_ops_per_input_spike)


def reset_signed_spike_stats(model, signed_if_type):
    for module in model.modules():
        if isinstance(module, signed_if_type):
            module.reset_stats()


def set_signed_spike_stats_enabled(model, signed_if_type, enabled):
    for module in model.modules():
        if isinstance(module, signed_if_type):
            module.set_collect_spike_stats(enabled)


def summarize_ftbc_storage(model, signed_if_type):
    parameters = 0
    storage_bytes = 0
    layers = 0
    for module in model.modules():
        if isinstance(module, signed_if_type):
            parameters += module.ftbc_parameter_count()
            storage_bytes += module.ftbc_storage_bytes()
            if module.ftbc_parameter_count() > 0:
                layers += 1
    return {
        "layers": layers,
        "parameters": int(parameters),
        "bytes": int(storage_bytes),
    }


def collect_signed_spike_stats(model, signed_if_type, conv2d_type, linear_type):
    stats = []
    pending_compute = None
    previous_spike_source = None
    previous_spike_source_by_time = None

    for name, module in model.named_modules():
        if isinstance(module, conv2d_type):
            pending_compute = (
                "Conv2d",
                estimate_conv2d_fanout(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.groups,
                ),
            )
        elif isinstance(module, linear_type):
            pending_compute = (
                "Linear",
                estimate_linear_fanout(module.in_features, module.out_features),
            )
        elif isinstance(module, signed_if_type):
            kind, ops_per_input_spike = pending_compute or ("Unknown", 0)
            # SOPs are input-driven: each layer is charged by spikes emitted
            # from the previous SignedIF layer. The raw image input before the
            # first spiking layer is not modeled as a spike source here, so the
            # first layer reports 0 SOPs unless an explicit input encoder is
            # added later.
            input_pos, input_neg = previous_spike_source or (0, 0)
            time_steps = max(int(getattr(module, "T", 0)), 1)
            pos_by_time = tuple(
                int(value)
                for value in getattr(module, "pos_spike_count_by_time", ())
            )
            neg_by_time = tuple(
                int(value)
                for value in getattr(module, "neg_spike_count_by_time", ())
            )
            output_by_time = tuple(
                pos + neg for pos, neg in zip(pos_by_time, neg_by_time)
            )
            input_by_time = previous_spike_source_by_time or tuple(
                0 for _ in range(time_steps)
            )
            scale_mode = (
                getattr(module, "refinement_schedule", "geometric")
                if module.uses_successive_refinement()
                else "rate"
            )
            time_scales = tuple(
                float(value)
                for value in make_time_scales(
                    time_steps,
                    mode=scale_mode,
                    ratio=getattr(module, "refinement_ratio", 2.0),
                )
            )
            total_neurons = int(getattr(module, "total_neurons", 0))
            output_neurons_per_step = max(total_neurons // time_steps, 1)
            pos_spikes = int(getattr(module, "pos_spike_count", 0))
            neg_spikes = int(getattr(module, "neg_spike_count", 0))
            stats.append(
                SpikeLayerStats(
                    name=name,
                    kind=kind,
                    time_steps=time_steps,
                    output_neurons_per_step=output_neurons_per_step,
                    positive_spikes=pos_spikes,
                    negative_spikes=neg_spikes,
                    input_positive_spikes=input_pos,
                    input_negative_spikes=input_neg,
                    synaptic_ops_per_input_spike=ops_per_input_spike,
                    input_spikes_by_time=input_by_time,
                    time_scales=time_scales,
                )
            )
            previous_spike_source = (pos_spikes, neg_spikes)
            previous_spike_source_by_time = output_by_time
            pending_compute = None

    return stats


def format_spike_stats_report(layer_stats):
    total_pos = sum(item.positive_spikes for item in layer_stats)
    total_neg = sum(item.negative_spikes for item in layer_stats)
    total_obs = sum(item.total_observations for item in layer_stats)
    total_sops = sum(item.sops for item in layer_stats)
    total_scale_ops = sum(item.scale_operations for item in layer_stats)

    lines = []
    lines.append("\nSpike/SOPs statistics")
    lines.append("=" * 120)
    lines.append(
        f"{'Layer':<28} {'Type':<8} {'PosRate':>10} {'NegRate':>10} "
        f"{'TotalRate':>10} {'Sparsity':>10} {'InputSpikes':>14} "
        f"{'SOPs':>16} {'ScaleOps':>16}"
    )
    lines.append("-" * 120)

    for item in layer_stats:
        lines.append(
            f"{item.name:<28} {item.kind:<8} "
            f"{item.positive_spike_rate:>9.4%} {item.negative_spike_rate:>9.4%} "
            f"{item.total_spike_rate:>9.4%} {item.spike_sparsity:>9.4%} "
            f"{item.total_input_spikes:>14,d} "
            f"{item.sops:>16,d} {item.scale_operations:>16,d}"
        )

    total_rate = (total_pos + total_neg) / max(total_obs, 1)
    lines.append("-" * 120)
    lines.append(f"Total positive spikes: {total_pos:,}")
    lines.append(f"Total negative spikes: {total_neg:,}")
    lines.append(f"Total spike rate: {total_rate:.4%}")
    lines.append(f"Overall spike sparsity: {1.0 - total_rate:.4%}")
    lines.append(f"Total SOPs (input-driven): {total_sops:,}")
    lines.append(f"Total time-scale operations: {total_scale_ops:,}")
    lines.append("=" * 120)
    return "\n".join(lines)
