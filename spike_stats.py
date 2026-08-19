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
    synaptic_ops_override: int = None
    has_spike_output: bool = True

    @property
    def total_observations(self):
        if not self.has_spike_output:
            return 0
        return max(int(self.time_steps) * int(self.output_neurons_per_step), 1)

    @property
    def total_spikes(self):
        return int(self.positive_spikes) + int(self.negative_spikes)

    @property
    def total_input_spikes(self):
        return int(self.input_positive_spikes) + int(self.input_negative_spikes)

    @property
    def positive_spike_rate(self):
        return self.positive_spikes / max(self.total_observations, 1)

    @property
    def negative_spike_rate(self):
        return self.negative_spikes / max(self.total_observations, 1)

    @property
    def total_spike_rate(self):
        return self.total_spikes / max(self.total_observations, 1)

    @property
    def spike_sparsity(self):
        if not self.has_spike_output:
            return 0.0
        return 1.0 - self.total_spike_rate

    @property
    def sops(self):
        if self.synaptic_ops_override is not None:
            return int(self.synaptic_ops_override)
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
    synthesis_macs = 0
    layers = 0
    for module in model.modules():
        if isinstance(module, signed_if_type):
            parameters += module.ftbc_parameter_count()
            storage_bytes += module.ftbc_storage_bytes()
            synthesis_macs += module.ftbc_synthesis_macs()
            if module.ftbc_parameter_count() > 0:
                layers += 1
    return {
        "layers": layers,
        "parameters": int(parameters),
        "bytes": int(storage_bytes),
        "synthesis_macs": int(synthesis_macs),
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


def collect_resnet20_spike_stats(
    model,
    signed_if_type,
    conv2d_type,
):
    """Collect activation statistics and graph-aware SOPs for CIFAR ResNet20."""
    named_modules = dict(model.named_modules())
    signed_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, signed_if_type)
    }

    expected_names = ["conv1.2"]
    for stage_name in ("conv2_x", "conv3_x", "conv4_x"):
        stage = getattr(model, stage_name)
        for block_index in range(len(stage)):
            prefix = f"{stage_name}.{block_index}"
            expected_names.extend(
                [f"{prefix}.residual_function.2", f"{prefix}.act"]
            )
    if set(signed_modules) != set(expected_names):
        missing = sorted(set(expected_names) - set(signed_modules))
        unexpected = sorted(set(signed_modules) - set(expected_names))
        raise RuntimeError(
            "ResNet20 SignedIF topology mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    def counts(module):
        pos = int(module.pos_spike_count)
        neg = int(module.neg_spike_count)
        pos_by_time = tuple(int(value) for value in module.pos_spike_count_by_time)
        neg_by_time = tuple(int(value) for value in module.neg_spike_count_by_time)
        events_by_time = tuple(
            pos_value + neg_value
            for pos_value, neg_value in zip(pos_by_time, neg_by_time)
        )
        return pos, neg, events_by_time

    def make_stats(
        name,
        input_sources=(),
        fanouts=(),
        kind="Conv2d",
    ):
        module = signed_modules[name]
        positive, negative, _ = counts(module)
        input_positive = 0
        input_negative = 0
        input_by_time = None
        synaptic_ops = 0
        for source_name, fanout in zip(input_sources, fanouts):
            source_positive, source_negative, source_by_time = counts(
                signed_modules[source_name]
            )
            input_positive += source_positive
            input_negative += source_negative
            synaptic_ops += (source_positive + source_negative) * fanout
            if input_by_time is None:
                input_by_time = [0 for _ in source_by_time]
            input_by_time = [
                current + source
                for current, source in zip(input_by_time, source_by_time)
            ]
        time_steps = max(int(module.T), 1)
        if input_by_time is None:
            input_by_time = [0 for _ in range(time_steps)]
        total_neurons = int(module.total_neurons)
        return SpikeLayerStats(
            name=name,
            kind=kind,
            time_steps=time_steps,
            output_neurons_per_step=max(total_neurons // time_steps, 1),
            positive_spikes=positive,
            negative_spikes=negative,
            input_positive_spikes=input_positive,
            input_negative_spikes=input_negative,
            input_spikes_by_time=tuple(input_by_time),
            time_scales=tuple(1.0 for _ in range(time_steps)),
            synaptic_ops_override=synaptic_ops,
        )

    stats = [make_stats("conv1.2")]
    block_input_name = "conv1.2"
    for stage_name in ("conv2_x", "conv3_x", "conv4_x"):
        stage = getattr(model, stage_name)
        for block_index in range(len(stage)):
            prefix = f"{stage_name}.{block_index}"
            internal_name = f"{prefix}.residual_function.2"
            output_name = f"{prefix}.act"

            first_conv = named_modules[f"{prefix}.residual_function.0"]
            if not isinstance(first_conv, conv2d_type):
                raise RuntimeError(f"Expected Conv2d at {prefix}.residual_function.0")
            first_fanout = estimate_conv2d_fanout(
                first_conv.in_channels,
                first_conv.out_channels,
                first_conv.kernel_size,
                first_conv.groups,
            )
            stats.append(
                make_stats(
                    internal_name,
                    input_sources=(block_input_name,),
                    fanouts=(first_fanout,),
                )
            )

            second_conv = named_modules[f"{prefix}.residual_function.3"]
            if not isinstance(second_conv, conv2d_type):
                raise RuntimeError(f"Expected Conv2d at {prefix}.residual_function.3")
            output_sources = [internal_name]
            output_fanouts = [
                estimate_conv2d_fanout(
                    second_conv.in_channels,
                    second_conv.out_channels,
                    second_conv.kernel_size,
                    second_conv.groups,
                )
            ]
            shortcut_name = f"{prefix}.shortcut.0"
            shortcut = named_modules.get(shortcut_name)
            if shortcut is not None:
                if not isinstance(shortcut, conv2d_type):
                    raise RuntimeError(f"Expected Conv2d at {shortcut_name}")
                output_sources.append(block_input_name)
                output_fanouts.append(
                    estimate_conv2d_fanout(
                        shortcut.in_channels,
                        shortcut.out_channels,
                        shortcut.kernel_size,
                        shortcut.groups,
                    )
                )
            stats.append(
                make_stats(
                    output_name,
                    input_sources=tuple(output_sources),
                    fanouts=tuple(output_fanouts),
                )
            )
            block_input_name = output_name
    classifier = getattr(model, "fc")
    source_positive, source_negative, source_by_time = counts(
        signed_modules[block_input_name]
    )
    classifier_fanout = estimate_linear_fanout(
        classifier.in_features,
        classifier.out_features,
    )
    stats.append(
        SpikeLayerStats(
            name="fc",
            kind="LinearReadout",
            time_steps=max(int(signed_modules[block_input_name].T), 1),
            output_neurons_per_step=0,
            positive_spikes=0,
            negative_spikes=0,
            input_positive_spikes=source_positive,
            input_negative_spikes=source_negative,
            input_spikes_by_time=source_by_time,
            time_scales=tuple(1.0 for _ in source_by_time),
            synaptic_ops_override=(
                (source_positive + source_negative) * classifier_fanout
            ),
            has_spike_output=False,
        )
    )
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
        positive_rate = (
            f"{item.positive_spike_rate:>9.4%}"
            if item.has_spike_output
            else f"{'-':>10}"
        )
        negative_rate = (
            f"{item.negative_spike_rate:>9.4%}"
            if item.has_spike_output
            else f"{'-':>10}"
        )
        total_rate_item = (
            f"{item.total_spike_rate:>9.4%}"
            if item.has_spike_output
            else f"{'-':>10}"
        )
        sparsity = (
            f"{item.spike_sparsity:>9.4%}"
            if item.has_spike_output
            else f"{'-':>10}"
        )
        lines.append(
            f"{item.name:<28} {item.kind:<8} "
            f"{positive_rate} {negative_rate} "
            f"{total_rate_item} {sparsity} "
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
