"""
FTBC (Forward Temporal Bias Correction) calibration module for SignedIF neurons.

Computes per-timestep, per-channel bias by comparing SNN spike output against
ANN activation at each layer, then stores the bias inside SignedIF.time_based_bias.
"""

import sys
import torch
import torch.nn as nn
from models.layer import SignedIF


def reshape_channel_bias(channel_bias, reference):
    """Broadcast a channel-level FTBC bias to match an activation tensor."""
    if reference.dim() == 4:
        return channel_bias.view(1, -1, 1, 1)
    if reference.dim() == 2:
        return channel_bias.view(1, -1)
    if reference.dim() == 1:
        return channel_bias
    raise ValueError(
        f"Unsupported activation rank {reference.dim()} for channel-level FTBC bias"
    )


def accumulate_state_low_rank_statistics(target, state, tau, weight=None):
    """Accumulate channel-wise X^T X and X^T y for X=[1, tau, state]."""
    if target.shape != state.shape:
        raise ValueError("target and state must have the same shape")
    if target.dim() < 2:
        raise ValueError("target must include batch and channel dimensions")

    channels = target.shape[1]
    target_by_channel = target.movedim(1, 0).reshape(channels, -1).float()
    state_by_channel = state.movedim(1, 0).reshape(channels, -1).float()
    samples = target_by_channel.shape[1]

    tau_tensor = torch.as_tensor(tau, device=target.device, dtype=torch.float32)
    if tau_tensor.numel() == 1:
        tau_flat = tau_tensor.expand(samples)
    elif tau_tensor.numel() == samples:
        tau_flat = tau_tensor.reshape(samples)
    else:
        raise ValueError("tau must be scalar or match the samples per channel")

    tau_by_channel = tau_flat.view(1, samples).expand(channels, samples)
    design = torch.stack(
        [torch.ones_like(state_by_channel), tau_by_channel, state_by_channel],
        dim=-1,
    )

    if weight is not None:
        weight_by_channel = weight.movedim(1, 0).reshape(channels, -1).float()
        sqrt_weight = weight_by_channel.clamp_min(0).sqrt().unsqueeze(-1)
        design = design * sqrt_weight
        target_by_channel = target_by_channel * sqrt_weight.squeeze(-1)

    xtx = torch.einsum("cni,cnj->cij", design, design)
    xty = torch.einsum("cni,cn->ci", design, target_by_channel)
    return xtx, xty


def solve_state_low_rank_coefficients(xtx, xty, ridge=1e-4):
    """Solve independent regularized 3x3 systems for every channel."""
    if xtx.dim() != 3 or xtx.shape[-2:] != (3, 3):
        raise ValueError("xtx must have shape [channels, 3, 3]")
    if xty.shape != (xtx.shape[0], 3):
        raise ValueError("xty must have shape [channels, 3]")

    eye = torch.eye(3, device=xtx.device, dtype=xtx.dtype).unsqueeze(0)
    regularized = xtx + float(ridge) * eye
    try:
        return torch.linalg.solve(regularized, xty.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        return torch.matmul(
            torch.linalg.pinv(regularized), xty.unsqueeze(-1)
        ).squeeze(-1)


class ActivationSaverHook:
    """Forward-hook that captures first-call input & output for a module."""
    def __init__(self):
        self.stored_output = None
        self.stored_input = None

    def __call__(self, module, input_batch, output_batch):
        if self.stored_output is None:
            self.stored_output = output_batch.detach().clone()
        if self.stored_input is None:
            self.stored_input = input_batch[0].detach().clone()

    def reset(self):
        self.stored_output = None
        self.stored_input = None


class GetLayerInputOutput:
    """Run a full forward pass and return (input, output) for *target_module*."""
    def __init__(self, model, target_module):
        self.model = model
        self.module = target_module
        self.saver = ActivationSaverHook()

    @torch.no_grad()
    def __call__(self, x):
        self.model.eval()
        h = self.module.register_forward_hook(self.saver)
        _ = self.model(x)
        h.remove()
        inp = self.saver.stored_input.detach()
        out = self.saver.stored_output.detach()
        self.saver.reset()
        return inp, out


def bias_corr_step_by_step(ann, module_ann, snn, module_snn,
                           T, train_data, curr_t_alpha=0.5):
    """
    Core FTBC calibration for one SignedIF layer.

    For signed neurons we use *cumulative-mean* deviation so that negative
    spikes (which are intentional corrections) do not explode the bias.

    For each timestep t:
      1. Locally replay the SignedIF dynamics
      2. Compute deviation:
           - unsigned: deviation_t = spike_t − ann_out       (original FTBC)
           - signed:   deviation_t = cumul_mean_t − ann_out  (stabilised)
      3. Average over spatial & batch dims → channel-level bias
      4. EMA-update module_snn.time_based_bias[t]
    """
    ann_getter = GetLayerInputOutput(ann, module_ann)
    ann_out = ann_getter(train_data.clone())[1]

    snn_getter = GetLayerInputOutput(snn, module_snn)
    snn_in = snn_getter(train_data.clone())[0]
    snn_in = snn_in.view(T, -1, *snn_in.shape[1:])
    if module_snn.uses_monotonic_refinement:
        snn_in = module_snn.prepare_temporal_input(snn_in)

    pos_thresh = module_snn.thresh.data
    neg_thresh = module_snn.neg_thresh.data
    enable_signed = module_snn.enable_signed
    enable_r0 = module_snn.enable_r0

    mem = (
        torch.zeros_like(snn_in[0])
        if module_snn.uses_monotonic_refinement
        else 0.5 * pos_thresh
    )
    transmitted = torch.zeros_like(snn_in[0])
    cumul_spike_sum = torch.zeros_like(snn_in[0])

    for t in range(T):
        if module_snn.uses_monotonic_refinement:
            mem_before = mem
            transmitted_before = transmitted
            bias_existing = module_snn.get_ftbc_bias(
                t,
                snn_in[t],
                transmitted_before,
            )
            _, _, uncorrected_transmitted = module_snn.refinement_step(
                input_t=snn_in[t],
                t=t,
                mem=mem_before,
                transmitted=transmitted_before,
                bias=bias_existing,
            )
            deviation = uncorrected_transmitted / T - ann_out
            if deviation.dim() == 4:
                bias_mean = deviation.mean(dim=[0, 2, 3])
            elif deviation.dim() == 2:
                bias_mean = deviation.mean(dim=0)
            else:
                bias_mean = deviation.mean(dim=0)

            module_snn.time_based_bias[t] = (
                curr_t_alpha * bias_mean + module_snn.time_based_bias[t]
            )
            corrected_bias = module_snn.get_ftbc_bias(
                t,
                snn_in[t],
                transmitted_before,
            )
            _, mem, transmitted = module_snn.refinement_step(
                input_t=snn_in[t],
                t=t,
                mem=mem_before,
                transmitted=transmitted_before,
                bias=corrected_bias,
            )
            continue

        bias_existing = module_snn.time_based_bias[t]
        if len(snn_in.shape) == 5:
            bias_existing = bias_existing.view(1, -1, 1, 1)
        elif len(snn_in.shape) == 3:
            bias_existing = bias_existing.view(1, -1)
        mem = mem - bias_existing

        mem = mem + snn_in[t]
        # 是否放电
        pos_spike = (mem >= pos_thresh).float() * pos_thresh
        if enable_signed:
            neg_spike = (mem <= neg_thresh).float() * neg_thresh
            neg_spike = neg_spike * (transmitted > 0).float()
        else:
            neg_spike = torch.zeros_like(pos_spike)
        spike = pos_spike + neg_spike
        
        # 累计脉冲和
        cumul_spike_sum = cumul_spike_sum + spike

        if enable_signed:
            cumul_mean = cumul_spike_sum / (t + 1)
            deviation = cumul_mean - ann_out
        else:
            deviation = spike - ann_out
        # 计算偏差的均值(bias_mean)
        if len(deviation.shape) == 4:
            bias_mean = deviation.mean(dim=[0, 2, 3])
        elif len(deviation.shape) == 2:
            bias_mean = deviation.mean(dim=0)
        else:
            bias_mean = deviation.mean(dim=0)

        module_snn.time_based_bias[t] = (
            curr_t_alpha * bias_mean + module_snn.time_based_bias[t]
        )

        # Replay with the same channel-level correction that is stored in
        # time_based_bias and later used during deployment inference.
        correction = reshape_channel_bias(curr_t_alpha * bias_mean, deviation)
        mem_corrected = mem - correction
        spike_c_pos = (mem_corrected >= pos_thresh).float() * pos_thresh
        if enable_signed:
            spike_c_neg = (mem_corrected <= neg_thresh).float() * neg_thresh
            spike_c_neg = spike_c_neg * (transmitted > 0).float()
        else:
            spike_c_neg = torch.zeros_like(spike_c_pos)
        spike_corrected = spike_c_pos + spike_c_neg

        mem = mem_corrected - spike_corrected
        transmitted = transmitted + spike_corrected

        if enable_r0:
            mem = torch.where(transmitted == 0,
                              torch.clamp(mem, min=0.0), mem)


def state_low_rank_corr_step_by_step(
    ann,
    module_ann,
    snn,
    module_snn,
    T,
    train_data,
    curr_t_alpha=0.5,
    ridge=1e-4,
    over_weight=1.0,
    under_weight=1.0,
    coefficient_clip=0.0,
):
    """Fit an additive state-conditioned low-rank correction for one layer."""
    ann_getter = GetLayerInputOutput(ann, module_ann)
    ann_out = ann_getter(train_data.clone())[1]

    snn_getter = GetLayerInputOutput(snn, module_snn)
    snn_in = snn_getter(train_data.clone())[0]
    snn_in = snn_in.view(T, -1, *snn_in.shape[1:])
    if module_snn.uses_monotonic_refinement:
        snn_in = module_snn.prepare_temporal_input(snn_in)

    pos_thresh = module_snn.thresh.data
    neg_thresh = module_snn.neg_thresh.data
    enable_signed = module_snn.enable_signed
    enable_r0 = module_snn.enable_r0

    mem = (
        torch.zeros_like(snn_in[0])
        if module_snn.uses_monotonic_refinement
        else 0.5 * pos_thresh
    )
    transmitted = torch.zeros_like(snn_in[0])
    cumulative_spike_sum = torch.zeros_like(snn_in[0])
    channels = snn_in.shape[2]
    xtx = torch.zeros(channels, 3, 3, device=snn_in.device)
    xty = torch.zeros(channels, 3, device=snn_in.device)

    for t in range(T):
        state_before_spike = (transmitted > 0).to(snn_in.dtype)
        bias = module_snn.get_ftbc_bias(t, snn_in[t], transmitted)
        if module_snn.uses_monotonic_refinement:
            spike, mem, transmitted = module_snn.refinement_step(
                input_t=snn_in[t],
                t=t,
                mem=mem,
                transmitted=transmitted,
                bias=bias,
            )
            deviation = transmitted / T - ann_out
        else:
            mem = mem - bias + snn_in[t]
            pos_spike = (mem >= pos_thresh).float() * pos_thresh
            if enable_signed:
                neg_spike = (mem <= neg_thresh).float() * neg_thresh
                neg_spike = neg_spike * state_before_spike
            else:
                neg_spike = torch.zeros_like(pos_spike)
            spike = pos_spike + neg_spike
            cumulative_spike_sum = cumulative_spike_sum + spike

            if enable_signed:
                deviation = cumulative_spike_sum / (t + 1) - ann_out
            else:
                deviation = spike - ann_out

        weight = torch.where(
            deviation > 0,
            torch.full_like(deviation, float(over_weight)),
            torch.full_like(deviation, float(under_weight)),
        )
        step_xtx, step_xty = accumulate_state_low_rank_statistics(
            target=deviation,
            state=state_before_spike,
            tau=float(t) / max(T - 1, 1),
            weight=weight,
        )
        xtx = xtx + step_xtx
        xty = xty + step_xty

        if not module_snn.uses_monotonic_refinement:
            mem = mem - spike
            transmitted = transmitted + spike
            if enable_r0:
                mem = torch.where(
                    transmitted == 0,
                    torch.clamp(mem, min=0.0),
                    mem,
                )

    delta = solve_state_low_rank_coefficients(xtx, xty, ridge=ridge)
    if coefficient_clip > 0:
        limit = float(coefficient_clip) * float(pos_thresh.abs().item())
        delta = delta.clamp(min=-limit, max=limit)

    module_snn.bias_base.add_(curr_t_alpha * delta[:, 0])
    module_snn.bias_slope.add_(curr_t_alpha * delta[:, 1])
    module_snn.bias_state.add_(curr_t_alpha * delta[:, 2])


@torch.no_grad()
def bias_corr_model(ann, snn, T, train_loader,
                    curr_t_alpha=0.5, num_cali_sample_batches=3,
                    ftbc_mode="full", ridge=1e-4,
                    over_weight=1.0, under_weight=1.0,
                    coefficient_clip=0.0):
    """
    Run FTBC calibration over the whole model, layer-by-layer.
    """
    device = next(ann.parameters()).device
    ann.eval()
    snn.eval()
    if hasattr(snn, "set_ftbc_mode"):
        snn.set_ftbc_mode(ftbc_mode)

    for i, (inputs, _) in enumerate(train_loader):
        if i >= num_cali_sample_batches:
            break
        print(f"  FTBC calibration batch {i+1}/{num_cali_sample_batches} ...",
              flush=True)
        inputs = inputs.to(device)

        for (name_a, mod_a), (name_s, mod_s) in zip(
                ann.named_modules(), snn.named_modules()):
            if isinstance(mod_s, SignedIF):
                if ftbc_mode == "state_low_rank":
                    state_low_rank_corr_step_by_step(
                        ann,
                        mod_a,
                        snn,
                        mod_s,
                        T,
                        inputs,
                        curr_t_alpha=curr_t_alpha,
                        ridge=ridge,
                        over_weight=over_weight,
                        under_weight=under_weight,
                        coefficient_clip=coefficient_clip,
                    )
                else:
                    bias_corr_step_by_step(
                        ann, mod_a, snn, mod_s, T, inputs,
                        curr_t_alpha=curr_t_alpha)

    print("  FTBC calibration done.", flush=True)
