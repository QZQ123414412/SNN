"""
FTBC (Forward Temporal Bias Correction) calibration module for SignedIF neurons.

Computes per-timestep, per-channel bias by comparing SNN spike output against
ANN activation at each layer, then stores the bias inside SignedIF.time_based_bias.
"""

import sys
import torch
import torch.nn as nn
from models.layer import SignedIF


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

    pos_thresh = module_snn.thresh.data
    neg_thresh = module_snn.neg_thresh.data
    enable_signed = module_snn.enable_signed
    enable_r0 = module_snn.enable_r0

    mem = 0.5 * pos_thresh
    transmitted = torch.zeros_like(snn_in[0])
    cumul_spike_sum = torch.zeros_like(snn_in[0])

    for t in range(T):
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

        # replay with corrected dynamics
        mem_corrected = mem - (curr_t_alpha * deviation)
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


@torch.no_grad()
def bias_corr_model(ann, snn, T, train_loader,
                    curr_t_alpha=0.5, num_cali_sample_batches=3):
    """
    Run FTBC calibration over the whole model, layer-by-layer.
    """
    device = next(ann.parameters()).device
    ann.eval()
    snn.eval()

    for i, (inputs, _) in enumerate(train_loader):
        if i >= num_cali_sample_batches:
            break
        print(f"  FTBC calibration batch {i+1}/{num_cali_sample_batches} ...",
              flush=True)
        inputs = inputs.to(device)

        for (name_a, mod_a), (name_s, mod_s) in zip(
                ann.named_modules(), snn.named_modules()):
            if isinstance(mod_s, SignedIF):
                bias_corr_step_by_step(
                    ann, mod_a, snn, mod_s, T, inputs,
                    curr_t_alpha=curr_t_alpha)

    print("  FTBC calibration done.", flush=True)
