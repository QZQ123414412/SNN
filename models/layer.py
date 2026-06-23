import torch
import torch.nn as nn

from models.temporal_coding import make_event_scales

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class IF(nn.Module):
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0):
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0
        self.refinement_proxy_enabled = False
        self.refinement_proxy_time_steps = 4
        self.refinement_proxy_schedule = "uniform"
        self.refinement_proxy_ratio = 1.0
        self.refinement_proxy_custom_weights = None
        self.refinement_proxy_positive_margin = 0.5
        self.refinement_proxy_negative_margin = 0.5
        self.refinement_proxy_r0_mode = "credit_only"
        self.refinement_event_count = None
        self.refinement_event_total = 0

    def set_refinement_proxy(
        self,
        enabled,
        time_steps=4,
        schedule="uniform",
        ratio=1.0,
        custom_weights=None,
        positive_margin=0.5,
        negative_margin=0.5,
        r0_mode="credit_only",
    ):
        if int(time_steps) < 1:
            raise ValueError("refinement proxy time_steps must be >= 1")
        if positive_margin <= 0 or negative_margin <= 0:
            raise ValueError("refinement proxy margins must be positive")
        if r0_mode not in {"credit_only", "legacy_clamp"}:
            raise ValueError(f"Unsupported refinement proxy R0 mode: {r0_mode}")
        self.refinement_proxy_enabled = bool(enabled)
        self.refinement_proxy_time_steps = int(time_steps)
        self.refinement_proxy_schedule = schedule
        self.refinement_proxy_ratio = float(ratio)
        self.refinement_proxy_custom_weights = (
            None if custom_weights is None else tuple(float(v) for v in custom_weights)
        )
        self.refinement_proxy_positive_margin = float(positive_margin)
        self.refinement_proxy_negative_margin = float(negative_margin)
        self.refinement_proxy_r0_mode = r0_mode

    def reset_refinement_proxy_stats(self):
        self.refinement_event_count = None
        self.refinement_event_total = 0

    def get_refinement_event_rate(self):
        if self.refinement_event_count is None or self.refinement_event_total == 0:
            return self.thresh.new_tensor(0.0)
        return self.refinement_event_count / float(self.refinement_event_total)

    def _qcfs_activation(self, x):
        x = x / self.thresh
        x = torch.clamp(x, 0, 1)
        x = myfloor(x * self.L + 0.5) / self.L
        return x * self.thresh

    def _ste_gate(self, signal):
        hard = (signal >= 0).to(signal.dtype)
        surrogate = torch.clamp(signal + 0.5, 0.0, 1.0)
        return hard + surrogate - surrogate.detach()

    def _record_refinement_events(self, event_count, event_total):
        if self.refinement_event_count is None:
            self.refinement_event_count = event_count
        else:
            self.refinement_event_count = self.refinement_event_count + event_count
        self.refinement_event_total += int(event_total)

    def _forward_refinement_proxy(self, x):
        target = self._qcfs_activation(x)
        T = self.refinement_proxy_time_steps
        if T <= 1:
            return target

        event_scales = make_event_scales(
            T,
            mode=self.refinement_proxy_schedule,
            ratio=self.refinement_proxy_ratio,
            custom_weights=self.refinement_proxy_custom_weights,
            device=target.device,
            dtype=target.dtype,
        )
        threshold = self.thresh.abs().to(device=target.device, dtype=target.dtype)
        eps = torch.finfo(target.dtype).eps * 8
        mem = target * float(T)
        transmitted = torch.zeros_like(target)
        events = []
        event_count = target.new_tensor(0.0)

        for t in range(T):
            quantum = torch.clamp(threshold * event_scales[t], min=eps)
            pos_signal = (mem - self.refinement_proxy_positive_margin * quantum) / quantum
            pos_gate = self._ste_gate(pos_signal)
            neg_signal = (-mem - self.refinement_proxy_negative_margin * quantum) / quantum
            credit_signal = (transmitted - quantum) / quantum
            neg_gate = self._ste_gate(neg_signal) * self._ste_gate(credit_signal)
            spike = (pos_gate - neg_gate) * quantum
            mem = mem - spike
            transmitted = transmitted + spike

            if self.refinement_proxy_r0_mode == "legacy_clamp":
                no_credit = transmitted.abs() <= eps
                transmitted = torch.where(no_credit, torch.zeros_like(transmitted), transmitted)
                mem = torch.where(no_credit, torch.clamp(mem, min=0.0), mem)

            events.append(spike)
            event_count = event_count + pos_gate.sum() + neg_gate.sum()

        self._record_refinement_events(event_count, target.numel() * T)
        return torch.stack(events, dim=0).mean(0)

    def forward(self, x):
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            mem = 0.5 * thre
            spike_pot = []
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre, self.gama) * thre
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        elif self.refinement_proxy_enabled:
            x = self._forward_refinement_proxy(x)
        else:
            x = self._qcfs_activation(x)
        return x

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

# SNM Signed Spike + Memory Neuron (with optional FTBC & R0)
class SignedIF(nn.Module):
    """Signed spike neuron with memory mechanism, optional FTBC bias correction and R0 rule."""
    def __init__(self, T=0, thresh=1.0, enable_signed=True, enable_r0=True):
        super(SignedIF, self).__init__()
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.neg_thresh = nn.Parameter(torch.tensor([-thresh]), requires_grad=True)
        self.T = T
        self.mem = None
        self.transmitted = None
        # ======================= [新增] 消融实验开关 =======================
        self.enable_signed = enable_signed  # 控制是否启用负脉冲（SNM）
        self.enable_r0 = enable_r0          # 控制是否启用 R0 无负债规则
        # ======================= [新增] FTBC 偏置存储 =====================
        self.ftbc_mode = "full"
        self.time_based_bias = None         # 逐时间步通道级偏置，由 calibration.py 写入
        self.bias_base = None
        self.bias_slope = None
        self.bias_state = None
        # ==================================================================

        self.coding_mode = "rate"
        self.refinement_schedule = "uniform"
        self.refinement_ratio = 2.0
        self.custom_time_weights = None
        self.positive_margin = 0.5
        self.negative_margin = 0.5
        self.refinement_r0_mode = "credit_only"
        self.is_input_layer = False
        self.time_scales = None

        self.pos_spike_count = 0
        self.neg_spike_count = 0
        self.total_neurons = 0
        self.pos_spike_count_by_time = [0 for _ in range(max(int(T), 0))]
        self.neg_spike_count_by_time = [0 for _ in range(max(int(T), 0))]
        self.scale_operation_count = 0
        self.collect_spike_stats = True

    def init_mem(self):
        self.mem = None
        self.transmitted = None

    # ========================= [新增] FTBC 重置方法 =======================
    def reset_bias(self):
        self.time_based_bias = None
        self.bias_base = None
        self.bias_slope = None
        self.bias_state = None
    # =====================================================================

    def reset_stats(self):
        self.pos_spike_count = 0
        self.neg_spike_count = 0
        self.total_neurons = 0
        self.pos_spike_count_by_time = [0 for _ in range(max(int(self.T), 0))]
        self.neg_spike_count_by_time = [0 for _ in range(max(int(self.T), 0))]
        self.scale_operation_count = 0

    def set_collect_spike_stats(self, enabled):
        self.collect_spike_stats = bool(enabled)

    def set_coding_mode(
        self,
        mode,
        schedule="uniform",
        ratio=2.0,
        custom_weights=None,
        positive_margin=0.5,
        negative_margin=0.5,
        r0_mode="credit_only",
    ):
        if mode not in {"rate", "monotonic_refinement"}:
            raise ValueError(f"Unsupported coding mode: {mode}")
        if positive_margin <= 0 or negative_margin <= 0:
            raise ValueError("refinement margins must be positive")
        if r0_mode not in {"credit_only", "legacy_clamp"}:
            raise ValueError(f"Unsupported refinement R0 mode: {r0_mode}")
        self.coding_mode = mode
        self.refinement_schedule = schedule
        self.refinement_ratio = float(ratio)
        self.custom_time_weights = (
            None if custom_weights is None else tuple(float(v) for v in custom_weights)
        )
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.refinement_r0_mode = r0_mode
        self.time_scales = None

    @property
    def uses_monotonic_refinement(self):
        return self.coding_mode == "monotonic_refinement" and self.T > 1

    def get_event_scales(self, reference):
        if (
            self.time_scales is None
            or self.time_scales.device != reference.device
            or self.time_scales.dtype != reference.dtype
            or self.time_scales.numel() != self.T
        ):
            self.time_scales = make_event_scales(
                self.T,
                mode=self.refinement_schedule,
                ratio=self.refinement_ratio,
                custom_weights=self.custom_time_weights,
                device=reference.device,
                dtype=reference.dtype,
            )
        return self.time_scales

    def prepare_temporal_input(self, sequence):
        if not (self.uses_monotonic_refinement and self.is_input_layer):
            return sequence
        consolidated = torch.zeros_like(sequence)
        consolidated[0] = sequence.sum(dim=0)
        return consolidated

    def refinement_step(self, input_t, t, mem, transmitted, bias=None):
        if not self.uses_monotonic_refinement:
            raise RuntimeError("refinement_step requires monotonic_refinement mode")
        if bias is None:
            bias = torch.zeros_like(input_t)

        quantum = self.thresh.data.abs() * self.get_event_scales(input_t)[t]
        mem = mem - bias + input_t
        pos_spike = torch.where(
            mem >= self.positive_margin * quantum,
            quantum,
            quantum.new_zeros(()),
        )

        if self.enable_signed:
            credit_tolerance = torch.finfo(mem.dtype).eps * 8
            has_credit = transmitted + credit_tolerance >= quantum
            neg_spike = torch.where(
                (mem <= -self.negative_margin * quantum) & has_credit,
                -quantum,
                quantum.new_zeros(()),
            )
        else:
            neg_spike = torch.zeros_like(pos_spike)

        spike = pos_spike + neg_spike
        mem = mem - spike
        transmitted = transmitted + spike

        if self.enable_r0:
            near_zero = transmitted.abs() <= torch.finfo(transmitted.dtype).eps * 8
            transmitted = torch.where(near_zero, torch.zeros_like(transmitted), transmitted)
            if self.refinement_r0_mode == "legacy_clamp":
                mem = torch.where(
                    transmitted == 0,
                    torch.clamp(mem, min=0.0),
                    mem,
                )
        return spike, mem, transmitted

    def get_stats(self):
        pos_spike_count = int(self.pos_spike_count)
        neg_spike_count = int(self.neg_spike_count)
        total_neurons = int(self.total_neurons)
        return {
            'pos_spike_count': pos_spike_count,
            'neg_spike_count': neg_spike_count,
            'total_neurons': total_neurons,
            'positive_spikes_by_time': [
                int(value) for value in self.pos_spike_count_by_time
            ],
            'negative_spikes_by_time': [
                int(value) for value in self.neg_spike_count_by_time
            ],
            'scale_operations': int(self.scale_operation_count),
            'pos_spike_rate': pos_spike_count / max(total_neurons, 1),
            'neg_spike_rate': neg_spike_count / max(total_neurons, 1)
        }

    def _record_spike_stats(self, spike, t, reference):
        positive = torch.count_nonzero(spike > 0)
        negative = torch.count_nonzero(spike < 0)
        self.pos_spike_count = self.pos_spike_count + positive
        self.neg_spike_count = self.neg_spike_count + negative
        self.total_neurons += reference.numel()
        if len(self.pos_spike_count_by_time) != self.T:
            self.pos_spike_count_by_time = [0 for _ in range(self.T)]
            self.neg_spike_count_by_time = [0 for _ in range(self.T)]
        self.pos_spike_count_by_time[t] = (
            self.pos_spike_count_by_time[t] + positive
        )
        self.neg_spike_count_by_time[t] = (
            self.neg_spike_count_by_time[t] + negative
        )
        if self.uses_monotonic_refinement:
            scale = self.get_event_scales(reference)[t]
            if not torch.isclose(scale, scale.new_tensor(1.0)):
                self.scale_operation_count = (
                    self.scale_operation_count + positive + negative
                )

    # ====================== [新增] FTBC 偏置初始化 =========================
    def set_ftbc_mode(self, mode):
        if mode not in {"none", "full", "state_low_rank"}:
            raise ValueError(f"Unsupported FTBC mode: {mode}")
        if self.ftbc_mode != mode:
            self.reset_bias()
        self.ftbc_mode = mode

    def _init_ftbc_bias(self, channels, device):
        if self.ftbc_mode == "full":
            if self.time_based_bias is None:
                self.time_based_bias = [
                    torch.zeros(channels, device=device) for _ in range(self.T)
                ]
        elif self.ftbc_mode == "state_low_rank":
            if self.bias_base is None:
                self.bias_base = torch.zeros(channels, device=device)
                self.bias_slope = torch.zeros(channels, device=device)
                self.bias_state = torch.zeros(channels, device=device)

    def _reshape_channel_bias(self, channel_bias, reference):
        if reference.dim() == 4:
            return channel_bias.view(1, -1, 1, 1)
        if reference.dim() == 2:
            return channel_bias.view(1, -1)
        if reference.dim() == 1:
            return channel_bias
        raise ValueError(
            f"Unsupported activation rank {reference.dim()} for FTBC bias"
        )

    def get_ftbc_bias(self, t, reference, transmitted=None):
        if self.ftbc_mode == "none":
            return torch.zeros_like(reference)
        if self.ftbc_mode == "full":
            return self._reshape_channel_bias(self.time_based_bias[t], reference)

        tau = float(t) / max(self.T - 1, 1)
        base = self._reshape_channel_bias(self.bias_base, reference)
        slope = self._reshape_channel_bias(self.bias_slope, reference)
        state_bias = self._reshape_channel_bias(self.bias_state, reference)
        if transmitted is None:
            state_term = torch.zeros_like(reference)
        else:
            state = transmitted if transmitted.dtype == torch.bool else transmitted > 0
            state_term = torch.where(state, state_bias, state_bias.new_zeros(()))
        state_term.add_(base + slope * tau)
        return state_term

    def ftbc_parameter_count(self):
        if self.ftbc_mode == "full":
            if self.time_based_bias is None:
                return 0
            return sum(item.numel() for item in self.time_based_bias)
        if self.ftbc_mode == "state_low_rank":
            tensors = (self.bias_base, self.bias_slope, self.bias_state)
            return sum(item.numel() for item in tensors if item is not None)
        return 0

    def ftbc_storage_bytes(self):
        if self.ftbc_mode == "full":
            if self.time_based_bias is None:
                return 0
            return sum(
                item.numel() * item.element_size() for item in self.time_based_bias
            )
        if self.ftbc_mode == "state_low_rank":
            tensors = (self.bias_base, self.bias_slope, self.bias_state)
            return sum(
                item.numel() * item.element_size()
                for item in tensors
                if item is not None
            )
        return 0
    # =====================================================================

    def forward(self, x):
        if self.T > 0:
            batch_size = x.shape[0] // self.T
            x = x.view(self.T, batch_size, *x.shape[1:])

            # ============== [新增] FTBC: 懒初始化偏置 =========================
            self._init_ftbc_bias(x.shape[2], x.device)
            # ==============================================================

            spike_pot = []
            pos_thresh = self.thresh.data
            neg_thresh = self.neg_thresh.data

            if self.uses_monotonic_refinement:
                x = self.prepare_temporal_input(x)
                self.mem = torch.zeros_like(x[0])
                self.transmitted = torch.zeros_like(x[0])
                for t in range(self.T):
                    bias_state = (
                        self.transmitted > 0
                        if self.ftbc_mode == "state_low_rank"
                        else self.transmitted
                    )
                    bias = self.get_ftbc_bias(t, x[t], bias_state)
                    spike, self.mem, self.transmitted = self.refinement_step(
                        input_t=x[t],
                        t=t,
                        mem=self.mem,
                        transmitted=self.transmitted,
                        bias=bias,
                    )
                    if not self.training and self.collect_spike_stats:
                        self._record_spike_stats(spike, t, x[t])
                    spike_pot.append(spike)

                x = torch.stack(spike_pot, dim=0)
                return x.view(self.T * batch_size, *x.shape[2:])

            for t in range(self.T):
                if t == 0:
                    self.mem = 0.5 * pos_thresh
                    self.transmitted = torch.zeros_like(x[t])

                positive_state = None
                if self.ftbc_mode == "state_low_rank" or self.enable_signed:
                    positive_state = self.transmitted > 0

                # ============== [新增] FTBC: 积分前减去时间步偏置 ==========
                # 这是 FTBC 的核心接入点：每个时间步 t，
                # 从膜电位中减去校准得到的通道级偏置 time_based_bias[t]，
                # 让 SNN 的逐步输出更逼近 ANN 的激活值。
                # 偏置初始为全零（不影响原行为），由 calibration.py 写入非零值。
                bias_state = (
                    positive_state
                    if self.ftbc_mode == "state_low_rank"
                    else self.transmitted
                )
                bias = self.get_ftbc_bias(t, x[t], bias_state)
                self.mem = self.mem - bias
                # ===========================================================

                self.mem = self.mem + x[t]

                # Positive spike
                pos_spike = (self.mem >= pos_thresh).float() * pos_thresh

                # ============== [修改] SNM 开关：可关闭负脉冲 ==============
                if self.enable_signed:
                    # Negative spike gated by memory (原 SNM 逻辑)
                    neg_spike = torch.where(
                        (self.mem <= neg_thresh) & positive_state,
                        neg_thresh,
                        neg_thresh.new_zeros(()),
                    )
                else:
                    # 关闭时退化为标准 QCFS 正脉冲
                    neg_spike = torch.zeros_like(pos_spike)
                # ===========================================================

                if not self.training and self.collect_spike_stats:
                    self._record_spike_stats(pos_spike + neg_spike, t, x[t])

                spike = pos_spike + neg_spike

                # Step 3: soft reset + memory update
                self.mem = self.mem - spike
                self.transmitted = self.transmitted + spike

                # ============== [修改] R0 开关：可关闭无负债规则 ============
                # Step 4 (R0): if m(t)==0, v(t) ← max(v(t), 0)
                if self.enable_r0:
                    self.mem = torch.where(
                        self.transmitted == 0,
                        torch.clamp(self.mem, min=0.0),
                        self.mem
                    )
                # ===========================================================

                spike_pot.append(spike)

            x = torch.stack(spike_pot, dim=0)
            x = x.view(self.T * batch_size, *x.shape[2:])
        else:
            # ANN mode
            x = x / self.thresh
            # ============== [修改] ANN 模式也受 SNM 开关控制 ==============
            if self.enable_signed:
                x = torch.clamp(x, -1, 1)   # signed: 允许负值
            else:
                x = torch.clamp(x, 0, 1)    # unsigned: 标准 QCFS
            # ==============================================================
            x = x * self.thresh

        return x
