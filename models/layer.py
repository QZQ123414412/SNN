import torch
import torch.nn as nn

from .temporal_coding import make_time_scales

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
        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh
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
        self.coding_mode = "rate"
        self.refinement_schedule = "geometric"
        self.refinement_ratio = 2.0
        self.refinement_positive_margin = 0.5
        self.refinement_negative_margin = 0.5
        self.refinement_r0_mode = "credit_only"
        self.time_scales = None
        self._time_scale_cache_key = None
        # ==================================================================

        self.pos_spike_count = 0
        self.neg_spike_count = 0
        self.total_neurons = 0
        self.pos_spike_count_by_time = [0 for _ in range(max(int(T), 0))]
        self.neg_spike_count_by_time = [0 for _ in range(max(int(T), 0))]
        self.collect_spike_stats = True

    def init_mem(self):
        self.mem = None
        self.transmitted = None

    def set_coding_mode(
        self,
        mode,
        schedule="geometric",
        ratio=2.0,
        positive_margin=0.5,
        negative_margin=0.5,
        r0_mode="credit_only",
    ):
        if mode not in {"rate", "successive_refinement"}:
            raise ValueError(f"Unsupported coding mode: {mode}")
        if schedule not in {"rate", "geometric"}:
            raise ValueError(f"Unsupported refinement schedule: {schedule}")
        if float(ratio) < 1.0:
            raise ValueError("refinement ratio must be greater than or equal to one")
        if float(positive_margin) <= 0 or float(negative_margin) <= 0:
            raise ValueError("refinement margins must be positive")
        if r0_mode not in {"credit_only", "legacy_clamp"}:
            raise ValueError(f"Unsupported refinement R0 mode: {r0_mode}")
        self.coding_mode = mode
        self.refinement_schedule = schedule
        self.refinement_ratio = float(ratio)
        self.refinement_positive_margin = float(positive_margin)
        self.refinement_negative_margin = float(negative_margin)
        self.refinement_r0_mode = r0_mode
        self.time_scales = None
        self._time_scale_cache_key = None
        self.init_mem()

    def get_time_scales(self, reference):
        schedule = (
            "rate"
            if self.coding_mode == "rate"
            else self.refinement_schedule
        )
        cache_key = (
            int(self.T),
            schedule,
            float(self.refinement_ratio),
            reference.device,
            reference.dtype,
        )
        if self.time_scales is None or self._time_scale_cache_key != cache_key:
            self.time_scales = make_time_scales(
                self.T,
                mode=schedule,
                ratio=self.refinement_ratio,
                device=reference.device,
                dtype=reference.dtype,
            )
            self._time_scale_cache_key = cache_key
        return self.time_scales

    def get_time_scale(self, t, reference):
        return self.get_time_scales(reference)[t]

    def uses_successive_refinement(self):
        return self.coding_mode == "successive_refinement" and int(self.T) > 1

    def decode_transmitted(self, transmitted, t):
        if not self.uses_successive_refinement():
            return transmitted / float(t + 1)
        scales = self.get_time_scales(transmitted)
        return transmitted / scales[:t + 1].sum()

    def _refinement_tolerance(self, reference):
        return (
            torch.finfo(reference.dtype).eps
            * self.thresh.data.abs().clamp_min(1.0)
            * 16
        )

    def refinement_step(self, input_t, t, mem, transmitted, bias=None):
        """Advance one signed successive-refinement step."""
        scale = self.get_time_scale(t, input_t)
        quantum = scale * self.thresh.data
        tolerance = self._refinement_tolerance(input_t)
        if bias is None:
            bias = torch.zeros_like(input_t)

        mem = mem - scale * bias + scale * input_t
        pos_mask = mem >= self.refinement_positive_margin * quantum
        pos_spike = pos_mask.to(input_t.dtype) * self.thresh.data

        if self.enable_signed:
            neg_mask = (
                (mem <= -self.refinement_negative_margin * quantum)
                & (transmitted + tolerance >= quantum)
            )
            neg_spike = torch.where(
                neg_mask,
                self.neg_thresh.data,
                self.neg_thresh.data.new_zeros(()),
            )
        else:
            neg_mask = torch.zeros_like(pos_mask)
            neg_spike = torch.zeros_like(pos_spike)

        spike = pos_spike + neg_spike
        signed_event = pos_mask.to(input_t.dtype) - neg_mask.to(input_t.dtype)
        weighted_event = signed_event * quantum
        mem = mem - weighted_event
        transmitted = transmitted + weighted_event
        transmitted = torch.where(
            transmitted.abs() <= tolerance,
            torch.zeros_like(transmitted),
            transmitted,
        )

        if self.enable_r0:
            no_credit = transmitted <= tolerance
            transmitted = torch.where(
                no_credit,
                torch.zeros_like(transmitted),
                transmitted,
            )
            if self.refinement_r0_mode == "legacy_clamp":
                mem = torch.where(
                    no_credit,
                    torch.clamp(mem, min=0.0),
                    mem,
                )
        return spike, mem, transmitted

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

    def set_collect_spike_stats(self, enabled):
        self.collect_spike_stats = bool(enabled)

    def get_stats(self):
        pos_spike_count = int(self.pos_spike_count)
        neg_spike_count = int(self.neg_spike_count)
        total_neurons = int(self.total_neurons)
        return {
            'pos_spike_count': pos_spike_count,
            'neg_spike_count': neg_spike_count,
            'total_neurons': total_neurons,
            'positive_spikes_by_time': tuple(
                int(value) for value in self.pos_spike_count_by_time
            ),
            'negative_spikes_by_time': tuple(
                int(value) for value in self.neg_spike_count_by_time
            ),
            'pos_spike_rate': pos_spike_count / max(total_neurons, 1),
            'neg_spike_rate': neg_spike_count / max(total_neurons, 1)
        }

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

            for t in range(self.T):
                if self.uses_successive_refinement():
                    if t == 0:
                        self.mem = torch.zeros_like(x[t])
                        self.transmitted = torch.zeros_like(x[t])

                    bias = self.get_ftbc_bias(t, x[t], self.transmitted)
                    spike, self.mem, self.transmitted = self.refinement_step(
                        x[t],
                        t,
                        self.mem,
                        self.transmitted,
                        bias=bias,
                    )

                    if not self.training and self.collect_spike_stats:
                        pos_count = torch.count_nonzero(spike > 0)
                        neg_count = torch.count_nonzero(spike < 0)
                        self.pos_spike_count = (
                            self.pos_spike_count + pos_count
                        )
                        self.neg_spike_count = (
                            self.neg_spike_count + neg_count
                        )
                        self.pos_spike_count_by_time[t] = (
                            self.pos_spike_count_by_time[t] + pos_count
                        )
                        self.neg_spike_count_by_time[t] = (
                            self.neg_spike_count_by_time[t] + neg_count
                        )
                        self.total_neurons += x[t].numel()

                    spike_pot.append(spike)
                    continue

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
                    pos_count = torch.count_nonzero(pos_spike)
                    neg_count = torch.count_nonzero(neg_spike)
                    self.pos_spike_count = (
                        self.pos_spike_count + pos_count
                    )
                    self.neg_spike_count = (
                        self.neg_spike_count + neg_count
                    )
                    self.pos_spike_count_by_time[t] = (
                        self.pos_spike_count_by_time[t] + pos_count
                    )
                    self.neg_spike_count_by_time[t] = (
                        self.neg_spike_count_by_time[t] + neg_count
                    )
                    self.total_neurons += x[t].numel()

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
