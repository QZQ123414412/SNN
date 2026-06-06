import torch
import torch.nn as nn

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
        self.time_based_bias = None         # 逐时间步通道级偏置，由 calibration.py 写入
        # ==================================================================

        self.pos_spike_count = 0
        self.neg_spike_count = 0
        self.total_neurons = 0

    def init_mem(self):
        self.mem = None
        self.transmitted = None

    # ========================= [新增] FTBC 重置方法 =======================
    def reset_bias(self):
        self.time_based_bias = None
    # =====================================================================

    def reset_stats(self):
        self.pos_spike_count = 0
        self.neg_spike_count = 0
        self.total_neurons = 0

    def get_stats(self):
        return {
            'pos_spike_count': self.pos_spike_count,
            'neg_spike_count': self.neg_spike_count,
            'total_neurons': self.total_neurons,
            'pos_spike_rate': self.pos_spike_count / max(self.total_neurons, 1),
            'neg_spike_rate': self.neg_spike_count / max(self.total_neurons, 1)
        }

    # ====================== [新增] FTBC 偏置初始化 =========================
    def _init_time_based_bias(self, channels, device):
        """懒初始化：首次 forward 时按通道数创建 T 个全零偏置向量"""
        if self.time_based_bias is None:
            self.time_based_bias = [torch.zeros(channels, device=device) for _ in range(self.T)]
    # =====================================================================

    def forward(self, x):
        if self.T > 0:
            batch_size = x.shape[0] // self.T
            x = x.view(self.T, batch_size, *x.shape[1:])

            # ============== [新增] FTBC: 懒初始化偏置 =========================
            self._init_time_based_bias(x.shape[2], x.device)
            # ==============================================================

            spike_pot = []
            pos_thresh = self.thresh.data
            neg_thresh = self.neg_thresh.data

            for t in range(self.T):
                if t == 0:
                    self.mem = 0.5 * pos_thresh
                    self.transmitted = torch.zeros_like(x[t])

                # ============== [新增] FTBC: 积分前减去时间步偏置 ==========
                # 这是 FTBC 的核心接入点：每个时间步 t，
                # 从膜电位中减去校准得到的通道级偏置 time_based_bias[t]，
                # 让 SNN 的逐步输出更逼近 ANN 的激活值。
                # 偏置初始为全零（不影响原行为），由 calibration.py 写入非零值。
                bias = self.time_based_bias[t]
                if len(x.shape) == 5:       # Conv 层: (T, B, C, H, W)
                    bias = bias.view(1, -1, 1, 1)
                elif len(x.shape) == 3:     # FC 层: (T, B, C)
                    bias = bias.view(1, -1)
                self.mem = self.mem - bias
                # ===========================================================

                self.mem = self.mem + x[t]

                # Positive spike
                pos_spike = (self.mem >= pos_thresh).float() * pos_thresh

                # ============== [修改] SNM 开关：可关闭负脉冲 ==============
                if self.enable_signed:
                    # Negative spike gated by memory (原 SNM 逻辑)
                    neg_spike = (self.mem <= neg_thresh).float() * neg_thresh
                    neg_spike = neg_spike * (self.transmitted > 0).float()
                else:
                    # 关闭时退化为标准 QCFS 正脉冲
                    neg_spike = torch.zeros_like(pos_spike)
                # ===========================================================

                if not self.training:
                    self.pos_spike_count += (pos_spike != 0).sum().item()
                    self.neg_spike_count += (neg_spike != 0).sum().item()
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
