# Error-aware Energy-efficient ANN-to-SNN Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 QCFS + SNM + R0 + FTBC 代码基础上，实现一个“误差感知、能耗约束、层级自适应”的 ANN-to-SNN 转换 pipeline，使模型在低时间步下获得更高精度，同时降低不必要的 SOPs、负脉冲、FTBC bias 存储量和实际推理延迟。

**Architecture:** 新 pipeline 不再把 SNM、R0、FTBC 作为全网统一开关，而是先进行层级转换误差诊断，再根据每层的精度收益、SOPs 成本、负脉冲成本和 bias 存储成本选择模块组合。核心流程为：QCFS 源 ANN 训练或加载 -> SNN 转换 -> 层级误差诊断 -> 选择性模块搜索 -> 压缩 FTBC 校准 -> 在统一固定时间步下进行精度、能耗和延迟评估。

**Tech Stack:** PyTorch, torchvision, existing QCFS code, `SignedIF`, `calibration.py`, `spike_stats.py`, `run_stats_ablation.py`, Python unittest, Markdown experiment reports.

---

## 1. 当前代码基础与主要瓶颈

当前分支已经具备以下基础能力：

- `models/layer.py` 中已有 `SignedIF`，支持正脉冲、负脉冲 SNM、R0 规则和 FTBC 的 `time_based_bias`。
- `models/VGG.py` 中已有 `VGG_Signed`，支持 `set_signed()`、`set_r0()`、`reset_all_bias()`。
- `calibration.py` 中已有 FTBC 校准逻辑，将每个时间步、每个通道的 bias 写入 `SignedIF.time_based_bias`。
- `spike_stats.py` 已能统计 input-driven SOPs、正/负脉冲率和每层 sparsity。
- `run_stats_ablation.py` 已能运行 A-F 六组配置：
  - `A_QCFS`
  - `B_QCFS+SNM`
  - `C_QCFS+SNM+R0`
  - `D_QCFS+FTBC`
  - `E_QCFS+SNM+FTBC`
  - `F_QCFS+SNM+R0+FTBC`

从已有 CIFAR-100/VGG16 结果看，现有全局开关 pipeline 有两个问题：

1. **模块收益不是全局一致的。**  
   `SNM/R0` 在低时间步通常提升精度，但也引入负脉冲、更多 SOPs 和更高延迟。`FTBC` 可明显降低 SOPs、提升 sparsity，但校准和 bias 存储有成本。

2. **缺少层级转换误差解释。**  
   目前知道最终 accuracy、SOPs 和 spike rate，但不知道是哪一层的 ANN activation 与 SNN rate output 偏差最大，也不知道某一层更适合 SNM、R0 还是 FTBC。

因此，后续创新不应继续堆模块，而应实现 **Error-aware Selective Calibration for QCFS-SNN，简称 EASC-QCFS**。

---

## 2. 目标创新点定义

建议论文主创新点表述为：

> 本文提出一种面向低时延、低能耗部署的误差感知选择性 ANN-to-SNN 转换框架。该框架首先量化每层 ANN-SNN 转换误差、脉冲活动和能耗成本，然后在精度、SOPs、负脉冲率和 FTBC bias 存储量约束下，按层选择 SNM、R0 和 FTBC 校准策略，并在相同固定时间步下与其他方法进行公平比较。

该创新点对应三个工程模块：

1. **Layer-wise Conversion Error Diagnosis**  
   层级 ANN-SNN 转换误差诊断。

2. **Energy-aware Selective Module Policy**  
   能耗约束的层级模块选择策略。

3. **Compressed FTBC**  
   压缩时序 bias 校准，降低 FTBC 存储和部署成本。

---

## 3. 文件结构规划

### 新建文件

- `conversion_diagnostics.py`  
  负责收集 ANN activation、SNN spike-rate output、membrane residual、逐层转换误差。

- `selective_policy.py`  
  负责定义每层模块开关策略、候选配置评分函数、贪心搜索或 Pareto 搜索。

- `compressed_ftbc.py`  
  负责实现 FTBC bias 的完整存储、线性压缩、分段常数压缩和重建逻辑。

- `energy_model.py`  
  负责把 SOPs、AC/MAC、负脉冲、bias 读取量转成可报告的能耗估计。

- `run_selective_pipeline.py`  
  负责端到端运行 EASC-QCFS：诊断、搜索、校准、验证、输出 Markdown。

- `tests/test_conversion_diagnostics.py`  
  验证层级误差公式、hook 收集、SNN rate output 形状。

- `tests/test_selective_policy.py`  
  验证候选策略评分和贪心选择逻辑。

- `tests/test_compressed_ftbc.py`  
  验证 bias 压缩和重建误差。

- `tests/test_energy_model.py`  
  验证 AC/MAC、SOPs 和 bias 存储估计公式。

### 修改文件

- `models/layer.py`  
  为 `SignedIF` 增加层级策略接口、膜电位 residual 统计、压缩 bias 支持。

- `models/VGG.py`  
  增加 `set_layer_policy(policy)`，使每个 `SignedIF` 层可以独立设置 `enable_signed`、`enable_r0`、`enable_ftbc` 和 `bias_mode`。

- `calibration.py`  
  将当前 FTBC 扩展为选择性 FTBC，并支持调用 `compressed_ftbc.py`。

- `spike_stats.py`  
  扩展统计量：加入每层转换误差、每层能耗和每层 bias 存储量。

- `run_stats_ablation.py`  
  保留为基线消融脚本，只增加可选的 energy report，不承担新 pipeline 搜索。

- `main_train.py`  
  增加能耗感知 QCFS 训练正则项，作为第二阶段增强，不应在第一阶段阻塞 pipeline。

---

## 4. 核心数据结构

### 4.1 层级误差统计

在 `conversion_diagnostics.py` 中定义：

```python
from dataclasses import dataclass


@dataclass
class LayerConversionStats:
    name: str
    kind: str
    time_steps: int
    ann_mean: float
    snn_mean: float
    l1_error: float
    l2_error: float
    relative_l2_error: float
    zero_to_positive_error: float
    positive_to_zero_error: float
    membrane_residual_mean: float
    membrane_residual_abs_mean: float
    positive_rate: float
    negative_rate: float
    sparsity: float
    sops: int
```

每层误差建议使用以下公式：

```text
ANN output:       A_l
SNN rate output:  R_l = (1 / T) * sum_t S_l(t)

L1 error:         E_l1 = mean(abs(R_l - A_l))
L2 error:         E_l2 = sqrt(mean((R_l - A_l)^2))
Relative L2:      E_rel = E_l2 / (sqrt(mean(A_l^2)) + eps)

Zero-to-positive: mean((A_l == 0) and (R_l > eps))
Positive-to-zero: mean((A_l > eps) and (R_l == 0))
```

为什么这样设计：

- QCFS 和 Parameter Calibration 相关论文都强调 ANN-SNN activation mismatch 是低时间步损失的核心来源。
- `zero_to_positive_error` 对应“ANN 输出应为 0，但 SNN 产生脉冲”的误触发问题。
- `positive_to_zero_error` 对应“ANN 有激活，但 SNN 低时间步未及时发放”的信息丢失问题。
- `membrane_residual_abs_mean` 用于判断是否需要 R0 或膜电位初始化类修正。

---

## 5. Task 1: 实现层级转换误差诊断

**Files:**

- Create: `conversion_diagnostics.py`
- Test: `tests/test_conversion_diagnostics.py`
- Modify: `run_selective_pipeline.py` 在后续任务中调用

- [ ] **Step 1: 写失败测试，验证误差公式**

在 `tests/test_conversion_diagnostics.py` 中加入：

```python
import unittest
import torch

from conversion_diagnostics import compute_activation_error


class ConversionDiagnosticsTest(unittest.TestCase):
    def test_compute_activation_error(self):
        ann = torch.tensor([0.0, 1.0, 2.0, 0.0])
        snn = torch.tensor([0.5, 1.0, 1.0, 0.0])
        result = compute_activation_error(ann, snn, eps=1e-6)

        self.assertAlmostEqual(result["l1_error"], 0.375, places=6)
        self.assertAlmostEqual(result["l2_error"], 0.5590169, places=5)
        self.assertAlmostEqual(result["zero_to_positive_error"], 0.25, places=6)
        self.assertAlmostEqual(result["positive_to_zero_error"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_conversion_diagnostics
```

Expected:

```text
ImportError: No module named conversion_diagnostics
```

- [ ] **Step 3: 实现最小误差函数**

在 `conversion_diagnostics.py` 中加入：

```python
import torch


def compute_activation_error(ann_output, snn_rate_output, eps=1e-8):
    ann = ann_output.detach().float()
    snn = snn_rate_output.detach().float()
    diff = snn - ann

    ann_energy = torch.sqrt(torch.mean(ann * ann)).item()
    l2_error = torch.sqrt(torch.mean(diff * diff)).item()

    zero_to_positive = ((ann.abs() <= eps) & (snn > eps)).float().mean().item()
    positive_to_zero = ((ann > eps) & (snn.abs() <= eps)).float().mean().item()

    return {
        "l1_error": torch.mean(torch.abs(diff)).item(),
        "l2_error": l2_error,
        "relative_l2_error": l2_error / (ann_energy + eps),
        "zero_to_positive_error": zero_to_positive,
        "positive_to_zero_error": positive_to_zero,
    }
```

- [ ] **Step 4: 运行测试，确认通过**

Run:

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_conversion_diagnostics
```

Expected:

```text
Ran 1 test
OK
```

- [ ] **Step 5: 扩展 hook 收集 ANN/SNN 层输出**

在 `conversion_diagnostics.py` 中继续加入：

```python
class SingleCallHook:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, module, inputs, output):
        if self.input is None:
            self.input = inputs[0].detach().clone()
        if self.output is None:
            self.output = output.detach().clone()

    def clear(self):
        self.input = None
        self.output = None


def reduce_snn_temporal_output(output, T):
    if T <= 0:
        return output
    if output.shape[0] % T != 0:
        raise ValueError(f"Temporal batch dimension {output.shape[0]} is not divisible by T={T}")
    batch_size = output.shape[0] // T
    return output.view(T, batch_size, *output.shape[1:]).mean(dim=0)
```

设计理由：

- 现有 `VGG_Signed` 在 SNN 模式下将时间维合并到 batch 维，诊断时必须还原为 `(T, B, ...)` 后求平均。
- 诊断层应该对齐 `SignedIF` 输出，而不是对齐卷积层输出，因为转换误差本质来自 ANN activation 与 SNN spike-rate activation 的不一致。

---

## 6. Task 2: 扩展 `SignedIF` 的层级策略接口

**Files:**

- Modify: `models/layer.py`
- Modify: `models/VGG.py`
- Test: `tests/test_selective_policy.py`

- [ ] **Step 1: 写测试，验证每层策略可以独立设置**

在 `tests/test_selective_policy.py` 中加入：

```python
import unittest

from models import modelpool
from models.layer import SignedIF


class SelectivePolicyTest(unittest.TestCase):
    def test_set_layer_policy(self):
        model = modelpool("vgg16_signed", "cifar10")
        first_name = None
        for name, module in model.named_modules():
            if isinstance(module, SignedIF):
                first_name = name
                break

        model.set_layer_policy({
            first_name: {
                "enable_signed": True,
                "enable_r0": False,
                "enable_ftbc": False,
                "bias_mode": "none",
            }
        })

        for name, module in model.named_modules():
            if isinstance(module, SignedIF) and name == first_name:
                self.assertTrue(module.enable_signed)
                self.assertFalse(module.enable_r0)
                self.assertFalse(module.enable_ftbc)
                self.assertEqual(module.bias_mode, "none")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 在 `SignedIF.__init__` 中增加策略字段**

在 `models/layer.py` 的 `SignedIF.__init__` 中加入：

```python
self.enable_ftbc = True
self.bias_mode = "full"
self.compressed_bias = None
self.last_mem_residual_mean = 0.0
self.last_mem_residual_abs_mean = 0.0
```

- [ ] **Step 3: 在 FTBC 应用处增加开关**

把当前无条件 bias 应用：

```python
bias = self.time_based_bias[t]
```

改成：

```python
if self.enable_ftbc and self.time_based_bias is not None:
    bias = self.get_time_bias(t, x)
    self.mem = self.mem - bias
```

并新增方法：

```python
def get_time_bias(self, t, x):
    if self.bias_mode == "none" or self.time_based_bias is None:
        return torch.zeros_like(x[t])
    if self.bias_mode == "full":
        bias = self.time_based_bias[t]
    else:
        bias = self.reconstruct_time_bias(t)

    if len(x.shape) == 5:
        return bias.view(1, -1, 1, 1)
    if len(x.shape) == 3:
        return bias.view(1, -1)
    return bias
```

- [ ] **Step 4: 在 `VGG_Signed` 中增加 `set_layer_policy`**

在 `models/VGG.py` 的 `VGG_Signed` 类中加入：

```python
def set_layer_policy(self, policy):
    for name, module in self.named_modules():
        if isinstance(module, SignedIF) and name in policy:
            cfg = policy[name]
            module.enable_signed = bool(cfg.get("enable_signed", module.enable_signed))
            module.enable_r0 = bool(cfg.get("enable_r0", module.enable_r0))
            module.enable_ftbc = bool(cfg.get("enable_ftbc", module.enable_ftbc))
            module.bias_mode = cfg.get("bias_mode", module.bias_mode)
```

- [ ] **Step 5: 运行测试**

Run:

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_selective_policy
```

Expected:

```text
Ran 1 test
OK
```

---

## 7. Task 3: 实现压缩 FTBC

**Files:**

- Create: `compressed_ftbc.py`
- Modify: `models/layer.py`
- Modify: `calibration.py`
- Test: `tests/test_compressed_ftbc.py`

### 7.1 压缩模式

完整 FTBC 存储量：

```text
B_full = num_layers * T * channels * bytes_per_value
```

建议支持两种压缩：

1. **Linear bias**

```text
b_l,c(t) = a_l,c * t + c_l,c
B_linear = num_layers * 2 * channels * bytes_per_value
```

2. **Piecewise constant bias**

```text
b_l,c(t) = p_l,c,k, where t in segment k
B_piecewise = num_layers * K * channels * bytes_per_value
```

### 7.2 测试

在 `tests/test_compressed_ftbc.py` 中加入：

```python
import unittest
import torch

from compressed_ftbc import compress_linear_bias, reconstruct_linear_bias


class CompressedFTBCTest(unittest.TestCase):
    def test_linear_bias_reconstruction_for_linear_sequence(self):
        full_bias = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([2.0, 4.0]),
            torch.tensor([3.0, 6.0]),
            torch.tensor([4.0, 8.0]),
        ]
        params = compress_linear_bias(full_bias)
        restored = [reconstruct_linear_bias(params, t) for t in range(4)]

        for expected, actual in zip(full_bias, restored):
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
```

### 7.3 实现

在 `compressed_ftbc.py` 中加入：

```python
import torch


def compress_linear_bias(full_bias):
    stacked = torch.stack([b.detach().float() for b in full_bias], dim=0)
    T = stacked.shape[0]
    t = torch.arange(T, device=stacked.device, dtype=stacked.dtype).view(T, 1)
    t_mean = t.mean(dim=0)
    y_mean = stacked.mean(dim=0)
    numerator = ((t - t_mean) * (stacked - y_mean)).sum(dim=0)
    denominator = ((t - t_mean) ** 2).sum(dim=0).clamp_min(1e-8)
    slope = numerator / denominator
    intercept = y_mean - slope * t_mean.squeeze(0)
    return {"mode": "linear", "slope": slope, "intercept": intercept}


def reconstruct_linear_bias(params, t):
    return params["slope"] * float(t) + params["intercept"]


def estimate_bias_storage(num_values, bytes_per_value=4):
    return int(num_values * bytes_per_value)
```

### 7.4 接入 FTBC

在 `calibration.py` 中保留当前完整 FTBC 校准，然后在每层校准结束后增加：

```python
if bias_mode == "linear":
    module_snn.compressed_bias = compress_linear_bias(module_snn.time_based_bias)
    module_snn.time_based_bias = None
    module_snn.bias_mode = "linear"
elif bias_mode == "full":
    module_snn.bias_mode = "full"
```

为什么这样改：

- 第一版先复用已有 FTBC 计算逻辑，降低风险。
- 压缩只作用于存储和推理读取，不改变校准数据采集。
- 后续可以再研究“直接学习压缩参数”，但第一版不应增加训练复杂度。

---

## 8. Task 4: 实现能耗模型

**Files:**

- Create: `energy_model.py`
- Modify: `spike_stats.py`
- Test: `tests/test_energy_model.py`

### 8.1 估计公式

建议报告两类能耗：

1. **硬件无关 proxy**

```text
Energy_proxy = SOPs + alpha_neg * NegSOPs + alpha_bias * BiasReads
```

2. **AC/MAC 估计**

```text
E_snn = SOPs * E_AC + BiasReads * E_BIAS
E_ann = MACs * E_MAC
EnergyRatio = E_snn / E_ann
```

默认常数建议可配置，不写死为唯一结论：

```text
E_AC  = 0.9 pJ
E_MAC = 4.6 pJ
E_BIAS = 0.9 pJ
```

论文中需要说明：这些数值是常见 45nm CMOS 级别估计，只用于相对比较，不等价于真实芯片测量。

### 8.2 测试

在 `tests/test_energy_model.py` 中加入：

```python
import unittest

from energy_model import estimate_snn_energy_pj


class EnergyModelTest(unittest.TestCase):
    def test_estimate_snn_energy_pj(self):
        result = estimate_snn_energy_pj(sops=100, bias_reads=10, e_ac_pj=0.9, e_bias_pj=0.9)
        self.assertAlmostEqual(result["snn_energy_pj"], 99.0, places=6)


if __name__ == "__main__":
    unittest.main()
```

### 8.3 实现

在 `energy_model.py` 中加入：

```python
def estimate_snn_energy_pj(sops, bias_reads=0, e_ac_pj=0.9, e_bias_pj=0.9):
    return {
        "sops": int(sops),
        "bias_reads": int(bias_reads),
        "snn_energy_pj": float(sops) * e_ac_pj + float(bias_reads) * e_bias_pj,
    }


def estimate_ann_energy_pj(macs, e_mac_pj=4.6):
    return {
        "macs": int(macs),
        "ann_energy_pj": float(macs) * e_mac_pj,
    }


def estimate_energy_ratio(snn_energy_pj, ann_energy_pj):
    return float(snn_energy_pj) / max(float(ann_energy_pj), 1e-12)
```

---

## 9. Task 5: 实现选择性模块搜索

**Files:**

- Create: `selective_policy.py`
- Modify: `run_selective_pipeline.py`
- Test: `tests/test_selective_policy.py`

### 9.1 候选集合

每层候选模块定义为：

```python
CANDIDATES = {
    "QCFS": {"enable_signed": False, "enable_r0": False, "enable_ftbc": False, "bias_mode": "none"},
    "SNM": {"enable_signed": True, "enable_r0": False, "enable_ftbc": False, "bias_mode": "none"},
    "SNM_R0": {"enable_signed": True, "enable_r0": True, "enable_ftbc": False, "bias_mode": "none"},
    "FTBC_FULL": {"enable_signed": False, "enable_r0": False, "enable_ftbc": True, "bias_mode": "full"},
    "SNM_FTBC_LINEAR": {"enable_signed": True, "enable_r0": False, "enable_ftbc": True, "bias_mode": "linear"},
    "SNM_R0_FTBC_LINEAR": {"enable_signed": True, "enable_r0": True, "enable_ftbc": True, "bias_mode": "linear"},
}
```

### 9.2 评分函数

对每层候选配置使用统一目标：

```text
Score = DeltaAccProxy
        - lambda_sops * DeltaSOPsNorm
        - lambda_neg * DeltaNegRate
        - lambda_bias * BiasStorageNorm
        - lambda_latency * DeltaLatencyNorm
```

第一版不直接用全验证集 accuracy 作为每层搜索目标，而用转换误差下降作为 `DeltaAccProxy`：

```text
DeltaAccProxy = E_l_before - E_l_after
```

为什么这样设计：

- 对每层每候选都跑完整测试集太慢。
- 转换误差是低时间步精度损失的直接代理。
- 最终仍需在搜索结束后跑完整 test accuracy 验证。

### 9.3 测试

在 `tests/test_selective_policy.py` 中增加：

```python
from selective_policy import score_candidate


def test_score_candidate_prefers_lower_cost_when_error_gain_equal():
    cheap = score_candidate(error_gain=0.1, sops_delta=1.0, neg_delta=0.0, bias_bytes=0)
    expensive = score_candidate(error_gain=0.1, sops_delta=10.0, neg_delta=0.0, bias_bytes=0)
    assert cheap > expensive
```

### 9.4 实现

在 `selective_policy.py` 中加入：

```python
def score_candidate(
    error_gain,
    sops_delta,
    neg_delta,
    bias_bytes,
    latency_delta=0.0,
    lambda_sops=0.05,
    lambda_neg=1.0,
    lambda_bias=0.001,
    lambda_latency=0.05,
):
    return (
        float(error_gain)
        - lambda_sops * float(sops_delta)
        - lambda_neg * float(neg_delta)
        - lambda_bias * float(bias_bytes)
        - lambda_latency * float(latency_delta)
    )
```

### 9.5 贪心搜索流程

第一版采用稳健贪心：

```text
1. Start from all-layer QCFS.
2. Run diagnostics and rank layers by relative_l2_error.
3. For top-K error layers, try candidate modules.
4. Keep candidate only if:
   error decreases by at least min_error_gain
   and SOPs increase is below max_sops_delta
   and negative spike rate below max_neg_rate
5. Re-run full validation after policy is fixed.
```

这样做的好处：

- 计算成本可控。
- 搜索结果可解释。
- 论文中可以展示“哪些层被选择了哪些模块”，比全局消融更有说服力。

---

## 10. Task 6: 能耗感知 QCFS 训练

**Files:**

- Modify: `main_train.py`
- Modify: `models/layer.py`
- Test: `tests/test_energy_aware_training.py`

该任务放在第二阶段，因为它会重新训练 checkpoint，成本高于前面几个转换后处理任务。

### 10.1 训练目标

在原交叉熵基础上加入脉冲代理正则：

```text
Loss = CE
       + lambda_act * mean(QCFS_activation)
       + lambda_sat * saturation_penalty
       + lambda_neg * negative_proxy
```

其中：

```text
saturation_penalty = mean(max(activation / threshold - 1, 0))
negative_proxy     = mean(max(-activation / threshold, 0))
```

### 10.2 为什么这样改

- 当前转换后 spike rate 大约 14%-15%，仍有降低空间。
- 如果只在转换后压脉冲，会牺牲精度；更好的方式是在 ANN 训练阶段让 activation 本身更适合低脉冲率转换。
- 这和 activity regularization / low-bit noise-aware ANN conversion 思路一致，但要适配 QCFS 和已有 `IF`/`SignedIF`。

### 10.3 命令设计

新增参数：

```powershell
D:\Anaconda\envs\ann2snn\python.exe main_train.py -data cifar100 -arch vgg16 -L 8 --lambda_act 1e-5 --lambda_sat 1e-4
```

输出 checkpoint 命名：

```text
cifar100-checkpoints/vgg16_L[8]_energy_aware.pth
```

评估时与原 checkpoint 对比：

```powershell
D:\Anaconda\envs\ann2snn\python.exe run_selective_pipeline.py -data cifar100 -id vgg16_L[8]_energy_aware -dev 0 --time_steps 1 2 4 8 16 32
```

---

## 11. Task 7: 端到端入口 `run_selective_pipeline.py`

**Files:**

- Create: `run_selective_pipeline.py`

### 11.1 CLI

建议入口：

```powershell
D:\Anaconda\envs\ann2snn\python.exe run_selective_pipeline.py `
  -data cifar100 `
  -id cifar100-vgg16-l8-example `
  -dev 0 `
  -b 200 `
  --time_steps 1 2 4 8 16 32 `
  --diagnostic_batches 5 `
  --search_topk_layers 6 `
  --bias_mode linear `
  --max_neg_rate 0.05 `
  --max_sops_increase 0.03 `
  --output docs/spike_stats/EASC_QCFS_cifar100_vgg16.md
```

### 11.2 输出 Markdown

必须包含：

1. Overall result table

```text
Config | T | Accuracy | SOPs | PosRate | NegRate | Sparsity | Energy(pJ) | BiasBytes | Elapsed
```

2. Layer decision table

```text
Layer | SelectedPolicy | ErrorBefore | ErrorAfter | SOPsDelta | NegRate | BiasMode | BiasBytes
```

3. Pareto table

```text
Method | T | Accuracy | SOPs | Energy | BiasBytes
```

4. Ablation table

```text
QCFS
QCFS+Global SNM
QCFS+Global FTBC
QCFS+Global SNM+R0+FTBC
EASC-QCFS without compressed FTBC
EASC-QCFS with compressed FTBC
```

---

## 12. 实验设计

所有方法必须在完全相同的固定时间步 `T` 下运行和比较。每个样本都执行完整的 `T` 个时间步，不设置提前退出，也不把高时间步作为方法生效条件。`T=16` 和 `T=32` 用于展示收敛趋势并与已有方法进行公平对比，主方法应同时报告低时间步和高时间步结果。

### 12.1 必做实验

1. **主结果**

```text
Dataset: CIFAR-10, CIFAR-100
Model: VGG16
T: 1, 2, 4, 8, 16, 32
Methods: A-F baseline + EASC-QCFS
```

2. **结构泛化**

```text
Dataset: CIFAR-10, CIFAR-100
Model: ResNet20 or ResNet34
T: 1, 2, 4, 8, 16, 32
```

3. **消融实验**

```text
EASC-QCFS
EASC without layer error diagnosis
EASC without energy penalty
EASC without compressed FTBC
```

4. **层级可解释性**

```text
Layer-wise error heatmap
Layer-wise selected module table
Layer-wise SOPs contribution
Layer-wise negative spike rate
```

5. **部署价值**

```text
Full FTBC bias vs Linear compressed FTBC
Accuracy-SOPs Pareto curve
Accuracy-Energy Pareto curve
```

### 12.2 成功标准

第一阶段成功标准：

```text
At T=4 or T=8:
EASC-QCFS accuracy >= best global A-F baseline - 0.1%
EASC-QCFS SOPs <= best global A-F baseline SOPs
EASC-QCFS bias storage < full FTBC bias storage by at least 50%
```

第二阶段成功标准：

```text
Energy-aware training:
Same or higher accuracy at T=8
Spike rate or SOPs reduction >= 5%
```

---

## 13. 验证命令

每完成一个任务都运行对应单测：

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_conversion_diagnostics
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_selective_policy
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_compressed_ftbc
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_energy_model
```

全部完成后运行：

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest discover tests
```

小规模 smoke test：

```powershell
D:\Anaconda\envs\ann2snn\python.exe run_selective_pipeline.py `
  -data cifar100 `
  -id cifar100-vgg16-l8-example `
  -dev 0 `
  -b 200 `
  --time_steps 4 `
  --diagnostic_batches 1 `
  --search_topk_layers 2 `
  --output docs/spike_stats/EASC_QCFS_smoke.md
```

完整实验：

```powershell
D:\Anaconda\envs\ann2snn\python.exe run_selective_pipeline.py `
  -data cifar100 `
  -id cifar100-vgg16-l8-example `
  -dev 0 `
  -b 200 `
  --time_steps 1 2 4 8 16 32 `
  --diagnostic_batches 5 `
  --search_topk_layers 6 `
  --bias_mode linear `
  --output docs/spike_stats/EASC_QCFS_cifar100_vgg16.md
```

---

## 14. 风险与处理方式

1. **风险：选择性模块搜索过慢。**  
   处理：第一版只对 top-K 高误差层搜索，每层最多 6 个候选，每个候选只用 calibration subset 估计误差，不跑完整 test set。

2. **风险：压缩 FTBC 降低精度。**  
   处理：同时报告 `full`、`linear`、`piecewise` 三种模式。若 `linear` 误差过大，优先使用 `piecewise K=4`。

3. **风险：SNM 负脉冲破坏能耗优势。**  
   处理：把 `negative_rate` 和 `negative SOPs` 加入选择策略约束，超过阈值的候选直接拒绝。

4. **风险：ANN energy-aware training 成本过高。**  
   处理：先完成转换后处理 pipeline，训练正则作为增强实验，不作为主方法必要条件。

---

## 15. 论文写作落点

建议论文贡献写成三点：

1. **误差感知层级诊断。**  
   提出一套 layer-wise ANN-SNN conversion mismatch 诊断指标，区分低时间步下的零激活误触发、正激活未触发和膜电位残差误差。

2. **能耗约束选择性校准。**  
   提出 EASC-QCFS，根据每层误差收益和 SOPs、负脉冲率、bias 存储量成本，自适应选择 QCFS、SNM、R0 和 FTBC，而不是全网统一叠加模块。

3. **压缩 FTBC。**  
   将完整时间步 bias 压缩为线性或分段形式，在相同固定时间步下减少校准参数存储和 bias 访问开销，并验证其对精度、SOPs 和延迟的影响。

---

## 16. 相关论文依据

- QCFS: [Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks](https://arxiv.org/abs/2303.04347)  
  依据：用更接近 SNN 估计激活的量化激活函数降低转换误差。

- FTBC: [Forward Temporal Bias Correction for Optimizing ANN-SNN Conversion](https://arxiv.org/abs/2403.18388)  
  依据：用逐时间步 bias 校正降低 ANN-SNN 转换误差。

- Parameter Calibration: [Converting Artificial Neural Networks to Spiking Neural Networks via Parameter Calibration](https://arxiv.org/abs/2205.10121)  
  依据：逐层分析 clipping/flooring error 和 activation mismatch，并做校准。

- OPI: [Optimized Potential Initialization for Low-latency Spiking Neural Networks](https://arxiv.org/abs/2202.01440)  
  依据：低时间步性能受膜电位初值和残差影响，膜电位相关策略可显著降低转换损失。

- RMP: [Residual Membrane Potential Neuron](https://arxiv.org/abs/2003.01811)  
  依据：软重置和残余膜电位保留可以减少 hard reset 信息损失。

- Activity Regularization: [Optimizing the Consumption of Spiking Neural Networks with Activity Regularization](https://arxiv.org/abs/2204.01460)  
  依据：训练阶段约束 activation/spike activity 能降低 SNN 消耗。

- Energy Estimation: [An Analytical Estimation of Spiking Neural Networks Energy Efficiency](https://arxiv.org/abs/2210.13107)  
  依据：需要在论文中报告硬件无关或半硬件相关的能耗估计，而不是只报告 accuracy。

---

## 17. 推荐实施顺序

推荐按以下提交粒度推进：

1. `Add conversion diagnostics`
2. `Add layer-wise module policy`
3. `Add compressed FTBC`
4. `Add energy model`
5. `Add selective pipeline search`
6. `Add energy-aware QCFS training`
7. `Add EASC-QCFS experiment reports`

每个提交都应保持以下标准：

- 单测通过。
- smoke experiment 能跑完。
- Markdown 报告能复现实验命令。
- 不提交 checkpoint、`.pyc` 或临时缓存。
