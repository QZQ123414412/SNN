# QCFS-SNN 最终创新与 Pipeline 改造方案

## 1. 最终目标

在现有 `QCFS + SNM + R0 + FTBC` 框架上，形成两个相互配合、但可以独立验证的创新方向：

1. **信用约束有符号残差精化**：保留膜电位残差，以净正累计输出作为负修正信用，并使用非对称 hysteresis 抑制无效的正负脉冲振荡。
2. **状态条件低秩 FTBC**：让 SNM、R0 和 FTBC 围绕同一个累计传输状态协同工作，并将 FTBC bias 从 `T×C` 压缩为 `3×C`。

最终方法不使用逐层配置搜索，不使用 early-exit，也不依赖高时间步才能生效。所有方法均在相同固定时间步：

```text
T = 1, 2, 4, 8, 16, 32
```

下进行公平比较。

---

## 2. 当前代码中三个模块的真实行为

当前核心实现在 `models/layer.py::SignedIF.forward()` 中。每个时间步的执行顺序为：

```text
1. mem = mem - time_based_bias[t]       # FTBC
2. mem = mem + input[t]                 # 积分
3. 根据正阈值产生正脉冲
4. 根据负阈值和 transmitted > 0 产生负脉冲  # SNM
5. mem = mem - spike                    # soft reset
6. transmitted = transmitted + spike
7. transmitted == 0 时截断负膜电位          # R0
```

### 2.1 SNM

当前负脉冲条件为：

```python
mem <= neg_thresh and transmitted > 0
```

因此 SNM 的作用是：当神经元当前仍有净正累计输出时，允许使用负脉冲撤销此前的部分正输出。

### 2.2 R0

当前规则为：

```python
if transmitted == 0:
    mem = max(mem, 0)
```

因此 R0 的作用是：当当前净累计输出为零时，清除没有可撤销正输出所对应的负膜电位，避免继续积累无意义的负债。

### 2.3 FTBC

当前 FTBC 为每层、每个时间步、每个通道保存一个 bias：

```text
time_based_bias[layer][t][channel]
```

推理时从膜电位中减去该 bias：

```text
mem = mem - bias
```

正 bias 会抑制发放，负 bias 会促进发放。

### 2.4 当前协同关系

三者已经存在间接影响：

```text
FTBC改变膜电位
→ 改变正负脉冲
→ 改变transmitted
→ 改变SNM和R0后续行为
```

但是 FTBC bias 目前只取决于层、时间步和通道，不直接读取 `transmitted`。因此它不知道神经元当前处于：

- 净累计输出为零的 R0 保护状态；
- 仍有净正累计输出、允许 SNM 撤销的状态。

这是三个模块尚未形成显式协同的核心问题。

---

## 3. 实施前必须修正的基线问题

以下修改属于实现一致性修正，不应包装为论文创新。

### 3.1 FTBC 校准重放与部署不一致

当前 `calibration.py` 中保存的是通道均值：

```python
bias_mean = deviation.mean(...)
time_based_bias[t] += alpha * bias_mean
```

但校准重放使用的是完整样本和空间位置上的 `deviation`：

```python
mem_corrected = mem - alpha * deviation
```

部署时实际使用的却是通道级 `bias_mean`。这会导致校准时模拟的神经元动力学与最终推理动力学不同。

应统一改为：

```python
correction = reshape_channel_bias(alpha * bias_mean)
mem_corrected = mem - correction
```

### 3.2 多校准批次更新方式不明确

当前更新是累加：

```python
bias = bias + alpha * bias_mean
```

批次数越多，bias 绝对值可能持续增长。应明确采用以下一种方式：

```text
运行均值：bias = accumulated_sum / batch_count
```

或：

```text
EMA：bias = (1-alpha) * bias + alpha * bias_mean
```

建议使用运行均值，保证校准结果对 batch 顺序不敏感。

### 3.3 ANN/SNN 层匹配方式

当前使用 `zip(ann.named_modules(), snn.named_modules())`，默认两个模型模块顺序完全一致。应改为按稳定层名或显式映射匹配 `IF` 与 `SignedIF`，并在不匹配时抛出错误。

### 3.4 状态重置

每个校准批次和验证批次前应显式重置：

```text
mem
transmitted
spike statistics
temporary calibration accumulators
```

避免上一批次状态污染下一批次。

---

## 4. 方向一：信用约束有符号残差精化

## 4.1 核心思想

现有 SignedIF 已允许负脉冲撤销过发放，但原 R0 在净累计输出为零时会直接清除负膜电位。该规则适合避免“负债”，却也会删除可供后续时间步使用的有效残差。

方向一把“是否允许负输出”和“是否保留负残差”分开：

```text
transmitted：已经对外发送的净正表示，即负修正信用
mem：尚未被脉冲表达的残差
```

规则为：

```text
没有足够transmitted时：禁止负脉冲
没有足够transmitted时：仍保留mem中的负残差
```

因此，后续输入可以抵消或利用该残差，而不会产生非法负输出。方向一不需要采集转换残差、不需要回到 ANN 注入噪声，也不需要微调原始 checkpoint。

## 4.2 非对称 hysteresis

正负修正使用不同决策边界：

```text
正脉冲：mem >= 0.55 * theta
负脉冲：mem <= -1.30 * theta 且 transmitted >= theta
```

较大的负阈值形成 hysteresis，避免正脉冲刚发出后立即被小幅负残差撤销，从而减少正负脉冲振荡和双重 SOPs 开销。所有层共享同一组 margin，不进行逐层配置搜索。

## 4.3 神经元更新

```text
1. mem = mem - FTBC_bias(t, transmitted)
2. mem = mem + input(t)
3. 若mem >= 0.55*theta，则发+theta
4. 若mem <= -1.30*theta且transmitted >= theta，则发-theta
5. mem = mem - spike
6. transmitted = transmitted + spike
7. transmitted接近0时置0，但不清空mem残差
```

`T=1` 时不存在后续修正过程，因此自动回退原 rate coding，保证单时间步基线不退化。

## 4.4 与 QCFS 的对应关系

```text
QCFS：目标激活属于哪个量化等级
残差记忆：保留尚未表达的正负信息
SNM：在信用充足时撤销过量表示
credit-only R0：禁止非法负输出，但不删除有效残差
hysteresis：减少正负脉冲来回振荡
FTBC：校正剩余通道和时间偏差
```

原始 checkpoint、网络拓扑和固定时间步协议保持不变。

## 4.5 代码改造

- `models/layer.py::SignedIF`：增加 CSRR 状态机、margin、credit-only R0 和 T=1 回退。
- `models/VGG.py`：增加全模型编码配置接口。
- `calibration.py`：让完整 FTBC 与状态低秩 FTBC 复用同一 CSRR 单步状态转移。
- `spike_stats.py`：增加逐时间步事件计数和独立 ScaleOps 统计。
- `scripts/experiments/run_successive_refinement_ablation.py`：实现方向一、方向二及组合消融。
- `preprocess/getdataloader.py`：增加跨平台数据目录和 worker 配置。
- `tests/test_successive_refinement.py`：验证状态守恒、R0、hysteresis、FTBC 重放和配置展开。

## 4.6 时间权重消融

实现中保留了全局几何时间尺度：

```text
w_t proportional to ratio^(-t)
```

实验表明 `ratio>1` 虽可提高部分低时间步准确率，但会显著增加负脉冲、SOPs 和尺度操作。因此最终取 `ratio=1.0`，即所有时间尺度恒为 1，ScaleOps 为零。几何时间权重仅作为失败消融保留。

最终方法在 CIFAR-10/100 上均取得 Accuracy-SOPs-sparsity 改善，完整结果见 `docs/results/successive_refinement/SR_FINAL_ANALYSIS.md`。

---

## 5. 方向二：状态条件低秩 FTBC

这是第二个核心创新，负责在逐次精化编码之后，以统一状态和低存储开销校正剩余系统性偏差。

## 5.1 核心状态

严格按照当前代码，状态定义为：

```text
g(t) = 1, transmitted(t-1) > 0
g(t) = 0, transmitted(t-1) <= 0
```

这里表示“当前是否仍有可撤销的净正累计输出”，不是简单的“历史上是否曾经发过脉冲”。

状态含义：

- `g=0`：SNM 不应产生负脉冲，R0 负责清除负膜电位；
- `g=1`：SNM 可以用负脉冲撤销过发放，R0 不截断负膜电位。

## 5.2 低秩状态条件 bias

将当前完整 FTBC：

```text
b_l,c(t), 参数量为 T×C
```

改为：

```text
b_l,c(t,g) = a_l,c + d_l,c * tau(t) + r_l,c * g(t)
tau(t) = t / max(T-1, 1)
```

三个参数分别表示：

- `a`：通道基础校正；
- `d`：随归一化时间变化的校正趋势；
- `r`：存在净正累计输出时的额外校正。

参数量从 `T×C` 变为 `3×C`。

理论存储压缩率：

```text
Compression = 1 - 3/T
```

| T | 相对完整 FTBC 的存储减少 |
|---:|---:|
| 4 | 25.0% |
| 8 | 62.5% |
| 16 | 81.25% |
| 32 | 90.625% |

`T=1/2` 时低秩形式没有存储优势，因此应允许退化为常数 bias 或直接使用完整 bias。不能声称所有时间步都压缩。

## 5.3 推理更新

每个时间步改为：

```text
g = transmitted > 0
bias = base + slope * normalized_time + state_bias * g
mem = mem - bias
mem = mem + input
产生正脉冲
若g为真且mem低于负阈值，则产生负脉冲
更新mem和transmitted
若transmitted==0，则执行R0
```

SNM、R0 和 FTBC 因此共享同一状态，而不是仅通过结果间接影响。

## 5.4 参数校准

校准时对每个通道积累：

```text
X = [1, tau(t), g(t)]
y = SNN累计平均输出 - ANN激活
```

求解带正则的最小二乘：

```text
beta = (X^T X + lambda I)^(-1) X^T y
beta = [a, d, r]
```

为避免保存所有样本，只需在线累积：

```text
XTX[channel, 3, 3]
XTy[channel, 3]
```

最终每层只求解一次 `3×3` 线性系统，不需要反向传播或配置搜索。

## 5.5 SOPs 约束

仅最小化激活误差可能通过增加脉冲提高准确率。建议对过发放和欠发放使用不同权重：

```text
L_cal = w_over * ReLU(R-A)^2
      + w_under * ReLU(A-R)^2
      + lambda_beta * ||beta||²
```

其中：

```text
w_over >= w_under
```

优先抑制过发放，因为过发放不仅增加正脉冲，还可能导致后续 SNM 负脉冲撤销，形成双重 SOPs 开销。

第一版必须先使用对称损失建立正确基线，再增加非对称权重作为独立消融。

## 5.6 代码改造

### `models/layer.py`

为 `SignedIF` 增加：

```python
self.bias_base
self.bias_slope
self.bias_state
self.bias_mode  # none, full, state_low_rank
```

增加：

```python
def get_ftbc_bias(self, t, transmitted):
    ...
```

### `calibration.py`

- 保留修正后的标准 FTBC，作为公平基线；
- 新增 `calibrate_state_low_rank_ftbc()`；
- 在线累计 `XTX` 和 `XTy`；
- 使用 `torch.linalg.solve()` 求解参数；
- 校准完成后报告拟合误差和参数量。

### `spike_stats.py`

新增：

```text
full_ftbc_parameter_count
low_rank_parameter_count
bias_storage_bytes
bias_read_count
estimated_bias_energy
positive_sops
negative_sops
```

### `scripts/experiments/run_stats_ablation.py`

增加新配置，不改变原 A-F：

```text
G_QCFS+SNM+R0+SCFTBC_FULL
H_QCFS+SNM+R0+SCFTBC_LR
I_SRQCFS+SNM+R0
J_SRQCFS+SNM+R0+SCFTBC_LR
```

其中：

- G：只验证状态条件是否有效，保留完整时间 bias；
- H：验证状态条件和低秩压缩；
- I：只加入方向一的单调有符号逐次精化编码；
- J：组合方向一与方向二。

---

## 6. 两个方向如何协同

两个方向分别处理不同阶段：

```text
方向一：编码阶段提高每个脉冲的信息量，并从粗到细修正表示
方向二：校准阶段用统一状态修正剩余系统性偏差
```

完整 pipeline：

```text
1. 加载原QCFS checkpoint
2. 将IF转换为支持逐次精化的SignedIF
3. 为固定T生成全局单调时间尺度
4. 使用SNM执行正负双向精化，使用R0约束非法负修正
5. 校准状态条件低秩FTBC
6. 在相同固定T下进行完整推理
7. 报告Accuracy、事件SOPs、尺度操作、Sparsity、bias开销和延迟
```

方向一不是方向二成立的前提。必须先分别验证，再验证组合，才能判断收益是否可叠加。

---

## 7. 能耗与效率评价

能耗估计只作为评价，不参与昂贵搜索。

```text
E_SNN = SOPs * E_AC
      + ScaleOps * E_Scale
      + BiasReads * E_Bias
E_ANN = MACs * E_MAC
```

默认可报告常见 45nm 估算：

```text
E_AC  = 0.9 pJ
E_MAC = 4.6 pJ
```

但必须明确这些是相对分析，不是真实芯片实测。

必须同时报告原始量：

```text
Accuracy
Input-driven SOPs
Time-scale operation count
Positive spike rate
Negative spike rate
Spike sparsity
FTBC parameter count
FTBC storage bytes
Elapsed
Estimated energy
```

低秩状态条件 FTBC 在推理时会比标准 FTBC 多出少量逐元素乘加和状态选择，因此不能只报告存储减少，还必须实测 PyTorch 延迟。若面向神经形态硬件，则应另外说明这些控制操作是否能被硬件原生支持。

---

## 8. 实验矩阵

### 8.1 基础配置

```text
A  QCFS
B  QCFS+SNM
C  QCFS+SNM+R0
D  QCFS+FTBC
E  QCFS+SNM+FTBC
F  QCFS+SNM+R0+FTBC
```

### 8.2 新方法

```text
G  QCFS+SNM+R0+状态条件完整FTBC
H  QCFS+SNM+R0+状态条件低秩FTBC
I  逐次精化QCFS+SNM+R0
J  逐次精化QCFS+SNM+R0+状态条件低秩FTBC
```

### 8.3 关键消融

```text
标准FTBC vs 修正后的标准FTBC
标准FTBC vs 状态条件完整FTBC
状态条件完整FTBC vs 状态条件低秩FTBC
对称校准损失 vs 非对称SOPs约束
等权rate coding vs 固定二进制逐次精化
固定二进制权重 vs 全局单调校准权重
无负校正 vs SNM有符号精化
逐次精化中关闭R0 vs 开启R0
方向一单独使用 vs 方向二单独使用 vs 两者组合
```

### 8.4 数据集和模型

最低要求：

```text
CIFAR-10 / VGG16
CIFAR-100 / VGG16
```

为了证明泛化性，建议增加：

```text
CIFAR-10或CIFAR-100 / ResNet20或ResNet34
```

所有配置均使用相同 checkpoint 来源、校准样本数、随机种子、batch size 和固定时间步。

---

## 9. 实施顺序

### 第一阶段：建立可信基线

1. 修正 FTBC 校准重放与部署不一致。
2. 修正多批次 bias 更新规则。
3. 增加显式状态重置和层映射检查。
4. 重跑 A-F，确认修正前后差异。

### 第二阶段：实现主创新

1. 实现状态条件完整 FTBC，验证状态协同本身。
2. 实现 `3×C` 低秩参数化。
3. 增加 bias 存储、读取和延迟统计。
4. 完成 G、H 与 F 的对比。

### 第三阶段：实现逐次精化编码

1. 在 rate 模式下建立严格兼容性测试。
2. 实现固定二进制式单调时间权重。
3. 修改 SignedIF 的带权正负校正、累计状态和 R0 逻辑。
4. 在 `T=2/4` 完成小规模可行性实验。
5. 若可行性成立，再实现全局单调权重校准。
6. 完成 I、J 以及全部时间步消融。

---

## 10. 成功标准

方向二至少应满足以下一项，否则不应宣称全面优于 FTBC：

```text
1. 相同Accuracy下，SOPs更低且bias存储更少；
2. 相同SOPs下，Accuracy更高且bias存储更少；
3. Accuracy基本不下降（建议<=0.1%），bias存储显著降低。
```

方向一至少应满足：

```text
在T=1/2/4/8中至少两个时间步改善Accuracy-SOPs折中；
T=16/32准确率不出现明显退化；
事件SOPs或Spike sparsity优于等权rate coding；
带权尺度的实际计算开销被单独报告，不能隐藏在SOPs中；
收益能够迁移到至少两个数据集。
```

最终方法应重点展示 Accuracy-SOPs 和 Accuracy-bias-storage 的 Pareto 优势，而不是只报告单一最高准确率。

---

## 11. 最终论文贡献表述

建议收敛为两点：

1. **Signed-state-conditioned temporal correction**
   利用 SignedIF 的净累计传输状态统一控制 SNM、R0 和 FTBC，使时序校正显式适应正脉冲撤销状态与 R0 保护状态。

2. **Credit-aware signed residual refinement**
   保留对后续修正有用的膜电位残差，以净正累计输出作为负修正信用，并使用非对称 hysteresis 抑制正负脉冲振荡；该机制与状态条件低秩 FTBC 共享同一累计输出状态。

低秩 bias 压缩、SOPs 和能耗评价是第一点的重要组成部分，但不应被单独夸大为第三个完全独立的算法贡献。
