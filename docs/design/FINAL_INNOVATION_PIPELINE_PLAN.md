# QCFS-SNN 最终创新与 Pipeline 改造方案

## 1. 最终目标

在现有 `QCFS + SNM + R0 + FTBC` 框架上，形成两个相互配合、但可以独立验证的创新方向：

1. **单调有符号逐次精化编码**：将等幅 rate coding 改为从粗到细的带权正负修正，使每个时间步承担不同精度的表示任务，提高低时间步的信息利用率。
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

## 4. 方向一：单调有符号逐次精化编码

## 4.1 核心思想

当前 rate coding 中，每个时间步的脉冲贡献均为 `1/T`，本质上是重复采样。低时间步时表示等级有限，早期过发放也缺少足够机会修正。

新方案为每个时间步分配全局共享的时间权重：

```text
w_t > 0
w_t >= w_(t+1)
sum_t w_t = 1
```

累计表示改为：

```text
R_t = R_(t-1) + w_t * s_t
s_t ∈ {-1, 0, +1}
```

早期大权重完成粗估计，后期小权重修正剩余误差。正脉冲向上修正，负脉冲向下撤回，不发放则保持当前结果。目标不是简单给输出乘时间系数，而是建立贯穿各层的逐次精化编码。

在现有代码仍使用时间平均读出的前提下，实际事件量子实现为：

```text
q_t = T * w_t * threshold
output = mean_t(signed_event_t)
       = threshold * sum_t(w_t * signed_digit_t)
```

因此时间平均只是实现形式，数学上仍是带权有符号累计。

## 4.2 编码与神经元更新

首版至少支持三种权重：

```text
uniform：w_t = 1/T，作为等幅有符号基线
binary：归一化的 2^(-t)，验证固定粗到细编码
calibrated：在校准集上求解受单调和归一化约束的全局权重
```

真实中间层不能读取 ANN 目标值，因此由膜电位保存尚未表达的残差。每个时间步根据当前尺度和膜电位决定：

```text
mem > positive_boundary(t)  -> s_t = +1
mem < negative_boundary(t)  -> s_t = -1
otherwise                   -> s_t = 0
```

发放后按当前 `q_t` 更新膜电位和累计表示。下游层直接接收带尺度事件；首个脉冲层先把重复静态输入汇总到第一步，使早期粗量子能够立即使用。最终分类器通过上述等价时间平均完成带权累计。

## 4.3 与 SNM、R0 和 FTBC 的关系

- **SNM**：从异常补偿机制变为正式的双向残差修正规则；
- **R0**：当不存在可撤回的正累计表示时，禁止非法负修正；
- **FTBC**：微调正负发放边界，补偿逐层传播产生的系统性时序偏差。

三者共同形成：

```text
双向修正 + 状态合法性约束 + 决策边界校正
```

## 4.4 权重校准与训练策略

第一阶段采用 **conversion-only**：保持 QCFS checkpoint、网络权重和阈值不变，只在训练集子集或独立校准集上选择全局时间比例、正边界和负边界。这样可以直接验证编码机制本身，而不是把收益归因于重新训练。

校准权重时应满足：

```text
min  activation_error + lambda_sop * event_cost
s.t. w_t > 0, w_t >= w_(t+1), sum(w_t) = 1
```

当前实现用一个全局几何比例生成权重，并联合校准全局正、负发放边界；准确率优先，在容差内以 SOPs 作为决策依据。该过程不进行逐层配置搜索。

现有实验中，固定递减权重直接作用于原 QCFS checkpoint 时性能下降，自动校准最终退化为 `ratio=1`。原因是原 checkpoint 按均匀 rate coding 训练，网络没有适应“早期大步估计、后期小步修正”的层间分布。因此第二阶段需要在固定递减编码下对 QCFS 进行轻量微调。

## 4.5 递减编码感知 QCFS 微调

微调从现有 QCFS checkpoint 开始，不从头训练。训练时在每个 QCFS 激活位置加入可微逐次精化代理：

```text
QCFS激活
→ 固定单调递减时间权重
→ STE正/负残差量化
→ 逐次精化等效激活
→ 传递到下一层
```

代理必须使用与部署一致的时间权重、正负修正规则和 R0 合法性约束。硬脉冲决策在前向中保持离散，反向使用 STE 或平滑替代梯度。

训练同时保留两条前向：

```text
clean branch：原始QCFS激活
refinement branch：递减有符号逐次精化代理
```

损失函数为：

```text
L = CE(logits_refinement, label)
  + lambda_clean * CE(logits_clean, label)
  + lambda_cons * KL(logits_refinement || stopgrad(logits_clean))
  + lambda_event * estimated_event_rate
```

- `CE(refinement)`：让网络适应递减时间编码；
- `CE(clean)`：保持原 QCFS 分类能力；
- `KL`：约束微调模型不要偏离原 checkpoint；
- `event_rate`：避免依靠增加正负脉冲换取准确率。

建议采用两阶段微调：

```text
阶段A：3～5 epoch，只训练QCFS阈值、BN和分类器；
阶段B：10～20 epoch，以较小学习率微调整个网络。
```

默认训练一个适配多个低时间步的共享 checkpoint，每个 batch 从 `T={2,4,8}` 中采样一个时间步；分别针对单一 `T` 微调只作为性能上限消融。训练数据只能来自训练集，模型选择使用验证集，测试集仅用于最终报告。

## 4.6 代码改造

- 新建 `models/temporal_coding.py`：生成、归一化并校验单调时间权重；
- 修改 `models/layer.py::SignedIF`：按当前尺度执行正、负或零修正；
- 修改模型输出读取：使用带权有符号累计值；
- 修改 FTBC 校准重放：与新的逐次精化动力学保持一致；
- 修改 `spike_stats.py`：单独统计事件 SOPs 和时间尺度操作；
- 新建可微逐次精化代理，并支持 clean/refinement 双分支前向；
- 新建轻量微调脚本，支持冻结参数阶段、全网络阶段和多时间步采样；
- 新建逐次精化消融脚本与单元测试。

## 4.7 开销与风险

- 预期提高 `T=2/4/8` 的准确率或在相近准确率下降低 SOPs；
- 二进制尺度可通过位移或权重预融合实现，但普通 GPU 上仍可能增加乘法；
- SOPs 只统计非零事件，尺度运算必须作为 `ScaleOps` 单独报告；
- 递减权重可能放大早期错误，因此必须检查 `T=16/32` 是否退化；
- 微调会增加训练开销，但部署模型不增加额外可训练分支；
- 必须同时报告微调前后的 ANN/QCFS 精度，防止只提高 SNN 而破坏源模型；
- `T=1` 应退化为原始编码，保证基线一致。

---

## 5. 方向二：状态条件低秩 FTBC

这是与方向一互补的另一项核心创新。

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
I_MSSR+SNM+R0+SCFTBC_LR
```

其中：

- G：只验证状态条件是否有效，保留完整时间 bias；
- H：验证状态条件和低秩压缩；
- I：加入方向一的单调有符号逐次精化编码。

---

## 6. 两个方向如何协同

两个方向处理不同层面的误差：

```text
方向一：提高每个时间步和每个脉冲携带的信息量
方向二：利用累计状态校正剩余时序偏差并压缩bias
```

完整 pipeline：

```text
1. 加载原QCFS checkpoint
2. 固定递减时间权重和逐次精化规则
3. 用可微代理进行两阶段QCFS轻量微调
4. 转换为支持单调带权事件的SignedIF SNN
5. 在校准集确定全局发放边界
6. 可选校准状态条件低秩FTBC
7. 在固定T下进行完整推理
8. 报告Accuracy、SOPs、ScaleOps、Sparsity、bias开销和延迟
```

方向一的 conversion-only 和微调版本必须分别报告；方向一也不是方向二成立的前提。必须先分别验证，再验证组合，才能判断收益是否可叠加。

---

## 7. 能耗与效率评价

能耗估计只作为评价，不参与昂贵搜索。

```text
E_SNN = SOPs * E_AC + BiasReads * E_Bias
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
ScaleOps
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
I  单调有符号逐次精化（直接转换）
J  单调有符号逐次精化（仅阈值/BN/分类器微调）
K  单调有符号逐次精化（全网络轻量微调）
L  微调逐次精化+SNM+R0+状态条件低秩FTBC
```

### 8.3 关键消融

```text
标准FTBC vs 修正后的标准FTBC
标准FTBC vs 状态条件完整FTBC
状态条件完整FTBC vs 状态条件低秩FTBC
对称校准损失 vs 非对称SOPs约束
均匀编码 vs 固定二进制递减权重 vs 校准单调权重
仅输出加权 vs 各层带权传播
递减权重直接转换 vs 仅阈值微调 vs 全网络微调
单时间步微调 vs 多时间步共享checkpoint
无clean约束 vs 无KL约束 vs 无event正则 vs 完整损失
微调前后ANN/QCFS精度与SNN精度
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

1. 实现统一、固定递减和校准单调三类时间权重。
2. 实现各层带权有符号传播和加权输出读取。
3. 统计 SOPs、ScaleOps、正负脉冲率、稀疏度和延迟。
4. 先在 `T=2/4` 筛选，再扩展到全部时间步并完成配置 I。

### 第四阶段：递减编码感知微调

1. 实现可微逐次精化代理和 clean/refinement 双分支。
2. 从现有 checkpoint 执行阈值、BN、分类器微调。
3. 在较小学习率下执行全网络微调。
4. 完成 I、J、K 的训练成本、ANN精度和 SNN 指标对比。
5. 只在微调后的递减编码有效时，再验证与方向二组合的配置 L。

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
在多个固定T上具有一致收益，而不是只改善单个T；
收益能够迁移到至少两个数据集；
Accuracy-SOPs或Accuracy-Sparsity形成稳定Pareto改善；
ScaleOps和实际延迟没有抵消事件数下降带来的收益。
微调后递减权重必须优于相同训练预算下的均匀权重；
源ANN/QCFS精度下降应受控，并完整报告训练成本。
```

最终方法应重点展示 Accuracy-SOPs 和 Accuracy-bias-storage 的 Pareto 优势，而不是只报告单一最高准确率。

---

## 11. 最终论文贡献表述

建议收敛为两点：

1. **Signed-state-conditioned temporal correction**
   利用 SignedIF 的净累计传输状态统一控制 SNM、R0 和 FTBC，使时序校正显式适应正脉冲撤销状态与 R0 保护状态。

2. **Monotonic signed successive refinement**
   使用非负、归一化且单调不增的时间尺度，将等幅重复发放改造成从粗到细的正负残差修正，并通过递减编码感知的轻量 QCFS 微调，使源网络主动适应该编码动力学。SNM、R0 与 FTBC 分别承担双向修正、状态约束与边界校正。

低秩 bias 压缩、SOPs 和能耗评价是第一点的重要组成部分，但不应被单独夸大为第三个完全独立的算法贡献。
