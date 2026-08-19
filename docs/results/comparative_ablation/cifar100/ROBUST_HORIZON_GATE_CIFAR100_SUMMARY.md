# Robust Horizon-Gated SNM：第二轮双架构实验总结

## 实验目的

本轮只验证第一轮遗留的问题：SNM能否在高时间步严格关闭，以及多验证子集能否稳定选择off、标准SNM和stage-gated SNM。QCFS、无符号Full-FTBC教师和Temporal-r4主体保持不变。

- 拟合：固定1,000张CIFAR-100训练图像，保留原校准增强。
- 验证：3个互斥子集，每组1,000张，使用轻度裁剪/翻转，不使用AutoAugment和Cutout。
- 选择目标：先比较 `mean accuracy - 0.5 x subset std`；0.1pp容差内先最小化ANN-SNN logit MSE，再比较SOP和负脉冲。
- 测试：完整10,000张测试图像，测试集不参与模式与margin选择。
- 两个架构使用相同的拟合与验证张量哈希。

## ResNet20

| Temporal-r4模式 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| SNM-off | 39.62 | 59.74 | 67.27 | **69.51** |
| 标准SNM | **41.98** | 61.73 | **68.30** | 69.53 |
| Stage-gated SNM | 41.23 | **62.26** | 68.29 | 69.09 |

T=8的stage gate为 `middle=2`，其余stage为0。相对标准SNM：

- 准确率提高0.53pp；
- 负脉冲率从0.3954%降到0.2983%，相对减少24.6%；
- SOP从792,939,954,624降到786,182,971,976，减少0.85%。

T=32的代理验证错误选择了stage gate，但完整测试显示off和标准SNM分别为69.51%和69.53%，stage只有69.09%。结合第一轮高时间步结果，应让horizon gate在T=32严格关闭SNM，而不是继续搜索有限margin。

Hybrid在本轮没有稳定优势：T=8只比纯Temporal高0.04pp，T=16/32反而更低，同时增加FTBC存储。因此不再推荐把Hybrid作为跨架构主配置。

## VGG16

| Temporal-r4模式 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| SNM-off | 73.34 | 76.27 | 77.34 | **77.64** |
| 标准SNM | **73.81** | **77.18** | **77.47** | 77.57 |
| Stage-gated SNM | 73.70 | **77.18** | **77.47** | 77.58 |

VGG16在T=8/16搜索出的stage margin全部为0，严格等价于标准SNM。T=4非零stage gate比标准SNM低0.11pp；T=32中off最高，而且完全消除负脉冲与SNM额外SOP。因此VGG16不需要stage级参数，只需要一个horizon开关。

## 跨两轮稳定结论

| 时间步 | 推荐SNM模式 | 依据 |
|---:|---|---|
| T=4 | 标准SNM | 两个架构、两轮实验均比off和stage gate更稳。 |
| T=8 | ResNet20使用middle-stage gate；VGG16使用标准SNM | unevenness最明显，SNM收益最大；ResNet门控同时提高准确率并降低事件。 |
| T=16 | 标准SNM | stage gate收益不超过0.01pp，额外搜索没有价值。 |
| T=32 | SNM-off | 两个架构均由off取得更稳的精度和最低事件开销。 |

因此当前最合理的主方法从复杂的四stage连续margin简化为：

```text
QCFS
  -> unsigned Full-FTBC teacher
  -> shared Temporal-r4 compression
  -> horizon SNM switch
       T=4: standard SNM
       T=8: standard SNM; ResNet可选middle-stage gate
       T=16: standard SNM
       T=32: SNM off
  -> R0
```

该结论支持最初的误差分解：Temporal-r4负责系统校准误差；SNM只在中低时间步仍存在明显异步过发放时开启；当unevenness已经很小时，SNM严格退化为恒等映射。

## 严谨性限制

- VGG16在训练来源验证图像上的准确率接近饱和，因此模式并列时主要依赖ANN-SNN logit MSE。正式论文仍应使用训练阶段预留、从未参与ANN训练的验证集。
- 上述时间步策略是综合第一轮和第二轮测试后得到的后验策略，不能把本轮数值直接当作其独立验证结果。下一次应固定该策略后更换校准子集种子，只测试一次，不再根据测试结果修改。
- 0.1pp左右的差异不能视为显著；需要至少三个校准子集种子报告均值和标准差。
