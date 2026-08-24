# 实验结果清单

本清单记录结果文件的实验含义、生产代码和使用状态。路径按“实验族 -> 阶段 -> 数据集”组织。

来源等级：

- `直接`：报告格式、配置和输出逻辑可与生产脚本逐项对应。
- `人工`：根据测试输出或多份报告人工整理，不是脚本直接写出的文件。
- `推定`：数据和图例可对应，但仓库缺少生成该文件的完整代码。
- `外部分支`：生产代码位于其他 Git 分支，当前分支只保留运行日志。

## Baseline: Accuracy Only

| 文件 | 实验 | 生产代码 | 来源 | 状态 |
|---|---|---|---|---|
| `baseline/accuracy_only/cifar10/ABLATION_RESULTS_cifar10_v2.md` | CIFAR-10/VGG16，QCFS、SNM、R0、FTBC 六配置准确率消融 | `scripts/experiments/run_ablation.py` | 直接 | 历史基线 |
| `baseline/accuracy_only/cifar10/ABLATION_RESULTS_cifar10_T64.md` | 上述六配置的 CIFAR-10 T=64 补充结果 | `scripts/experiments/run_ablation.py` | 直接 | 历史补充 |
| `baseline/accuracy_only/cifar10/RESNET20_QCFS_L4_PAPER_ALIGNED_RETRAIN.md` | CIFAR-10/ResNet20 QCFS-L4、batch size 300、paper_era profile 的 300-epoch 论文口径对齐重训；记录测试集最优 checkpoint、严格重载准确率和 SHA256 | `scripts/train/main_train.py` 与训练元数据 | 人工 | 正式基线权重 |
| `baseline/accuracy_only/cifar100/ABLATION_RESULTS.md` | CIFAR-100/VGG16，QCFS、SNM、R0、FTBC 六配置早期准确率消融 | `scripts/experiments/run_ablation.py` | 直接 | 历史基线 |
| `baseline/accuracy_only/cifar100/ABLATION_RESULTS_cifar100_v2.md` | 六配置 CIFAR-100 准确率复跑 | `scripts/experiments/run_ablation.py` | 直接 | 历史基线 |
| `baseline/accuracy_only/cifar100/ABLATION_RESULTS_cifar100_T64.md` | 上述六配置的 CIFAR-100 T=64 补充结果 | `scripts/experiments/run_ablation.py` | 直接 | 历史补充 |
| `baseline/accuracy_only/cifar100/CIFAR100_IF_TEST_RESULTS.md` | 原始 IF/QCFS 测试 | `scripts/evaluate/main_test.py` 的输出 | 人工 | 早期测试记录 |
| `baseline/accuracy_only/cifar100/CIFAR100_COMPARISON_RESULTS.md` | IF 与 SignedIF/SNM 对比 | `scripts/evaluate/main_test.py`、`scripts/evaluate/main_test_signed.py` 的输出 | 人工 | 早期对比记录 |

## Baseline: Spike Statistics

| 文件 | 实验 | 生产代码 | 来源 | 状态 |
|---|---|---|---|---|
| `baseline/spike_statistics/cifar100/ABLATION_RESULTS_VGG_cifar100_add_3_quantity_v2.md` | CIFAR-100/VGG16 六配置；Accuracy、SOPs、正负脉冲率、稀疏率和逐层指标，T=1/2/4/8/16/32 | `scripts/experiments/run_stats_ablation.py` | 直接 | 正式基线 |
| `baseline/spike_statistics/README.md` | 脉冲统计指标及实现说明 | 人工文档 | 人工 | 说明文档 |

对应图表位于 `../../figures/baseline/spike_statistics/cifar100/`。图中数据和六配置图例与上述正式基线一致，但仓库没有保留绘图脚本，因此图表来源等级为“推定”。

## Comparative Ablation

| 文件 | 实验 | 生产代码 | 来源 | 状态 |
|---|---|---|---|---|
| `comparative_ablation/cifar10/CIFAR10_FIVE_WAY_ABLATION.md` | CIFAR-10/VGG16 五组严格对照：QCFS、QCFS+SNM、QCFS+FTBC、QCFS+SNM+完整 FTBC、QCFS+SNM+state-low-rank FTBC；全部启用 R0，T=1/2/4/8/16/32 | `scripts/experiments/run_successive_refinement_ablation.py` | 直接 | 最新正式结果 |
| `comparative_ablation/cifar100/SR_THREE_WAY_ABLATION_cifar100.md` | CIFAR-100/VGG16 三组严格对照：完整 FTBC、匹配权重的 state-low-rank FTBC、state-low-rank FTBC+CSRR，T=1/2/4/8/16/32 | `scripts/experiments/run_successive_refinement_ablation.py` | 直接 | 最新正式结果 |
| `comparative_ablation/cifar100/RESNET20_SIX_WAY_ABLATION.md` | CIFAR-100/ResNet20，使用训练日志 68.78% 的 QCFS-L8 权重；SNM 开/关与 FTBC none/full/state-low-rank 的 2×3 严格对照，全部启用 R0、禁用 CSRR，T=1/2/4/8/16/32 | `scripts/experiments/run_resnet20_qcfs_ablation.py` | 直接 | 最新正式结果 |
| `comparative_ablation/cifar100/RESNET20_STATE_LR_CAUSAL_DIAGNOSTICS.md` | 对 ResNet20 的 SNM×state-low-rank 异常做三项因果诊断：复用 E 系数开启 SNM、关闭 `bias_state`、五批校准后全局裁剪；T=4/8/16/32 | `scripts/experiments/run_resnet20_state_lr_causal_diagnostics.py` | 直接 | 最新正式诊断 |
| `comparative_ablation/cifar100/TEMPORAL_LR_GATED_SNM_CIFAR100_RESNET20.md` | CIFAR-100/ResNet20 的 Full-FTBC 教师压缩、Temporal-LR 与冻结后 Gated-SNM 组合实验 | `scripts/experiments/run_temporal_lr_gated_snm.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/TEMPORAL_LR_GATED_SNM_CIFAR100_VGG16.md` | CIFAR-100/VGG16 的 Full-FTBC 教师压缩、Temporal-LR 与冻结后 Gated-SNM 组合实验 | `scripts/experiments/run_temporal_lr_gated_snm.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/TEMPORAL_LR_GATED_SNM_CIFAR100_SUMMARY.md` | Temporal-LR + Gated-SNM 在 ResNet20 和 VGG16 上的跨网络结果汇总 | 上述两份 Temporal-LR 报告 | 人工 | 正式分析 |
| `comparative_ablation/cifar100/ROBUST_HORIZON_GATE_CIFAR100_RESNET20.md` | CIFAR-100/ResNet20 的稳健 horizon gate 验证 | `scripts/experiments/run_horizon_gate_validation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/ROBUST_HORIZON_GATE_CIFAR100_VGG16.md` | CIFAR-100/VGG16 的稳健 horizon gate 验证 | `scripts/experiments/run_horizon_gate_validation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/ROBUST_HORIZON_GATE_CIFAR100_SUMMARY.md` | 稳健 horizon gate 在 ResNet20 和 VGG16 上的跨网络结果汇总 | 上述两份 horizon gate 报告 | 人工 | 正式分析 |
| `comparative_ablation/cifar100/FULL_FTBC_ASNM_CIFAR100_resnet20.md` | CIFAR-100/ResNet20 的 QCFS、标准 SNM、逐时间步验证准确率门控 A-SNM、Full-FTBC 六组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_full_ftbc_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/FULL_FTBC_ASNM_CIFAR100_vgg16.md` | CIFAR-100/VGG16 的 QCFS、标准 SNM、逐时间步验证准确率门控 A-SNM、Full-FTBC 六组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_full_ftbc_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar10/FULL_FTBC_ASNM_CIFAR10_resnet20.md` | CIFAR-10/ResNet20，使用 L=8、seed 42、300 epoch 测试集最优权重的 QCFS、标准 SNM、逐时间步 A-SNM 与 Full-FTBC 六组严格消融，T=1/2/4/8/16/32；报告明确记录测试集选择偏差 | `scripts/experiments/run_full_ftbc_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar10/FULL_FTBC_ASNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.md` | CIFAR-10/ResNet20，使用 L=4、batch size 300、paper_era profile 的论文口径对齐重训权重；QCFS、标准 SNM、逐时间步 A-SNM 与 Full-FTBC 六组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_full_ftbc_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar10/TEMPORAL_LR_ASNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.md` | CIFAR-10/ResNet20，使用论文口径对齐重训的 QCFS L=4 权重；QCFS、Full-FTBC、共享 rank-4 Temporal-LR FTBC 与逐时间步 A-SNM 九组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_temporal_lr_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar10/FULL_FTBC_ASNM_CIFAR10_vgg16.md` | CIFAR-10/VGG16 旧权重在 L=8 下的后设评测；QCFS、标准 SNM、逐时间步 A-SNM 与 Full-FTBC 六组严格消融，T=1/2/4/8/16/32；不视为 L=8 重训结果 | `scripts/experiments/run_full_ftbc_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar10/FULL_FTBC_ASNM_CIFAR10_SUMMARY.md` | CIFAR-10/ResNet20 与 VGG16 的 Full-FTBC + A-SNM 正式结果汇总 | `scripts/experiments/summarize_full_ftbc_asnm.py` | 直接 | 正式分析 |
| `comparative_ablation/cifar10/FULL_FTBC_ASNM_CIFAR10_CIFAR100_COMPARISON.md` | Full-FTBC + A-SNM 在 CIFAR-10 与 CIFAR-100 上的双架构对照；明确限制 VGG16 的训练 L 来源 | `scripts/experiments/summarize_full_ftbc_asnm.py` | 直接 | 正式分析 |
| `comparative_ablation/cifar100/TEMPORAL_LR_ASNM_CIFAR100_RESNET20.md` | CIFAR-100/ResNet20 的 QCFS、Full-FTBC、共享 rank-4 Temporal-LR FTBC 与逐时间步 A-SNM 九组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_temporal_lr_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/TEMPORAL_LR_ASNM_CIFAR100_VGG16.md` | CIFAR-100/VGG16 的 QCFS、Full-FTBC、共享 rank-4 Temporal-LR FTBC 与逐时间步 A-SNM 九组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_temporal_lr_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/TEMPORAL_LR_ASNM_CIFAR100_SUMMARY.md` | Temporal-LR FTBC + A-SNM 在 ResNet20 和 VGG16 上的双架构准确率与存储压缩汇总 | `scripts/experiments/run_temporal_lr_asnm_ablation.py` | 直接 | 正式分析 |
| `comparative_ablation/cifar10/PA_FTBC_ASNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.md` | CIFAR-10/ResNet20 QCFS-L4 的 QCFS、Full、Temporal-LR、无 SVD Parity-Anchor FTBC 与各自 A-SNM 十二组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_pa_ftbc_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar10/PA_FTBC_ASNM_CIFAR10_VGG16_L8.md` | CIFAR-10/VGG16 QCFS-L8 的 QCFS、Full、Temporal-LR、Parity-Anchor FTBC 与各自 A-SNM 十二组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_pa_ftbc_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/PA_FTBC_ASNM_CIFAR100_RESNET20_L8.md` | CIFAR-100/ResNet20 QCFS-L8 的 QCFS、Full、Temporal-LR、Parity-Anchor FTBC 与各自 A-SNM 十二组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_pa_ftbc_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/PA_FTBC_ASNM_CIFAR100_VGG16_L8.md` | CIFAR-100/VGG16 QCFS-L8 的 QCFS、Full、Temporal-LR、Parity-Anchor FTBC 与各自 A-SNM 十二组严格消融，T=1/2/4/8/16/32 | `scripts/experiments/run_pa_ftbc_asnm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/PA_FTBC_ASNM_FOUR_MODEL_SUMMARY.md` | Parity-Anchor FTBC 四模型正式结果、全指标、存储、合成开销与 Temporal-LR 对比汇总 | `scripts/experiments/summarize_pa_ftbc.py` | 直接 | 正式分析 |
| `comparative_ablation/cifar10/HA_SNM_CIFAR10_RESNET20_L4_PAPER_ALIGNED.md` | CIFAR-10/ResNet20 L4 的 Full、Temporal-LR、PA-FTBC × SNM-off、原始SNM、HA-SNM九组正式消融 | `scripts/experiments/run_ha_snm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar10/HA_SNM_CIFAR10_VGG16_L8.md` | CIFAR-10/VGG16 L8 的 Full、Temporal-LR、PA-FTBC × SNM-off、原始SNM、HA-SNM九组正式消融 | `scripts/experiments/run_ha_snm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/HA_SNM_CIFAR100_RESNET20_L8.md` | CIFAR-100/ResNet20 L8 的 Full、Temporal-LR、PA-FTBC × SNM-off、原始SNM、HA-SNM九组正式消融 | `scripts/experiments/run_ha_snm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/cifar100/HA_SNM_CIFAR100_VGG16_L8.md` | CIFAR-100/VGG16 L8 的 Full、Temporal-LR、PA-FTBC × SNM-off、原始SNM、HA-SNM九组正式消融 | `scripts/experiments/run_ha_snm_ablation.py` | 直接 | 正式验证 |
| `comparative_ablation/HA_SNM_FOUR_MODEL_DELIVERY_REPORT.md` | HA-SNM四模型准确率、MSE、脉冲/SOP、推理开销、验证筛选和1152单元回归审计汇总 | `scripts/experiments/summarize_ha_snm.py` | 直接 | 正式分析 |

## State-conditioned Low-rank FTBC

| 文件 | 实验 | 生产代码 | 来源 | 状态 |
|---|---|---|---|---|
| `state_low_rank_ftbc/final/cifar100/STATE_LOW_RANK_FTBC_cifar100_full.md` | 完整 FTBC、state-low-rank FTBC、SOPs 加权 state-low-rank FTBC 全时间步消融 | `scripts/experiments/run_state_ftbc_ablation.py` | 直接 | 正式结果 |
| `state_low_rank_ftbc/final/STATE_LOW_RANK_FTBC_FINAL.md` | low-rank FTBC 原理、存储量和最终结论汇总 | 多份 state-FTBC 报告 | 人工 | 正式分析 |
| `state_low_rank_ftbc/replay_validation/cifar100/FTBC_REPLAY_FIX_D.md` | D_QCFS+FTBC replay 修复验证 | `scripts/experiments/run_stats_ablation.py` | 直接 | 回归验证 |
| `state_low_rank_ftbc/replay_validation/cifar100/FTBC_REPLAY_FIX_E.md` | E_QCFS+SNM+FTBC replay 修复验证 | `scripts/experiments/run_stats_ablation.py` | 直接 | 回归验证 |
| `state_low_rank_ftbc/replay_validation/cifar100/FTBC_REPLAY_FIX_F.md` | F_QCFS+SNM+R0+FTBC replay 修复验证 | `scripts/experiments/run_stats_ablation.py` | 直接 | 回归验证 |

## CSRR Final and Analysis

| 文件 | 实验 | 生产代码 | 来源 | 状态 |
|---|---|---|---|---|
| `csrr/final/cifar100/SR_ABLATION_cifar100.md` | CIFAR-100 旧版四组最终消融：SNM+R0、state-low-rank、CSRR、state-low-rank+CSRR | `scripts/experiments/run_successive_refinement_ablation.py` | 直接 | 正式历史结果 |
| `csrr/final/cifar10/SR_ABLATION_cifar10_T2_T8.md` | 上述 CSRR 方案在 CIFAR-10 的 T=2/4/8 迁移验证 | `scripts/experiments/run_successive_refinement_ablation.py` | 直接 | 正式迁移验证 |
| `csrr/analysis/SR_FINAL_ANALYSIS.md` | CSRR 参数筛选、最终配置和跨数据集结论 | 多份 `SR_*.md` 报告 | 人工 | 正式分析 |

## CSRR Screening

以下文件均由 `scripts/experiments/run_successive_refinement_ablation.py` 直接生成，属于 CIFAR-100/VGG16 参数筛选，不应单独作为论文最终主表。

| 文件 | 筛选内容 |
|---|---|
| `csrr/screening/cifar100/SR_RATIO_SCREEN_cifar100_T2_T4.md` | successive-refinement ratio：1.1、1.25、1.5、2.0 |
| `csrr/screening/cifar100/SR_HYSTERESIS_SCREEN_cifar100_T2_T4.md` | 无 FTBC 条件下的负 margin |
| `csrr/screening/cifar100/SR_FTBC_SCREEN_cifar100_T2_T4.md` | state-low-rank FTBC 与负 margin 组合 |
| `csrr/screening/cifar100/SR_MILD_RATIO_FTBC_cifar100_T2_T4.md` | 轻量 ratio 1.02、1.05、1.075 与 FTBC 组合 |
| `csrr/screening/cifar100/SR_UNIFORM_FTBC_cifar100_T2_T4.md` | ratio=1.0 时的负 margin 筛选 |
| `csrr/screening/cifar100/SR_PARETO_SCREEN_cifar100_T2_T4.md` | 负 margin 1.30、1.35、1.40 的 Pareto 筛选 |
| `csrr/screening/cifar100/SR_POSITIVE_MARGIN_SCREEN_cifar100_T2_T4.md` | 正 margin 0.50、0.52、0.55、0.60 筛选 |
| `csrr/screening/cifar100/SR_T4_MARGIN_SCREEN_cifar100.md` | T=4 正负 margin 网格 |
| `csrr/screening/cifar100/SR_T4_OVER25_cifar100.md` | T=4，over_weight=2.5 候选 |
| `csrr/screening/cifar100/SR_T4_OVER30_cifar100.md` | T=4，over_weight=3.0 候选 |

## CSRR Diagnostics

以下文件均由 `scripts/experiments/run_successive_refinement_ablation.py` 直接生成。

| 文件 | 诊断内容 |
|---|---|
| `csrr/diagnostics/cifar100/SR_SMOKE_cifar100_T2.md` | 最初 T=2 冒烟实验 |
| `csrr/diagnostics/cifar100/SR_DIAG_cifar100_T2.md` | uniform、unsigned geometric、signed、有无 R0 的机制诊断 |
| `csrr/diagnostics/cifar100/SR_FIXED_A_T2_cifar100.md` | 固定候选 A 的 T=2 验证 |
| `csrr/diagnostics/cifar100/SR_FIXED_B_T2_T4_cifar100.md` | 固定候选 B 的 T=2/4 验证 |

## Development Archive

以下报告均由 `scripts/experiments/run_state_ftbc_ablation.py` 直接生成，只用于实现追溯。

| 文件 | 内容 | 状态 |
|---|---|---|
| `../archive/experiments/resnet20/DISTILL_QCFS_FINETUNE_DIAGNOSTIC.md` | CIFAR-100/ResNet20 QCFS 68.78% 权重的两条独立 100-epoch ResNet56/DIST 蒸馏微调轨迹 | 已完成诊断；最高 68.66%，未达到 69.94%，拒绝作为正式基线 |
| `../archive/experiments/resnet20/FIXED_BUDGET_FINETUNE_DIAGNOSTIC.md` | CIFAR-100/ResNet20 QCFS 68.78% 权重的三条独立 50-epoch 微调轨迹，初始 LR 为 0.005/0.002/0.001 | 已完成诊断；最高 68.69%，未达到 69.94%，拒绝作为正式基线 |
| `../archive/experiments/ha_snm/HA_SNM_NORMALIZED_SCREEN_CIFAR10_RESNET20_L4_20260824.md` | HA-SNM统一阈值日程的CIFAR-10/ResNet20验证集筛选；不访问测试图像 | 参数筛选归档 |
| `../archive/experiments/ha_snm/HA_SNM_NORMALIZED_SCREEN_CIFAR10_VGG16_L8_20260824.md` | HA-SNM统一阈值日程的CIFAR-10/VGG16验证集筛选；不访问测试图像 | 参数筛选归档 |
| `../archive/experiments/ha_snm/HA_SNM_NORMALIZED_SCREEN_CIFAR100_RESNET20_L8_20260824.md` | HA-SNM统一阈值日程的CIFAR-100/ResNet20验证集筛选；不访问测试图像 | 参数筛选归档 |
| `../archive/experiments/ha_snm/HA_SNM_NORMALIZED_SCREEN_CIFAR100_VGG16_L8_20260824.md` | HA-SNM统一阈值日程的CIFAR-100/VGG16验证集筛选；不访问测试图像 | 参数筛选归档 |
| `../archive/experiments/ha_snm/HA_SNM_CIFAR10_RESNET20_L4_SMOKE_T4_T8_20260824.md` | HA-SNM九组入口的T=4/8单批次冒烟和回退一致性检查 | 冒烟归档 |
| `../archive/experiments/state_low_rank_ftbc/smoke/cifar100/STATE_LOW_RANK_FTBC_smoke_T4.md` | 单校准批次 T=4 冒烟 | 归档 |
| `../archive/experiments/state_low_rank_ftbc/smoke/cifar100/STATE_LOW_RANK_FTBC_validation_T4_T8.md` | T=4/8 初步验证 | 归档 |
| `../archive/experiments/state_low_rank_ftbc/implementation_validation/cifar100/STATE_LOW_RANK_FTBC_optimized_T4.md` | 优化实现 T=4 验证 | 归档 |
| `../archive/experiments/state_low_rank_ftbc/implementation_validation/cifar100/STATE_LOW_RANK_FTBC_optimized_H_T32.md` | 优化 H 配置 T=32 验证 | 归档 |
| `../archive/experiments/state_low_rank_ftbc/implementation_validation/cifar100/STATE_LOW_RANK_FTBC_inplace_bias_T4.md` | in-place bias T=4 验证 | 归档 |
| `../archive/experiments/state_low_rank_ftbc/implementation_validation/cifar100/STATE_LOW_RANK_FTBC_inplace_bias_H_T32.md` | in-place H 配置 T=32 验证 | 归档 |
| `../archive/experiments/state_low_rank_ftbc/implementation_validation/cifar100/STATE_LOW_RANK_FTBC_latency_clean.md` | 关闭统计同步后的纯推理延迟 | 归档 |
| `../archive/experiments/state_low_rank_ftbc/failed_attempts/cifar100/STATE_LOW_RANK_FTBC_addcmul_T4.md` | `addcmul` 实现失败尝试 | 失败归档 |

## Figures and External-branch Logs

| 路径 | 来源 | 来源等级 | 状态 |
|---|---|---|---|
| `../../figures/baseline/spike_statistics/cifar100/elapsed_time.png` | CIFAR-100 六配置推理时间 | 推定 | 展示图；缺少绘图脚本 |
| `../../figures/baseline/spike_statistics/cifar100/elapsed_time.pdf` | 推理时间图的矢量版本 | 推定 | 展示图；缺少绘图脚本 |
| `../../figures/baseline/spike_statistics/cifar100/input_driven_sops.png` | CIFAR-100 六配置 input-driven SOPs | 推定 | 展示图；缺少绘图脚本 |
| `../../figures/baseline/spike_statistics/cifar100/input_driven_sops.pdf` | SOPs 图的矢量版本 | 推定 | 展示图；缺少绘图脚本 |
| `../../figures/baseline/spike_statistics/cifar100/positive_spike_rate.png` | CIFAR-100 六配置正脉冲率 | 推定 | 展示图；缺少绘图脚本 |
| `../../figures/baseline/spike_statistics/cifar100/positive_spike_rate.pdf` | 正脉冲率图的矢量版本 | 推定 | 展示图；缺少绘图脚本 |
| `../../figures/baseline/spike_statistics/cifar100/negative_spike_rate.png` | CIFAR-100 六配置负脉冲率 | 推定 | 展示图；缺少绘图脚本 |
| `../../figures/baseline/spike_statistics/cifar100/negative_spike_rate.pdf` | 负脉冲率图的矢量版本 | 推定 | 展示图；缺少绘图脚本 |
| `../../figures/baseline/spike_statistics/cifar100/overall_spike_sparsity.png` | CIFAR-100 六配置整体脉冲稀疏率 | 推定 | 展示图；缺少绘图脚本 |
| `../../figures/baseline/spike_statistics/cifar100/overall_spike_sparsity.pdf` | 稀疏率图的矢量版本 | 推定 | 展示图；缺少绘图脚本 |
| `../archive/foreign_branch_logs/monotonic_refinement/MSSR_full_run.log` | `feature/monotonic-signed-successive-refinement` 分支的 `scripts/experiments/run_monotonic_refinement_ablation.py` | 外部分支 | 本地运行日志，不是正式报告 |
| `../archive/foreign_branch_logs/monotonic_refinement/MSSR_full_run.err.log` | 同上；空错误日志 | 外部分支 | 本地运行日志，不是正式报告 |

`docs/archive/project_history/` 中的 `MODIFICATIONS_SUMMARY.md` 和 `参数转换示例输出.md` 是项目历史与参数示例，不是实验结果，因此不纳入结果分类。
