# QCFS + Full-FTBC + Temporal-LR FTBC + A-SNM CIFAR-10 Ablation

- Status: complete
- Dataset: CIFAR-10
- Architecture: resnet20
- QCFS levels: L=4
- ResNet20 evaluation profile: paper_era
- Checkpoint: `resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth`
- Checkpoint SHA256: `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3`
- ANN accuracy on the 200-image test set: 89.50%
- Time steps: [4, 8]
- Full-FTBC fit: 1 x 200, alpha=0.4
- A-SNM validation: 1 x 200
- Temporal-LR: shared rank-4 basis, threshold-normalized, no exempt layer.
- Temporal-LR falls back to Full-FTBC at T<=4 and is active at T>4.
- Fit batch SHA256: `91909d93eb8fb74d65302d1548e3d026296c22235f21255153411bff6ad97905`
- Validation batch SHA256: `2680aa74ee4792d98247b8f0fe96b3727ae6f1e43319d93759f379e940ba4405`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-10 normalization, and Cutout(1,16).
- Test uses only ToTensor and normalization with shuffle=False.
- Every SNN uses QCFS L=4, rate coding/schedule, ratio=1.0, R0=True, SNM margin=0, FP32.
- Full-FTBC is independently fitted at every T with SNM off; Temporal-LR is compressed from that frozen teacher.
- Each family enables SNM only when SNM-on has strictly higher validation accuracy; ties select off.
- Test data is first accessed after all three A-SNM families are frozen.
- A-SNM guarantees validation-set selection only; test-set reversals are reported diagnostically and never retuned.
- Checkpoint-selection note: the checkpoint is selected by the highest accuracy observed on the 10,000-image CIFAR-10 test set during 300 training epochs; this creates model-selection bias.
- Checkpoint-interpretation note: the checkpoint is the CIFAR-10/ResNet20 QCFS-L4 paper-aligned retrained model evaluated with the paper_era profile; it is not a strict reproduction of the paper's reported accuracy.

## Primary accuracy table

| Config | T=4 | T=8 | SNM-on T |
|---|---:|---:|---|
| A_QCFS_R0 | 83.00% | 89.50% | none |
| B_QCFS_STANDARD_SNM_R0 | 88.50% | 89.50% | 4, 8 |
| C_QCFS_ASNM_R0 | 88.50% | 89.50% | 4, 8 |
| D_QCFS_FULL_FTBC_R0 | 84.00% | 88.00% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 86.00% | 91.50% | 4, 8 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 86.00% | 91.50% | 4, 8 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 84.00% | 90.00% | none |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 86.00% | 91.50% | 4, 8 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 86.00% | 91.50% | 4, 8 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 86.25% |
| B_QCFS_STANDARD_SNM_R0 | 89.00% |
| C_QCFS_ASNM_R0 | 89.00% |
| D_QCFS_FULL_FTBC_R0 | 86.00% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 88.75% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 88.75% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 87.00% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 88.75% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 88.75% |

## Temporal-LR accuracy comparisons

| Comparison | T=4 | T=8 | Mean |
|---|---:|---:|---:|
| G-D | +0.00pp | +2.00pp | +1.00pp |
| H-E | +0.00pp | +0.00pp | +0.00pp |
| I-F | +0.00pp | +0.00pp | +0.00pp |

## ANN-SNN logit MSE

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 2.704767 | 1.156449 |
| B_QCFS_STANDARD_SNM_R0 | 2.399095 | 0.792880 |
| C_QCFS_ASNM_R0 | 2.399095 | 0.792880 |
| D_QCFS_FULL_FTBC_R0 | 2.437396 | 1.049796 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2.221032 | 0.812190 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2.221032 | 0.812190 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 2.437396 | 1.017297 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 2.221032 | 0.782394 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2.221032 | 0.782394 |

## Positive spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 28.0501% | 27.7145% |
| B_QCFS_STANDARD_SNM_R0 | 28.0634% | 27.8103% |
| C_QCFS_ASNM_R0 | 28.0634% | 27.8103% |
| D_QCFS_FULL_FTBC_R0 | 27.7163% | 27.5429% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 27.7436% | 27.6423% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 27.7436% | 27.6423% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 27.7163% | 27.5222% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 27.7436% | 27.6184% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 27.7436% | 27.6184% |

## Negative spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.1743% | 0.2676% |
| C_QCFS_ASNM_R0 | 0.1743% | 0.2676% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.1768% | 0.2681% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.1768% | 0.2681% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.0000% | 0.0000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.1768% | 0.2674% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.1768% | 0.2674% |

## Overall spike sparsity

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 71.9499% | 72.2855% |
| B_QCFS_STANDARD_SNM_R0 | 71.7623% | 71.9220% |
| C_QCFS_ASNM_R0 | 71.7623% | 71.9220% |
| D_QCFS_FULL_FTBC_R0 | 72.2837% | 72.4571% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 72.0796% | 72.0897% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 72.0796% | 72.0897% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 72.2837% | 72.4778% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 72.0796% | 72.1142% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 72.0796% | 72.1142% |

## Input-driven SOPs

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 10,426,035,882 | 20,603,191,794 |
| B_QCFS_STANDARD_SNM_R0 | 10,500,810,602 | 20,926,707,668 |
| C_QCFS_ASNM_R0 | 10,500,810,602 | 20,926,707,668 |
| D_QCFS_FULL_FTBC_R0 | 10,274,686,026 | 20,454,749,958 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 10,356,663,346 | 20,775,820,030 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 10,356,663,346 | 20,775,820,030 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 10,274,686,026 | 20,442,720,938 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 10,356,663,346 | 20,758,772,146 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 10,356,663,346 | 20,758,772,146 |

## FTBC parameters

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 2,752 | 5,504 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2,752 | 5,504 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2,752 | 5,504 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 2,752 | 2,784 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 2,752 | 2,784 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2,752 | 2,784 |

## FTBC storage bytes

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 11,008 | 22,016 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 11,008 | 22,016 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 11,008 | 22,016 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 11,008 | 11,136 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 11,008 | 11,136 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 11,008 | 11,136 |

## Temporal bias synthesis MACs

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 0 | 0 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0 | 0 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0 | 0 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0 | 22,016 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0 | 22,016 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0 | 22,016 |

## Full-teacher calibration elapsed

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.486s | 0.922s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.486s | 0.922s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.486s | 0.922s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.486s | 0.922s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.486s | 0.922s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.486s | 0.922s |

## Temporal compression elapsed

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.000s | 0.000s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000s | 0.000s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000s | 0.000s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000s | 0.127s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000s | 0.127s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000s | 0.127s |

## Inference elapsed (statistics disabled)

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.020s | 0.044s |
| B_QCFS_STANDARD_SNM_R0 | 0.025s | 0.049s |
| C_QCFS_ASNM_R0 | 0.025s | 0.049s |
| D_QCFS_FULL_FTBC_R0 | 0.019s | 0.043s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.028s | 0.047s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.028s | 0.047s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.022s | 0.043s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.022s | 0.050s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.022s | 0.050s |

## Temporal-LR compression

| T | Effective mode | Effective rank | Explained energy | Full parameters | Temporal parameters | Storage ratio | Storage reduction | Synthesis MACs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | full fallback | 4 | 1.000000 | 2,752 | 2,752 | 1.000000 | 0.00% | 0 |
| 8 | temporal_low_rank | 4 | 0.855774 | 5,504 | 2,784 | 0.505814 | 49.42% | 22,016 |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 4, 8 | 0.464s |
| Full-FTBC | 4, 8 | 0.220s |
| Temporal-LR FTBC | 4, 8 | 0.227s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 4 | 59.0000% | 62.0000% | +3.0000pp | on |
| 8 | 68.5000% | 81.0000% | +12.5000pp | on |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 4 | 70.5000% | 72.0000% | +1.5000pp | on |
| 8 | 80.5000% | 83.5000% | +3.0000pp | on |

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 4 | 70.5000% | 72.0000% | +1.5000pp | on |
| 8 | 79.0000% | 83.0000% | +4.0000pp | on |

## Validation-selection generalization audit

This table is diagnostic only. Test accuracy never changes a frozen A-SNM decision.

| Family | T | Selected from validation | Test off | Test on | Test-best mode | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 4 | on | 83.00% | 88.50% | on | yes |
| Full-FTBC | 4 | on | 84.00% | 86.00% | on | yes |
| Temporal-LR FTBC | 4 | on | 84.00% | 86.00% | on | yes |
| QCFS | 8 | on | 89.50% | 89.50% | off | no |
| Full-FTBC | 8 | on | 88.00% | 91.50% | on | yes |
| Temporal-LR FTBC | 8 | on | 90.00% | 91.50% | on | yes |

## Equivalence checks

| Kind | Config/family | T | Expected source | Exact |
|---|---|---:|---|---|
| validation fallback | off:full=temporal | 4 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 4 | Full-FTBC on | yes |
| gate fallback | full=temporal | 4 | identical validation metrics | yes |
| test fallback | off:full=temporal | 4 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 4 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 4 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 8 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 8 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 8 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |

## Per-layer Temporal-LR reconstruction

### T=4

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |

### T=8

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | temporal_low_rank | 16 | 0.00047703 | 0.49429998 | 0.07374726 |
| `conv2_x.0.residual_function.2` | temporal_low_rank | 16 | 0.00010506 | 0.58651346 | 0.04153423 |
| `conv2_x.0.act` | temporal_low_rank | 16 | 0.00031850 | 0.44397634 | 0.05688540 |
| `conv2_x.1.residual_function.2` | temporal_low_rank | 16 | 0.00006852 | 0.39816552 | 0.02945636 |
| `conv2_x.1.act` | temporal_low_rank | 16 | 0.00019084 | 0.33927485 | 0.03919053 |
| `conv2_x.2.residual_function.2` | temporal_low_rank | 16 | 0.00000985 | 0.25561839 | 0.01458037 |
| `conv2_x.2.act` | temporal_low_rank | 16 | 0.00007914 | 0.27495101 | 0.03397037 |
| `conv3_x.0.residual_function.2` | temporal_low_rank | 32 | 0.00000976 | 0.24571653 | 0.01210847 |
| `conv3_x.0.act` | temporal_low_rank | 32 | 0.00001442 | 0.21947806 | 0.02108851 |
| `conv3_x.1.residual_function.2` | temporal_low_rank | 32 | 0.00000495 | 0.24685015 | 0.01725496 |
| `conv3_x.1.act` | temporal_low_rank | 32 | 0.00001822 | 0.23235802 | 0.01985191 |
| `conv3_x.2.residual_function.2` | temporal_low_rank | 32 | 0.00000102 | 0.17096809 | 0.00659266 |
| `conv3_x.2.act` | temporal_low_rank | 32 | 0.00001955 | 0.21455874 | 0.02020727 |
| `conv4_x.0.residual_function.2` | temporal_low_rank | 64 | 0.00000196 | 0.16111486 | 0.00581663 |
| `conv4_x.0.act` | temporal_low_rank | 64 | 0.00000188 | 0.21685952 | 0.00588441 |
| `conv4_x.1.residual_function.2` | temporal_low_rank | 64 | 0.00000093 | 0.25207958 | 0.00407085 |
| `conv4_x.1.act` | temporal_low_rank | 64 | 0.00000233 | 0.21167934 | 0.00556504 |
| `conv4_x.2.residual_function.2` | temporal_low_rank | 64 | 0.00000060 | 0.23401162 | 0.00329312 |
| `conv4_x.2.act` | temporal_low_rank | 64 | 0.00002905 | 0.32776144 | 0.02029308 |
