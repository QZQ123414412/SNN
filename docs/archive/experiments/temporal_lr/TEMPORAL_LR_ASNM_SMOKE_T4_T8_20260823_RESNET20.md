# QCFS + Full-FTBC + Temporal-LR FTBC + A-SNM CIFAR-100 Ablation

- Status: complete
- Architecture: resnet20
- Checkpoint: `resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth`
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- ANN accuracy on the 200-image test set: 68.00%
- Time steps: [4, 8]
- Full-FTBC fit: 1 x 200, alpha=0.4
- A-SNM validation: 1 x 200
- Temporal-LR: shared rank-4 basis, threshold-normalized, no exempt layer.
- Temporal-LR falls back to Full-FTBC at T<=4 and is active at T>4.
- Fit batch SHA256: `42e35ed3bdcda2e94471199d0ce318fef1b60aa43ec493ec52631af8e5b10049`
- Validation batch SHA256: `ed8a6c033924c980bc943bf1c48e1fff63f1a04baf4e661dd89ae16f9f52742e`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-100 normalization, and Cutout(1,16).
- Test uses only ToTensor and normalization with shuffle=False.
- Every SNN uses QCFS L=8, rate coding/schedule, ratio=1.0, R0=True, SNM margin=0, FP32.
- Full-FTBC is independently fitted at every T with SNM off; Temporal-LR is compressed from that frozen teacher.
- Each family enables SNM only when SNM-on has strictly higher validation accuracy; ties select off.
- Test data is first accessed after all three A-SNM families are frozen.
- A-SNM guarantees validation-set selection only; test-set reversals are reported diagnostically and never retuned.

## Primary accuracy table

| Config | T=4 | T=8 | SNM-on T |
|---|---:|---:|---|
| A_QCFS_R0 | 23.50% | 50.50% | none |
| B_QCFS_STANDARD_SNM_R0 | 26.00% | 58.50% | 4, 8 |
| C_QCFS_ASNM_R0 | 26.00% | 58.50% | 4, 8 |
| D_QCFS_FULL_FTBC_R0 | 40.00% | 58.50% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 44.50% | 63.00% | 4, 8 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 44.50% | 63.00% | 4, 8 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 40.00% | 57.50% | none |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 44.50% | 63.00% | 4, 8 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 44.50% | 63.00% | 4, 8 |

## ANN-SNN logit MSE

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 9.488415 | 4.990201 |
| B_QCFS_STANDARD_SNM_R0 | 8.581809 | 3.176345 |
| C_QCFS_ASNM_R0 | 8.581809 | 3.176345 |
| D_QCFS_FULL_FTBC_R0 | 6.118006 | 2.965895 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 5.632106 | 2.291849 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 5.632106 | 2.291849 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 6.118006 | 2.971408 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 5.632106 | 2.256723 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 5.632106 | 2.256723 |

## Positive spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 21.6252% | 21.0858% |
| B_QCFS_STANDARD_SNM_R0 | 22.0128% | 21.6052% |
| C_QCFS_ASNM_R0 | 22.0128% | 21.6052% |
| D_QCFS_FULL_FTBC_R0 | 21.0231% | 20.7070% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 21.3954% | 21.1961% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 21.3954% | 21.1961% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 21.0231% | 20.7097% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 21.3954% | 21.1912% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 21.3954% | 21.1912% |

## Negative spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.2698% | 0.4753% |
| C_QCFS_ASNM_R0 | 0.2698% | 0.4753% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.2542% | 0.4302% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.2542% | 0.4302% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.0000% | 0.0000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.2542% | 0.4295% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.2542% | 0.4295% |

## Overall spike sparsity

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 78.3748% | 78.9142% |
| B_QCFS_STANDARD_SNM_R0 | 77.7174% | 77.9195% |
| C_QCFS_ASNM_R0 | 77.7174% | 77.9195% |
| D_QCFS_FULL_FTBC_R0 | 78.9769% | 79.2930% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 78.3504% | 78.3737% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 78.3504% | 78.3737% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 78.9769% | 79.2903% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 78.3504% | 78.3793% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 78.3504% | 78.3793% |

## Input-driven SOPs

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 8,250,192,212 | 15,985,678,304 |
| B_QCFS_STANDARD_SNM_R0 | 8,492,615,220 | 16,841,309,656 |
| C_QCFS_ASNM_R0 | 8,492,615,220 | 16,841,309,656 |
| D_QCFS_FULL_FTBC_R0 | 7,906,459,788 | 15,564,386,516 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 8,139,469,840 | 16,328,287,924 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 8,139,469,840 | 16,328,287,924 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 7,906,459,788 | 15,567,802,320 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 8,139,469,840 | 16,324,182,304 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 8,139,469,840 | 16,324,182,304 |

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
| D_QCFS_FULL_FTBC_R0 | 0.528s | 1.027s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.528s | 1.027s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.528s | 1.027s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.528s | 1.027s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.528s | 1.027s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.528s | 1.027s |

## Temporal compression elapsed

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.000s | 0.000s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000s | 0.000s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000s | 0.000s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000s | 0.071s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000s | 0.071s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000s | 0.071s |

## Inference elapsed (statistics disabled)

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.025s | 0.052s |
| B_QCFS_STANDARD_SNM_R0 | 0.027s | 0.057s |
| C_QCFS_ASNM_R0 | 0.027s | 0.057s |
| D_QCFS_FULL_FTBC_R0 | 0.025s | 0.051s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.025s | 0.054s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.025s | 0.054s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.025s | 0.049s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.028s | 0.055s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.028s | 0.055s |

## Temporal-LR compression

| T | Effective mode | Effective rank | Explained energy | Full parameters | Temporal parameters | Storage ratio | Storage reduction | Synthesis MACs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | full fallback | 4 | 1.000000 | 2,752 | 2,752 | 1.000000 | 0.00% | 0 |
| 8 | temporal_low_rank | 4 | 0.864889 | 5,504 | 2,784 | 0.505814 | 49.42% | 22,016 |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 4, 8 | 0.497s |
| Full-FTBC | 4, 8 | 0.230s |
| Temporal-LR FTBC | 4, 8 | 0.238s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 4 | 10.0000% | 11.0000% | +1.0000pp | on |
| 8 | 24.5000% | 35.5000% | +11.0000pp | on |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 4 | 22.0000% | 22.5000% | +0.5000pp | on |
| 8 | 37.0000% | 42.0000% | +5.0000pp | on |

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 4 | 22.0000% | 22.5000% | +0.5000pp | on |
| 8 | 38.5000% | 46.0000% | +7.5000pp | on |

## Validation-selection generalization audit

This table is diagnostic only. Test accuracy never changes a frozen A-SNM decision.

| Family | T | Selected from validation | Test off | Test on | Test-best mode | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 4 | on | 23.50% | 26.00% | on | yes |
| Full-FTBC | 4 | on | 40.00% | 44.50% | on | yes |
| Temporal-LR FTBC | 4 | on | 40.00% | 44.50% | on | yes |
| QCFS | 8 | on | 50.50% | 58.50% | on | yes |
| Full-FTBC | 8 | on | 58.50% | 63.00% | on | yes |
| Temporal-LR FTBC | 8 | on | 57.50% | 63.00% | on | yes |

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
| `conv1.2` | temporal_low_rank | 16 | 0.00012881 | 0.44927579 | 0.03868946 |
| `conv2_x.0.residual_function.2` | temporal_low_rank | 16 | 0.00005323 | 0.55194539 | 0.02825547 |
| `conv2_x.0.act` | temporal_low_rank | 16 | 0.00007084 | 0.52508193 | 0.03520408 |
| `conv2_x.1.residual_function.2` | temporal_low_rank | 16 | 0.00001044 | 0.52182156 | 0.01279256 |
| `conv2_x.1.act` | temporal_low_rank | 16 | 0.00000752 | 0.23816743 | 0.01065121 |
| `conv2_x.2.residual_function.2` | temporal_low_rank | 16 | 0.00000079 | 0.27524349 | 0.00583022 |
| `conv2_x.2.act` | temporal_low_rank | 16 | 0.00004813 | 0.41843480 | 0.03021349 |
| `conv3_x.0.residual_function.2` | temporal_low_rank | 32 | 0.00000796 | 0.26869291 | 0.01537848 |
| `conv3_x.0.act` | temporal_low_rank | 32 | 0.00000825 | 0.22605693 | 0.01148913 |
| `conv3_x.1.residual_function.2` | temporal_low_rank | 32 | 0.00000060 | 0.15626289 | 0.00312247 |
| `conv3_x.1.act` | temporal_low_rank | 32 | 0.00000691 | 0.17700791 | 0.01251845 |
| `conv3_x.2.residual_function.2` | temporal_low_rank | 32 | 0.00000079 | 0.17965209 | 0.00564511 |
| `conv3_x.2.act` | temporal_low_rank | 32 | 0.00000871 | 0.15380539 | 0.01338892 |
| `conv4_x.0.residual_function.2` | temporal_low_rank | 64 | 0.00000276 | 0.17031899 | 0.00843113 |
| `conv4_x.0.act` | temporal_low_rank | 64 | 0.00001096 | 0.22525993 | 0.01306066 |
| `conv4_x.1.residual_function.2` | temporal_low_rank | 64 | 0.00000254 | 0.29758018 | 0.00903225 |
| `conv4_x.1.act` | temporal_low_rank | 64 | 0.00004840 | 0.24102062 | 0.02849360 |
| `conv4_x.2.residual_function.2` | temporal_low_rank | 64 | 0.00000244 | 0.32168218 | 0.00666154 |
| `conv4_x.2.act` | temporal_low_rank | 64 | 0.00049702 | 0.27382740 | 0.08275238 |
