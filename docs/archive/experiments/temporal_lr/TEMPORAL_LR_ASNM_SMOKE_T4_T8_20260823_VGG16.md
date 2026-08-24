# QCFS + Full-FTBC + Temporal-LR FTBC + A-SNM CIFAR-100 Ablation

- Status: complete
- Architecture: vgg16
- Checkpoint: `cifar100-vgg16-l8-example.pth`
- Checkpoint SHA256: `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339`
- ANN accuracy on the 200-image test set: 75.50%
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
| A_QCFS_R0 | 71.50% | 76.00% | none |
| B_QCFS_STANDARD_SNM_R0 | 75.00% | 77.50% | 4, 8 |
| C_QCFS_ASNM_R0 | 75.00% | 77.50% | 4, 8 |
| D_QCFS_FULL_FTBC_R0 | 74.50% | 75.50% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 72.50% | 78.00% | 4, 8 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 72.50% | 78.00% | 4, 8 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 74.50% | 74.50% | none |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 72.50% | 79.00% | 4, 8 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 72.50% | 79.00% | 4, 8 |

## ANN-SNN logit MSE

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 1.501546 | 0.564945 |
| B_QCFS_STANDARD_SNM_R0 | 1.300599 | 0.309677 |
| C_QCFS_ASNM_R0 | 1.300599 | 0.309677 |
| D_QCFS_FULL_FTBC_R0 | 1.346281 | 0.502271 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.139520 | 0.287994 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.139520 | 0.287994 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.346281 | 0.529029 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.139520 | 0.302332 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.139520 | 0.302332 |

## Positive spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 15.2381% | 15.1730% |
| B_QCFS_STANDARD_SNM_R0 | 15.2607% | 15.1944% |
| C_QCFS_ASNM_R0 | 15.2607% | 15.1944% |
| D_QCFS_FULL_FTBC_R0 | 15.1682% | 15.1211% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 15.1877% | 15.1431% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 15.1877% | 15.1431% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 15.1682% | 15.1424% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 15.1877% | 15.1642% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 15.1877% | 15.1642% |

## Negative spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0329% | 0.0446% |
| C_QCFS_ASNM_R0 | 0.0329% | 0.0446% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0307% | 0.0415% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0307% | 0.0415% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.0000% | 0.0000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.0307% | 0.0426% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.0307% | 0.0426% |

## Overall spike sparsity

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 84.7619% | 84.8270% |
| B_QCFS_STANDARD_SNM_R0 | 84.7064% | 84.7610% |
| C_QCFS_ASNM_R0 | 84.7064% | 84.7610% |
| D_QCFS_FULL_FTBC_R0 | 84.8318% | 84.8789% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 84.7816% | 84.8155% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 84.7816% | 84.8155% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 84.8318% | 84.8576% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 84.7816% | 84.7932% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 84.7816% | 84.7932% |

## Input-driven SOPs

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 52,086,159,488 | 103,365,927,232 |
| B_QCFS_STANDARD_SNM_R0 | 52,460,676,992 | 104,364,419,520 |
| C_QCFS_ASNM_R0 | 52,460,676,992 | 104,364,419,520 |
| D_QCFS_FULL_FTBC_R0 | 51,679,822,208 | 102,832,383,744 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 52,013,768,192 | 103,775,609,856 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 52,013,768,192 | 103,775,609,856 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 51,679,822,208 | 102,986,841,088 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 52,013,768,192 | 103,950,213,504 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 52,013,768,192 | 103,950,213,504 |

## FTBC parameters

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 49,664 | 99,328 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 49,664 | 99,328 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 49,664 | 99,328 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 49,664 | 49,696 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 49,664 | 49,696 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 49,664 | 49,696 |

## FTBC storage bytes

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 198,656 | 397,312 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 198,656 | 397,312 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 198,656 | 397,312 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 198,656 | 198,784 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 198,656 | 198,784 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 198,656 | 198,784 |

## Temporal bias synthesis MACs

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 0 | 0 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0 | 0 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0 | 0 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0 | 397,312 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0 | 397,312 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0 | 397,312 |

## Full-teacher calibration elapsed

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 1.003s | 1.851s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.003s | 1.851s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.003s | 1.851s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.003s | 1.851s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.003s | 1.851s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.003s | 1.851s |

## Temporal compression elapsed

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.000s | 0.000s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000s | 0.000s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000s | 0.000s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000s | 0.025s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000s | 0.025s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000s | 0.025s |

## Inference elapsed (statistics disabled)

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.058s | 0.120s |
| B_QCFS_STANDARD_SNM_R0 | 0.063s | 0.132s |
| C_QCFS_ASNM_R0 | 0.063s | 0.132s |
| D_QCFS_FULL_FTBC_R0 | 0.059s | 0.120s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.064s | 0.129s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.064s | 0.129s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.059s | 0.118s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.063s | 0.127s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.063s | 0.127s |

## Temporal-LR compression

| T | Effective mode | Effective rank | Explained energy | Full parameters | Temporal parameters | Storage ratio | Storage reduction | Synthesis MACs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | full fallback | 4 | 1.000000 | 49,664 | 49,664 | 1.000000 | 0.00% | 0 |
| 8 | temporal_low_rank | 4 | 0.901210 | 99,328 | 49,696 | 0.500322 | 49.97% | 397,312 |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 4, 8 | 0.497s |
| Full-FTBC | 4, 8 | 0.459s |
| Temporal-LR FTBC | 4, 8 | 0.466s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 4 | 76.5000% | 80.0000% | +3.5000pp | on |
| 8 | 85.0000% | 87.5000% | +2.5000pp | on |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 4 | 80.0000% | 81.0000% | +1.0000pp | on |
| 8 | 86.0000% | 87.5000% | +1.5000pp | on |

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 4 | 80.0000% | 81.0000% | +1.0000pp | on |
| 8 | 86.0000% | 87.5000% | +1.5000pp | on |

## Validation-selection generalization audit

This table is diagnostic only. Test accuracy never changes a frozen A-SNM decision.

| Family | T | Selected from validation | Test off | Test on | Test-best mode | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 4 | on | 71.50% | 75.00% | on | yes |
| Full-FTBC | 4 | on | 74.50% | 72.50% | off | no |
| Temporal-LR FTBC | 4 | on | 74.50% | 72.50% | off | no |
| QCFS | 8 | on | 76.00% | 77.50% | on | yes |
| Full-FTBC | 8 | on | 75.50% | 78.00% | on | yes |
| Temporal-LR FTBC | 8 | on | 74.50% | 79.00% | on | yes |

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
| `layer1.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer1.6` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer2.2` | full | 128 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer2.6` | full | 128 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer3.2` | full | 256 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer3.6` | full | 256 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer3.10` | full | 256 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer4.2` | full | 512 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer4.6` | full | 512 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer4.10` | full | 512 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer5.2` | full | 512 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer5.6` | full | 512 | 0.00000000 | 0.00000000 | 0.00000000 |
| `layer5.10` | full | 512 | 0.00000000 | 0.00000000 | 0.00000000 |
| `classifier.2` | full | 4096 | 0.00000000 | 0.00000000 | 0.00000000 |
| `classifier.5` | full | 4096 | 0.00000000 | 0.00000000 | 0.00000000 |

### T=8

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | temporal_low_rank | 64 | 0.00000786 | 0.55313128 | 0.01365386 |
| `layer1.6` | temporal_low_rank | 64 | 0.00001996 | 0.68587267 | 0.02901806 |
| `layer2.2` | temporal_low_rank | 128 | 0.00000318 | 0.51114595 | 0.01599938 |
| `layer2.6` | temporal_low_rank | 128 | 0.00000034 | 0.31057599 | 0.00556781 |
| `layer3.2` | temporal_low_rank | 256 | 0.00000038 | 0.31972337 | 0.00336071 |
| `layer3.6` | temporal_low_rank | 256 | 0.00000030 | 0.33816060 | 0.00287478 |
| `layer3.10` | temporal_low_rank | 256 | 0.00000022 | 0.34328941 | 0.00258831 |
| `layer4.2` | temporal_low_rank | 512 | 0.00000024 | 0.43743229 | 0.00408475 |
| `layer4.6` | temporal_low_rank | 512 | 0.00000010 | 0.42041796 | 0.00182802 |
| `layer4.10` | temporal_low_rank | 512 | 0.00000005 | 0.47744536 | 0.00146778 |
| `layer5.2` | temporal_low_rank | 512 | 0.00000014 | 0.49881566 | 0.00205149 |
| `layer5.6` | temporal_low_rank | 512 | 0.00000031 | 0.52093482 | 0.00254970 |
| `layer5.10` | temporal_low_rank | 512 | 0.00002115 | 0.45828554 | 0.02143718 |
| `classifier.2` | temporal_low_rank | 4096 | 0.00003767 | 0.26606411 | 0.04676090 |
| `classifier.5` | temporal_low_rank | 4096 | 0.00007562 | 0.36481640 | 0.06859276 |
