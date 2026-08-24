# QCFS + Full-FTBC + Accuracy-Gated SNM CIFAR-10 Ablation

- Status: complete
- Dataset: CIFAR-10
- Architecture: vgg16
- Checkpoint: `cifar10-vgg16-example.pth`
- Checkpoint SHA256: `093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84`
- ANN accuracy on the 10,000-image test set: 95.51%
- Time steps: [1, 2, 4, 8, 16, 32]
- Full-FTBC fit: 5 x 200, alpha=0.4
- A-SNM validation: 5 x 200
- Fit batch SHA256: `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df`
- Validation batch SHA256: `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-10 normalization, and Cutout(1,16).
- The test loader uses only ToTensor and normalization, with shuffle=False.
- Every SNN uses QCFS L=8, rate coding, rate schedule, ratio=1.0, R0=True, FP32.
- ResNet20 evaluation profile: not-applicable.
- Full-FTBC is independently fitted for every T with SNM off and frozen before validation/test.
- A-SNM independently enables SNM at each T only when SNM-on has strictly higher validation accuracy; ties select off.
- A-SNM changes only the standard SNM on/off state, uses margin=0, and stores one frozen Boolean per evaluated T.
- During ablation, test images are first accessed after both families' A-SNM decisions are frozen.
- Checkpoint-selection note: the existing VGG16 checkpoint selection procedure is not recorded
- Checkpoint-interpretation note: the legacy VGG16 checkpoint was probably trained with L=4 and is evaluated post-hoc with L=8; it is not an L=8-trained model

## Primary accuracy table

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | SNM-on T |
|---|---:|---:|---:|---:|---:|---:|---|
| A_QCFS_R0 | 88.26% | 91.06% | 93.81% | 95.00% | 95.42% | 95.48% | none |
| B_QCFS_STANDARD_SNM_R0 | 88.26% | 91.11% | 94.11% | 95.28% | 95.56% | 95.58% | 1, 2, 4, 8, 16, 32 |
| C_QCFS_ASNM_R0 | 88.26% | 91.11% | 94.11% | 95.00% | 95.56% | 95.48% | 2, 4, 16 |
| D_QCFS_FULL_FTBC_R0 | 89.87% | 91.98% | 94.27% | 95.24% | 95.51% | 95.47% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 89.87% | 92.05% | 94.54% | 95.48% | 95.51% | 95.53% | 1, 2, 4, 8, 16, 32 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 89.87% | 92.05% | 94.54% | 95.48% | 95.51% | 95.47% | 2, 4, 8 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 93.17% |
| B_QCFS_STANDARD_SNM_R0 | 93.32% |
| C_QCFS_ASNM_R0 | 93.25% |
| D_QCFS_FULL_FTBC_R0 | 93.72% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 93.83% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 93.82% |

## Accuracy comparisons

| Comparison | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| C-A | +0.00pp | +0.05pp | +0.30pp | +0.00pp | +0.14pp | +0.00pp | +0.08pp |
| C-B | +0.00pp | +0.00pp | +0.00pp | -0.28pp | +0.00pp | -0.10pp | -0.06pp |
| F-D | +0.00pp | +0.07pp | +0.27pp | +0.24pp | +0.00pp | +0.00pp | +0.10pp |
| F-E | +0.00pp | +0.00pp | +0.00pp | +0.00pp | +0.00pp | -0.06pp | -0.01pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 2.246263 | 1.111278 | 0.360277 | 0.096545 | 0.036447 | 0.020578 |
| B_QCFS_STANDARD_SNM_R0 | 2.246263 | 1.098213 | 0.319911 | 0.056658 | 0.020730 | 0.016335 |
| C_QCFS_ASNM_R0 | 2.246263 | 1.098213 | 0.319911 | 0.096545 | 0.020730 | 0.020578 |
| D_QCFS_FULL_FTBC_R0 | 2.630639 | 1.139098 | 0.313663 | 0.080230 | 0.031981 | 0.019420 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2.630639 | 1.135152 | 0.284983 | 0.050940 | 0.020611 | 0.016584 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2.630639 | 1.135152 | 0.284983 | 0.050940 | 0.031981 | 0.019420 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 12.2864% | 12.5910% | 12.5868% | 12.5415% | 12.5040% | 12.4825% |
| B_QCFS_STANDARD_SNM_R0 | 12.2864% | 12.6074% | 12.6206% | 12.5753% | 12.5292% | 12.5036% |
| C_QCFS_ASNM_R0 | 12.2864% | 12.6074% | 12.6206% | 12.5415% | 12.5292% | 12.4825% |
| D_QCFS_FULL_FTBC_R0 | 12.6000% | 12.5567% | 12.5235% | 12.4498% | 12.4584% | 12.4491% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12.6000% | 12.5659% | 12.5496% | 12.4807% | 12.4837% | 12.4704% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12.6000% | 12.5659% | 12.5496% | 12.4807% | 12.4584% | 12.4491% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0188% | 0.0433% | 0.0517% | 0.0440% | 0.0349% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0188% | 0.0433% | 0.0000% | 0.0440% | 0.0000% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0129% | 0.0360% | 0.0464% | 0.0407% | 0.0327% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0129% | 0.0360% | 0.0464% | 0.0000% | 0.0000% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 87.7136% | 87.4090% | 87.4132% | 87.4585% | 87.4960% | 87.5175% |
| B_QCFS_STANDARD_SNM_R0 | 87.7136% | 87.3738% | 87.3361% | 87.3730% | 87.4268% | 87.4614% |
| C_QCFS_ASNM_R0 | 87.7136% | 87.3738% | 87.3361% | 87.4585% | 87.4268% | 87.5175% |
| D_QCFS_FULL_FTBC_R0 | 87.4000% | 87.4433% | 87.4765% | 87.5502% | 87.5416% | 87.5509% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 87.4000% | 87.4212% | 87.4144% | 87.4729% | 87.4756% | 87.4968% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 87.4000% | 87.4212% | 87.4144% | 87.4729% | 87.5416% | 87.5509% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 580,815,093,184 | 1,215,986,399,872 | 2,447,218,210,752 | 4,878,577,169,280 | 9,727,441,706,688 | 19,421,506,591,232 |
| B_QCFS_STANDARD_SNM_R0 | 580,815,093,184 | 1,219,063,444,096 | 2,467,137,795,648 | 4,927,484,989,312 | 9,803,314,477,888 | 19,534,760,398,464 |
| C_QCFS_ASNM_R0 | 580,815,093,184 | 1,219,063,444,096 | 2,467,137,795,648 | 4,878,577,169,280 | 9,803,314,477,888 | 19,421,506,591,232 |
| D_QCFS_FULL_FTBC_R0 | 601,143,818,688 | 1,218,071,939,456 | 2,424,122,737,344 | 4,836,589,713,856 | 9,684,943,809,024 | 19,369,458,039,168 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,880,159,407,680 | 9,756,058,592,000 | 19,477,757,894,656 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,880,159,407,680 | 9,684,943,809,024 | 19,369,458,039,168 |

## Full-FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |

## Full-FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |

## Full-FTBC calibration elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 1.651s | 2.752s | 4.836s | 8.866s | 16.802s | 71.426s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.651s | 2.752s | 4.836s | 8.866s | 16.802s | 71.426s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.651s | 2.752s | 4.836s | 8.866s | 16.802s | 71.426s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.428s | 2.138s | 3.471s | 6.124s | 11.288s | 21.766s |
| B_QCFS_STANDARD_SNM_R0 | 1.485s | 2.234s | 3.708s | 6.604s | 12.246s | 23.733s |
| C_QCFS_ASNM_R0 | 1.428s | 2.234s | 3.708s | 6.124s | 12.246s | 21.766s |
| D_QCFS_FULL_FTBC_R0 | 1.380s | 2.078s | 3.349s | 5.903s | 10.982s | 20.856s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.417s | 2.172s | 3.665s | 6.388s | 11.936s | 22.768s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.380s | 2.172s | 3.665s | 6.388s | 10.982s | 20.856s |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 2, 4, 16 | 11.729s |
| Full-FTBC | 2, 4, 8 | 31.491s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 77.5000% | 77.5000% | +0.0000pp | off |
| 2 | 82.5000% | 82.7000% | +0.2000pp | on |
| 4 | 89.4000% | 90.4000% | +1.0000pp | on |
| 8 | 92.8000% | 92.8000% | +0.0000pp | off |
| 16 | 92.8000% | 93.1000% | +0.3000pp | on |
| 32 | 92.9000% | 92.6000% | -0.3000pp | off |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 80.3000% | 80.3000% | +0.0000pp | off |
| 2 | 83.7000% | 83.9000% | +0.2000pp | on |
| 4 | 90.6000% | 90.8000% | +0.2000pp | on |
| 8 | 92.6000% | 92.8000% | +0.2000pp | on |
| 16 | 92.6000% | 92.6000% | +0.0000pp | off |
| 32 | 92.7000% | 92.5000% | -0.2000pp | off |

## Validation-gate versus test-oracle diagnostic

This table is post-hoc only. Test accuracy never changes the frozen A-SNM gate.

| Family | T | Validation selection | Test off | Test on | Test oracle | A-SNM oracle gap |
|---|---:|---|---:|---:|---|---:|
| qcfs | 1 | off | 88.26% | 88.26% | off | +0.00pp |
| qcfs | 2 | on | 91.06% | 91.11% | on | +0.00pp |
| qcfs | 4 | on | 93.81% | 94.11% | on | +0.00pp |
| qcfs | 8 | off | 95.00% | 95.28% | on | -0.28pp |
| qcfs | 16 | on | 95.42% | 95.56% | on | +0.00pp |
| qcfs | 32 | off | 95.48% | 95.58% | on | -0.10pp |
| full | 1 | off | 89.87% | 89.87% | off | +0.00pp |
| full | 2 | on | 91.98% | 92.05% | on | +0.00pp |
| full | 4 | on | 94.27% | 94.54% | on | +0.00pp |
| full | 8 | on | 95.24% | 95.48% | on | +0.00pp |
| full | 16 | off | 95.51% | 95.51% | off | +0.00pp |
| full | 32 | off | 95.47% | 95.53% | on | -0.06pp |

## Deployment equivalence checks

| Config | T | Expected source | Exact cached result |
|---|---:|---|---|
| C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 4 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 8 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 8 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 16 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 32 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 32 | D_QCFS_FULL_FTBC_R0 | yes |
