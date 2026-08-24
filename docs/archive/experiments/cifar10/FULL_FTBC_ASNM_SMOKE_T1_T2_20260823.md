# QCFS + Full-FTBC + Accuracy-Gated SNM CIFAR-10 Ablation

- Status: complete
- Dataset: CIFAR-10
- Architecture: vgg16
- Checkpoint: `cifar10-vgg16-example.pth`
- Checkpoint SHA256: `093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84`
- ANN accuracy on the 200-image test set: 95.50%
- Time steps: [1, 2]
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

| Config | T=1 | T=2 | SNM-on T |
|---|---:|---:|---|
| A_QCFS_R0 | 90.50% | 92.50% | none |
| B_QCFS_STANDARD_SNM_R0 | 90.50% | 92.00% | 1, 2 |
| C_QCFS_ASNM_R0 | 90.50% | 92.00% | 2 |
| D_QCFS_FULL_FTBC_R0 | 91.50% | 93.00% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 91.50% | 93.00% | 1, 2 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 91.50% | 93.00% | 2 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 91.50% |
| B_QCFS_STANDARD_SNM_R0 | 91.25% |
| C_QCFS_ASNM_R0 | 91.25% |
| D_QCFS_FULL_FTBC_R0 | 92.25% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 92.25% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 92.25% |

## Accuracy comparisons

| Comparison | T=1 | T=2 | Mean |
|---|---:|---:|---:|
| C-A | +0.00pp | -0.50pp | -0.25pp |
| C-B | +0.00pp | +0.00pp | +0.00pp |
| F-D | +0.00pp | +0.00pp | +0.00pp |
| F-E | +0.00pp | +0.00pp | +0.00pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 2.206230 | 1.014212 |
| B_QCFS_STANDARD_SNM_R0 | 2.206230 | 0.994570 |
| C_QCFS_ASNM_R0 | 2.206230 | 0.994570 |
| D_QCFS_FULL_FTBC_R0 | 2.389502 | 1.041626 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2.389502 | 1.051107 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2.389502 | 1.051107 |

## Positive spike rate

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 12.3045% | 12.6130% |
| B_QCFS_STANDARD_SNM_R0 | 12.3045% | 12.6305% |
| C_QCFS_ASNM_R0 | 12.3045% | 12.6305% |
| D_QCFS_FULL_FTBC_R0 | 12.6154% | 12.5730% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12.6154% | 12.5796% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12.6154% | 12.5796% |

## Negative spike rate

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0184% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0184% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0130% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0130% |

## Overall spike sparsity

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 87.6955% | 87.3870% |
| B_QCFS_STANDARD_SNM_R0 | 87.6955% | 87.3511% |
| C_QCFS_ASNM_R0 | 87.6955% | 87.3511% |
| D_QCFS_FULL_FTBC_R0 | 87.3846% | 87.4270% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 87.3846% | 87.4074% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 87.3846% | 87.4074% |

## Input-driven SOPs

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 11,673,989,952 | 24,432,348,800 |
| B_QCFS_STANDARD_SNM_R0 | 11,673,989,952 | 24,496,564,864 |
| C_QCFS_ASNM_R0 | 11,673,989,952 | 24,496,564,864 |
| D_QCFS_FULL_FTBC_R0 | 12,070,835,072 | 24,440,854,208 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12,070,835,072 | 24,473,518,912 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12,070,835,072 | 24,473,518,912 |

## Full-FTBC parameters

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 12,416 | 24,832 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12,416 | 24,832 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12,416 | 24,832 |

## Full-FTBC storage bytes

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 49,664 | 99,328 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 49,664 | 99,328 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 49,664 | 99,328 |

## Full-FTBC calibration elapsed

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 1.713s | 2.787s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.713s | 2.787s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.713s | 2.787s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0.018s | 0.030s |
| B_QCFS_STANDARD_SNM_R0 | 0.018s | 0.031s |
| C_QCFS_ASNM_R0 | 0.018s | 0.031s |
| D_QCFS_FULL_FTBC_R0 | 0.016s | 0.029s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.015s | 0.033s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.016s | 0.033s |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 2 | 1.066s |
| Full-FTBC | 2 | 0.726s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 77.5000% | 77.5000% | +0.0000pp | off |
| 2 | 82.5000% | 82.7000% | +0.2000pp | on |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 80.3000% | 80.3000% | +0.0000pp | off |
| 2 | 83.7000% | 83.9000% | +0.2000pp | on |

## Validation-gate versus test-oracle diagnostic

This table is post-hoc only. Test accuracy never changes the frozen A-SNM gate.

| Family | T | Validation selection | Test off | Test on | Test oracle | A-SNM oracle gap |
|---|---:|---|---:|---:|---|---:|
| qcfs | 1 | off | 90.50% | 90.50% | off | +0.00pp |
| qcfs | 2 | on | 92.50% | 92.00% | off | -0.50pp |
| full | 1 | off | 91.50% | 91.50% | off | +0.00pp |
| full | 2 | on | 93.00% | 93.00% | off | +0.00pp |

## Deployment equivalence checks

| Config | T | Expected source | Exact cached result |
|---|---:|---|---|
| C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
