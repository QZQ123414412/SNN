# QCFS + Full-FTBC + Accuracy-Gated SNM CIFAR-100 Ablation

- Status: complete
- Architecture: resnet20
- Checkpoint: `resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth`
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- ANN accuracy on the 200-image test set: 68.00%
- Time steps: [1]
- Full-FTBC fit: 1 x 200, alpha=0.4
- A-SNM validation: 1 x 200
- Fit batch SHA256: `42e35ed3bdcda2e94471199d0ce318fef1b60aa43ec493ec52631af8e5b10049`
- Validation batch SHA256: `ed8a6c033924c980bc943bf1c48e1fff63f1a04baf4e661dd89ae16f9f52742e`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-100 normalization, and Cutout(1,16).
- The test loader uses only ToTensor and normalization, with shuffle=False.
- Every SNN uses QCFS L=8, rate coding, rate schedule, ratio=1.0, R0=True, FP32.
- Full-FTBC is independently fitted for every T with SNM off and frozen before validation/test.
- A-SNM independently enables SNM at each T only when SNM-on has strictly higher validation accuracy; ties select off.
- A-SNM changes only the standard SNM on/off state, uses margin=0, and stores one frozen Boolean per evaluated T.
- Test images are first accessed after both families' A-SNM decisions are frozen.

## Primary accuracy table

| Config | T=1 | SNM-on T |
|---|---:|---:|
| A_QCFS_R0 | 5.50% | none |
| B_QCFS_STANDARD_SNM_R0 | 5.50% | 1 |
| C_QCFS_ASNM_R0 | 5.50% | none |
| D_QCFS_FULL_FTBC_R0 | 12.00% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12.00% | 1 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12.00% | none |

## ANN-SNN logit MSE

| Config | T=1 |
|---|---:|
| A_QCFS_R0 | 16.362998 |
| B_QCFS_STANDARD_SNM_R0 | 16.362998 |
| C_QCFS_ASNM_R0 | 16.362998 |
| D_QCFS_FULL_FTBC_R0 | 12.697731 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12.697731 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12.697731 |

## Positive spike rate

| Config | T=1 |
|---|---:|
| A_QCFS_R0 | 21.8945% |
| B_QCFS_STANDARD_SNM_R0 | 21.8945% |
| C_QCFS_ASNM_R0 | 21.8945% |
| D_QCFS_FULL_FTBC_R0 | 22.0360% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 22.0360% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 22.0360% |

## Negative spike rate

| Config | T=1 |
|---|---:|
| A_QCFS_R0 | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% |
| C_QCFS_ASNM_R0 | 0.0000% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% |

## Overall spike sparsity

| Config | T=1 |
|---|---:|
| A_QCFS_R0 | 78.1055% |
| B_QCFS_STANDARD_SNM_R0 | 78.1055% |
| C_QCFS_ASNM_R0 | 78.1055% |
| D_QCFS_FULL_FTBC_R0 | 77.9640% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 77.9640% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 77.9640% |

## Input-driven SOPs

| Config | T=1 |
|---|---:|
| A_QCFS_R0 | 2,132,789,816 |
| B_QCFS_STANDARD_SNM_R0 | 2,132,789,816 |
| C_QCFS_ASNM_R0 | 2,132,789,816 |
| D_QCFS_FULL_FTBC_R0 | 2,076,334,488 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2,076,334,488 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2,076,334,488 |

## Full-FTBC parameters

| Config | T=1 |
|---|---:|
| A_QCFS_R0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 |
| C_QCFS_ASNM_R0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 688 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 688 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 688 |

## Full-FTBC storage bytes

| Config | T=1 |
|---|---:|
| A_QCFS_R0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 |
| C_QCFS_ASNM_R0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 2,752 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2,752 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2,752 |

## Full-FTBC calibration elapsed

| Config | T=1 |
|---|---:|
| A_QCFS_R0 | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.184s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.184s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.184s |

## Inference elapsed (statistics disabled)

| Config | T=1 |
|---|---:|
| A_QCFS_R0 | 0.009s |
| B_QCFS_STANDARD_SNM_R0 | 0.010s |
| C_QCFS_ASNM_R0 | 0.009s |
| D_QCFS_FULL_FTBC_R0 | 0.008s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.011s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.008s |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | none | 0.271s |
| Full-FTBC | none | 0.026s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 2.5000% | 2.5000% | +0.0000pp | off |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 8.5000% | 8.5000% | +0.0000pp | off |

## Deployment equivalence checks

| Config | T | Expected source | Exact cached result |
|---|---:|---|---|
| C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
