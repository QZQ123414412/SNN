# QCFS + Full-FTBC + Accuracy-Gated SNM CIFAR-10 Ablation

- Status: complete
- Dataset: CIFAR-10
- Architecture: resnet20
- Checkpoint: `resnet20_L[8]_bs128_fixed_repo_seed42_testbest.pth`
- Checkpoint SHA256: `eb8301ebda8ae91e52f2f273306befa5d349931c05b829a9440dafa05df70631`
- ANN accuracy on the 200-image test set: 93.00%
- Time steps: [1, 2]
- Full-FTBC fit: 5 x 200, alpha=0.4
- A-SNM validation: 5 x 200
- Fit batch SHA256: `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df`
- Validation batch SHA256: `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-10 normalization, and Cutout(1,16).
- The test loader uses only ToTensor and normalization, with shuffle=False.
- Every SNN uses QCFS L=8, rate coding, rate schedule, ratio=1.0, R0=True, FP32.
- ResNet20 evaluation profile: fixed_repo.
- Full-FTBC is independently fitted for every T with SNM off and frozen before validation/test.
- A-SNM independently enables SNM at each T only when SNM-on has strictly higher validation accuracy; ties select off.
- A-SNM changes only the standard SNM on/off state, uses margin=0, and stores one frozen Boolean per evaluated T.
- During ablation, test images are first accessed after both families' A-SNM decisions are frozen.
- Checkpoint-selection note: the checkpoint is selected by the highest accuracy observed on the 10,000-image CIFAR-10 test set during 300 training epochs; this creates model-selection bias
- Checkpoint-interpretation note: the ResNet20 checkpoint is trained and evaluated with QCFS L=8

## Primary accuracy table

| Config | T=1 | T=2 | SNM-on T |
|---|---:|---:|---|
| A_QCFS_R0 | 42.50% | 51.50% | none |
| B_QCFS_STANDARD_SNM_R0 | 42.50% | 54.00% | 1, 2 |
| C_QCFS_ASNM_R0 | 42.50% | 54.00% | 2 |
| D_QCFS_FULL_FTBC_R0 | 63.50% | 76.50% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 63.50% | 74.50% | 1, 2 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 63.50% | 76.50% | none |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 47.00% |
| B_QCFS_STANDARD_SNM_R0 | 48.25% |
| C_QCFS_ASNM_R0 | 48.25% |
| D_QCFS_FULL_FTBC_R0 | 70.00% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 69.00% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 70.00% |

## Accuracy comparisons

| Comparison | T=1 | T=2 | Mean |
|---|---:|---:|---:|
| C-A | +0.00pp | +2.50pp | +1.25pp |
| C-B | +0.00pp | +0.00pp | +0.00pp |
| F-D | +0.00pp | +0.00pp | +0.00pp |
| F-E | +0.00pp | +2.00pp | +1.00pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 17.718693 | 11.902816 |
| B_QCFS_STANDARD_SNM_R0 | 17.718693 | 11.857477 |
| C_QCFS_ASNM_R0 | 17.718693 | 11.857477 |
| D_QCFS_FULL_FTBC_R0 | 8.446140 | 6.043681 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 8.446140 | 6.007808 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 8.446140 | 6.043681 |

## Positive spike rate

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 20.2870% | 20.1029% |
| B_QCFS_STANDARD_SNM_R0 | 20.2870% | 20.1345% |
| C_QCFS_ASNM_R0 | 20.2870% | 20.1345% |
| D_QCFS_FULL_FTBC_R0 | 19.6750% | 18.9734% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 19.6750% | 18.9985% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 19.6750% | 18.9734% |

## Negative spike rate

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0622% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0622% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0414% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0000% |

## Overall spike sparsity

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 79.7130% | 79.8971% |
| B_QCFS_STANDARD_SNM_R0 | 79.7130% | 79.8032% |
| C_QCFS_ASNM_R0 | 79.7130% | 79.8032% |
| D_QCFS_FULL_FTBC_R0 | 80.3250% | 81.0266% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 80.3250% | 80.9601% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 80.3250% | 81.0266% |

## Input-driven SOPs

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 1,823,380,480 | 3,593,206,344 |
| B_QCFS_STANDARD_SNM_R0 | 1,823,380,480 | 3,608,346,302 |
| C_QCFS_ASNM_R0 | 1,823,380,480 | 3,608,346,302 |
| D_QCFS_FULL_FTBC_R0 | 1,725,998,192 | 3,353,926,204 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1,725,998,192 | 3,365,009,554 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1,725,998,192 | 3,353,926,204 |

## Full-FTBC parameters

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 688 | 1,376 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 688 | 1,376 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 688 | 1,376 |

## Full-FTBC storage bytes

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 2,752 | 5,504 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2,752 | 5,504 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2,752 | 5,504 |

## Full-FTBC calibration elapsed

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.969s | 1.301s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.969s | 1.301s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.969s | 1.301s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0.009s | 0.012s |
| B_QCFS_STANDARD_SNM_R0 | 0.010s | 0.016s |
| C_QCFS_ASNM_R0 | 0.009s | 0.016s |
| D_QCFS_FULL_FTBC_R0 | 0.009s | 0.012s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.013s | 0.013s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.009s | 0.012s |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 2 | 0.583s |
| Full-FTBC | none | 0.334s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 27.4000% | 27.4000% | +0.0000pp | off |
| 2 | 35.9000% | 36.1000% | +0.2000pp | on |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 40.8000% | 40.8000% | +0.0000pp | off |
| 2 | 54.4000% | 53.8000% | -0.6000pp | off |

## Validation-gate versus test-oracle diagnostic

This table is post-hoc only. Test accuracy never changes the frozen A-SNM gate.

| Family | T | Validation selection | Test off | Test on | Test oracle | A-SNM oracle gap |
|---|---:|---|---:|---:|---|---:|
| qcfs | 1 | off | 42.50% | 42.50% | off | +0.00pp |
| qcfs | 2 | on | 51.50% | 54.00% | on | +0.00pp |
| full | 1 | off | 63.50% | 63.50% | off | +0.00pp |
| full | 2 | off | 76.50% | 74.50% | off | +0.00pp |

## Deployment equivalence checks

| Config | T | Expected source | Exact cached result |
|---|---:|---|---|
| C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2 | D_QCFS_FULL_FTBC_R0 | yes |
