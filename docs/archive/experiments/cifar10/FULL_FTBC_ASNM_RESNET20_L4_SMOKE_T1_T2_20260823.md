# QCFS + Full-FTBC + Accuracy-Gated SNM CIFAR-10 Ablation

- Status: complete
- Dataset: CIFAR-10
- Architecture: resnet20
- Checkpoint: `resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth`
- Checkpoint SHA256: `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3`
- ANN accuracy on the 200-image test set: 89.50%
- Time steps: [1, 2]
- Full-FTBC fit: 5 x 200, alpha=0.4
- A-SNM validation: 5 x 200
- Fit batch SHA256: `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df`
- Validation batch SHA256: `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-10 normalization, and Cutout(1,16).
- The test loader uses only ToTensor and normalization, with shuffle=False.
- Every SNN uses QCFS L=4, rate coding, rate schedule, ratio=1.0, R0=True, FP32.
- ResNet20 evaluation profile: paper_era.
- Full-FTBC is independently fitted for every T with SNM off and frozen before validation/test.
- A-SNM independently enables SNM at each T only when SNM-on has strictly higher validation accuracy; ties select off.
- A-SNM changes only the standard SNM on/off state, uses margin=0, and stores one frozen Boolean per evaluated T.
- During ablation, test images are first accessed after both families' A-SNM decisions are frozen.
- Checkpoint-selection note: the checkpoint is selected by the highest accuracy observed on the 10,000-image CIFAR-10 test set during 300 training epochs; this creates model-selection bias
- Checkpoint-interpretation note: the ResNet20 checkpoint is evaluated with QCFS L=4 and the paper_era profile; the checkpoint training provenance is recorded separately

## Primary accuracy table

| Config | T=1 | T=2 | SNM-on T |
|---|---:|---:|---|
| A_QCFS_R0 | 64.00% | 72.00% | none |
| B_QCFS_STANDARD_SNM_R0 | 64.00% | 75.50% | 1, 2 |
| C_QCFS_ASNM_R0 | 64.00% | 75.50% | 2 |
| D_QCFS_FULL_FTBC_R0 | 67.00% | 79.50% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 67.00% | 79.50% | 1, 2 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 67.00% | 79.50% | 2 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 68.00% |
| B_QCFS_STANDARD_SNM_R0 | 69.75% |
| C_QCFS_ASNM_R0 | 69.75% |
| D_QCFS_FULL_FTBC_R0 | 73.25% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 73.25% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 73.25% |

## Accuracy comparisons

| Comparison | T=1 | T=2 | Mean |
|---|---:|---:|---:|
| C-A | +0.00pp | +3.50pp | +1.75pp |
| C-B | +0.00pp | +0.00pp | +0.00pp |
| F-D | +0.00pp | +0.00pp | +0.00pp |
| F-E | +0.00pp | +0.00pp | +0.00pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 6.588621 | 4.579474 |
| B_QCFS_STANDARD_SNM_R0 | 6.588621 | 4.436104 |
| C_QCFS_ASNM_R0 | 6.588621 | 4.436104 |
| D_QCFS_FULL_FTBC_R0 | 6.131937 | 4.063236 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 6.131937 | 4.035259 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 6.131937 | 4.035259 |

## Positive spike rate

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 28.3813% | 28.5974% |
| B_QCFS_STANDARD_SNM_R0 | 28.3813% | 28.5881% |
| C_QCFS_ASNM_R0 | 28.3813% | 28.5881% |
| D_QCFS_FULL_FTBC_R0 | 27.9464% | 27.3270% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 27.9464% | 27.3321% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 27.9464% | 27.3321% |

## Negative spike rate

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0760% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0760% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0658% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0658% |

## Overall spike sparsity

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 71.6187% | 71.4026% |
| B_QCFS_STANDARD_SNM_R0 | 71.6187% | 71.3359% |
| C_QCFS_ASNM_R0 | 71.6187% | 71.3359% |
| D_QCFS_FULL_FTBC_R0 | 72.0536% | 72.6730% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 72.0536% | 72.6020% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 72.0536% | 72.6020% |

## Input-driven SOPs

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 2,648,469,842 | 5,317,477,318 |
| B_QCFS_STANDARD_SNM_R0 | 2,648,469,842 | 5,325,792,158 |
| C_QCFS_ASNM_R0 | 2,648,469,842 | 5,325,792,158 |
| D_QCFS_FULL_FTBC_R0 | 2,571,089,884 | 5,055,989,070 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2,571,089,884 | 5,065,738,060 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2,571,089,884 | 5,065,738,060 |

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
| D_QCFS_FULL_FTBC_R0 | 1.157s | 1.590s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.157s | 1.590s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.157s | 1.590s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 |
|---|---:|---:|
| A_QCFS_R0 | 0.009s | 0.023s |
| B_QCFS_STANDARD_SNM_R0 | 0.009s | 0.014s |
| C_QCFS_ASNM_R0 | 0.009s | 0.014s |
| D_QCFS_FULL_FTBC_R0 | 0.008s | 0.015s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.011s | 0.025s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.008s | 0.025s |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 2 | 0.611s |
| Full-FTBC | 2 | 0.403s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 37.7000% | 37.7000% | +0.0000pp | off |
| 2 | 45.8000% | 46.6000% | +0.8000pp | on |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 47.6000% | 47.6000% | +0.0000pp | off |
| 2 | 59.6000% | 60.3000% | +0.7000pp | on |

## Validation-gate versus test-oracle diagnostic

This table is post-hoc only. Test accuracy never changes the frozen A-SNM gate.

| Family | T | Validation selection | Test off | Test on | Test oracle | A-SNM oracle gap |
|---|---:|---|---:|---:|---|---:|
| qcfs | 1 | off | 64.00% | 64.00% | off | +0.00pp |
| qcfs | 2 | on | 72.00% | 75.50% | on | +0.00pp |
| full | 1 | off | 67.00% | 67.00% | off | +0.00pp |
| full | 2 | on | 79.50% | 79.50% | off | +0.00pp |

## Deployment equivalence checks

| Config | T | Expected source | Exact cached result |
|---|---:|---|---|
| C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
