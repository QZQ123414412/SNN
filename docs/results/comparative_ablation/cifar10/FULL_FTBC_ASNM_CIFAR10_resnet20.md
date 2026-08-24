# QCFS + Full-FTBC + Accuracy-Gated SNM CIFAR-10 Ablation

- Status: complete
- Dataset: CIFAR-10
- Architecture: resnet20
- Checkpoint: `resnet20_L[8]_bs128_fixed_repo_seed42_testbest.pth`
- Checkpoint SHA256: `eb8301ebda8ae91e52f2f273306befa5d349931c05b829a9440dafa05df70631`
- ANN accuracy on the 10,000-image test set: 92.79%
- Time steps: [1, 2, 4, 8, 16, 32]
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

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | SNM-on T |
|---|---:|---:|---:|---:|---:|---:|---|
| A_QCFS_R0 | 40.96% | 53.49% | 73.11% | 88.50% | 92.32% | 92.92% | none |
| B_QCFS_STANDARD_SNM_R0 | 40.96% | 54.30% | 76.66% | 90.82% | 92.57% | 92.90% | 1, 2, 4, 8, 16, 32 |
| C_QCFS_ASNM_R0 | 40.96% | 54.30% | 76.66% | 90.82% | 92.57% | 92.92% | 2, 4, 8, 16 |
| D_QCFS_FULL_FTBC_R0 | 58.84% | 70.87% | 83.59% | 90.39% | 92.50% | 92.93% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 58.84% | 71.17% | 84.90% | 91.38% | 92.76% | 93.09% | 1, 2, 4, 8, 16, 32 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 58.84% | 70.87% | 84.90% | 90.39% | 92.50% | 93.09% | 4, 32 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 73.55% |
| B_QCFS_STANDARD_SNM_R0 | 74.70% |
| C_QCFS_ASNM_R0 | 74.70% |
| D_QCFS_FULL_FTBC_R0 | 81.52% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 82.02% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 81.77% |

## Accuracy comparisons

| Comparison | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| C-A | +0.00pp | +0.81pp | +3.55pp | +2.32pp | +0.25pp | +0.00pp | +1.15pp |
| C-B | +0.00pp | +0.00pp | +0.00pp | +0.00pp | +0.00pp | +0.02pp | +0.00pp |
| F-D | +0.00pp | +0.00pp | +1.31pp | +0.00pp | +0.00pp | +0.16pp | +0.24pp |
| F-E | +0.00pp | -0.30pp | +0.00pp | -0.99pp | -0.26pp | +0.00pp | -0.26pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 17.737108 | 11.954760 | 6.325454 | 2.525087 | 0.847078 | 0.317739 |
| B_QCFS_STANDARD_SNM_R0 | 17.737108 | 11.819607 | 5.449061 | 1.360612 | 0.407052 | 0.189817 |
| C_QCFS_ASNM_R0 | 17.737108 | 11.819607 | 5.449061 | 1.360612 | 0.407052 | 0.317739 |
| D_QCFS_FULL_FTBC_R0 | 9.285819 | 6.511190 | 3.838482 | 1.674092 | 0.617189 | 0.259502 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 9.285819 | 6.452257 | 3.477459 | 1.231475 | 0.430131 | 0.200429 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 9.285819 | 6.511190 | 3.477459 | 1.674092 | 0.617189 | 0.200429 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 20.2108% | 20.0772% | 19.4935% | 19.1379% | 18.9639% | 18.8762% |
| B_QCFS_STANDARD_SNM_R0 | 20.2108% | 20.1091% | 19.6460% | 19.4073% | 19.2558% | 19.1544% |
| C_QCFS_ASNM_R0 | 20.2108% | 20.1091% | 19.6460% | 19.4073% | 19.2558% | 18.8762% |
| D_QCFS_FULL_FTBC_R0 | 19.6161% | 18.9619% | 18.5896% | 18.6434% | 18.6756% | 18.7328% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 19.6161% | 18.9896% | 18.7157% | 18.8670% | 18.9405% | 19.0044% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 19.6161% | 18.9619% | 18.7157% | 18.6434% | 18.6756% | 19.0044% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0619% | 0.1773% | 0.2983% | 0.3350% | 0.3155% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0619% | 0.1773% | 0.2983% | 0.3350% | 0.0000% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0418% | 0.1424% | 0.2552% | 0.3029% | 0.2999% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0000% | 0.1424% | 0.0000% | 0.0000% | 0.2999% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 79.7892% | 79.9228% | 80.5065% | 80.8621% | 81.0361% | 81.1238% |
| B_QCFS_STANDARD_SNM_R0 | 79.7892% | 79.8290% | 80.1767% | 80.2944% | 80.4091% | 80.5300% |
| C_QCFS_ASNM_R0 | 79.7892% | 79.8290% | 80.1767% | 80.2944% | 80.4091% | 81.1238% |
| D_QCFS_FULL_FTBC_R0 | 80.3839% | 81.0381% | 81.4104% | 81.3566% | 81.3244% | 81.2672% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 80.3839% | 80.9686% | 81.1418% | 80.8778% | 80.7566% | 80.6957% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 80.3839% | 81.0381% | 81.1418% | 81.3566% | 81.3244% | 80.6957% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 90,874,950,414 | 179,425,115,648 | 347,902,231,480 | 682,038,635,110 | 1,349,940,154,412 | 2,685,574,383,668 |
| B_QCFS_STANDARD_SNM_R0 | 90,874,950,414 | 180,224,012,988 | 355,062,017,810 | 708,022,347,248 | 1,406,117,806,962 | 2,788,914,626,766 |
| C_QCFS_ASNM_R0 | 90,874,950,414 | 180,224,012,988 | 355,062,017,810 | 708,022,347,248 | 1,406,117,806,962 | 2,685,574,383,668 |
| D_QCFS_FULL_FTBC_R0 | 86,145,787,408 | 167,632,966,032 | 329,069,910,668 | 660,431,743,250 | 1,326,056,317,012 | 2,662,594,534,172 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 86,145,787,408 | 168,218,298,884 | 334,745,959,386 | 681,790,843,212 | 1,375,890,519,018 | 2,760,612,821,674 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 86,145,787,408 | 167,632,966,032 | 334,745,959,386 | 660,431,743,250 | 1,326,056,317,012 | 2,760,612,821,674 |

## Full-FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |

## Full-FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |

## Full-FTBC calibration elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.984s | 1.296s | 2.330s | 4.551s | 8.965s | 18.409s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.984s | 1.296s | 2.330s | 4.551s | 8.965s | 18.409s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.984s | 1.296s | 2.330s | 4.551s | 8.965s | 18.409s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.163s | 1.333s | 1.901s | 3.123s | 5.494s | 10.284s |
| B_QCFS_STANDARD_SNM_R0 | 1.220s | 1.459s | 2.085s | 3.334s | 6.137s | 11.526s |
| C_QCFS_ASNM_R0 | 1.163s | 1.459s | 2.085s | 3.334s | 6.137s | 10.284s |
| D_QCFS_FULL_FTBC_R0 | 1.156s | 1.318s | 1.873s | 3.005s | 5.399s | 9.940s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.224s | 1.530s | 2.041s | 3.283s | 5.792s | 11.271s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.156s | 1.318s | 2.041s | 3.005s | 5.399s | 11.271s |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 2, 4, 8, 16 | 5.725s |
| Full-FTBC | 4, 32 | 5.680s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 27.4000% | 27.4000% | +0.0000pp | off |
| 2 | 35.9000% | 36.1000% | +0.2000pp | on |
| 4 | 50.6000% | 52.5000% | +1.9000pp | on |
| 8 | 68.4000% | 75.6000% | +7.2000pp | on |
| 16 | 80.3000% | 83.5000% | +3.2000pp | on |
| 32 | 85.4000% | 85.3000% | -0.1000pp | off |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 40.8000% | 40.8000% | +0.0000pp | off |
| 2 | 54.4000% | 53.8000% | -0.6000pp | off |
| 4 | 70.2000% | 70.8000% | +0.6000pp | on |
| 8 | 81.2000% | 80.4000% | -0.8000pp | off |
| 16 | 84.8000% | 84.8000% | +0.0000pp | off |
| 32 | 85.7000% | 85.8000% | +0.1000pp | on |

## Validation-gate versus test-oracle diagnostic

This table is post-hoc only. Test accuracy never changes the frozen A-SNM gate.

| Family | T | Validation selection | Test off | Test on | Test oracle | A-SNM oracle gap |
|---|---:|---|---:|---:|---|---:|
| qcfs | 1 | off | 40.96% | 40.96% | off | +0.00pp |
| qcfs | 2 | on | 53.49% | 54.30% | on | +0.00pp |
| qcfs | 4 | on | 73.11% | 76.66% | on | +0.00pp |
| qcfs | 8 | on | 88.50% | 90.82% | on | +0.00pp |
| qcfs | 16 | on | 92.32% | 92.57% | on | +0.00pp |
| qcfs | 32 | off | 92.92% | 92.90% | off | +0.00pp |
| full | 1 | off | 58.84% | 58.84% | off | +0.00pp |
| full | 2 | off | 70.87% | 71.17% | on | -0.30pp |
| full | 4 | on | 83.59% | 84.90% | on | +0.00pp |
| full | 8 | off | 90.39% | 91.38% | on | -0.99pp |
| full | 16 | off | 92.50% | 92.76% | on | -0.26pp |
| full | 32 | on | 92.93% | 93.09% | on | +0.00pp |

## Deployment equivalence checks

| Config | T | Expected source | Exact cached result |
|---|---:|---|---|
| C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 4 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 8 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 8 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 16 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 32 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
