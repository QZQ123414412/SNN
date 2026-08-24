# QCFS + Full-FTBC + Accuracy-Gated SNM CIFAR-10 Ablation

- Status: complete
- Dataset: CIFAR-10
- Architecture: resnet20
- Checkpoint: `resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth`
- Checkpoint SHA256: `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3`
- ANN accuracy on the 10,000-image test set: 90.72%
- Time steps: [1, 2, 4, 8, 16, 32]
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

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | SNM-on T |
|---|---:|---:|---:|---:|---:|---:|---|
| A_QCFS_R0 | 63.89% | 73.29% | 83.38% | 89.82% | 91.32% | 91.69% | none |
| B_QCFS_STANDARD_SNM_R0 | 63.89% | 73.65% | 84.47% | 90.66% | 91.49% | 91.61% | 1, 2, 4, 8, 16, 32 |
| C_QCFS_ASNM_R0 | 63.89% | 73.65% | 84.47% | 90.66% | 91.49% | 91.69% | 2, 4, 8, 16 |
| D_QCFS_FULL_FTBC_R0 | 66.87% | 76.91% | 85.48% | 89.95% | 91.14% | 91.52% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 66.87% | 77.23% | 86.43% | 90.32% | 91.28% | 91.50% | 1, 2, 4, 8, 16, 32 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 66.87% | 77.23% | 86.43% | 90.32% | 91.14% | 91.50% | 2, 4, 8, 32 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 82.23% |
| B_QCFS_STANDARD_SNM_R0 | 82.63% |
| C_QCFS_ASNM_R0 | 82.64% |
| D_QCFS_FULL_FTBC_R0 | 83.64% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 83.94% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 83.92% |

## Accuracy comparisons

| Comparison | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| C-A | +0.00pp | +0.36pp | +1.09pp | +0.84pp | +0.17pp | +0.00pp | +0.41pp |
| C-B | +0.00pp | +0.00pp | +0.00pp | +0.00pp | +0.00pp | +0.08pp | +0.01pp |
| F-D | +0.00pp | +0.32pp | +0.95pp | +0.37pp | +0.00pp | -0.02pp | +0.27pp |
| F-E | +0.00pp | +0.00pp | +0.00pp | +0.00pp | -0.14pp | +0.00pp | -0.02pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 7.000631 | 4.907970 | 2.883020 | 1.246616 | 0.524325 | 0.336480 |
| B_QCFS_STANDARD_SNM_R0 | 7.000631 | 4.830027 | 2.537667 | 0.812023 | 0.377964 | 0.316965 |
| C_QCFS_ASNM_R0 | 7.000631 | 4.830027 | 2.537667 | 0.812023 | 0.377964 | 0.336480 |
| D_QCFS_FULL_FTBC_R0 | 6.037342 | 4.095437 | 2.283818 | 1.014533 | 0.479735 | 0.330866 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 6.037342 | 4.045237 | 2.062640 | 0.784920 | 0.401490 | 0.321341 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 6.037342 | 4.045237 | 2.062640 | 0.784920 | 0.479735 | 0.321341 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 28.3549% | 28.5556% | 28.0161% | 27.6859% | 27.5660% | 27.5040% |
| B_QCFS_STANDARD_SNM_R0 | 28.3549% | 28.5393% | 28.0347% | 27.7835% | 27.7120% | 27.6699% |
| C_QCFS_ASNM_R0 | 28.3549% | 28.5393% | 28.0347% | 27.7835% | 27.7120% | 27.5040% |
| D_QCFS_FULL_FTBC_R0 | 27.9753% | 27.3101% | 27.1507% | 27.2582% | 27.3730% | 27.4461% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 27.9753% | 27.3099% | 27.1896% | 27.3693% | 27.5392% | 27.6315% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 27.9753% | 27.3099% | 27.1896% | 27.3693% | 27.3730% | 27.6315% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0762% | 0.1740% | 0.2678% | 0.2910% | 0.2709% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0762% | 0.1740% | 0.2678% | 0.2910% | 0.0000% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0675% | 0.1695% | 0.2606% | 0.2906% | 0.2772% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0675% | 0.1695% | 0.2606% | 0.0000% | 0.2772% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 71.6451% | 71.4444% | 71.9839% | 72.3141% | 72.4340% | 72.4960% |
| B_QCFS_STANDARD_SNM_R0 | 71.6451% | 71.3845% | 71.7913% | 71.9487% | 71.9971% | 72.0591% |
| C_QCFS_ASNM_R0 | 71.6451% | 71.3845% | 71.7913% | 71.9487% | 71.9971% | 72.4960% |
| D_QCFS_FULL_FTBC_R0 | 72.0247% | 72.6899% | 72.8493% | 72.7418% | 72.6270% | 72.5539% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 72.0247% | 72.6226% | 72.6409% | 72.3701% | 72.1702% | 72.0913% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 72.0247% | 72.6226% | 72.6409% | 72.3701% | 72.6270% | 72.0913% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 132,340,624,864 | 265,631,172,838 | 520,897,861,530 | 1,029,514,514,622 | 2,051,636,957,312 | 4,096,720,484,146 |
| B_QCFS_STANDARD_SNM_R0 | 132,340,624,864 | 265,995,411,938 | 524,840,201,812 | 1,045,867,621,090 | 2,089,739,057,842 | 4,168,858,567,532 |
| C_QCFS_ASNM_R0 | 132,340,624,864 | 265,995,411,938 | 524,840,201,812 | 1,045,867,621,090 | 2,089,739,057,842 | 4,096,720,484,146 |
| D_QCFS_FULL_FTBC_R0 | 128,672,920,314 | 252,801,717,190 | 503,546,961,836 | 1,013,125,919,270 | 2,037,728,556,012 | 4,089,509,842,670 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,028,676,462,258 | 2,075,858,402,230 | 4,163,986,383,636 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,028,676,462,258 | 2,037,728,556,012 | 4,163,986,383,636 |

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
| D_QCFS_FULL_FTBC_R0 | 1.189s | 1.511s | 2.588s | 4.997s | 9.802s | 19.464s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.189s | 1.511s | 2.588s | 4.997s | 9.802s | 19.464s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.189s | 1.511s | 2.588s | 4.997s | 9.802s | 19.464s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.131s | 1.329s | 1.931s | 3.311s | 5.682s | 10.596s |
| B_QCFS_STANDARD_SNM_R0 | 1.198s | 1.450s | 2.087s | 3.536s | 6.302s | 11.739s |
| C_QCFS_ASNM_R0 | 1.131s | 1.450s | 2.087s | 3.536s | 6.302s | 10.596s |
| D_QCFS_FULL_FTBC_R0 | 1.137s | 1.309s | 1.854s | 3.180s | 5.553s | 10.406s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.192s | 1.426s | 2.016s | 3.434s | 6.093s | 11.541s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.137s | 1.426s | 2.016s | 3.434s | 5.553s | 11.541s |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 2, 4, 8, 16 | 6.385s |
| Full-FTBC | 2, 4, 8, 32 | 5.795s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 37.7000% | 37.7000% | +0.0000pp | off |
| 2 | 45.8000% | 46.6000% | +0.8000pp | on |
| 4 | 56.7000% | 58.8000% | +2.1000pp | on |
| 8 | 70.2000% | 75.8000% | +5.6000pp | on |
| 16 | 81.6000% | 83.4000% | +1.8000pp | on |
| 32 | 84.6000% | 84.4000% | -0.2000pp | off |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 47.6000% | 47.6000% | +0.0000pp | off |
| 2 | 59.6000% | 60.3000% | +0.7000pp | on |
| 4 | 72.9000% | 73.3000% | +0.4000pp | on |
| 8 | 80.8000% | 81.5000% | +0.7000pp | on |
| 16 | 83.9000% | 83.9000% | +0.0000pp | off |
| 32 | 84.7000% | 85.2000% | +0.5000pp | on |

## Validation-gate versus test-oracle diagnostic

This table is post-hoc only. Test accuracy never changes the frozen A-SNM gate.

| Family | T | Validation selection | Test off | Test on | Test oracle | A-SNM oracle gap |
|---|---:|---|---:|---:|---|---:|
| qcfs | 1 | off | 63.89% | 63.89% | off | +0.00pp |
| qcfs | 2 | on | 73.29% | 73.65% | on | +0.00pp |
| qcfs | 4 | on | 83.38% | 84.47% | on | +0.00pp |
| qcfs | 8 | on | 89.82% | 90.66% | on | +0.00pp |
| qcfs | 16 | on | 91.32% | 91.49% | on | +0.00pp |
| qcfs | 32 | off | 91.69% | 91.61% | off | +0.00pp |
| full | 1 | off | 66.87% | 66.87% | off | +0.00pp |
| full | 2 | on | 76.91% | 77.23% | on | +0.00pp |
| full | 4 | on | 85.48% | 86.43% | on | +0.00pp |
| full | 8 | on | 89.95% | 90.32% | on | +0.00pp |
| full | 16 | off | 91.14% | 91.28% | on | -0.14pp |
| full | 32 | on | 91.52% | 91.50% | off | -0.02pp |

## Deployment equivalence checks

| Config | T | Expected source | Exact cached result |
|---|---:|---|---|
| C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 4 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 8 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 8 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 16 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 32 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
