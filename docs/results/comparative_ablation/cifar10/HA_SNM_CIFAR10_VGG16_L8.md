# QCFS + Full/Temporal-LR/PA-FTBC + HA-SNM Ablation

Status: complete

- Dataset/architecture: CIFAR-10/vgg16
- QCFS L: 8
- ANN accuracy: 95.51%
- Checkpoint: `cifar10-vgg16-example.pth`
- Checkpoint SHA256: `093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84`
- Fit/validation SHA256: `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df` / `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c`
- Test samples: 10,000
- Evaluation profile: `not-applicable`
- HA-SNM threshold schedule: start=1.25, end=0.5, reference horizon=8.0, linear.
- HA-SNM keeps the original transmitted-credit/R0 rule and changes only the negative-spike decision threshold.
- It uses the original -theta event amplitude, adds no dense neuron state, and has two global FP32 deployment constants plus one fixed reference horizon (12 bytes if stored).
- Full-FTBC is fitted independently at every T with SNM off; Temporal-LR and PA are compressed from that same teacher.
- Temporal-LR and PA fall back exactly to Full-FTBC at T<=4.
- Checkpoint note: existing frozen repository checkpoint and evaluation protocol.

## Primary accuracy

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 89.87% | 91.98% | 94.27% | 95.24% | 95.51% | 95.47% | 93.72% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 89.87% | 92.05% | 94.54% | 95.48% | 95.51% | 95.53% | 93.83% |
| C_QCFS_FULL_FTBC_HA_R0 | 89.87% | 92.35% | 94.72% | 95.54% | 95.56% | 95.58% | 93.94% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 89.87% | 91.98% | 94.27% | 95.26% | 95.46% | 95.50% | 93.72% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 89.87% | 92.05% | 94.54% | 95.30% | 95.50% | 95.56% | 93.80% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 89.87% | 92.35% | 94.72% | 95.36% | 95.52% | 95.54% | 93.89% |
| G_QCFS_PA_FTBC_OFF_R0 | 89.87% | 91.98% | 94.27% | 95.15% | 95.46% | 95.45% | 93.70% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 89.87% | 92.05% | 94.54% | 95.39% | 95.52% | 95.60% | 93.83% |
| I_QCFS_PA_FTBC_HA_R0 | 89.87% | 92.35% | 94.72% | 95.43% | 95.53% | 95.55% | 93.91% |

## HA-SNM accuracy gain

| Family | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-FTBC: HA - standard | +0.00pp | +0.30pp | +0.18pp | +0.06pp | +0.05pp | +0.05pp | +0.107pp |
| Full-FTBC: HA - off | +0.00pp | +0.37pp | +0.45pp | +0.30pp | +0.05pp | +0.11pp | +0.213pp |
| Temporal-LR FTBC: HA - standard | +0.00pp | +0.30pp | +0.18pp | +0.06pp | +0.02pp | -0.02pp | +0.090pp |
| Temporal-LR FTBC: HA - off | +0.00pp | +0.37pp | +0.45pp | +0.10pp | +0.06pp | +0.04pp | +0.170pp |
| PA-FTBC: HA - standard | +0.00pp | +0.30pp | +0.18pp | +0.04pp | +0.01pp | -0.05pp | +0.080pp |
| PA-FTBC: HA - off | +0.00pp | +0.37pp | +0.45pp | +0.28pp | +0.07pp | +0.10pp | +0.212pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 2.63063927 | 1.13909849 | 0.31366280 | 0.08023012 | 0.03198067 | 0.01942024 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 2.63063927 | 1.13515204 | 0.28498339 | 0.05093954 | 0.02061055 | 0.01658427 |
| C_QCFS_FULL_FTBC_HA_R0 | 2.63063927 | 1.06907919 | 0.25548690 | 0.04129602 | 0.02026197 | 0.01661202 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 2.63063927 | 1.13909849 | 0.31366280 | 0.08119668 | 0.03297856 | 0.01976635 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 2.63063927 | 1.13515204 | 0.28498339 | 0.05088627 | 0.02030310 | 0.01617842 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 2.63063927 | 1.06907919 | 0.25548690 | 0.03981442 | 0.01985015 | 0.01618265 |
| G_QCFS_PA_FTBC_OFF_R0 | 2.63063927 | 1.13909849 | 0.31366280 | 0.07998374 | 0.03180451 | 0.01944934 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 2.63063927 | 1.13515204 | 0.28498339 | 0.05118486 | 0.02061253 | 0.01667086 |
| I_QCFS_PA_FTBC_HA_R0 | 2.63063927 | 1.06907919 | 0.25548690 | 0.04131592 | 0.02033308 | 0.01664832 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 12.600040% | 12.556655% | 12.523482% | 12.449802% | 12.458355% | 12.449096% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 12.600040% | 12.565884% | 12.549598% | 12.480699% | 12.483730% | 12.470422% |
| C_QCFS_FULL_FTBC_HA_R0 | 12.600040% | 12.611016% | 12.587088% | 12.503244% | 12.488699% | 12.471916% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 12.600040% | 12.556655% | 12.523482% | 12.535064% | 12.498412% | 12.477644% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 12.600040% | 12.565884% | 12.549598% | 12.565684% | 12.523235% | 12.498941% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 12.600040% | 12.611016% | 12.587088% | 12.588241% | 12.528478% | 12.500452% |
| G_QCFS_PA_FTBC_OFF_R0 | 12.600040% | 12.556655% | 12.523482% | 12.448153% | 12.458413% | 12.449981% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 12.600040% | 12.565884% | 12.549598% | 12.478778% | 12.483316% | 12.471323% |
| I_QCFS_PA_FTBC_HA_R0 | 12.600040% | 12.611016% | 12.587088% | 12.501050% | 12.488261% | 12.472789% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 0.000000% | 0.012932% | 0.036022% | 0.046445% | 0.040664% | 0.032731% |
| C_QCFS_FULL_FTBC_HA_R0 | 0.000000% | 0.120759% | 0.112647% | 0.092581% | 0.050019% | 0.035224% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 0.000000% | 0.012932% | 0.036022% | 0.047028% | 0.041850% | 0.034041% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 0.000000% | 0.120759% | 0.112647% | 0.094308% | 0.051611% | 0.036635% |
| G_QCFS_PA_FTBC_OFF_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 0.000000% | 0.012932% | 0.036022% | 0.046485% | 0.040506% | 0.032810% |
| I_QCFS_PA_FTBC_HA_R0 | 0.000000% | 0.120759% | 0.112647% | 0.092372% | 0.049750% | 0.035288% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 87.399960% | 87.443345% | 87.476518% | 87.550198% | 87.541645% | 87.550904% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 87.399960% | 87.421185% | 87.414380% | 87.472856% | 87.475606% | 87.496847% |
| C_QCFS_FULL_FTBC_HA_R0 | 87.399960% | 87.268225% | 87.300265% | 87.404175% | 87.461281% | 87.492860% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 87.399960% | 87.443345% | 87.476518% | 87.464936% | 87.501588% | 87.522356% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 87.399960% | 87.421185% | 87.414380% | 87.387288% | 87.434914% | 87.467018% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 87.399960% | 87.268225% | 87.300265% | 87.317451% | 87.419911% | 87.462913% |
| G_QCFS_PA_FTBC_OFF_R0 | 87.399960% | 87.443345% | 87.476518% | 87.551847% | 87.541587% | 87.550019% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 87.399960% | 87.421185% | 87.414380% | 87.474736% | 87.476178% | 87.495867% |
| I_QCFS_PA_FTBC_HA_R0 | 87.399960% | 87.268225% | 87.300265% | 87.406579% | 87.461989% | 87.491922% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 601,143,818,688 | 1,218,071,939,456 | 2,424,122,737,344 | 4,836,589,713,856 | 9,684,943,809,024 | 19,369,458,039,168 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,880,159,407,680 | 9,756,058,592,000 | 19,477,757,894,656 |
| C_QCFS_FULL_FTBC_HA_R0 | 601,143,818,688 | 1,236,372,320,128 | 2,471,073,149,760 | 4,917,821,274,432 | 9,771,031,535,488 | 19,485,590,158,208 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 601,143,818,688 | 1,218,071,939,456 | 2,424,122,737,344 | 4,870,621,568,832 | 9,716,270,214,464 | 19,405,005,845,440 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,914,777,840,832 | 9,788,453,731,904 | 19,515,672,258,752 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 601,143,818,688 | 1,236,372,320,128 | 2,471,073,149,760 | 4,953,004,746,304 | 9,804,214,559,680 | 19,523,577,305,664 |
| G_QCFS_PA_FTBC_OFF_R0 | 601,143,818,688 | 1,218,071,939,456 | 2,424,122,737,344 | 4,835,465,688,832 | 9,685,213,645,248 | 19,370,597,874,752 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,878,739,892,480 | 9,755,715,430,720 | 19,478,630,660,032 |
| I_QCFS_PA_FTBC_HA_R0 | 601,143,818,688 | 1,236,372,320,128 | 2,471,073,149,760 | 4,916,239,868,544 | 9,770,586,676,160 | 19,486,328,785,216 |

## FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| C_QCFS_FULL_FTBC_HA_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 12,416 | 24,832 | 49,664 | 49,696 | 49,728 | 49,792 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 12,416 | 24,832 | 49,664 | 49,696 | 49,728 | 49,792 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 12,416 | 24,832 | 49,664 | 49,696 | 49,728 | 49,792 |
| G_QCFS_PA_FTBC_OFF_R0 | 12,416 | 24,832 | 49,664 | 49,664 | 49,664 | 49,664 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 12,416 | 24,832 | 49,664 | 49,664 | 49,664 | 49,664 |
| I_QCFS_PA_FTBC_HA_R0 | 12,416 | 24,832 | 49,664 | 49,664 | 49,664 | 49,664 |

## FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| C_QCFS_FULL_FTBC_HA_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 49,664 | 99,328 | 198,656 | 198,784 | 198,912 | 199,168 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 49,664 | 99,328 | 198,656 | 198,784 | 198,912 | 199,168 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 49,664 | 99,328 | 198,656 | 198,784 | 198,912 | 199,168 |
| G_QCFS_PA_FTBC_OFF_R0 | 49,664 | 99,328 | 198,656 | 198,656 | 198,656 | 198,656 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 49,664 | 99,328 | 198,656 | 198,656 | 198,656 | 198,656 |
| I_QCFS_PA_FTBC_HA_R0 | 49,664 | 99,328 | 198,656 | 198,656 | 198,656 | 198,656 |

## Bias synthesis MACs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_FULL_FTBC_HA_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 0 | 0 | 0 | 397,312 | 794,624 | 1,589,248 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 0 | 0 | 0 | 397,312 | 794,624 | 1,589,248 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 0 | 0 | 0 | 397,312 | 794,624 | 1,589,248 |
| G_QCFS_PA_FTBC_OFF_R0 | 0 | 0 | 0 | 173,824 | 372,480 | 769,792 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 0 | 0 | 0 | 173,824 | 372,480 | 769,792 |
| I_QCFS_PA_FTBC_HA_R0 | 0 | 0 | 0 | 173,824 | 372,480 | 769,792 |

## Inference elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 1.388190 | 2.095444 | 3.365654 | 5.969375 | 11.199242 | 21.103548 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 1.472541 | 2.168553 | 3.598039 | 6.428870 | 11.891463 | 22.900701 |
| C_QCFS_FULL_FTBC_HA_R0 | 1.483846 | 2.216839 | 3.615289 | 6.449720 | 11.990293 | 23.255873 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 1.399843 | 2.085176 | 3.380208 | 6.033064 | 11.042711 | 21.258138 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 1.477162 | 2.144595 | 3.608505 | 6.460038 | 11.983323 | 23.125169 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 1.426713 | 2.186584 | 3.593303 | 6.519448 | 12.017690 | 23.302052 |
| G_QCFS_PA_FTBC_OFF_R0 | 1.385668 | 2.076038 | 3.367410 | 5.996046 | 11.105463 | 21.380378 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 1.462797 | 2.175439 | 3.583119 | 6.469442 | 12.059288 | 23.415481 |
| I_QCFS_PA_FTBC_HA_R0 | 1.455196 | 2.143888 | 3.599919 | 6.517135 | 12.090539 | 23.554600 |

## HA-SNM overhead

| Item | Value |
|---|---:|
| Additional dense per-neuron state | 0 bytes |
| Global constants | 3 (12 bytes if all stored as FP32) |
| SignedIF layers | 15 |
| Per layer/time decision overhead | one scalar threshold interpolation and the existing comparison |

## Exact fallback checks

| T | Mode | Full=Temporal | Full=PA |
|---:|---|---|---|
| 1 | SNM-off | yes | yes |
| 1 | standard SNM | yes | yes |
| 1 | HA-SNM | yes | yes |
| 2 | SNM-off | yes | yes |
| 2 | standard SNM | yes | yes |
| 2 | HA-SNM | yes | yes |
| 4 | SNM-off | yes | yes |
| 4 | standard SNM | yes | yes |
| 4 | HA-SNM | yes | yes |
