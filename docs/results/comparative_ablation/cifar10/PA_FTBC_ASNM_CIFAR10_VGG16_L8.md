# QCFS + Full-FTBC + Temporal-LR FTBC + Parity-Anchor FTBC + A-SNM Ablation

Status: complete

- Dataset/architecture: CIFAR-10/vgg16
- QCFS L: 8
- ANN accuracy: 95.51%
- Checkpoint: `cifar10-vgg16-example.pth`
- Checkpoint SHA256: `093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84`
- Fit/validation SHA256: `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df` / `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c`
- Test samples: 10,000
- Evaluation profile: `not-applicable`
- Full-FTBC is fitted independently at every T with SNM off.
- Temporal-LR uses a shared learned rank-4 SVD basis with threshold normalization.
- PA-FTBC uses no SVD or stored basis: t=0/t=1 anchors plus tail mean and tail parity.
- Both compressed methods fall back exactly to Full-FTBC at T<=4.
- Every family freezes its own strict accuracy-gated A-SNM decisions before test inference.
- Checkpoint note: existing frozen repository checkpoint and evaluation protocol.

## Primary accuracy table

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | SNM-on T |
|---|---:|---:|---:|---:|---:|---:|---|
| A_QCFS_R0 | 88.26% | 91.06% | 93.81% | 95.00% | 95.42% | 95.48% | none |
| B_QCFS_STANDARD_SNM_R0 | 88.26% | 91.11% | 94.11% | 95.28% | 95.56% | 95.58% | 1, 2, 4, 8, 16, 32 |
| C_QCFS_ASNM_R0 | 88.26% | 91.11% | 94.11% | 95.00% | 95.56% | 95.48% | 2, 4, 16 |
| D_QCFS_FULL_FTBC_R0 | 89.87% | 91.98% | 94.27% | 95.24% | 95.51% | 95.47% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 89.87% | 92.05% | 94.54% | 95.48% | 95.51% | 95.53% | 1, 2, 4, 8, 16, 32 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 89.87% | 92.05% | 94.54% | 95.48% | 95.51% | 95.47% | 2, 4, 8 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 89.87% | 91.98% | 94.27% | 95.26% | 95.46% | 95.50% | none |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 89.87% | 92.05% | 94.54% | 95.30% | 95.50% | 95.56% | 1, 2, 4, 8, 16, 32 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 89.87% | 92.05% | 94.54% | 95.30% | 95.46% | 95.50% | 2, 4, 8 |
| J_QCFS_PA_FTBC_R0 | 89.87% | 91.98% | 94.27% | 95.15% | 95.46% | 95.45% | none |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 89.87% | 92.05% | 94.54% | 95.39% | 95.52% | 95.60% | 1, 2, 4, 8, 16, 32 |
| L_QCFS_PA_FTBC_ASNM_R0 | 89.87% | 92.05% | 94.54% | 95.39% | 95.52% | 95.45% | 2, 4, 8, 16 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 93.17% |
| B_QCFS_STANDARD_SNM_R0 | 93.32% |
| C_QCFS_ASNM_R0 | 93.25% |
| D_QCFS_FULL_FTBC_R0 | 93.72% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 93.83% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 93.82% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 93.72% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 93.80% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 93.79% |
| J_QCFS_PA_FTBC_R0 | 93.70% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 93.83% |
| L_QCFS_PA_FTBC_ASNM_R0 | 93.80% |

## PA-FTBC accuracy comparisons

| T | PA off - Temporal off | PA standard - Temporal standard | PA A-SNM - Temporal A-SNM |
|---:|---:|---:|---:|
| 1 | +0.00pp | +0.00pp | +0.00pp |
| 2 | +0.00pp | +0.00pp | +0.00pp |
| 4 | +0.00pp | +0.00pp | +0.00pp |
| 8 | -0.11pp | +0.09pp | +0.09pp |
| 16 | +0.00pp | +0.02pp | +0.06pp |
| 32 | -0.05pp | +0.04pp | -0.05pp |
| Mean | -0.03pp | +0.02pp | +0.02pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 2.24626339 | 1.11127808 | 0.36027746 | 0.09654537 | 0.03644716 | 0.02057767 |
| B_QCFS_STANDARD_SNM_R0 | 2.24626339 | 1.09821262 | 0.31991142 | 0.05665841 | 0.02073001 | 0.01633455 |
| C_QCFS_ASNM_R0 | 2.24626339 | 1.09821262 | 0.31991142 | 0.09654537 | 0.02073001 | 0.02057767 |
| D_QCFS_FULL_FTBC_R0 | 2.63063927 | 1.13909849 | 0.31366280 | 0.08023012 | 0.03198067 | 0.01942024 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2.63063927 | 1.13515204 | 0.28498339 | 0.05093954 | 0.02061055 | 0.01658427 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2.63063927 | 1.13515204 | 0.28498339 | 0.05093954 | 0.03198067 | 0.01942024 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 2.63063927 | 1.13909849 | 0.31366280 | 0.08119668 | 0.03297856 | 0.01976635 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 2.63063927 | 1.13515204 | 0.28498339 | 0.05088627 | 0.02030310 | 0.01617842 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2.63063927 | 1.13515204 | 0.28498339 | 0.05088627 | 0.03297856 | 0.01976635 |
| J_QCFS_PA_FTBC_R0 | 2.63063927 | 1.13909849 | 0.31366280 | 0.07998374 | 0.03180451 | 0.01944934 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 2.63063927 | 1.13515204 | 0.28498339 | 0.05118486 | 0.02061253 | 0.01667086 |
| L_QCFS_PA_FTBC_ASNM_R0 | 2.63063927 | 1.13515204 | 0.28498339 | 0.05118486 | 0.02061253 | 0.01944934 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 12.286397% | 12.591046% | 12.586811% | 12.541459% | 12.504041% | 12.482544% |
| B_QCFS_STANDARD_SNM_R0 | 12.286397% | 12.607389% | 12.620579% | 12.575277% | 12.529154% | 12.503634% |
| C_QCFS_ASNM_R0 | 12.286397% | 12.607389% | 12.620579% | 12.541459% | 12.529154% | 12.482544% |
| D_QCFS_FULL_FTBC_R0 | 12.600040% | 12.556655% | 12.523482% | 12.449802% | 12.458355% | 12.449096% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12.600040% | 12.565884% | 12.549598% | 12.480699% | 12.483730% | 12.470422% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12.600040% | 12.565884% | 12.549598% | 12.480699% | 12.458355% | 12.449096% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 12.600040% | 12.556655% | 12.523482% | 12.535064% | 12.498412% | 12.477644% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 12.600040% | 12.565884% | 12.549598% | 12.565684% | 12.523235% | 12.498941% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 12.600040% | 12.565884% | 12.549598% | 12.565684% | 12.498412% | 12.477644% |
| J_QCFS_PA_FTBC_R0 | 12.600040% | 12.556655% | 12.523482% | 12.448153% | 12.458413% | 12.449981% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 12.600040% | 12.565884% | 12.549598% | 12.478778% | 12.483316% | 12.471323% |
| L_QCFS_PA_FTBC_ASNM_R0 | 12.600040% | 12.565884% | 12.549598% | 12.478778% | 12.483316% | 12.449981% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_STANDARD_SNM_R0 | 0.000000% | 0.018832% | 0.043282% | 0.051697% | 0.044005% | 0.034928% |
| C_QCFS_ASNM_R0 | 0.000000% | 0.018832% | 0.043282% | 0.000000% | 0.044005% | 0.000000% |
| D_QCFS_FULL_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.012932% | 0.036022% | 0.046445% | 0.040664% | 0.032731% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000000% | 0.012932% | 0.036022% | 0.046445% | 0.000000% | 0.000000% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.012932% | 0.036022% | 0.047028% | 0.041850% | 0.034041% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000000% | 0.012932% | 0.036022% | 0.047028% | 0.000000% | 0.000000% |
| J_QCFS_PA_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.012932% | 0.036022% | 0.046485% | 0.040506% | 0.032810% |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.000000% | 0.012932% | 0.036022% | 0.046485% | 0.040506% | 0.000000% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 87.713603% | 87.408954% | 87.413189% | 87.458541% | 87.495959% | 87.517456% |
| B_QCFS_STANDARD_SNM_R0 | 87.713603% | 87.373779% | 87.336139% | 87.373026% | 87.426841% | 87.461438% |
| C_QCFS_ASNM_R0 | 87.713603% | 87.373779% | 87.336139% | 87.458541% | 87.426841% | 87.517456% |
| D_QCFS_FULL_FTBC_R0 | 87.399960% | 87.443345% | 87.476518% | 87.550198% | 87.541645% | 87.550904% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 87.399960% | 87.421185% | 87.414380% | 87.472856% | 87.475606% | 87.496847% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 87.399960% | 87.421185% | 87.414380% | 87.472856% | 87.541645% | 87.550904% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 87.399960% | 87.443345% | 87.476518% | 87.464936% | 87.501588% | 87.522356% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 87.399960% | 87.421185% | 87.414380% | 87.387288% | 87.434914% | 87.467018% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 87.399960% | 87.421185% | 87.414380% | 87.387288% | 87.501588% | 87.522356% |
| J_QCFS_PA_FTBC_R0 | 87.399960% | 87.443345% | 87.476518% | 87.551847% | 87.541587% | 87.550019% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 87.399960% | 87.421185% | 87.414380% | 87.474736% | 87.476178% | 87.495867% |
| L_QCFS_PA_FTBC_ASNM_R0 | 87.399960% | 87.421185% | 87.414380% | 87.474736% | 87.476178% | 87.550019% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 580,815,093,184 | 1,215,986,399,872 | 2,447,218,210,752 | 4,878,577,169,280 | 9,727,441,706,688 | 19,421,506,591,232 |
| B_QCFS_STANDARD_SNM_R0 | 580,815,093,184 | 1,219,063,444,096 | 2,467,137,795,648 | 4,927,484,989,312 | 9,803,314,477,888 | 19,534,760,398,464 |
| C_QCFS_ASNM_R0 | 580,815,093,184 | 1,219,063,444,096 | 2,467,137,795,648 | 4,878,577,169,280 | 9,803,314,477,888 | 19,421,506,591,232 |
| D_QCFS_FULL_FTBC_R0 | 601,143,818,688 | 1,218,071,939,456 | 2,424,122,737,344 | 4,836,589,713,856 | 9,684,943,809,024 | 19,369,458,039,168 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,880,159,407,680 | 9,756,058,592,000 | 19,477,757,894,656 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,880,159,407,680 | 9,684,943,809,024 | 19,369,458,039,168 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 601,143,818,688 | 1,218,071,939,456 | 2,424,122,737,344 | 4,870,621,568,832 | 9,716,270,214,464 | 19,405,005,845,440 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,914,777,840,832 | 9,788,453,731,904 | 19,515,672,258,752 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,914,777,840,832 | 9,716,270,214,464 | 19,405,005,845,440 |
| J_QCFS_PA_FTBC_R0 | 601,143,818,688 | 1,218,071,939,456 | 2,424,122,737,344 | 4,835,465,688,832 | 9,685,213,645,248 | 19,370,597,874,752 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,878,739,892,480 | 9,755,715,430,720 | 19,478,630,660,032 |
| L_QCFS_PA_FTBC_ASNM_R0 | 601,143,818,688 | 1,220,262,201,088 | 2,440,311,782,080 | 4,878,739,892,480 | 9,755,715,430,720 | 19,370,597,874,752 |

## FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 12,416 | 24,832 | 49,664 | 49,696 | 49,728 | 49,792 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 12,416 | 24,832 | 49,664 | 49,696 | 49,728 | 49,792 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 12,416 | 24,832 | 49,664 | 49,696 | 49,728 | 49,792 |
| J_QCFS_PA_FTBC_R0 | 12,416 | 24,832 | 49,664 | 49,664 | 49,664 | 49,664 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 12,416 | 24,832 | 49,664 | 49,664 | 49,664 | 49,664 |
| L_QCFS_PA_FTBC_ASNM_R0 | 12,416 | 24,832 | 49,664 | 49,664 | 49,664 | 49,664 |

## FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 49,664 | 99,328 | 198,656 | 198,784 | 198,912 | 199,168 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 49,664 | 99,328 | 198,656 | 198,784 | 198,912 | 199,168 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 49,664 | 99,328 | 198,656 | 198,784 | 198,912 | 199,168 |
| J_QCFS_PA_FTBC_R0 | 49,664 | 99,328 | 198,656 | 198,656 | 198,656 | 198,656 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 49,664 | 99,328 | 198,656 | 198,656 | 198,656 | 198,656 |
| L_QCFS_PA_FTBC_ASNM_R0 | 49,664 | 99,328 | 198,656 | 198,656 | 198,656 | 198,656 |

## Bias synthesis MACs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0 | 0 | 0 | 397,312 | 794,624 | 1,589,248 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 397,312 | 794,624 | 1,589,248 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0 | 0 | 0 | 397,312 | 794,624 | 1,589,248 |
| J_QCFS_PA_FTBC_R0 | 0 | 0 | 0 | 173,824 | 372,480 | 769,792 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 173,824 | 372,480 | 769,792 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0 | 0 | 0 | 173,824 | 372,480 | 769,792 |

## Full-teacher calibration elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B_QCFS_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C_QCFS_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| D_QCFS_FULL_FTBC_R0 | 1.949491 | 2.810395 | 4.925607 | 9.125611 | 17.478434 | 55.598987 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.949491 | 2.810395 | 4.925607 | 9.125611 | 17.478434 | 55.598987 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.949491 | 2.810395 | 4.925607 | 9.125611 | 17.478434 | 55.598987 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.949491 | 2.810395 | 4.925607 | 9.125611 | 17.478434 | 55.598987 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.949491 | 2.810395 | 4.925607 | 9.125611 | 17.478434 | 55.598987 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.949491 | 2.810395 | 4.925607 | 9.125611 | 17.478434 | 55.598987 |
| J_QCFS_PA_FTBC_R0 | 1.949491 | 2.810395 | 4.925607 | 9.125611 | 17.478434 | 55.598987 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 1.949491 | 2.810395 | 4.925607 | 9.125611 | 17.478434 | 55.598987 |
| L_QCFS_PA_FTBC_ASNM_R0 | 1.949491 | 2.810395 | 4.925607 | 9.125611 | 17.478434 | 55.598987 |

## Compression elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B_QCFS_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C_QCFS_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| D_QCFS_FULL_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.051324 | 0.019494 | 0.027102 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.051324 | 0.019494 | 0.027102 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.051324 | 0.019494 | 0.027102 |
| J_QCFS_PA_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.031880 | 0.008291 | 0.017768 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.031880 | 0.008291 | 0.017768 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.031880 | 0.008291 | 0.017768 |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.443878 | 2.216478 | 3.512613 | 6.280340 | 11.533843 | 22.491884 |
| B_QCFS_STANDARD_SNM_R0 | 1.482945 | 2.243742 | 3.776928 | 6.703899 | 12.470140 | 24.402663 |
| C_QCFS_ASNM_R0 | 1.443878 | 2.243742 | 3.776928 | 6.280340 | 12.470140 | 22.491884 |
| D_QCFS_FULL_FTBC_R0 | 1.405198 | 2.057065 | 3.419377 | 6.143473 | 11.139633 | 21.614041 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.464744 | 2.179746 | 3.666132 | 6.483438 | 12.058244 | 23.309785 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.405198 | 2.179746 | 3.666132 | 6.483438 | 11.139633 | 21.614041 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.402959 | 2.077223 | 3.393878 | 6.082133 | 11.198752 | 21.565381 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.454620 | 2.176327 | 3.644104 | 6.540448 | 12.127174 | 23.501017 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.402959 | 2.176327 | 3.644104 | 6.540448 | 11.198752 | 21.565381 |
| J_QCFS_PA_FTBC_R0 | 1.388863 | 2.057306 | 3.410826 | 6.060872 | 11.360840 | 21.726582 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 1.447220 | 2.184850 | 3.651510 | 6.577830 | 12.311200 | 23.615636 |
| L_QCFS_PA_FTBC_ASNM_R0 | 1.388863 | 2.184850 | 3.651510 | 6.577830 | 12.311200 | 21.726582 |

## Compression summary

| T | Full params | Temporal params | PA params | Temporal saving | PA saving | Temporal MACs | PA MACs | Temporal energy | PA energy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12,416 | 12,416 | 12,416 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 2 | 24,832 | 24,832 | 24,832 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 4 | 49,664 | 49,664 | 49,664 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 8 | 99,328 | 49,696 | 49,664 | 49.97% | 50.00% | 397,312 | 173,824 | 0.957244 | 0.912799 |
| 16 | 198,656 | 49,728 | 49,664 | 74.97% | 75.00% | 794,624 | 372,480 | 0.894582 | 0.837065 |
| 32 | 397,312 | 49,792 | 49,664 | 87.47% | 87.50% | 1,589,248 | 769,792 | 0.797287 | 0.738871 |

## A-SNM selection

- QCFS SNM-on T: 2, 4, 16; selection elapsed: 12.106970s.

### QCFS accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 77.50% | 77.50% | off |
| 2 | 82.50% | 82.70% | on |
| 4 | 89.40% | 90.40% | on |
| 8 | 92.80% | 92.80% | off |
| 16 | 92.80% | 93.10% | on |
| 32 | 92.90% | 92.60% | off |

- Full-FTBC SNM-on T: 2, 4, 8; selection elapsed: 11.553755s.

### Full-FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 80.30% | 80.30% | off |
| 2 | 83.70% | 83.90% | on |
| 4 | 90.60% | 90.80% | on |
| 8 | 92.60% | 92.80% | on |
| 16 | 92.60% | 92.60% | off |
| 32 | 92.70% | 92.50% | off |

- Temporal-LR FTBC SNM-on T: 2, 4, 8; selection elapsed: 11.608251s.

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 80.30% | 80.30% | off |
| 2 | 83.70% | 83.90% | on |
| 4 | 90.60% | 90.80% | on |
| 8 | 91.80% | 92.70% | on |
| 16 | 92.90% | 92.70% | off |
| 32 | 92.80% | 92.70% | off |

- Parity-Anchor FTBC SNM-on T: 2, 4, 8, 16; selection elapsed: 11.533643s.

### Parity-Anchor FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 80.30% | 80.30% | off |
| 2 | 83.70% | 83.90% | on |
| 4 | 90.60% | 90.80% | on |
| 8 | 92.70% | 92.90% | on |
| 16 | 92.70% | 92.80% | on |
| 32 | 92.90% | 92.70% | off |

## Validation-selection generalization audit

| Family | T | Selected | Test off | Test on | Test-best | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 1 | off | 88.26% | 88.26% | off | yes |
| Full-FTBC | 1 | off | 89.87% | 89.87% | off | yes |
| Temporal-LR FTBC | 1 | off | 89.87% | 89.87% | off | yes |
| Parity-Anchor FTBC | 1 | off | 89.87% | 89.87% | off | yes |
| QCFS | 2 | on | 91.06% | 91.11% | on | yes |
| Full-FTBC | 2 | on | 91.98% | 92.05% | on | yes |
| Temporal-LR FTBC | 2 | on | 91.98% | 92.05% | on | yes |
| Parity-Anchor FTBC | 2 | on | 91.98% | 92.05% | on | yes |
| QCFS | 4 | on | 93.81% | 94.11% | on | yes |
| Full-FTBC | 4 | on | 94.27% | 94.54% | on | yes |
| Temporal-LR FTBC | 4 | on | 94.27% | 94.54% | on | yes |
| Parity-Anchor FTBC | 4 | on | 94.27% | 94.54% | on | yes |
| QCFS | 8 | off | 95.00% | 95.28% | on | no |
| Full-FTBC | 8 | on | 95.24% | 95.48% | on | yes |
| Temporal-LR FTBC | 8 | on | 95.26% | 95.30% | on | yes |
| Parity-Anchor FTBC | 8 | on | 95.15% | 95.39% | on | yes |
| QCFS | 16 | on | 95.42% | 95.56% | on | yes |
| Full-FTBC | 16 | off | 95.51% | 95.51% | off | yes |
| Temporal-LR FTBC | 16 | off | 95.46% | 95.50% | on | no |
| Parity-Anchor FTBC | 16 | on | 95.46% | 95.52% | on | yes |
| QCFS | 32 | off | 95.48% | 95.58% | on | no |
| Full-FTBC | 32 | off | 95.47% | 95.53% | on | no |
| Temporal-LR FTBC | 32 | off | 95.50% | 95.56% | on | no |
| Parity-Anchor FTBC | 32 | off | 95.45% | 95.60% | on | no |

## Equivalence checks

| Kind | Name | T | Source | Exact |
|---|---|---:|---|---|
| validation fallback | off:full=temporal | 1 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 1 | Full-FTBC on | yes |
| validation fallback | off:full=pa | 1 | Full-FTBC off | yes |
| validation fallback | on:full=pa | 1 | Full-FTBC on | yes |
| validation fallback | off:full=temporal | 2 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 2 | Full-FTBC on | yes |
| validation fallback | off:full=pa | 2 | Full-FTBC off | yes |
| validation fallback | on:full=pa | 2 | Full-FTBC on | yes |
| validation fallback | off:full=temporal | 4 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 4 | Full-FTBC on | yes |
| validation fallback | off:full=pa | 4 | Full-FTBC off | yes |
| validation fallback | on:full=pa | 4 | Full-FTBC on | yes |
| gate fallback | full=temporal | 1 | identical validation metrics | yes |
| gate fallback | full=pa | 1 | identical validation metrics | yes |
| gate fallback | full=temporal | 2 | identical validation metrics | yes |
| gate fallback | full=pa | 2 | identical validation metrics | yes |
| gate fallback | full=temporal | 4 | identical validation metrics | yes |
| gate fallback | full=pa | 4 | identical validation metrics | yes |
| test fallback | off:full=temporal | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 1 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| test fallback | off:full=pa | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=pa | 1 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1 | G_QCFS_TEMPORAL_LR_FTBC_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 1 | J_QCFS_PA_FTBC_R0 | yes |
| test fallback | off:full=temporal | 2 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| test fallback | off:full=pa | 2 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=pa | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 2 | K_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| test fallback | off:full=temporal | 4 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| test fallback | off:full=pa | 4 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=pa | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 4 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 4 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 4 | K_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 8 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 8 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 8 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 8 | K_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 16 | D_QCFS_FULL_FTBC_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 16 | G_QCFS_TEMPORAL_LR_FTBC_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 16 | K_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 32 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 32 | D_QCFS_FULL_FTBC_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 32 | G_QCFS_TEMPORAL_LR_FTBC_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 32 | J_QCFS_PA_FTBC_R0 | yes |

## Per-layer Temporal-LR FTBC reconstruction

### T=1

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
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

### T=2

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
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

### T=4

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | temporal_low_rank | 64 | 0.00005439 | 0.60708362 | 0.04468700 |
| `layer1.6` | temporal_low_rank | 64 | 0.00007161 | 0.57492965 | 0.07353038 |
| `layer2.2` | temporal_low_rank | 128 | 0.00002243 | 0.56464994 | 0.06183887 |
| `layer2.6` | temporal_low_rank | 128 | 0.00000517 | 0.47192290 | 0.01614423 |
| `layer3.2` | temporal_low_rank | 256 | 0.00000223 | 0.26757994 | 0.00961851 |
| `layer3.6` | temporal_low_rank | 256 | 0.00000435 | 0.48294967 | 0.00981965 |
| `layer3.10` | temporal_low_rank | 256 | 0.00000258 | 0.48871025 | 0.00860377 |
| `layer4.2` | temporal_low_rank | 512 | 0.00000084 | 0.41368625 | 0.00390954 |
| `layer4.6` | temporal_low_rank | 512 | 0.00000028 | 0.49930659 | 0.00460547 |
| `layer4.10` | temporal_low_rank | 512 | 0.00000126 | 0.18334934 | 0.00435363 |
| `layer5.2` | temporal_low_rank | 512 | 0.00000022 | 0.23457204 | 0.00251635 |
| `layer5.6` | temporal_low_rank | 512 | 0.00000033 | 0.29081279 | 0.00288730 |
| `layer5.10` | temporal_low_rank | 512 | 0.00001277 | 0.28285623 | 0.02592510 |
| `classifier.2` | temporal_low_rank | 4096 | 0.00000774 | 0.19750936 | 0.02489288 |
| `classifier.5` | temporal_low_rank | 4096 | 0.00001395 | 0.18373449 | 0.02512525 |

### T=16

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | temporal_low_rank | 64 | 0.00008740 | 0.77228147 | 0.04255624 |
| `layer1.6` | temporal_low_rank | 64 | 0.00010324 | 0.76956260 | 0.07710592 |
| `layer2.2` | temporal_low_rank | 128 | 0.00002774 | 0.74242675 | 0.06158996 |
| `layer2.6` | temporal_low_rank | 128 | 0.00000353 | 0.53130239 | 0.01639307 |
| `layer3.2` | temporal_low_rank | 256 | 0.00000175 | 0.32932848 | 0.00940560 |
| `layer3.6` | temporal_low_rank | 256 | 0.00000247 | 0.50877994 | 0.00947419 |
| `layer3.10` | temporal_low_rank | 256 | 0.00000147 | 0.51445925 | 0.00807219 |
| `layer4.2` | temporal_low_rank | 512 | 0.00000057 | 0.46663320 | 0.00374141 |
| `layer4.6` | temporal_low_rank | 512 | 0.00000019 | 0.55244482 | 0.00423461 |
| `layer4.10` | temporal_low_rank | 512 | 0.00000103 | 0.23222809 | 0.00494017 |
| `layer5.2` | temporal_low_rank | 512 | 0.00000028 | 0.35541150 | 0.00249990 |
| `layer5.6` | temporal_low_rank | 512 | 0.00000045 | 0.45040968 | 0.00290087 |
| `layer5.10` | temporal_low_rank | 512 | 0.00001495 | 0.41178119 | 0.02698170 |
| `classifier.2` | temporal_low_rank | 4096 | 0.00001073 | 0.31775144 | 0.02486896 |
| `classifier.5` | temporal_low_rank | 4096 | 0.00001930 | 0.29660127 | 0.02642959 |

### T=32

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | temporal_low_rank | 64 | 0.00011219 | 0.88026488 | 0.04944229 |
| `layer1.6` | temporal_low_rank | 64 | 0.00010431 | 0.85930121 | 0.07810079 |
| `layer2.2` | temporal_low_rank | 128 | 0.00002484 | 0.83047676 | 0.06108933 |
| `layer2.6` | temporal_low_rank | 128 | 0.00000234 | 0.58552122 | 0.01644607 |
| `layer3.2` | temporal_low_rank | 256 | 0.00000128 | 0.38918769 | 0.00930003 |
| `layer3.6` | temporal_low_rank | 256 | 0.00000142 | 0.53665125 | 0.00910177 |
| `layer3.10` | temporal_low_rank | 256 | 0.00000087 | 0.54618829 | 0.00774039 |
| `layer4.2` | temporal_low_rank | 512 | 0.00000043 | 0.54215235 | 0.00356822 |
| `layer4.6` | temporal_low_rank | 512 | 0.00000013 | 0.61877006 | 0.00415261 |
| `layer4.10` | temporal_low_rank | 512 | 0.00000065 | 0.25965914 | 0.00497444 |
| `layer5.2` | temporal_low_rank | 512 | 0.00000031 | 0.49182454 | 0.00249497 |
| `layer5.6` | temporal_low_rank | 512 | 0.00000050 | 0.60001165 | 0.00370676 |
| `layer5.10` | temporal_low_rank | 512 | 0.00001364 | 0.52112430 | 0.02708282 |
| `classifier.2` | temporal_low_rank | 4096 | 0.00001165 | 0.44212627 | 0.02504881 |
| `classifier.5` | temporal_low_rank | 4096 | 0.00002169 | 0.42185748 | 0.02648063 |

## Per-layer Parity-Anchor FTBC reconstruction

### T=1

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
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

### T=2

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
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

### T=4

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | parity_anchor | 64 | 0.00004643 | 0.56092536 | 0.04188660 |
| `layer1.6` | parity_anchor | 64 | 0.00007408 | 0.58475709 | 0.07339866 |
| `layer2.2` | parity_anchor | 128 | 0.00002728 | 0.62274492 | 0.04873643 |
| `layer2.6` | parity_anchor | 128 | 0.00000189 | 0.28564745 | 0.01483372 |
| `layer3.2` | parity_anchor | 256 | 0.00000211 | 0.26048014 | 0.01225682 |
| `layer3.6` | parity_anchor | 256 | 0.00000062 | 0.18171126 | 0.00646715 |
| `layer3.10` | parity_anchor | 256 | 0.00000027 | 0.15940461 | 0.00502933 |
| `layer4.2` | parity_anchor | 512 | 0.00000022 | 0.21012758 | 0.00386604 |
| `layer4.6` | parity_anchor | 512 | 0.00000006 | 0.22691251 | 0.00151849 |
| `layer4.10` | parity_anchor | 512 | 0.00000436 | 0.34031233 | 0.00746243 |
| `layer5.2` | parity_anchor | 512 | 0.00000028 | 0.26153648 | 0.00247352 |
| `layer5.6` | parity_anchor | 512 | 0.00000060 | 0.39480659 | 0.00460963 |
| `layer5.10` | parity_anchor | 512 | 0.00002303 | 0.37985113 | 0.03798834 |
| `classifier.2` | parity_anchor | 4096 | 0.00001756 | 0.29757410 | 0.03799067 |
| `classifier.5` | parity_anchor | 4096 | 0.00003261 | 0.28087837 | 0.04012556 |

### T=16

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | parity_anchor | 64 | 0.00007307 | 0.70615822 | 0.05353688 |
| `layer1.6` | parity_anchor | 64 | 0.00011086 | 0.79745233 | 0.08522723 |
| `layer2.2` | parity_anchor | 128 | 0.00003106 | 0.78570771 | 0.06288888 |
| `layer2.6` | parity_anchor | 128 | 0.00000189 | 0.38880271 | 0.01876301 |
| `layer3.2` | parity_anchor | 256 | 0.00000178 | 0.33219957 | 0.01424845 |
| `layer3.6` | parity_anchor | 256 | 0.00000058 | 0.24569656 | 0.00813334 |
| `layer3.10` | parity_anchor | 256 | 0.00000032 | 0.24046583 | 0.00667986 |
| `layer4.2` | parity_anchor | 512 | 0.00000029 | 0.33149219 | 0.00566164 |
| `layer4.6` | parity_anchor | 512 | 0.00000008 | 0.35414383 | 0.00222807 |
| `layer4.10` | parity_anchor | 512 | 0.00000399 | 0.45631281 | 0.00902757 |
| `layer5.2` | parity_anchor | 512 | 0.00000032 | 0.38241816 | 0.00281127 |
| `layer5.6` | parity_anchor | 512 | 0.00000064 | 0.53769016 | 0.00589008 |
| `layer5.10` | parity_anchor | 512 | 0.00002222 | 0.50190634 | 0.03517611 |
| `classifier.2` | parity_anchor | 4096 | 0.00001742 | 0.40479630 | 0.04848132 |
| `classifier.5` | parity_anchor | 4096 | 0.00003193 | 0.38150981 | 0.04819747 |

### T=32

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | parity_anchor | 64 | 0.00010289 | 0.84296322 | 0.05152206 |
| `layer1.6` | parity_anchor | 64 | 0.00010634 | 0.86762303 | 0.08153518 |
| `layer2.2` | parity_anchor | 128 | 0.00002497 | 0.83268374 | 0.06588411 |
| `layer2.6` | parity_anchor | 128 | 0.00000145 | 0.46143591 | 0.01773191 |
| `layer3.2` | parity_anchor | 256 | 0.00000126 | 0.38609400 | 0.01466545 |
| `layer3.6` | parity_anchor | 256 | 0.00000046 | 0.30564007 | 0.00846338 |
| `layer3.10` | parity_anchor | 256 | 0.00000029 | 0.31644461 | 0.00754275 |
| `layer4.2` | parity_anchor | 512 | 0.00000029 | 0.44963756 | 0.00649165 |
| `layer4.6` | parity_anchor | 512 | 0.00000007 | 0.46170852 | 0.00246743 |
| `layer4.10` | parity_anchor | 512 | 0.00000238 | 0.49512631 | 0.01004827 |
| `layer5.2` | parity_anchor | 512 | 0.00000033 | 0.51204103 | 0.00293270 |
| `layer5.6` | parity_anchor | 512 | 0.00000060 | 0.66208649 | 0.00642616 |
| `layer5.10` | parity_anchor | 512 | 0.00001761 | 0.59218627 | 0.03567813 |
| `classifier.2` | parity_anchor | 4096 | 0.00001548 | 0.50963503 | 0.05131925 |
| `classifier.5` | parity_anchor | 4096 | 0.00002891 | 0.48704684 | 0.05164460 |
