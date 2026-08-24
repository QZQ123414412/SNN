# QCFS + Full-FTBC + Temporal-LR FTBC + Parity-Anchor FTBC + A-SNM Ablation

Status: complete

- Dataset/architecture: CIFAR-100/vgg16
- QCFS L: 8
- ANN accuracy: 77.35%
- Checkpoint: `cifar100-vgg16-l8-example.pth`
- Checkpoint SHA256: `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339`
- Fit/validation SHA256: `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a` / `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3`
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
| A_QCFS_R0 | 58.82% | 65.00% | 71.09% | 75.20% | 77.11% | 77.66% | none |
| B_QCFS_STANDARD_SNM_R0 | 58.82% | 65.00% | 72.29% | 76.59% | 77.49% | 77.50% | 1, 2, 4, 8, 16, 32 |
| C_QCFS_ASNM_R0 | 58.82% | 65.00% | 72.29% | 76.59% | 77.49% | 77.66% | 2, 4, 8, 16 |
| D_QCFS_FULL_FTBC_R0 | 61.72% | 67.58% | 73.39% | 76.47% | 77.40% | 77.53% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 61.72% | 67.68% | 74.01% | 77.16% | 77.64% | 77.61% | 1, 2, 4, 8, 16, 32 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 61.72% | 67.68% | 74.01% | 77.16% | 77.64% | 77.61% | 2, 4, 8, 16, 32 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 61.72% | 67.58% | 73.39% | 76.37% | 77.62% | 77.75% | none |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 61.72% | 67.68% | 74.01% | 77.12% | 77.65% | 77.57% | 1, 2, 4, 8, 16, 32 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 61.72% | 67.68% | 74.01% | 77.12% | 77.65% | 77.57% | 2, 4, 8, 16, 32 |
| J_QCFS_PA_FTBC_R0 | 61.72% | 67.58% | 73.39% | 76.24% | 77.49% | 77.61% | none |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 61.72% | 67.68% | 74.01% | 76.98% | 77.49% | 77.54% | 1, 2, 4, 8, 16, 32 |
| L_QCFS_PA_FTBC_ASNM_R0 | 61.72% | 67.68% | 74.01% | 76.98% | 77.49% | 77.61% | 2, 4, 8, 16 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 70.81% |
| B_QCFS_STANDARD_SNM_R0 | 71.28% |
| C_QCFS_ASNM_R0 | 71.31% |
| D_QCFS_FULL_FTBC_R0 | 72.35% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 72.64% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 72.64% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 72.41% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 72.63% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 72.63% |
| J_QCFS_PA_FTBC_R0 | 72.34% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 72.57% |
| L_QCFS_PA_FTBC_ASNM_R0 | 72.58% |

## PA-FTBC accuracy comparisons

| T | PA off - Temporal off | PA standard - Temporal standard | PA A-SNM - Temporal A-SNM |
|---:|---:|---:|---:|
| 1 | +0.00pp | +0.00pp | +0.00pp |
| 2 | +0.00pp | +0.00pp | +0.00pp |
| 4 | +0.00pp | +0.00pp | +0.00pp |
| 8 | -0.13pp | -0.14pp | -0.14pp |
| 16 | -0.13pp | -0.16pp | -0.16pp |
| 32 | -0.14pp | -0.03pp | +0.04pp |
| Mean | -0.07pp | -0.06pp | -0.04pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 6.33891227 | 4.17280399 | 1.72800092 | 0.64333612 | 0.21531409 | 0.09927765 |
| B_QCFS_STANDARD_SNM_R0 | 6.33891227 | 4.09635542 | 1.44470267 | 0.33731325 | 0.09995643 | 0.07218033 |
| C_QCFS_ASNM_R0 | 6.33891227 | 4.09635542 | 1.44470267 | 0.33731325 | 0.09995643 | 0.09927765 |
| D_QCFS_FULL_FTBC_R0 | 11.70284070 | 3.59960155 | 1.24905064 | 0.44585386 | 0.16363363 | 0.08881311 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 11.70284070 | 3.53799891 | 1.05203704 | 0.25555007 | 0.09663141 | 0.07206773 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 11.70284070 | 3.53799891 | 1.05203704 | 0.25555007 | 0.09663141 | 0.07206773 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 11.70284070 | 3.59960155 | 1.24905064 | 0.45768403 | 0.17105388 | 0.09102433 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 11.70284070 | 3.53799891 | 1.05203704 | 0.25781043 | 0.09522265 | 0.07109507 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 11.70284070 | 3.53799891 | 1.05203704 | 0.25781043 | 0.09522265 | 0.07109507 |
| J_QCFS_PA_FTBC_R0 | 11.70284070 | 3.59960155 | 1.24905064 | 0.44832980 | 0.16476413 | 0.08922123 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 11.70284070 | 3.53799891 | 1.05203704 | 0.25539854 | 0.09720389 | 0.07264778 |
| L_QCFS_PA_FTBC_ASNM_R0 | 11.70284070 | 3.53799891 | 1.05203704 | 0.25539854 | 0.09720389 | 0.08922123 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 14.554943% | 15.156253% | 15.150919% | 15.081787% | 15.039481% | 15.016009% |
| B_QCFS_STANDARD_SNM_R0 | 14.554943% | 15.167775% | 15.173302% | 15.104789% | 15.055121% | 15.028278% |
| C_QCFS_ASNM_R0 | 14.554943% | 15.167775% | 15.173302% | 15.104789% | 15.055121% | 15.016009% |
| D_QCFS_FULL_FTBC_R0 | 14.966055% | 15.129794% | 14.991640% | 14.972103% | 14.968577% | 14.966596% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 14.966055% | 15.138903% | 15.010658% | 14.994965% | 14.985965% | 14.980309% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 14.966055% | 15.138903% | 15.010658% | 14.994965% | 14.985965% | 14.980309% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 14.966055% | 15.129794% | 14.991640% | 15.029483% | 15.028983% | 15.014473% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 14.966055% | 15.138903% | 15.010658% | 15.053012% | 15.046128% | 15.028009% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 14.966055% | 15.138903% | 15.010658% | 15.053012% | 15.046128% | 15.028009% |
| J_QCFS_PA_FTBC_R0 | 14.966055% | 15.129794% | 14.991640% | 14.973450% | 14.968318% | 14.967167% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 14.966055% | 15.138903% | 15.010658% | 14.995877% | 14.985760% | 14.981129% |
| L_QCFS_PA_FTBC_ASNM_R0 | 14.966055% | 15.138903% | 15.010658% | 14.995877% | 14.985760% | 14.967167% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_STANDARD_SNM_R0 | 0.000000% | 0.010462% | 0.034467% | 0.046934% | 0.040118% | 0.029654% |
| C_QCFS_ASNM_R0 | 0.000000% | 0.010462% | 0.034467% | 0.046934% | 0.040118% | 0.000000% |
| D_QCFS_FULL_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.008617% | 0.028697% | 0.039150% | 0.033912% | 0.025950% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000000% | 0.008617% | 0.028697% | 0.039150% | 0.033912% | 0.025950% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.008617% | 0.028697% | 0.039831% | 0.034813% | 0.026705% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000000% | 0.008617% | 0.028697% | 0.039831% | 0.034813% | 0.026705% |
| J_QCFS_PA_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.008617% | 0.028697% | 0.039176% | 0.033983% | 0.026043% |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.000000% | 0.008617% | 0.028697% | 0.039176% | 0.033983% | 0.000000% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 85.445057% | 84.843747% | 84.849081% | 84.918213% | 84.960519% | 84.983991% |
| B_QCFS_STANDARD_SNM_R0 | 85.445057% | 84.821762% | 84.792231% | 84.848277% | 84.904760% | 84.942068% |
| C_QCFS_ASNM_R0 | 85.445057% | 84.821762% | 84.792231% | 84.848277% | 84.904760% | 84.983991% |
| D_QCFS_FULL_FTBC_R0 | 85.033945% | 84.870206% | 85.008360% | 85.027897% | 85.031423% | 85.033404% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 85.033945% | 84.852480% | 84.960645% | 84.965886% | 84.980124% | 84.993742% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 85.033945% | 84.852480% | 84.960645% | 84.965886% | 84.980124% | 84.993742% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 85.033945% | 84.870206% | 85.008360% | 84.970517% | 84.971017% | 84.985527% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 85.033945% | 84.852480% | 84.960645% | 84.907157% | 84.919059% | 84.945286% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 85.033945% | 84.852480% | 84.960645% | 84.907157% | 84.919059% | 84.945286% |
| J_QCFS_PA_FTBC_R0 | 85.033945% | 84.870206% | 85.008360% | 85.026550% | 85.031682% | 85.032833% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 85.033945% | 84.852480% | 84.960645% | 84.964947% | 84.980257% | 84.992827% |
| L_QCFS_PA_FTBC_ASNM_R0 | 85.033945% | 84.852480% | 84.960645% | 84.964947% | 84.980257% | 85.032833% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 624,247,119,488 | 1,305,067,150,592 | 2,588,715,654,400 | 5,135,019,932,608 | 10,223,717,533,760 | 20,394,968,227,648 |
| B_QCFS_STANDARD_SNM_R0 | 624,247,119,488 | 1,307,468,667,520 | 2,607,638,985,856 | 5,187,982,392,000 | 10,308,318,464,832 | 20,517,034,551,232 |
| C_QCFS_ASNM_R0 | 624,247,119,488 | 1,307,468,667,520 | 2,607,638,985,856 | 5,187,982,392,000 | 10,308,318,464,832 | 20,394,968,227,648 |
| D_QCFS_FULL_FTBC_R0 | 642,073,748,672 | 1,287,142,886,272 | 2,547,186,255,872 | 5,084,745,842,240 | 10,167,219,618,240 | 20,328,287,895,616 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,130,873,717,056 | 10,243,080,312,896 | 20,441,318,701,888 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,130,873,717,056 | 10,243,080,312,896 | 20,441,318,701,888 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 642,073,748,672 | 1,287,142,886,272 | 2,547,186,255,872 | 5,107,622,627,008 | 10,212,382,280,256 | 20,398,244,034,944 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,154,410,295,232 | 10,289,123,421,376 | 20,513,091,425,792 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,154,410,295,232 | 10,289,123,421,376 | 20,513,091,425,792 |
| J_QCFS_PA_FTBC_R0 | 642,073,748,672 | 1,287,142,886,272 | 2,547,186,255,872 | 5,085,945,299,392 | 10,166,830,189,312 | 20,328,826,782,912 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,131,772,443,840 | 10,242,960,741,888 | 20,442,878,013,632 |
| L_QCFS_PA_FTBC_ASNM_R0 | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,131,772,443,840 | 10,242,960,741,888 | 20,328,826,782,912 |

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
| D_QCFS_FULL_FTBC_R0 | 1.698228 | 2.797335 | 4.910169 | 9.117518 | 17.441275 | 55.268921 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.698228 | 2.797335 | 4.910169 | 9.117518 | 17.441275 | 55.268921 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.698228 | 2.797335 | 4.910169 | 9.117518 | 17.441275 | 55.268921 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.698228 | 2.797335 | 4.910169 | 9.117518 | 17.441275 | 55.268921 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.698228 | 2.797335 | 4.910169 | 9.117518 | 17.441275 | 55.268921 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.698228 | 2.797335 | 4.910169 | 9.117518 | 17.441275 | 55.268921 |
| J_QCFS_PA_FTBC_R0 | 1.698228 | 2.797335 | 4.910169 | 9.117518 | 17.441275 | 55.268921 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 1.698228 | 2.797335 | 4.910169 | 9.117518 | 17.441275 | 55.268921 |
| L_QCFS_PA_FTBC_ASNM_R0 | 1.698228 | 2.797335 | 4.910169 | 9.117518 | 17.441275 | 55.268921 |

## Compression elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B_QCFS_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C_QCFS_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| D_QCFS_FULL_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.060058 | 0.020675 | 0.030592 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.060058 | 0.020675 | 0.030592 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.060058 | 0.020675 | 0.030592 |
| J_QCFS_PA_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.027630 | 0.007354 | 0.013161 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.027630 | 0.007354 | 0.013161 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.027630 | 0.007354 | 0.013161 |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.436215 | 2.154251 | 3.436989 | 6.115376 | 11.273076 | 21.552300 |
| B_QCFS_STANDARD_SNM_R0 | 1.481507 | 2.207031 | 3.656411 | 6.586787 | 12.343506 | 23.494625 |
| C_QCFS_ASNM_R0 | 1.436215 | 2.207031 | 3.656411 | 6.586787 | 12.343506 | 21.552300 |
| D_QCFS_FULL_FTBC_R0 | 1.379022 | 2.064986 | 3.344592 | 5.892708 | 11.049426 | 20.906994 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.423421 | 2.157990 | 3.528799 | 6.383897 | 11.758403 | 22.733542 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.379022 | 2.157990 | 3.528799 | 6.383897 | 11.758403 | 22.733542 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.381067 | 2.025983 | 3.307752 | 5.941788 | 10.876166 | 21.008800 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.412005 | 2.125486 | 3.533950 | 6.429533 | 11.755892 | 23.149301 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.381067 | 2.125486 | 3.533950 | 6.429533 | 11.755892 | 23.149301 |
| J_QCFS_PA_FTBC_R0 | 1.370942 | 2.011298 | 3.339379 | 6.032712 | 10.907589 | 21.361146 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 1.440538 | 2.140894 | 3.563331 | 6.419270 | 11.814046 | 23.312147 |
| L_QCFS_PA_FTBC_ASNM_R0 | 1.370942 | 2.140894 | 3.563331 | 6.419270 | 11.814046 | 21.361146 |

## Compression summary

| T | Full params | Temporal params | PA params | Temporal saving | PA saving | Temporal MACs | PA MACs | Temporal energy | PA energy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12,416 | 12,416 | 12,416 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 2 | 24,832 | 24,832 | 24,832 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 4 | 49,664 | 49,664 | 49,664 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 8 | 99,328 | 49,696 | 49,664 | 49.97% | 50.00% | 397,312 | 173,824 | 0.948015 | 0.908405 |
| 16 | 198,656 | 49,728 | 49,664 | 74.97% | 75.00% | 794,624 | 372,480 | 0.855806 | 0.796124 |
| 32 | 397,312 | 49,792 | 49,664 | 87.47% | 87.50% | 1,589,248 | 769,792 | 0.720206 | 0.651369 |

## A-SNM selection

- QCFS SNM-on T: 2, 4, 8, 16; selection elapsed: 12.006027s.

### QCFS accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 53.60% | 53.60% | off |
| 2 | 65.00% | 65.90% | on |
| 4 | 75.60% | 77.70% | on |
| 8 | 84.90% | 86.80% | on |
| 16 | 88.10% | 88.30% | on |
| 32 | 88.70% | 87.90% | off |

- Full-FTBC SNM-on T: 2, 4, 8, 16, 32; selection elapsed: 11.392327s.

### Full-FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 57.90% | 57.90% | off |
| 2 | 67.70% | 68.20% | on |
| 4 | 77.80% | 78.10% | on |
| 8 | 83.80% | 87.40% | on |
| 16 | 87.30% | 88.40% | on |
| 32 | 87.90% | 88.10% | on |

- Temporal-LR FTBC SNM-on T: 2, 4, 8, 16, 32; selection elapsed: 11.521169s.

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 57.90% | 57.90% | off |
| 2 | 67.70% | 68.20% | on |
| 4 | 77.80% | 78.10% | on |
| 8 | 83.80% | 86.70% | on |
| 16 | 87.40% | 88.00% | on |
| 32 | 88.30% | 88.50% | on |

- Parity-Anchor FTBC SNM-on T: 2, 4, 8, 16; selection elapsed: 11.708027s.

### Parity-Anchor FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 57.90% | 57.90% | off |
| 2 | 67.70% | 68.20% | on |
| 4 | 77.80% | 78.10% | on |
| 8 | 84.80% | 87.10% | on |
| 16 | 87.70% | 88.30% | on |
| 32 | 88.00% | 88.00% | off |

## Validation-selection generalization audit

| Family | T | Selected | Test off | Test on | Test-best | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 1 | off | 58.82% | 58.82% | off | yes |
| Full-FTBC | 1 | off | 61.72% | 61.72% | off | yes |
| Temporal-LR FTBC | 1 | off | 61.72% | 61.72% | off | yes |
| Parity-Anchor FTBC | 1 | off | 61.72% | 61.72% | off | yes |
| QCFS | 2 | on | 65.00% | 65.00% | off | no |
| Full-FTBC | 2 | on | 67.58% | 67.68% | on | yes |
| Temporal-LR FTBC | 2 | on | 67.58% | 67.68% | on | yes |
| Parity-Anchor FTBC | 2 | on | 67.58% | 67.68% | on | yes |
| QCFS | 4 | on | 71.09% | 72.29% | on | yes |
| Full-FTBC | 4 | on | 73.39% | 74.01% | on | yes |
| Temporal-LR FTBC | 4 | on | 73.39% | 74.01% | on | yes |
| Parity-Anchor FTBC | 4 | on | 73.39% | 74.01% | on | yes |
| QCFS | 8 | on | 75.20% | 76.59% | on | yes |
| Full-FTBC | 8 | on | 76.47% | 77.16% | on | yes |
| Temporal-LR FTBC | 8 | on | 76.37% | 77.12% | on | yes |
| Parity-Anchor FTBC | 8 | on | 76.24% | 76.98% | on | yes |
| QCFS | 16 | on | 77.11% | 77.49% | on | yes |
| Full-FTBC | 16 | on | 77.40% | 77.64% | on | yes |
| Temporal-LR FTBC | 16 | on | 77.62% | 77.65% | on | yes |
| Parity-Anchor FTBC | 16 | on | 77.49% | 77.49% | off | no |
| QCFS | 32 | off | 77.66% | 77.50% | off | yes |
| Full-FTBC | 32 | on | 77.53% | 77.61% | on | yes |
| Temporal-LR FTBC | 32 | on | 77.75% | 77.57% | off | no |
| Parity-Anchor FTBC | 32 | off | 77.61% | 77.54% | off | yes |

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
| A-SNM cache | C_QCFS_ASNM_R0 | 8 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 8 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 8 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 8 | K_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 16 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 16 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 16 | K_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 32 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 32 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
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
| `layer1.2` | temporal_low_rank | 64 | 0.00003863 | 0.46992025 | 0.03438693 |
| `layer1.6` | temporal_low_rank | 64 | 0.00008772 | 0.49195278 | 0.06751649 |
| `layer2.2` | temporal_low_rank | 128 | 0.00002332 | 0.45783076 | 0.03641290 |
| `layer2.6` | temporal_low_rank | 128 | 0.00000563 | 0.33228531 | 0.02292853 |
| `layer3.2` | temporal_low_rank | 256 | 0.00000303 | 0.25637326 | 0.01054865 |
| `layer3.6` | temporal_low_rank | 256 | 0.00000263 | 0.32543764 | 0.00701699 |
| `layer3.10` | temporal_low_rank | 256 | 0.00000138 | 0.25827396 | 0.00648163 |
| `layer4.2` | temporal_low_rank | 512 | 0.00000123 | 0.29361448 | 0.00857863 |
| `layer4.6` | temporal_low_rank | 512 | 0.00000061 | 0.31081694 | 0.00464453 |
| `layer4.10` | temporal_low_rank | 512 | 0.00000035 | 0.39141965 | 0.00333785 |
| `layer5.2` | temporal_low_rank | 512 | 0.00000071 | 0.46290806 | 0.00470774 |
| `layer5.6` | temporal_low_rank | 512 | 0.00000048 | 0.44555300 | 0.00421798 |
| `layer5.10` | temporal_low_rank | 512 | 0.00003723 | 0.31814703 | 0.03504444 |
| `classifier.2` | temporal_low_rank | 4096 | 0.00005906 | 0.15942287 | 0.05383148 |
| `classifier.5` | temporal_low_rank | 4096 | 0.00015108 | 0.34477940 | 0.08590782 |

### T=16

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | temporal_low_rank | 64 | 0.00006416 | 0.60441858 | 0.03972055 |
| `layer1.6` | temporal_low_rank | 64 | 0.00016737 | 0.73688883 | 0.08619776 |
| `layer2.2` | temporal_low_rank | 128 | 0.00003148 | 0.65145504 | 0.04806526 |
| `layer2.6` | temporal_low_rank | 128 | 0.00000596 | 0.46193838 | 0.02464197 |
| `layer3.2` | temporal_low_rank | 256 | 0.00000299 | 0.35166916 | 0.01147353 |
| `layer3.6` | temporal_low_rank | 256 | 0.00000232 | 0.42377961 | 0.00688755 |
| `layer3.10` | temporal_low_rank | 256 | 0.00000146 | 0.36764997 | 0.00704847 |
| `layer4.2` | temporal_low_rank | 512 | 0.00000137 | 0.41947278 | 0.00918491 |
| `layer4.6` | temporal_low_rank | 512 | 0.00000071 | 0.45280218 | 0.00511252 |
| `layer4.10` | temporal_low_rank | 512 | 0.00000038 | 0.54239482 | 0.00354017 |
| `layer5.2` | temporal_low_rank | 512 | 0.00000065 | 0.59621584 | 0.00532279 |
| `layer5.6` | temporal_low_rank | 512 | 0.00000060 | 0.62429059 | 0.00431067 |
| `layer5.10` | temporal_low_rank | 512 | 0.00004803 | 0.47696248 | 0.03414527 |
| `classifier.2` | temporal_low_rank | 4096 | 0.00009862 | 0.28283137 | 0.05978307 |
| `classifier.5` | temporal_low_rank | 4096 | 0.00024660 | 0.54844034 | 0.08744896 |

### T=32

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | temporal_low_rank | 64 | 0.00008631 | 0.74031520 | 0.04157029 |
| `layer1.6` | temporal_low_rank | 64 | 0.00018916 | 0.85646528 | 0.08834864 |
| `layer2.2` | temporal_low_rank | 128 | 0.00003329 | 0.77964079 | 0.04751571 |
| `layer2.6` | temporal_low_rank | 128 | 0.00000504 | 0.56386083 | 0.02572789 |
| `layer3.2` | temporal_low_rank | 256 | 0.00000246 | 0.43549323 | 0.00999077 |
| `layer3.6` | temporal_low_rank | 256 | 0.00000174 | 0.50387198 | 0.00673596 |
| `layer3.10` | temporal_low_rank | 256 | 0.00000127 | 0.46850201 | 0.00642607 |
| `layer4.2` | temporal_low_rank | 512 | 0.00000135 | 0.54668510 | 0.00763997 |
| `layer4.6` | temporal_low_rank | 512 | 0.00000070 | 0.58583486 | 0.00481695 |
| `layer4.10` | temporal_low_rank | 512 | 0.00000034 | 0.66168326 | 0.00313048 |
| `layer5.2` | temporal_low_rank | 512 | 0.00000051 | 0.68953621 | 0.00479884 |
| `layer5.6` | temporal_low_rank | 512 | 0.00000060 | 0.75344455 | 0.00400574 |
| `layer5.10` | temporal_low_rank | 512 | 0.00004487 | 0.59758931 | 0.03393731 |
| `classifier.2` | temporal_low_rank | 4096 | 0.00011823 | 0.41527060 | 0.05918596 |
| `classifier.5` | temporal_low_rank | 4096 | 0.00029228 | 0.70482814 | 0.09517122 |

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
| `layer1.2` | parity_anchor | 64 | 0.00005895 | 0.58055902 | 0.04084072 |
| `layer1.6` | parity_anchor | 64 | 0.00019226 | 0.72833097 | 0.10503355 |
| `layer2.2` | parity_anchor | 128 | 0.00003788 | 0.58349329 | 0.04509082 |
| `layer2.6` | parity_anchor | 128 | 0.00000328 | 0.25372031 | 0.01776535 |
| `layer3.2` | parity_anchor | 256 | 0.00000228 | 0.22230694 | 0.01126232 |
| `layer3.6` | parity_anchor | 256 | 0.00000096 | 0.19639422 | 0.00469999 |
| `layer3.10` | parity_anchor | 256 | 0.00000072 | 0.18598025 | 0.00441720 |
| `layer4.2` | parity_anchor | 512 | 0.00000078 | 0.23402104 | 0.00433054 |
| `layer4.6` | parity_anchor | 512 | 0.00000043 | 0.26249599 | 0.00478156 |
| `layer4.10` | parity_anchor | 512 | 0.00000022 | 0.31155437 | 0.00206025 |
| `layer5.2` | parity_anchor | 512 | 0.00000033 | 0.31302142 | 0.00299525 |
| `layer5.6` | parity_anchor | 512 | 0.00000037 | 0.38975742 | 0.00246481 |
| `layer5.10` | parity_anchor | 512 | 0.00004929 | 0.36604971 | 0.03446434 |
| `classifier.2` | parity_anchor | 4096 | 0.00011677 | 0.22417197 | 0.06871063 |
| `classifier.5` | parity_anchor | 4096 | 0.00020689 | 0.40346453 | 0.08229867 |

### T=16

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | parity_anchor | 64 | 0.00009313 | 0.72820377 | 0.04569613 |
| `layer1.6` | parity_anchor | 64 | 0.00020955 | 0.82453191 | 0.12033794 |
| `layer2.2` | parity_anchor | 128 | 0.00003657 | 0.70214534 | 0.04806599 |
| `layer2.6` | parity_anchor | 128 | 0.00000374 | 0.36606494 | 0.02182779 |
| `layer3.2` | parity_anchor | 256 | 0.00000223 | 0.30385402 | 0.01350719 |
| `layer3.6` | parity_anchor | 256 | 0.00000103 | 0.28276116 | 0.00643030 |
| `layer3.10` | parity_anchor | 256 | 0.00000087 | 0.28340521 | 0.00603703 |
| `layer4.2` | parity_anchor | 512 | 0.00000107 | 0.37045500 | 0.00491864 |
| `layer4.6` | parity_anchor | 512 | 0.00000056 | 0.40208656 | 0.00587720 |
| `layer4.10` | parity_anchor | 512 | 0.00000026 | 0.45422268 | 0.00271553 |
| `layer5.2` | parity_anchor | 512 | 0.00000035 | 0.43712941 | 0.00356910 |
| `layer5.6` | parity_anchor | 512 | 0.00000051 | 0.57315212 | 0.00316904 |
| `layer5.10` | parity_anchor | 512 | 0.00005532 | 0.51189399 | 0.03805263 |
| `classifier.2` | parity_anchor | 4096 | 0.00013813 | 0.33472162 | 0.08301368 |
| `classifier.5` | parity_anchor | 4096 | 0.00027546 | 0.57963771 | 0.08982312 |

### T=32

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `layer1.2` | parity_anchor | 64 | 0.00010484 | 0.81588548 | 0.04664877 |
| `layer1.6` | parity_anchor | 64 | 0.00021361 | 0.91011703 | 0.12509322 |
| `layer2.2` | parity_anchor | 128 | 0.00003595 | 0.81018984 | 0.05464749 |
| `layer2.6` | parity_anchor | 128 | 0.00000361 | 0.47700876 | 0.02511421 |
| `layer3.2` | parity_anchor | 256 | 0.00000198 | 0.39050126 | 0.01462223 |
| `layer3.6` | parity_anchor | 256 | 0.00000093 | 0.36762199 | 0.00688658 |
| `layer3.10` | parity_anchor | 256 | 0.00000085 | 0.38179800 | 0.00647856 |
| `layer4.2` | parity_anchor | 512 | 0.00000116 | 0.50656825 | 0.00510041 |
| `layer4.6` | parity_anchor | 512 | 0.00000059 | 0.53899604 | 0.00585150 |
| `layer4.10` | parity_anchor | 512 | 0.00000026 | 0.58172530 | 0.00279106 |
| `layer5.2` | parity_anchor | 512 | 0.00000033 | 0.55801684 | 0.00364188 |
| `layer5.6` | parity_anchor | 512 | 0.00000055 | 0.71805614 | 0.00405907 |
| `layer5.10` | parity_anchor | 512 | 0.00004824 | 0.61965680 | 0.03802760 |
| `classifier.2` | parity_anchor | 4096 | 0.00013924 | 0.45066181 | 0.09109068 |
| `classifier.5` | parity_anchor | 4096 | 0.00030154 | 0.71589881 | 0.10205035 |
