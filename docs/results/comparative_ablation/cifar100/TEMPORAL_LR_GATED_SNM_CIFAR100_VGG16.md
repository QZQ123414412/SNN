# QCFS Temporal-LR + Gated-SNM CIFAR-100 Experiment

- Architecture: vgg16
- Checkpoint: cifar100-vgg16-l8-example.pth
- Checkpoint SHA256: `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339`
- ANN accuracy: 77.35%
- Time steps: [1, 2, 4, 8, 16, 32]
- Calibration: 5 x 200
- Gate validation: 5 x 200
- Fit data SHA256: `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a`
- Validation data SHA256: `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3`
- SNM is disabled for Full-FTBC teacher calibration.
- Rank and gate margins use calibration-validation data only.
- Runtime state is the existing R0 membrane plus transmitted-credit state; Gated-SNM adds only four FP32 margins (16 bytes), not another dense state.

## Accuracy

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 58.82% | 65.00% | 71.09% | 75.20% | 77.11% | 77.66% |
| B_QCFS_STANDARD_SNM_R0 | 58.82% | 65.00% | 72.29% | 76.59% | 77.49% | 77.50% |
| C_FULL_UNSIGNED_TEACHER | 61.72% | 67.58% | 73.39% | 76.47% | 77.40% | 77.53% |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 61.72% | 67.68% | 74.01% | 77.16% | 77.64% | 77.61% |
| E_TEMPORAL_R4_UNSIGNED | 61.72% | 67.58% | 73.39% | 76.37% | 77.62% | 77.75% |
| F_TEMPORAL_R4_STANDARD_SNM | 61.72% | 67.68% | 74.01% | 77.12% | 77.65% | 77.57% |
| G_TEMPORAL_R4_GATED_SNM | 61.72% | 67.68% | 73.87% | 76.90% | 77.65% | 77.60% |
| H_HYBRID_R4_UNSIGNED | 61.72% | 67.58% | 73.39% | 76.42% | 77.36% | 77.62% |
| I_HYBRID_R4_GATED_SNM | 61.72% | 67.68% | 73.87% | 77.11% | 77.51% | 77.49% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 624,247,119,488 | 1,305,067,150,592 | 2,588,715,654,400 | 5,135,019,932,608 | 10,223,717,533,760 | 20,394,968,227,648 |
| B_QCFS_STANDARD_SNM_R0 | 624,247,119,488 | 1,307,468,667,520 | 2,607,638,985,856 | 5,187,982,392,000 | 10,308,318,464,832 | 20,517,034,551,232 |
| C_FULL_UNSIGNED_TEACHER | 642,073,748,672 | 1,287,142,886,272 | 2,547,186,255,872 | 5,084,745,842,240 | 10,167,219,618,240 | 20,328,287,895,616 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,130,873,717,056 | 10,243,080,312,896 | 20,441,318,701,888 |
| E_TEMPORAL_R4_UNSIGNED | 642,073,748,672 | 1,287,142,886,272 | 2,547,186,255,872 | 5,107,622,627,008 | 10,212,382,280,256 | 20,398,244,034,944 |
| F_TEMPORAL_R4_STANDARD_SNM | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,154,410,295,232 | 10,289,123,421,376 | 20,513,091,425,792 |
| G_TEMPORAL_R4_GATED_SNM | 642,073,748,672 | 1,289,084,024,192 | 2,560,047,383,168 | 5,149,221,393,600 | 10,289,123,421,376 | 20,513,025,635,840 |
| H_HYBRID_R4_UNSIGNED | 642,073,748,672 | 1,287,142,886,272 | 2,547,186,255,872 | 5,093,907,679,360 | 10,195,549,022,272 | 20,372,241,305,408 |
| I_HYBRID_R4_GATED_SNM | 642,073,748,672 | 1,289,084,024,192 | 2,560,047,383,168 | 5,135,523,809,664 | 10,272,082,167,488 | 20,485,362,427,712 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 14.5549% | 15.1563% | 15.1509% | 15.0818% | 15.0395% | 15.0160% |
| B_QCFS_STANDARD_SNM_R0 | 14.5549% | 15.1678% | 15.1733% | 15.1048% | 15.0551% | 15.0283% |
| C_FULL_UNSIGNED_TEACHER | 14.9661% | 15.1298% | 14.9916% | 14.9721% | 14.9686% | 14.9666% |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 14.9661% | 15.1389% | 15.0107% | 14.9950% | 14.9860% | 14.9803% |
| E_TEMPORAL_R4_UNSIGNED | 14.9661% | 15.1298% | 14.9916% | 15.0295% | 15.0290% | 15.0145% |
| F_TEMPORAL_R4_STANDARD_SNM | 14.9661% | 15.1389% | 15.0107% | 15.0530% | 15.0461% | 15.0280% |
| G_TEMPORAL_R4_GATED_SNM | 14.9661% | 15.1389% | 15.0075% | 15.0510% | 15.0461% | 15.0280% |
| H_HYBRID_R4_UNSIGNED | 14.9661% | 15.1298% | 14.9916% | 15.0009% | 15.0166% | 15.0063% |
| I_HYBRID_R4_GATED_SNM | 14.9661% | 15.1389% | 15.0075% | 15.0225% | 15.0339% | 15.0195% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0105% | 0.0345% | 0.0469% | 0.0401% | 0.0297% |
| C_FULL_UNSIGNED_TEACHER | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 0.0000% | 0.0086% | 0.0287% | 0.0391% | 0.0339% | 0.0259% |
| E_TEMPORAL_R4_UNSIGNED | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| F_TEMPORAL_R4_STANDARD_SNM | 0.0000% | 0.0086% | 0.0287% | 0.0398% | 0.0348% | 0.0267% |
| G_TEMPORAL_R4_GATED_SNM | 0.0000% | 0.0086% | 0.0174% | 0.0351% | 0.0348% | 0.0266% |
| H_HYBRID_R4_UNSIGNED | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| I_HYBRID_R4_GATED_SNM | 0.0000% | 0.0086% | 0.0174% | 0.0350% | 0.0343% | 0.0262% |

## Overall sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 85.4451% | 84.8437% | 84.8491% | 84.9182% | 84.9605% | 84.9840% |
| B_QCFS_STANDARD_SNM_R0 | 85.4451% | 84.8218% | 84.7922% | 84.8483% | 84.9048% | 84.9421% |
| C_FULL_UNSIGNED_TEACHER | 85.0339% | 84.8702% | 85.0084% | 85.0279% | 85.0314% | 85.0334% |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 85.0339% | 84.8525% | 84.9606% | 84.9659% | 84.9801% | 84.9937% |
| E_TEMPORAL_R4_UNSIGNED | 85.0339% | 84.8702% | 85.0084% | 84.9705% | 84.9710% | 84.9855% |
| F_TEMPORAL_R4_STANDARD_SNM | 85.0339% | 84.8525% | 84.9606% | 84.9072% | 84.9191% | 84.9453% |
| G_TEMPORAL_R4_GATED_SNM | 85.0339% | 84.8525% | 84.9751% | 84.9138% | 84.9191% | 84.9454% |
| H_HYBRID_R4_UNSIGNED | 85.0339% | 84.8702% | 85.0084% | 84.9991% | 84.9834% | 84.9937% |
| I_HYBRID_R4_GATED_SNM | 85.0339% | 84.8525% | 84.9751% | 84.9425% | 84.9318% | 84.9542% |

## FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_FULL_UNSIGNED_TEACHER | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| E_TEMPORAL_R4_UNSIGNED | 12,416 | 24,832 | 49,664 | 49,696 | 49,728 | 49,792 |
| F_TEMPORAL_R4_STANDARD_SNM | 12,416 | 24,832 | 49,664 | 49,696 | 49,728 | 49,792 |
| G_TEMPORAL_R4_GATED_SNM | 12,416 | 24,832 | 49,664 | 49,696 | 49,728 | 49,792 |
| H_HYBRID_R4_UNSIGNED | 12,416 | 24,832 | 49,664 | 66,080 | 98,880 | 164,480 |
| I_HYBRID_R4_GATED_SNM | 12,416 | 24,832 | 49,664 | 66,080 | 98,880 | 164,480 |

## FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_FULL_UNSIGNED_TEACHER | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| E_TEMPORAL_R4_UNSIGNED | 49,664 | 99,328 | 198,656 | 198,784 | 198,912 | 199,168 |
| F_TEMPORAL_R4_STANDARD_SNM | 49,664 | 99,328 | 198,656 | 198,784 | 198,912 | 199,168 |
| G_TEMPORAL_R4_GATED_SNM | 49,664 | 99,328 | 198,656 | 198,784 | 198,912 | 199,168 |
| H_HYBRID_R4_UNSIGNED | 49,664 | 99,328 | 198,656 | 264,320 | 395,520 | 657,920 |
| I_HYBRID_R4_GATED_SNM | 49,664 | 99,328 | 198,656 | 264,320 | 395,520 | 657,920 |

## Temporal bias synthesis MACs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_FULL_UNSIGNED_TEACHER | 0 | 0 | 0 | 0 | 0 | 0 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 0 | 0 | 0 | 0 | 0 | 0 |
| E_TEMPORAL_R4_UNSIGNED | 0 | 0 | 0 | 397,312 | 794,624 | 1,589,248 |
| F_TEMPORAL_R4_STANDARD_SNM | 0 | 0 | 0 | 397,312 | 794,624 | 1,589,248 |
| G_TEMPORAL_R4_GATED_SNM | 0 | 0 | 0 | 397,312 | 794,624 | 1,589,248 |
| H_HYBRID_R4_UNSIGNED | 0 | 0 | 0 | 266,240 | 532,480 | 1,064,960 |
| I_HYBRID_R4_GATED_SNM | 0 | 0 | 0 | 266,240 | 532,480 | 1,064,960 |

## SNM gate parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_FULL_UNSIGNED_TEACHER | 0 | 0 | 0 | 0 | 0 | 0 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 0 | 0 | 0 | 0 | 0 | 0 |
| E_TEMPORAL_R4_UNSIGNED | 0 | 0 | 0 | 0 | 0 | 0 |
| F_TEMPORAL_R4_STANDARD_SNM | 0 | 0 | 0 | 0 | 0 | 0 |
| G_TEMPORAL_R4_GATED_SNM | 4 | 4 | 4 | 4 | 4 | 4 |
| H_HYBRID_R4_UNSIGNED | 0 | 0 | 0 | 0 | 0 | 0 |
| I_HYBRID_R4_GATED_SNM | 4 | 4 | 4 | 4 | 4 | 4 |

## R0 neuron runtime state bytes per sample

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 |
| B_QCFS_STANDARD_SNM_R0 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 |
| C_FULL_UNSIGNED_TEACHER | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 |
| E_TEMPORAL_R4_UNSIGNED | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 |
| F_TEMPORAL_R4_STANDARD_SNM | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 |
| G_TEMPORAL_R4_GATED_SNM | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 |
| H_HYBRID_R4_UNSIGNED | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 |
| I_HYBRID_R4_GATED_SNM | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 | 2,277,376 |

## Full-teacher calibration elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s |
| B_QCFS_STANDARD_SNM_R0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s |
| C_FULL_UNSIGNED_TEACHER | 1.5s | 2.5s | 4.3s | 8.0s | 15.4s | 59.8s |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 1.5s | 2.5s | 4.3s | 8.0s | 15.4s | 59.8s |
| E_TEMPORAL_R4_UNSIGNED | 1.5s | 2.5s | 4.3s | 8.0s | 15.4s | 59.8s |
| F_TEMPORAL_R4_STANDARD_SNM | 1.5s | 2.5s | 4.3s | 8.0s | 15.4s | 59.8s |
| G_TEMPORAL_R4_GATED_SNM | 1.5s | 2.5s | 4.3s | 8.0s | 15.4s | 59.8s |
| H_HYBRID_R4_UNSIGNED | 1.5s | 2.5s | 4.3s | 8.0s | 15.4s | 59.8s |
| I_HYBRID_R4_GATED_SNM | 1.5s | 2.5s | 4.3s | 8.0s | 15.4s | 59.8s |

## Temporal compression elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_FULL_UNSIGNED_TEACHER | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| E_TEMPORAL_R4_UNSIGNED | 0.000s | 0.000s | 0.000s | 0.018s | 0.022s | 0.034s |
| F_TEMPORAL_R4_STANDARD_SNM | 0.000s | 0.000s | 0.000s | 0.016s | 0.019s | 0.032s |
| G_TEMPORAL_R4_GATED_SNM | 0.000s | 0.000s | 0.000s | 0.017s | 0.021s | 0.030s |
| H_HYBRID_R4_UNSIGNED | 0.000s | 0.000s | 0.000s | 0.017s | 0.023s | 0.031s |
| I_HYBRID_R4_GATED_SNM | 0.000s | 0.000s | 0.000s | 0.017s | 0.023s | 0.030s |

## Inference elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.2s | 1.9s | 3.1s | 5.5s | 10.3s | 37.4s |
| B_QCFS_STANDARD_SNM_R0 | 1.3s | 2.0s | 3.3s | 5.9s | 11.1s | 21.8s |
| C_FULL_UNSIGNED_TEACHER | 1.2s | 1.8s | 3.0s | 5.3s | 9.9s | 19.3s |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 1.2s | 1.9s | 3.2s | 5.7s | 10.8s | 21.0s |
| E_TEMPORAL_R4_UNSIGNED | 1.2s | 1.8s | 3.0s | 5.4s | 10.0s | 19.5s |
| F_TEMPORAL_R4_STANDARD_SNM | 1.2s | 1.9s | 3.2s | 5.8s | 10.8s | 21.2s |
| G_TEMPORAL_R4_GATED_SNM | 1.2s | 1.9s | 3.2s | 5.8s | 11.0s | 21.3s |
| H_HYBRID_R4_UNSIGNED | 1.2s | 1.8s | 3.0s | 5.4s | 10.2s | 19.5s |
| I_HYBRID_R4_GATED_SNM | 1.2s | 2.0s | 3.2s | 5.8s | 10.9s | 21.2s |

## Rank screen on calibration validation

### T=1

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 57.90% | 10.174579 | 1.000000 | 49,668 |
| 4 | 1 | 57.90% | 10.174579 | 1.000000 | 49,668 |
| 6 | 1 | 57.90% | 10.174579 | 1.000000 | 49,668 |

### T=2

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 67.70% | 3.698648 | 1.000000 | 99,344 |
| 4 | 2 | 67.70% | 3.698648 | 1.000000 | 99,344 |
| 6 | 2 | 67.70% | 3.698648 | 1.000000 | 99,344 |

### T=4

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 74.40% | 1.629833 | 0.952338 | 99,360 |
| 4 | 4 | 77.80% | 1.360926 | 1.000000 | 198,720 |
| 6 | 4 | 77.80% | 1.360926 | 1.000000 | 198,720 |

### T=8

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 82.90% | 0.660861 | 0.894611 | 99,392 |
| 4 | 4 | 83.80% | 0.571045 | 0.948015 | 198,784 |
| 6 | 6 | 85.00% | 0.477182 | 0.981738 | 298,176 |

### T=16

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 87.40% | 0.218863 | 0.800413 | 99,456 |
| 4 | 4 | 87.40% | 0.193159 | 0.855806 | 198,912 |
| 6 | 6 | 87.50% | 0.196330 | 0.896212 | 298,368 |

### T=32

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 88.40% | 0.103066 | 0.669229 | 99,584 |
| 4 | 4 | 88.30% | 0.106001 | 0.720206 | 199,168 |
| 6 | 6 | 88.00% | 0.094303 | 0.758509 | 298,752 |

## Selected SNM margins

| T | Early | Middle | Late | Final | Baseline val acc. | Gated val acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 57.90% | 57.90% |
| 2 | 0.0 | 0.0 | 0.0 | 0.0 | 68.20% | 68.20% |
| 4 | 2.0 | 0.0 | 0.0 | 0.0 | 78.10% | 79.20% |
| 8 | 0.0 | 0.5 | 0.0 | 0.0 | 86.70% | 87.10% |
| 16 | 0.0 | 0.0 | 0.0 | 0.0 | 88.00% | 88.00% |
| 32 | 0.0 | 0.0 | 0.0 | 2.0 | 88.50% | 88.40% |
