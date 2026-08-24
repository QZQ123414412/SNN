# QCFS + Full-FTBC + Temporal-LR FTBC + A-SNM CIFAR-100 Ablation

- Status: complete
- Architecture: vgg16
- Checkpoint: `cifar100-vgg16-l8-example.pth`
- Checkpoint SHA256: `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339`
- ANN accuracy on the 10,000-image test set: 77.35%
- Time steps: [1, 2, 4, 8, 16, 32]
- Full-FTBC fit: 5 x 200, alpha=0.4
- A-SNM validation: 5 x 200
- Temporal-LR: shared rank-4 basis, threshold-normalized, no exempt layer.
- Temporal-LR falls back to Full-FTBC at T<=4 and is active at T>4.
- Fit batch SHA256: `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a`
- Validation batch SHA256: `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-100 normalization, and Cutout(1,16).
- Test uses only ToTensor and normalization with shuffle=False.
- Every SNN uses QCFS L=8, rate coding/schedule, ratio=1.0, R0=True, SNM margin=0, FP32.
- Full-FTBC is independently fitted at every T with SNM off; Temporal-LR is compressed from that frozen teacher.
- Each family enables SNM only when SNM-on has strictly higher validation accuracy; ties select off.
- Test data is first accessed after all three A-SNM families are frozen.
- A-SNM guarantees validation-set selection only; test-set reversals are reported diagnostically and never retuned.

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

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 6.338912 | 4.172804 | 1.728001 | 0.643336 | 0.215314 | 0.099278 |
| B_QCFS_STANDARD_SNM_R0 | 6.338912 | 4.096355 | 1.444703 | 0.337313 | 0.099956 | 0.072180 |
| C_QCFS_ASNM_R0 | 6.338912 | 4.096355 | 1.444703 | 0.337313 | 0.099956 | 0.099278 |
| D_QCFS_FULL_FTBC_R0 | 11.702841 | 3.599602 | 1.249051 | 0.445854 | 0.163634 | 0.088813 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 11.702841 | 3.537999 | 1.052037 | 0.255550 | 0.096631 | 0.072068 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 11.702841 | 3.537999 | 1.052037 | 0.255550 | 0.096631 | 0.072068 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 11.702841 | 3.599602 | 1.249051 | 0.457684 | 0.171054 | 0.091024 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 11.702841 | 3.537999 | 1.052037 | 0.257810 | 0.095223 | 0.071095 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 11.702841 | 3.537999 | 1.052037 | 0.257810 | 0.095223 | 0.071095 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 14.5549% | 15.1563% | 15.1509% | 15.0818% | 15.0395% | 15.0160% |
| B_QCFS_STANDARD_SNM_R0 | 14.5549% | 15.1678% | 15.1733% | 15.1048% | 15.0551% | 15.0283% |
| C_QCFS_ASNM_R0 | 14.5549% | 15.1678% | 15.1733% | 15.1048% | 15.0551% | 15.0160% |
| D_QCFS_FULL_FTBC_R0 | 14.9661% | 15.1298% | 14.9916% | 14.9721% | 14.9686% | 14.9666% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 14.9661% | 15.1389% | 15.0107% | 14.9950% | 14.9860% | 14.9803% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 14.9661% | 15.1389% | 15.0107% | 14.9950% | 14.9860% | 14.9803% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 14.9661% | 15.1298% | 14.9916% | 15.0295% | 15.0290% | 15.0145% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 14.9661% | 15.1389% | 15.0107% | 15.0530% | 15.0461% | 15.0280% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 14.9661% | 15.1389% | 15.0107% | 15.0530% | 15.0461% | 15.0280% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0105% | 0.0345% | 0.0469% | 0.0401% | 0.0297% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0105% | 0.0345% | 0.0469% | 0.0401% | 0.0000% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0086% | 0.0287% | 0.0391% | 0.0339% | 0.0259% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0086% | 0.0287% | 0.0391% | 0.0339% | 0.0259% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0086% | 0.0287% | 0.0398% | 0.0348% | 0.0267% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.0000% | 0.0086% | 0.0287% | 0.0398% | 0.0348% | 0.0267% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 85.4451% | 84.8437% | 84.8491% | 84.9182% | 84.9605% | 84.9840% |
| B_QCFS_STANDARD_SNM_R0 | 85.4451% | 84.8218% | 84.7922% | 84.8483% | 84.9048% | 84.9421% |
| C_QCFS_ASNM_R0 | 85.4451% | 84.8218% | 84.7922% | 84.8483% | 84.9048% | 84.9840% |
| D_QCFS_FULL_FTBC_R0 | 85.0339% | 84.8702% | 85.0084% | 85.0279% | 85.0314% | 85.0334% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 85.0339% | 84.8525% | 84.9606% | 84.9659% | 84.9801% | 84.9937% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 85.0339% | 84.8525% | 84.9606% | 84.9659% | 84.9801% | 84.9937% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 85.0339% | 84.8702% | 85.0084% | 84.9705% | 84.9710% | 84.9855% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 85.0339% | 84.8525% | 84.9606% | 84.9072% | 84.9191% | 84.9453% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 85.0339% | 84.8525% | 84.9606% | 84.9072% | 84.9191% | 84.9453% |

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

## Temporal bias synthesis MACs

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

## Full-teacher calibration elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 1.691s | 2.722s | 4.753s | 9.006s | 17.477s | 67.222s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.691s | 2.722s | 4.753s | 9.006s | 17.477s | 67.222s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.691s | 2.722s | 4.753s | 9.006s | 17.477s | 67.222s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.691s | 2.722s | 4.753s | 9.006s | 17.477s | 67.222s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.691s | 2.722s | 4.753s | 9.006s | 17.477s | 67.222s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.691s | 2.722s | 4.753s | 9.006s | 17.477s | 67.222s |

## Temporal compression elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000s | 0.000s | 0.000s | 0.022s | 0.026s | 0.028s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.022s | 0.026s | 0.028s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.022s | 0.026s | 0.028s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.662s | 2.174s | 3.586s | 6.274s | 11.530s | 22.924s |
| B_QCFS_STANDARD_SNM_R0 | 1.547s | 2.329s | 3.889s | 6.779s | 12.517s | 24.699s |
| C_QCFS_ASNM_R0 | 1.662s | 2.329s | 3.889s | 6.779s | 12.517s | 22.924s |
| D_QCFS_FULL_FTBC_R0 | 1.398s | 2.099s | 3.417s | 6.137s | 11.125s | 21.499s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.449s | 2.199s | 3.726s | 6.551s | 12.055s | 25.665s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.398s | 2.199s | 3.726s | 6.551s | 12.055s | 25.665s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.359s | 2.087s | 3.501s | 6.710s | 11.206s | 24.733s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.431s | 2.213s | 3.822s | 7.138s | 12.222s | 27.392s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.359s | 2.213s | 3.822s | 7.138s | 12.222s | 27.392s |

## Temporal-LR compression

| T | Effective mode | Effective rank | Explained energy | Full parameters | Temporal parameters | Storage ratio | Storage reduction | Synthesis MACs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | full fallback | 1 | 1.000000 | 12,416 | 12,416 | 1.000000 | 0.00% | 0 |
| 2 | full fallback | 2 | 1.000000 | 24,832 | 24,832 | 1.000000 | 0.00% | 0 |
| 4 | full fallback | 4 | 1.000000 | 49,664 | 49,664 | 1.000000 | 0.00% | 0 |
| 8 | temporal_low_rank | 4 | 0.948015 | 99,328 | 49,696 | 0.500322 | 49.97% | 397,312 |
| 16 | temporal_low_rank | 4 | 0.855806 | 198,656 | 49,728 | 0.250322 | 74.97% | 794,624 |
| 32 | temporal_low_rank | 4 | 0.720206 | 397,312 | 49,792 | 0.125322 | 87.47% | 1,589,248 |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 2, 4, 8, 16 | 11.798s |
| Full-FTBC | 2, 4, 8, 16, 32 | 15.116s |
| Temporal-LR FTBC | 2, 4, 8, 16, 32 | 14.983s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 53.6000% | 53.6000% | +0.0000pp | off |
| 2 | 65.0000% | 65.9000% | +0.9000pp | on |
| 4 | 75.6000% | 77.7000% | +2.1000pp | on |
| 8 | 84.9000% | 86.8000% | +1.9000pp | on |
| 16 | 88.1000% | 88.3000% | +0.2000pp | on |
| 32 | 88.7000% | 87.9000% | -0.8000pp | off |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 57.9000% | 57.9000% | +0.0000pp | off |
| 2 | 67.7000% | 68.2000% | +0.5000pp | on |
| 4 | 77.8000% | 78.1000% | +0.3000pp | on |
| 8 | 83.8000% | 87.4000% | +3.6000pp | on |
| 16 | 87.3000% | 88.4000% | +1.1000pp | on |
| 32 | 87.9000% | 88.1000% | +0.2000pp | on |

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 57.9000% | 57.9000% | +0.0000pp | off |
| 2 | 67.7000% | 68.2000% | +0.5000pp | on |
| 4 | 77.8000% | 78.1000% | +0.3000pp | on |
| 8 | 83.8000% | 86.7000% | +2.9000pp | on |
| 16 | 87.4000% | 88.0000% | +0.6000pp | on |
| 32 | 88.3000% | 88.5000% | +0.2000pp | on |

## Validation-selection generalization audit

This table is diagnostic only. Test accuracy never changes a frozen A-SNM decision.

| Family | T | Selected from validation | Test off | Test on | Test-best mode | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 1 | off | 58.82% | 58.82% | off | yes |
| Full-FTBC | 1 | off | 61.72% | 61.72% | off | yes |
| Temporal-LR FTBC | 1 | off | 61.72% | 61.72% | off | yes |
| QCFS | 2 | on | 65.00% | 65.00% | off | no |
| Full-FTBC | 2 | on | 67.58% | 67.68% | on | yes |
| Temporal-LR FTBC | 2 | on | 67.58% | 67.68% | on | yes |
| QCFS | 4 | on | 71.09% | 72.29% | on | yes |
| Full-FTBC | 4 | on | 73.39% | 74.01% | on | yes |
| Temporal-LR FTBC | 4 | on | 73.39% | 74.01% | on | yes |
| QCFS | 8 | on | 75.20% | 76.59% | on | yes |
| Full-FTBC | 8 | on | 76.47% | 77.16% | on | yes |
| Temporal-LR FTBC | 8 | on | 76.37% | 77.12% | on | yes |
| QCFS | 16 | on | 77.11% | 77.49% | on | yes |
| Full-FTBC | 16 | on | 77.40% | 77.64% | on | yes |
| Temporal-LR FTBC | 16 | on | 77.62% | 77.65% | on | yes |
| QCFS | 32 | off | 77.66% | 77.50% | off | yes |
| Full-FTBC | 32 | on | 77.53% | 77.61% | on | yes |
| Temporal-LR FTBC | 32 | on | 77.75% | 77.57% | off | no |

## Equivalence checks

| Kind | Config/family | T | Expected source | Exact |
|---|---|---:|---|---|
| validation fallback | off:full=temporal | 1 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 1 | Full-FTBC on | yes |
| validation fallback | off:full=temporal | 2 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 2 | Full-FTBC on | yes |
| validation fallback | off:full=temporal | 4 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 4 | Full-FTBC on | yes |
| gate fallback | full=temporal | 1 | identical validation metrics | yes |
| gate fallback | full=temporal | 2 | identical validation metrics | yes |
| gate fallback | full=temporal | 4 | identical validation metrics | yes |
| test fallback | off:full=temporal | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 1 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1 | G_QCFS_TEMPORAL_LR_FTBC_R0 | yes |
| test fallback | off:full=temporal | 2 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| test fallback | off:full=temporal | 4 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 4 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 4 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 8 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 8 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 8 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 16 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 16 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 32 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 32 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |

## Per-layer Temporal-LR reconstruction

### T=1

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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
