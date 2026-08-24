# QCFS + Full-FTBC + Accuracy-Gated SNM CIFAR-100 Ablation

- Status: complete
- Architecture: vgg16
- Checkpoint: `cifar100-vgg16-l8-example.pth`
- Checkpoint SHA256: `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339`
- ANN accuracy on the 10,000-image test set: 77.35%
- Time steps: [1, 2, 4, 8, 16, 32]
- Full-FTBC fit: 5 x 200, alpha=0.4
- A-SNM validation: 5 x 200
- Fit batch SHA256: `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a`
- Validation batch SHA256: `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-100 normalization, and Cutout(1,16).
- The test loader uses only ToTensor and normalization, with shuffle=False.
- Every SNN uses QCFS L=8, rate coding, rate schedule, ratio=1.0, R0=True, FP32.
- Full-FTBC is independently fitted for every T with SNM off and frozen before validation/test.
- A-SNM independently enables SNM at each T only when SNM-on has strictly higher validation accuracy; ties select off.
- A-SNM changes only the standard SNM on/off state, uses margin=0, and stores one frozen Boolean per evaluated T.
- Test images are first accessed after both families' A-SNM decisions are frozen.

## Primary accuracy table

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | SNM-on T |
|---|---:|---:|---:|---:|---:|---:|---|
| A_QCFS_R0 | 58.82% | 65.00% | 71.09% | 75.20% | 77.11% | 77.66% | none |
| B_QCFS_STANDARD_SNM_R0 | 58.82% | 65.00% | 72.29% | 76.59% | 77.49% | 77.50% | 1, 2, 4, 8, 16, 32 |
| C_QCFS_ASNM_R0 | 58.82% | 65.00% | 72.29% | 76.59% | 77.49% | 77.66% | 2, 4, 8, 16 |
| D_QCFS_FULL_FTBC_R0 | 61.72% | 67.58% | 73.39% | 76.47% | 77.40% | 77.53% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 61.72% | 67.68% | 74.01% | 77.16% | 77.64% | 77.61% | 1, 2, 4, 8, 16, 32 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 61.72% | 67.68% | 74.01% | 77.16% | 77.64% | 77.61% | 2, 4, 8, 16, 32 |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 6.338912 | 4.172804 | 1.728001 | 0.643336 | 0.215314 | 0.099278 |
| B_QCFS_STANDARD_SNM_R0 | 6.338912 | 4.096355 | 1.444703 | 0.337313 | 0.099956 | 0.072180 |
| C_QCFS_ASNM_R0 | 6.338912 | 4.096355 | 1.444703 | 0.337313 | 0.099956 | 0.099278 |
| D_QCFS_FULL_FTBC_R0 | 11.702841 | 3.599602 | 1.249051 | 0.445854 | 0.163634 | 0.088813 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 11.702841 | 3.537999 | 1.052037 | 0.255550 | 0.096631 | 0.072068 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 11.702841 | 3.537999 | 1.052037 | 0.255550 | 0.096631 | 0.072068 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 14.5549% | 15.1563% | 15.1509% | 15.0818% | 15.0395% | 15.0160% |
| B_QCFS_STANDARD_SNM_R0 | 14.5549% | 15.1678% | 15.1733% | 15.1048% | 15.0551% | 15.0283% |
| C_QCFS_ASNM_R0 | 14.5549% | 15.1678% | 15.1733% | 15.1048% | 15.0551% | 15.0160% |
| D_QCFS_FULL_FTBC_R0 | 14.9661% | 15.1298% | 14.9916% | 14.9721% | 14.9686% | 14.9666% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 14.9661% | 15.1389% | 15.0107% | 14.9950% | 14.9860% | 14.9803% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 14.9661% | 15.1389% | 15.0107% | 14.9950% | 14.9860% | 14.9803% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0105% | 0.0345% | 0.0469% | 0.0401% | 0.0297% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0105% | 0.0345% | 0.0469% | 0.0401% | 0.0000% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0086% | 0.0287% | 0.0391% | 0.0339% | 0.0259% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0086% | 0.0287% | 0.0391% | 0.0339% | 0.0259% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 85.4451% | 84.8437% | 84.8491% | 84.9182% | 84.9605% | 84.9840% |
| B_QCFS_STANDARD_SNM_R0 | 85.4451% | 84.8218% | 84.7922% | 84.8483% | 84.9048% | 84.9421% |
| C_QCFS_ASNM_R0 | 85.4451% | 84.8218% | 84.7922% | 84.8483% | 84.9048% | 84.9840% |
| D_QCFS_FULL_FTBC_R0 | 85.0339% | 84.8702% | 85.0084% | 85.0279% | 85.0314% | 85.0334% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 85.0339% | 84.8525% | 84.9606% | 84.9659% | 84.9801% | 84.9937% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 85.0339% | 84.8525% | 84.9606% | 84.9659% | 84.9801% | 84.9937% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 624,247,119,488 | 1,305,067,150,592 | 2,588,715,654,400 | 5,135,019,932,608 | 10,223,717,533,760 | 20,394,968,227,648 |
| B_QCFS_STANDARD_SNM_R0 | 624,247,119,488 | 1,307,468,667,520 | 2,607,638,985,856 | 5,187,982,392,000 | 10,308,318,464,832 | 20,517,034,551,232 |
| C_QCFS_ASNM_R0 | 624,247,119,488 | 1,307,468,667,520 | 2,607,638,985,856 | 5,187,982,392,000 | 10,308,318,464,832 | 20,394,968,227,648 |
| D_QCFS_FULL_FTBC_R0 | 642,073,748,672 | 1,287,142,886,272 | 2,547,186,255,872 | 5,084,745,842,240 | 10,167,219,618,240 | 20,328,287,895,616 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,130,873,717,056 | 10,243,080,312,896 | 20,441,318,701,888 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 642,073,748,672 | 1,289,084,024,192 | 2,563,266,864,000 | 5,130,873,717,056 | 10,243,080,312,896 | 20,441,318,701,888 |

## Full-FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |

## Full-FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |

## Full-FTBC calibration elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 1.396s | 2.260s | 3.962s | 7.418s | 14.055s | 30.038s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.396s | 2.260s | 3.962s | 7.418s | 14.055s | 30.038s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.396s | 2.260s | 3.962s | 7.418s | 14.055s | 30.038s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.288s | 1.859s | 2.977s | 5.176s | 9.483s | 18.256s |
| B_QCFS_STANDARD_SNM_R0 | 1.327s | 1.960s | 3.162s | 5.565s | 10.332s | 19.837s |
| C_QCFS_ASNM_R0 | 1.288s | 1.960s | 3.162s | 5.565s | 10.332s | 18.256s |
| D_QCFS_FULL_FTBC_R0 | 1.219s | 1.803s | 2.869s | 4.996s | 9.172s | 17.576s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.265s | 1.888s | 3.045s | 5.351s | 9.943s | 19.176s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.219s | 1.888s | 3.045s | 5.351s | 9.943s | 19.176s |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 2, 4, 8, 16 | 9.596s |
| Full-FTBC | 2, 4, 8, 16, 32 | 9.537s |

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
| F_QCFS_FULL_FTBC_ASNM_R0 | 16 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 32 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
