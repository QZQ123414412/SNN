# QCFS + Full-FTBC + Accuracy-Gated SNM CIFAR-100 Ablation

- Status: complete
- Architecture: resnet20
- Checkpoint: `resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth`
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- ANN accuracy on the 10,000-image test set: 68.68%
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
| A_QCFS_R0 | 7.93% | 11.66% | 22.42% | 46.37% | 64.03% | 68.78% | none |
| B_QCFS_STANDARD_SNM_R0 | 7.93% | 12.04% | 25.41% | 57.27% | 66.52% | 69.00% | 1, 2, 4, 8, 16, 32 |
| C_QCFS_ASNM_R0 | 7.93% | 11.66% | 25.41% | 57.27% | 66.52% | 69.00% | 4, 8, 16, 32 |
| D_QCFS_FULL_FTBC_R0 | 14.66% | 22.69% | 38.81% | 59.15% | 67.58% | 69.37% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 14.66% | 22.92% | 40.82% | 60.50% | 67.43% | 68.50% | 1, 2, 4, 8, 16, 32 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 14.66% | 22.92% | 40.82% | 60.50% | 67.43% | 68.50% | 2, 4, 8, 16, 32 |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 17.060971 | 13.857015 | 9.981356 | 5.256991 | 1.980072 | 0.713141 |
| B_QCFS_STANDARD_SNM_R0 | 17.060971 | 13.665451 | 8.934425 | 3.276014 | 1.072969 | 0.396984 |
| C_QCFS_ASNM_R0 | 17.060971 | 13.857015 | 8.934425 | 3.276014 | 1.072969 | 0.396984 |
| D_QCFS_FULL_FTBC_R0 | 11.704163 | 8.864907 | 5.987022 | 2.978636 | 1.153339 | 0.473841 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 11.704163 | 8.808012 | 5.633623 | 2.561259 | 0.864456 | 0.368019 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 11.704163 | 8.808012 | 5.633623 | 2.561259 | 0.864456 | 0.368019 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 21.7705% | 22.3887% | 21.5543% | 20.9977% | 20.7175% | 20.5833% |
| B_QCFS_STANDARD_SNM_R0 | 21.7705% | 22.6098% | 21.9433% | 21.5252% | 21.2464% | 21.0660% |
| C_QCFS_ASNM_R0 | 21.7705% | 22.3887% | 21.9433% | 21.5252% | 21.2464% | 21.0660% |
| D_QCFS_FULL_FTBC_R0 | 21.1393% | 20.3062% | 20.1176% | 20.1557% | 20.3057% | 20.3607% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 21.1393% | 20.4988% | 20.5168% | 20.6697% | 20.8247% | 20.8403% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 21.1393% | 20.4988% | 20.5168% | 20.6697% | 20.8247% | 20.8403% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.1214% | 0.2756% | 0.4826% | 0.5466% | 0.5036% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0000% | 0.2756% | 0.4826% | 0.5466% | 0.5036% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0820% | 0.2394% | 0.3983% | 0.4626% | 0.4568% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0820% | 0.2394% | 0.3983% | 0.4626% | 0.4568% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 78.2295% | 77.6113% | 78.4457% | 79.0023% | 79.2825% | 79.4167% |
| B_QCFS_STANDARD_SNM_R0 | 78.2295% | 77.2688% | 77.7811% | 77.9922% | 78.2070% | 78.4304% |
| C_QCFS_ASNM_R0 | 78.2295% | 77.6113% | 77.7811% | 77.9922% | 78.2070% | 78.4304% |
| D_QCFS_FULL_FTBC_R0 | 78.8607% | 79.6938% | 79.8824% | 79.8443% | 79.6943% | 79.6393% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 78.8607% | 79.4192% | 79.2439% | 78.9320% | 78.7127% | 78.7029% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 78.8607% | 79.4192% | 79.2439% | 78.9320% | 78.7127% | 78.7029% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 106,203,817,528 | 214,969,665,148 | 411,274,195,716 | 795,974,261,116 | 1,566,147,585,492 | 3,108,635,563,864 |
| B_QCFS_STANDARD_SNM_R0 | 106,203,817,528 | 217,403,594,480 | 423,540,233,072 | 839,558,553,528 | 1,658,935,588,120 | 3,271,706,702,556 |
| C_QCFS_ASNM_R0 | 106,203,817,528 | 214,969,665,148 | 423,540,233,072 | 839,558,553,528 | 1,658,935,588,120 | 3,271,706,702,556 |
| D_QCFS_FULL_FTBC_R0 | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 754,678,672,336 | 1,525,854,968,124 | 3,067,405,523,052 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,838,738,384 | 1,606,584,487,312 | 3,216,935,311,060 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,838,738,384 | 1,606,584,487,312 | 3,216,935,311,060 |

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
| D_QCFS_FULL_FTBC_R0 | 0.903s | 1.200s | 2.335s | 4.372s | 8.738s | 16.702s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.903s | 1.200s | 2.335s | 4.372s | 8.738s | 16.702s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.903s | 1.200s | 2.335s | 4.372s | 8.738s | 16.702s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.111s | 1.234s | 1.828s | 2.926s | 4.983s | 8.456s |
| B_QCFS_STANDARD_SNM_R0 | 1.125s | 1.337s | 1.970s | 3.198s | 5.631s | 9.559s |
| C_QCFS_ASNM_R0 | 1.111s | 1.234s | 1.970s | 3.198s | 5.631s | 9.559s |
| D_QCFS_FULL_FTBC_R0 | 1.084s | 1.236s | 1.776s | 2.812s | 5.002s | 8.346s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.163s | 1.351s | 1.869s | 3.100s | 5.448s | 9.185s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.084s | 1.351s | 1.869s | 3.100s | 5.448s | 9.185s |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 4, 8, 16, 32 | 5.502s |
| Full-FTBC | 2, 4, 8, 16, 32 | 5.085s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 4.7000% | 4.7000% | +0.0000pp | off |
| 2 | 6.4000% | 6.1000% | -0.3000pp | off |
| 4 | 10.1000% | 12.6000% | +2.5000pp | on |
| 8 | 21.4000% | 30.1000% | +8.7000pp | on |
| 16 | 39.2000% | 45.7000% | +6.5000pp | on |
| 32 | 49.3000% | 51.3000% | +2.0000pp | on |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 6.2000% | 6.2000% | +0.0000pp | off |
| 2 | 11.0000% | 11.1000% | +0.1000pp | on |
| 4 | 20.9000% | 23.4000% | +2.5000pp | on |
| 8 | 36.6000% | 39.7000% | +3.1000pp | on |
| 16 | 47.1000% | 48.2000% | +1.1000pp | on |
| 32 | 51.2000% | 52.4000% | +1.2000pp | on |

## Deployment equivalence checks

| Config | T | Expected source | Exact cached result |
|---|---:|---|---|
| C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| C_QCFS_ASNM_R0 | 2 | A_QCFS_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 4 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 8 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 8 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 16 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| C_QCFS_ASNM_R0 | 32 | B_QCFS_STANDARD_SNM_R0 | yes |
| F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
