# QCFS Temporal-LR + Gated-SNM CIFAR-100 Experiment

- Architecture: resnet20
- Checkpoint: resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- ANN accuracy: 68.68%
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
| A_QCFS_R0 | 7.93% | 11.66% | 22.42% | 46.37% | 64.03% | 68.78% |
| B_QCFS_STANDARD_SNM_R0 | 7.93% | 12.04% | 25.41% | 57.27% | 66.52% | 69.00% |
| C_FULL_UNSIGNED_TEACHER | 14.66% | 22.69% | 38.81% | 59.15% | 67.58% | 69.37% |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 14.66% | 22.92% | 40.82% | 60.50% | 67.43% | 68.50% |
| E_TEMPORAL_R4_UNSIGNED | 14.66% | 22.69% | 38.81% | 58.94% | 67.64% | 69.37% |
| F_TEMPORAL_R4_STANDARD_SNM | 14.66% | 22.92% | 40.82% | 60.38% | 68.15% | 69.19% |
| G_TEMPORAL_R4_GATED_SNM | 14.66% | 22.40% | 40.58% | 61.22% | 67.41% | 68.66% |
| H_HYBRID_R4_UNSIGNED | 14.66% | 22.69% | 38.81% | 59.09% | 67.83% | 69.50% |
| I_HYBRID_R4_GATED_SNM | 14.66% | 22.40% | 40.58% | 61.19% | 67.42% | 68.96% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 106,203,817,528 | 214,969,665,148 | 411,274,195,716 | 795,974,261,116 | 1,566,147,585,492 | 3,108,635,563,864 |
| B_QCFS_STANDARD_SNM_R0 | 106,203,817,528 | 217,403,594,480 | 423,540,233,072 | 839,558,553,528 | 1,658,935,588,120 | 3,271,706,702,556 |
| C_FULL_UNSIGNED_TEACHER | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 754,678,672,336 | 1,525,854,968,124 | 3,067,405,523,052 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,838,738,384 | 1,606,584,487,312 | 3,216,935,311,060 |
| E_TEMPORAL_R4_UNSIGNED | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 754,490,343,208 | 1,535,400,878,448 | 3,076,064,879,156 |
| F_TEMPORAL_R4_STANDARD_SNM | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,685,405,184 | 1,617,026,271,984 | 3,226,155,411,812 |
| G_TEMPORAL_R4_GATED_SNM | 97,023,771,544 | 190,547,355,760 | 386,948,493,492 | 784,756,817,580 | 1,613,145,820,724 | 3,221,953,693,300 |
| H_HYBRID_R4_UNSIGNED | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 754,277,051,768 | 1,532,639,564,296 | 3,070,128,888,636 |
| I_HYBRID_R4_GATED_SNM | 97,023,771,544 | 190,547,355,760 | 386,948,493,492 | 784,434,958,572 | 1,608,921,625,080 | 3,215,824,426,512 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 21.7705% | 22.3887% | 21.5543% | 20.9977% | 20.7175% | 20.5833% |
| B_QCFS_STANDARD_SNM_R0 | 21.7705% | 22.6098% | 21.9433% | 21.5252% | 21.2464% | 21.0660% |
| C_FULL_UNSIGNED_TEACHER | 21.1393% | 20.3062% | 20.1176% | 20.1557% | 20.3057% | 20.3607% |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 21.1393% | 20.4988% | 20.5168% | 20.6697% | 20.8247% | 20.8403% |
| E_TEMPORAL_R4_UNSIGNED | 21.1393% | 20.3062% | 20.1176% | 20.1573% | 20.4246% | 20.4157% |
| F_TEMPORAL_R4_STANDARD_SNM | 21.1393% | 20.4988% | 20.5168% | 20.6698% | 20.9471% | 20.8925% |
| G_TEMPORAL_R4_GATED_SNM | 21.1393% | 20.5220% | 20.5732% | 20.6343% | 20.9559% | 20.9175% |
| H_HYBRID_R4_UNSIGNED | 21.1393% | 20.3062% | 20.1176% | 20.1539% | 20.3962% | 20.3813% |
| I_HYBRID_R4_GATED_SNM | 21.1393% | 20.5220% | 20.5732% | 20.6289% | 20.9130% | 20.8824% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.1214% | 0.2756% | 0.4826% | 0.5466% | 0.5036% |
| C_FULL_UNSIGNED_TEACHER | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 0.0000% | 0.0820% | 0.2394% | 0.3983% | 0.4626% | 0.4568% |
| E_TEMPORAL_R4_UNSIGNED | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| F_TEMPORAL_R4_STANDARD_SNM | 0.0000% | 0.0820% | 0.2394% | 0.3978% | 0.4719% | 0.4646% |
| G_TEMPORAL_R4_GATED_SNM | 0.0000% | 0.0485% | 0.1878% | 0.2930% | 0.4001% | 0.4257% |
| H_HYBRID_R4_UNSIGNED | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| I_HYBRID_R4_GATED_SNM | 0.0000% | 0.0485% | 0.1878% | 0.2925% | 0.3944% | 0.4247% |

## Overall sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 78.2295% | 77.6113% | 78.4457% | 79.0023% | 79.2825% | 79.4167% |
| B_QCFS_STANDARD_SNM_R0 | 78.2295% | 77.2688% | 77.7811% | 77.9922% | 78.2070% | 78.4304% |
| C_FULL_UNSIGNED_TEACHER | 78.8607% | 79.6938% | 79.8824% | 79.8443% | 79.6943% | 79.6393% |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 78.8607% | 79.4192% | 79.2439% | 78.9320% | 78.7127% | 78.7029% |
| E_TEMPORAL_R4_UNSIGNED | 78.8607% | 79.6938% | 79.8824% | 79.8427% | 79.5754% | 79.5843% |
| F_TEMPORAL_R4_STANDARD_SNM | 78.8607% | 79.4192% | 79.2439% | 78.9324% | 78.5809% | 78.6429% |
| G_TEMPORAL_R4_GATED_SNM | 78.8607% | 79.4295% | 79.2389% | 79.0727% | 78.6440% | 78.6568% |
| H_HYBRID_R4_UNSIGNED | 78.8607% | 79.6938% | 79.8824% | 79.8461% | 79.6038% | 79.6187% |
| I_HYBRID_R4_GATED_SNM | 78.8607% | 79.4295% | 79.2389% | 79.0787% | 78.6926% | 78.6929% |

## FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_FULL_UNSIGNED_TEACHER | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| E_TEMPORAL_R4_UNSIGNED | 688 | 1,376 | 2,752 | 2,784 | 2,816 | 2,880 |
| F_TEMPORAL_R4_STANDARD_SNM | 688 | 1,376 | 2,752 | 2,784 | 2,816 | 2,880 |
| G_TEMPORAL_R4_GATED_SNM | 688 | 1,376 | 2,752 | 2,784 | 2,816 | 2,880 |
| H_HYBRID_R4_UNSIGNED | 688 | 1,376 | 2,752 | 3,040 | 3,584 | 4,672 |
| I_HYBRID_R4_GATED_SNM | 688 | 1,376 | 2,752 | 3,040 | 3,584 | 4,672 |

## FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_FULL_UNSIGNED_TEACHER | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| E_TEMPORAL_R4_UNSIGNED | 2,752 | 5,504 | 11,008 | 11,136 | 11,264 | 11,520 |
| F_TEMPORAL_R4_STANDARD_SNM | 2,752 | 5,504 | 11,008 | 11,136 | 11,264 | 11,520 |
| G_TEMPORAL_R4_GATED_SNM | 2,752 | 5,504 | 11,008 | 11,136 | 11,264 | 11,520 |
| H_HYBRID_R4_UNSIGNED | 2,752 | 5,504 | 11,008 | 12,160 | 14,336 | 18,688 |
| I_HYBRID_R4_GATED_SNM | 2,752 | 5,504 | 11,008 | 12,160 | 14,336 | 18,688 |

## Temporal bias synthesis MACs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_FULL_UNSIGNED_TEACHER | 0 | 0 | 0 | 0 | 0 | 0 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 0 | 0 | 0 | 0 | 0 | 0 |
| E_TEMPORAL_R4_UNSIGNED | 0 | 0 | 0 | 22,016 | 44,032 | 88,064 |
| F_TEMPORAL_R4_STANDARD_SNM | 0 | 0 | 0 | 22,016 | 44,032 | 88,064 |
| G_TEMPORAL_R4_GATED_SNM | 0 | 0 | 0 | 22,016 | 44,032 | 88,064 |
| H_HYBRID_R4_UNSIGNED | 0 | 0 | 0 | 19,968 | 39,936 | 79,872 |
| I_HYBRID_R4_GATED_SNM | 0 | 0 | 0 | 19,968 | 39,936 | 79,872 |

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
| A_QCFS_R0 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 |
| B_QCFS_STANDARD_SNM_R0 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 |
| C_FULL_UNSIGNED_TEACHER | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 |
| E_TEMPORAL_R4_UNSIGNED | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 |
| F_TEMPORAL_R4_STANDARD_SNM | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 |
| G_TEMPORAL_R4_GATED_SNM | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 |
| H_HYBRID_R4_UNSIGNED | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 |
| I_HYBRID_R4_GATED_SNM | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 | 1,507,328 |

## Full-teacher calibration elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s |
| B_QCFS_STANDARD_SNM_R0 | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s | 0.0s |
| C_FULL_UNSIGNED_TEACHER | 1.1s | 1.2s | 2.2s | 4.4s | 8.6s | 16.9s |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 1.1s | 1.2s | 2.2s | 4.4s | 8.6s | 16.9s |
| E_TEMPORAL_R4_UNSIGNED | 1.1s | 1.2s | 2.2s | 4.4s | 8.6s | 16.9s |
| F_TEMPORAL_R4_STANDARD_SNM | 1.1s | 1.2s | 2.2s | 4.4s | 8.6s | 16.9s |
| G_TEMPORAL_R4_GATED_SNM | 1.1s | 1.2s | 2.2s | 4.4s | 8.6s | 16.9s |
| H_HYBRID_R4_UNSIGNED | 1.1s | 1.2s | 2.2s | 4.4s | 8.6s | 16.9s |
| I_HYBRID_R4_GATED_SNM | 1.1s | 1.2s | 2.2s | 4.4s | 8.6s | 16.9s |

## Temporal compression elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_FULL_UNSIGNED_TEACHER | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| E_TEMPORAL_R4_UNSIGNED | 0.000s | 0.000s | 0.000s | 0.027s | 0.024s | 0.031s |
| F_TEMPORAL_R4_STANDARD_SNM | 0.000s | 0.000s | 0.000s | 0.022s | 0.029s | 0.034s |
| G_TEMPORAL_R4_GATED_SNM | 0.000s | 0.000s | 0.000s | 0.024s | 0.025s | 0.032s |
| H_HYBRID_R4_UNSIGNED | 0.000s | 0.000s | 0.000s | 0.020s | 0.022s | 0.035s |
| I_HYBRID_R4_GATED_SNM | 0.000s | 0.000s | 0.000s | 0.022s | 0.027s | 0.029s |

## Inference elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.9s | 1.1s | 1.7s | 2.7s | 4.8s | 9.0s |
| B_QCFS_STANDARD_SNM_R0 | 1.0s | 1.2s | 1.8s | 3.0s | 5.3s | 10.0s |
| C_FULL_UNSIGNED_TEACHER | 0.9s | 1.1s | 1.6s | 2.6s | 4.7s | 8.8s |
| D_FULL_STANDARD_SNM_AFTER_UNSIGNED | 1.0s | 1.2s | 1.7s | 2.9s | 5.2s | 9.8s |
| E_TEMPORAL_R4_UNSIGNED | 0.9s | 1.1s | 1.6s | 2.7s | 4.8s | 9.0s |
| F_TEMPORAL_R4_STANDARD_SNM | 1.0s | 1.2s | 1.7s | 3.0s | 5.3s | 10.2s |
| G_TEMPORAL_R4_GATED_SNM | 1.0s | 1.2s | 1.7s | 3.1s | 5.3s | 10.4s |
| H_HYBRID_R4_UNSIGNED | 0.9s | 1.1s | 1.6s | 2.7s | 4.8s | 8.5s |
| I_HYBRID_R4_GATED_SNM | 1.0s | 1.2s | 1.7s | 3.0s | 5.3s | 10.1s |

## Rank screen on calibration validation

### T=1

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 6.20% | 7.958133 | 1.000000 | 2,756 |
| 4 | 1 | 6.20% | 7.958133 | 1.000000 | 2,756 |
| 6 | 1 | 6.20% | 7.958133 | 1.000000 | 2,756 |

### T=2

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 11.00% | 6.065233 | 1.000000 | 5,520 |
| 4 | 2 | 11.00% | 6.065233 | 1.000000 | 5,520 |
| 6 | 2 | 11.00% | 6.065233 | 1.000000 | 5,520 |

### T=4

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 21.60% | 4.267351 | 0.771275 | 5,536 |
| 4 | 4 | 20.90% | 4.145582 | 1.000000 | 11,072 |
| 6 | 4 | 20.90% | 4.145582 | 1.000000 | 11,072 |

### T=8

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 35.30% | 2.449816 | 0.579042 | 5,568 |
| 4 | 4 | 37.00% | 2.247333 | 0.842512 | 11,136 |
| 6 | 6 | 37.60% | 2.216409 | 0.961962 | 16,704 |

### T=16

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 46.10% | 1.165338 | 0.464600 | 5,632 |
| 4 | 4 | 45.80% | 1.142655 | 0.693008 | 11,264 |
| 6 | 6 | 45.00% | 1.080231 | 0.843664 | 16,896 |

### T=32

| Rank | Effective rank | Val acc. | Logit MSE | Explained energy | Storage |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 48.50% | 0.549941 | 0.390421 | 5,760 |
| 4 | 4 | 50.10% | 0.490188 | 0.586493 | 11,520 |
| 6 | 6 | 51.80% | 0.445829 | 0.724079 | 17,280 |

## Selected SNM margins

| T | Early | Middle | Late | Final | Baseline val acc. | Gated val acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 6.20% | 6.20% |
| 2 | 0.25 | 0.5 | 0.0 | 0.0 | 11.10% | 11.80% |
| 4 | 0.25 | 0.0 | 0.0 | 0.0 | 23.40% | 23.60% |
| 8 | 0.0 | 2.0 | 0.0 | 0.5 | 39.20% | 39.50% |
| 16 | 0.0 | 1.0 | 0.0 | 1.0 | 49.00% | 49.20% |
| 32 | 0.0 | 0.0 | 1.0 | 2.0 | 51.80% | 53.40% |
