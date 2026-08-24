# QCFS + Full/Temporal-LR/PA-FTBC + HA-SNM Ablation

Status: complete

- Dataset/architecture: CIFAR-100/resnet20
- QCFS L: 8
- ANN accuracy: 68.68%
- Checkpoint: `resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth`
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- Fit/validation SHA256: `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a` / `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3`
- Test samples: 10,000
- Evaluation profile: `paper_era`
- HA-SNM threshold schedule: start=1.25, end=0.5, reference horizon=8.0, linear.
- HA-SNM keeps the original transmitted-credit/R0 rule and changes only the negative-spike decision threshold.
- It uses the original -theta event amplitude, adds no dense neuron state, and has two global FP32 deployment constants plus one fixed reference horizon (12 bytes if stored).
- Full-FTBC is fitted independently at every T with SNM off; Temporal-LR and PA are compressed from that same teacher.
- Temporal-LR and PA fall back exactly to Full-FTBC at T<=4.
- Checkpoint note: existing frozen repository checkpoint and evaluation protocol.

## Primary accuracy

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 14.66% | 22.69% | 38.81% | 59.15% | 67.58% | 69.37% | 45.38% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 14.66% | 22.92% | 40.82% | 60.50% | 67.43% | 68.50% | 45.81% |
| C_QCFS_FULL_FTBC_HA_R0 | 14.66% | 25.03% | 44.55% | 61.05% | 67.25% | 68.50% | 46.84% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 14.66% | 22.69% | 38.81% | 58.94% | 67.64% | 69.37% | 45.35% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 14.66% | 22.92% | 40.82% | 60.38% | 68.15% | 69.19% | 46.02% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 14.66% | 25.03% | 44.55% | 61.60% | 68.08% | 69.19% | 47.19% |
| G_QCFS_PA_FTBC_OFF_R0 | 14.66% | 22.69% | 38.81% | 59.26% | 67.84% | 69.31% | 45.43% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 14.66% | 22.92% | 40.82% | 61.22% | 67.70% | 68.58% | 45.98% |
| I_QCFS_PA_FTBC_HA_R0 | 14.66% | 25.03% | 44.55% | 61.74% | 67.58% | 68.54% | 47.02% |

## HA-SNM accuracy gain

| Family | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-FTBC: HA - standard | +0.00pp | +2.11pp | +3.73pp | +0.55pp | -0.18pp | +0.00pp | +1.035pp |
| Full-FTBC: HA - off | +0.00pp | +2.34pp | +5.74pp | +1.90pp | -0.33pp | -0.87pp | +1.463pp |
| Temporal-LR FTBC: HA - standard | +0.00pp | +2.11pp | +3.73pp | +1.22pp | -0.07pp | +0.00pp | +1.165pp |
| Temporal-LR FTBC: HA - off | +0.00pp | +2.34pp | +5.74pp | +2.66pp | +0.44pp | -0.18pp | +1.833pp |
| PA-FTBC: HA - standard | +0.00pp | +2.11pp | +3.73pp | +0.52pp | -0.12pp | -0.04pp | +1.033pp |
| PA-FTBC: HA - off | +0.00pp | +2.34pp | +5.74pp | +2.48pp | -0.26pp | -0.77pp | +1.588pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 11.70416273 | 8.86490727 | 5.98702230 | 2.97863559 | 1.15333930 | 0.47384057 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 11.70416273 | 8.80801217 | 5.63362295 | 2.56125889 | 0.86445645 | 0.36801880 |
| C_QCFS_FULL_FTBC_HA_R0 | 11.70416273 | 8.35827625 | 5.09637033 | 2.42343589 | 0.84978109 | 0.37006042 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 11.70416273 | 8.86490727 | 5.98702230 | 2.98889267 | 1.18884983 | 0.53166937 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 11.70416273 | 8.80801217 | 5.63362295 | 2.56022004 | 0.79474718 | 0.35449877 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 11.70416273 | 8.35827625 | 5.09637033 | 2.42206655 | 0.78398228 | 0.35567604 |
| G_QCFS_PA_FTBC_OFF_R0 | 11.70416273 | 8.86490727 | 5.98702230 | 2.95617253 | 1.13105523 | 0.47085537 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 11.70416273 | 8.80801217 | 5.63362295 | 2.50593807 | 0.83397612 | 0.37642431 |
| I_QCFS_PA_FTBC_HA_R0 | 11.70416273 | 8.35827625 | 5.09637033 | 2.35544397 | 0.83383469 | 0.38097281 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 21.139312% | 20.306214% | 20.117610% | 20.155742% | 20.305703% | 20.360652% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 21.139312% | 20.498814% | 20.516761% | 20.669711% | 20.824673% | 20.840262% |
| C_QCFS_FULL_FTBC_HA_R0 | 21.139312% | 20.650288% | 20.739014% | 20.899361% | 20.902024% | 20.867646% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 21.139312% | 20.306214% | 20.117610% | 20.157332% | 20.424614% | 20.415701% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 21.139312% | 20.498814% | 20.516761% | 20.669757% | 20.947116% | 20.892465% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 21.139312% | 20.650288% | 20.739014% | 20.899022% | 21.024812% | 20.920049% |
| G_QCFS_PA_FTBC_OFF_R0 | 21.139312% | 20.306214% | 20.117610% | 20.166644% | 20.317820% | 20.364202% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 21.139312% | 20.498814% | 20.516761% | 20.681454% | 20.830831% | 20.837079% |
| I_QCFS_PA_FTBC_HA_R0 | 21.139312% | 20.650288% | 20.739014% | 20.909555% | 20.907964% | 20.863952% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 0.000000% | 0.082010% | 0.239388% | 0.398309% | 0.462623% | 0.456818% |
| C_QCFS_FULL_FTBC_HA_R0 | 0.000000% | 0.459579% | 0.718177% | 0.816228% | 0.578072% | 0.493391% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 0.000000% | 0.082010% | 0.239388% | 0.397810% | 0.471938% | 0.464590% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 0.000000% | 0.459579% | 0.718177% | 0.818563% | 0.590071% | 0.501675% |
| G_QCFS_PA_FTBC_OFF_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 0.000000% | 0.082010% | 0.239388% | 0.398387% | 0.464022% | 0.455477% |
| I_QCFS_PA_FTBC_HA_R0 | 0.000000% | 0.459579% | 0.718177% | 0.820503% | 0.580271% | 0.491626% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 78.860688% | 79.693786% | 79.882390% | 79.844258% | 79.694297% | 79.639348% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 78.860688% | 79.419176% | 79.243852% | 78.931980% | 78.712704% | 78.702920% |
| C_QCFS_FULL_FTBC_HA_R0 | 78.860688% | 78.890133% | 78.542810% | 78.284411% | 78.519904% | 78.638963% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 78.860688% | 79.693786% | 79.882390% | 79.842668% | 79.575386% | 79.584299% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 78.860688% | 79.419176% | 79.243852% | 78.932433% | 78.580946% | 78.642945% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 78.860688% | 78.890133% | 78.542810% | 78.282416% | 78.385118% | 78.578276% |
| G_QCFS_PA_FTBC_OFF_R0 | 78.860688% | 79.693786% | 79.882390% | 79.833356% | 79.682180% | 79.635798% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 78.860688% | 79.419176% | 79.243852% | 78.920159% | 78.705147% | 78.707444% |
| I_QCFS_PA_FTBC_HA_R0 | 78.860688% | 78.890133% | 78.542810% | 78.269942% | 78.511765% | 78.644421% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 754,678,672,336 | 1,525,854,968,124 | 3,067,405,523,052 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,838,738,384 | 1,606,584,487,312 | 3,216,935,311,060 |
| C_QCFS_FULL_FTBC_HA_R0 | 97,023,771,544 | 195,354,172,652 | 401,553,185,844 | 819,865,383,336 | 1,622,817,808,884 | 3,227,153,209,228 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 754,490,343,208 | 1,535,400,878,448 | 3,076,064,879,156 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,685,405,184 | 1,617,026,271,984 | 3,226,155,411,812 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 97,023,771,544 | 195,354,172,652 | 401,553,185,844 | 819,822,366,712 | 1,633,517,292,472 | 3,236,543,672,456 |
| G_QCFS_PA_FTBC_OFF_R0 | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 755,331,242,684 | 1,527,125,941,820 | 3,068,367,534,872 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 792,589,351,960 | 1,607,623,945,972 | 3,216,860,154,316 |
| I_QCFS_PA_FTBC_HA_R0 | 97,023,771,544 | 195,354,172,652 | 401,553,185,844 | 820,675,702,684 | 1,623,921,485,752 | 3,226,958,758,148 |

## FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| C_QCFS_FULL_FTBC_HA_R0 | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 688 | 1,376 | 2,752 | 2,784 | 2,816 | 2,880 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 688 | 1,376 | 2,752 | 2,784 | 2,816 | 2,880 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 688 | 1,376 | 2,752 | 2,784 | 2,816 | 2,880 |
| G_QCFS_PA_FTBC_OFF_R0 | 688 | 1,376 | 2,752 | 2,752 | 2,752 | 2,752 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 688 | 1,376 | 2,752 | 2,752 | 2,752 | 2,752 |
| I_QCFS_PA_FTBC_HA_R0 | 688 | 1,376 | 2,752 | 2,752 | 2,752 | 2,752 |

## FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| C_QCFS_FULL_FTBC_HA_R0 | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 2,752 | 5,504 | 11,008 | 11,136 | 11,264 | 11,520 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 2,752 | 5,504 | 11,008 | 11,136 | 11,264 | 11,520 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 2,752 | 5,504 | 11,008 | 11,136 | 11,264 | 11,520 |
| G_QCFS_PA_FTBC_OFF_R0 | 2,752 | 5,504 | 11,008 | 11,008 | 11,008 | 11,008 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 2,752 | 5,504 | 11,008 | 11,008 | 11,008 | 11,008 |
| I_QCFS_PA_FTBC_HA_R0 | 2,752 | 5,504 | 11,008 | 11,008 | 11,008 | 11,008 |

## Bias synthesis MACs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_FULL_FTBC_HA_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 0 | 0 | 0 | 22,016 | 44,032 | 88,064 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 0 | 0 | 0 | 22,016 | 44,032 | 88,064 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 0 | 0 | 0 | 22,016 | 44,032 | 88,064 |
| G_QCFS_PA_FTBC_OFF_R0 | 0 | 0 | 0 | 9,632 | 20,640 | 42,656 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 0 | 0 | 0 | 9,632 | 20,640 | 42,656 |
| I_QCFS_PA_FTBC_HA_R0 | 0 | 0 | 0 | 9,632 | 20,640 | 42,656 |

## Inference elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 1.109413 | 1.339741 | 1.898229 | 3.113334 | 5.442133 | 10.049286 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 1.190092 | 1.458006 | 2.032050 | 3.392012 | 5.939936 | 11.226145 |
| C_QCFS_FULL_FTBC_HA_R0 | 1.191175 | 1.459593 | 2.026877 | 3.426281 | 6.112597 | 11.592685 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 1.134682 | 1.280923 | 1.882664 | 3.129210 | 5.641273 | 10.313038 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 1.180841 | 1.403259 | 2.046237 | 3.482064 | 6.187872 | 11.646043 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 1.204413 | 1.464166 | 2.082320 | 3.521841 | 6.349156 | 12.154993 |
| G_QCFS_PA_FTBC_OFF_R0 | 1.107934 | 1.315066 | 1.876875 | 3.194518 | 5.612667 | 10.493010 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 1.210965 | 1.430341 | 2.055509 | 3.507732 | 6.271614 | 11.814280 |
| I_QCFS_PA_FTBC_HA_R0 | 1.192459 | 1.463931 | 2.193260 | 3.531867 | 6.294250 | 12.256554 |

## HA-SNM overhead

| Item | Value |
|---|---:|
| Additional dense per-neuron state | 0 bytes |
| Global constants | 3 (12 bytes if all stored as FP32) |
| SignedIF layers | 19 |
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
