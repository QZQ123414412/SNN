# QCFS + Full-FTBC + Temporal-LR FTBC + Parity-Anchor FTBC + A-SNM Ablation

Status: complete

- Dataset/architecture: CIFAR-100/resnet20
- QCFS L: 8
- ANN accuracy: 68.68%
- Checkpoint: `resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth`
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- Fit/validation SHA256: `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a` / `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3`
- Test samples: 10,000
- Evaluation profile: `paper_era`
- Full-FTBC is fitted independently at every T with SNM off.
- Temporal-LR uses a shared learned rank-4 SVD basis with threshold normalization.
- PA-FTBC uses no SVD or stored basis: t=0/t=1 anchors plus tail mean and tail parity.
- Both compressed methods fall back exactly to Full-FTBC at T<=4.
- Every family freezes its own strict accuracy-gated A-SNM decisions before test inference.
- Checkpoint note: existing frozen repository checkpoint and evaluation protocol.

## Primary accuracy table

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | SNM-on T |
|---|---:|---:|---:|---:|---:|---:|---|
| A_QCFS_R0 | 7.93% | 11.66% | 22.42% | 46.37% | 64.03% | 68.78% | none |
| B_QCFS_STANDARD_SNM_R0 | 7.93% | 12.04% | 25.41% | 57.27% | 66.52% | 69.00% | 1, 2, 4, 8, 16, 32 |
| C_QCFS_ASNM_R0 | 7.93% | 11.66% | 25.41% | 57.27% | 66.52% | 69.00% | 4, 8, 16, 32 |
| D_QCFS_FULL_FTBC_R0 | 14.66% | 22.69% | 38.81% | 59.15% | 67.58% | 69.37% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 14.66% | 22.92% | 40.82% | 60.50% | 67.43% | 68.50% | 1, 2, 4, 8, 16, 32 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 14.66% | 22.92% | 40.82% | 60.50% | 67.43% | 68.50% | 2, 4, 8, 16, 32 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 14.66% | 22.69% | 38.81% | 58.94% | 67.64% | 69.37% | none |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 14.66% | 22.92% | 40.82% | 60.38% | 68.15% | 69.19% | 1, 2, 4, 8, 16, 32 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 14.66% | 22.92% | 40.82% | 60.38% | 68.15% | 69.19% | 2, 4, 8, 16, 32 |
| J_QCFS_PA_FTBC_R0 | 14.66% | 22.69% | 38.81% | 59.26% | 67.84% | 69.31% | none |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 14.66% | 22.92% | 40.82% | 61.22% | 67.70% | 68.58% | 1, 2, 4, 8, 16, 32 |
| L_QCFS_PA_FTBC_ASNM_R0 | 14.66% | 22.92% | 40.82% | 61.22% | 67.70% | 68.58% | 2, 4, 8, 16, 32 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 36.87% |
| B_QCFS_STANDARD_SNM_R0 | 39.70% |
| C_QCFS_ASNM_R0 | 39.63% |
| D_QCFS_FULL_FTBC_R0 | 45.38% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 45.81% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 45.81% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 45.35% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 46.02% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 46.02% |
| J_QCFS_PA_FTBC_R0 | 45.43% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 45.98% |
| L_QCFS_PA_FTBC_ASNM_R0 | 45.98% |

## PA-FTBC accuracy comparisons

| T | PA off - Temporal off | PA standard - Temporal standard | PA A-SNM - Temporal A-SNM |
|---:|---:|---:|---:|
| 1 | +0.00pp | +0.00pp | +0.00pp |
| 2 | +0.00pp | +0.00pp | +0.00pp |
| 4 | +0.00pp | +0.00pp | +0.00pp |
| 8 | +0.32pp | +0.84pp | +0.84pp |
| 16 | +0.20pp | -0.45pp | -0.45pp |
| 32 | -0.06pp | -0.61pp | -0.61pp |
| Mean | +0.08pp | -0.04pp | -0.04pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 17.06097112 | 13.85701483 | 9.98135594 | 5.25699140 | 1.98007151 | 0.71314130 |
| B_QCFS_STANDARD_SNM_R0 | 17.06097112 | 13.66545084 | 8.93442475 | 3.27601362 | 1.07296898 | 0.39698374 |
| C_QCFS_ASNM_R0 | 17.06097112 | 13.85701483 | 8.93442475 | 3.27601362 | 1.07296898 | 0.39698374 |
| D_QCFS_FULL_FTBC_R0 | 11.70416273 | 8.86490727 | 5.98702230 | 2.97863559 | 1.15333930 | 0.47384057 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 11.70416273 | 8.80801217 | 5.63362295 | 2.56125889 | 0.86445645 | 0.36801880 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 11.70416273 | 8.80801217 | 5.63362295 | 2.56125889 | 0.86445645 | 0.36801880 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 11.70416273 | 8.86490727 | 5.98702230 | 2.98889267 | 1.18884983 | 0.53166937 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 11.70416273 | 8.80801217 | 5.63362295 | 2.56022004 | 0.79474718 | 0.35449877 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 11.70416273 | 8.80801217 | 5.63362295 | 2.56022004 | 0.79474718 | 0.35449877 |
| J_QCFS_PA_FTBC_R0 | 11.70416273 | 8.86490727 | 5.98702230 | 2.95617253 | 1.13105523 | 0.47085537 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 11.70416273 | 8.80801217 | 5.63362295 | 2.50593807 | 0.83397612 | 0.37642431 |
| L_QCFS_PA_FTBC_ASNM_R0 | 11.70416273 | 8.80801217 | 5.63362295 | 2.50593807 | 0.83397612 | 0.37642431 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 21.770521% | 22.388661% | 21.554266% | 20.997709% | 20.717495% | 20.583338% |
| B_QCFS_STANDARD_SNM_R0 | 21.770521% | 22.609807% | 21.943301% | 21.525177% | 21.246427% | 21.066010% |
| C_QCFS_ASNM_R0 | 21.770521% | 22.388661% | 21.943301% | 21.525177% | 21.246427% | 21.066010% |
| D_QCFS_FULL_FTBC_R0 | 21.139312% | 20.306214% | 20.117610% | 20.155742% | 20.305703% | 20.360652% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 21.139312% | 20.498814% | 20.516761% | 20.669711% | 20.824673% | 20.840262% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 21.139312% | 20.498814% | 20.516761% | 20.669711% | 20.824673% | 20.840262% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 21.139312% | 20.306214% | 20.117610% | 20.157332% | 20.424614% | 20.415701% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 21.139312% | 20.498814% | 20.516761% | 20.669757% | 20.947116% | 20.892465% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 21.139312% | 20.498814% | 20.516761% | 20.669757% | 20.947116% | 20.892465% |
| J_QCFS_PA_FTBC_R0 | 21.139312% | 20.306214% | 20.117610% | 20.166644% | 20.317820% | 20.364202% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 21.139312% | 20.498814% | 20.516761% | 20.681454% | 20.830831% | 20.837079% |
| L_QCFS_PA_FTBC_ASNM_R0 | 21.139312% | 20.498814% | 20.516761% | 20.681454% | 20.830831% | 20.837079% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_STANDARD_SNM_R0 | 0.000000% | 0.121382% | 0.275648% | 0.482599% | 0.546600% | 0.503620% |
| C_QCFS_ASNM_R0 | 0.000000% | 0.000000% | 0.275648% | 0.482599% | 0.546600% | 0.503620% |
| D_QCFS_FULL_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.082010% | 0.239388% | 0.398309% | 0.462623% | 0.456818% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000000% | 0.082010% | 0.239388% | 0.398309% | 0.462623% | 0.456818% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.082010% | 0.239388% | 0.397810% | 0.471938% | 0.464590% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000000% | 0.082010% | 0.239388% | 0.397810% | 0.471938% | 0.464590% |
| J_QCFS_PA_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.082010% | 0.239388% | 0.398387% | 0.464022% | 0.455477% |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.000000% | 0.082010% | 0.239388% | 0.398387% | 0.464022% | 0.455477% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 78.229479% | 77.611339% | 78.445734% | 79.002291% | 79.282505% | 79.416662% |
| B_QCFS_STANDARD_SNM_R0 | 78.229479% | 77.268811% | 77.781050% | 77.992224% | 78.206973% | 78.430370% |
| C_QCFS_ASNM_R0 | 78.229479% | 77.611339% | 77.781050% | 77.992224% | 78.206973% | 78.430370% |
| D_QCFS_FULL_FTBC_R0 | 78.860688% | 79.693786% | 79.882390% | 79.844258% | 79.694297% | 79.639348% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 78.860688% | 79.419176% | 79.243852% | 78.931980% | 78.712704% | 78.702920% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 78.860688% | 79.419176% | 79.243852% | 78.931980% | 78.712704% | 78.702920% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 78.860688% | 79.693786% | 79.882390% | 79.842668% | 79.575386% | 79.584299% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 78.860688% | 79.419176% | 79.243852% | 78.932433% | 78.580946% | 78.642945% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 78.860688% | 79.419176% | 79.243852% | 78.932433% | 78.580946% | 78.642945% |
| J_QCFS_PA_FTBC_R0 | 78.860688% | 79.693786% | 79.882390% | 79.833356% | 79.682180% | 79.635798% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 78.860688% | 79.419176% | 79.243852% | 78.920159% | 78.705147% | 78.707444% |
| L_QCFS_PA_FTBC_ASNM_R0 | 78.860688% | 79.419176% | 79.243852% | 78.920159% | 78.705147% | 78.707444% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 106,203,817,528 | 214,969,665,148 | 411,274,195,716 | 795,974,261,116 | 1,566,147,585,492 | 3,108,635,563,864 |
| B_QCFS_STANDARD_SNM_R0 | 106,203,817,528 | 217,403,594,480 | 423,540,233,072 | 839,558,553,528 | 1,658,935,588,120 | 3,271,706,702,556 |
| C_QCFS_ASNM_R0 | 106,203,817,528 | 214,969,665,148 | 423,540,233,072 | 839,558,553,528 | 1,658,935,588,120 | 3,271,706,702,556 |
| D_QCFS_FULL_FTBC_R0 | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 754,678,672,336 | 1,525,854,968,124 | 3,067,405,523,052 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,838,738,384 | 1,606,584,487,312 | 3,216,935,311,060 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,838,738,384 | 1,606,584,487,312 | 3,216,935,311,060 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 754,490,343,208 | 1,535,400,878,448 | 3,076,064,879,156 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,685,405,184 | 1,617,026,271,984 | 3,226,155,411,812 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 791,685,405,184 | 1,617,026,271,984 | 3,226,155,411,812 |
| J_QCFS_PA_FTBC_R0 | 97,023,771,544 | 188,492,574,456 | 374,813,123,172 | 755,331,242,684 | 1,527,125,941,820 | 3,068,367,534,872 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 792,589,351,960 | 1,607,623,945,972 | 3,216,860,154,316 |
| L_QCFS_PA_FTBC_ASNM_R0 | 97,023,771,544 | 190,634,878,908 | 386,621,167,572 | 792,589,351,960 | 1,607,623,945,972 | 3,216,860,154,316 |

## FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 688 | 1,376 | 2,752 | 5,504 | 11,008 | 22,016 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 688 | 1,376 | 2,752 | 2,784 | 2,816 | 2,880 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 688 | 1,376 | 2,752 | 2,784 | 2,816 | 2,880 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 688 | 1,376 | 2,752 | 2,784 | 2,816 | 2,880 |
| J_QCFS_PA_FTBC_R0 | 688 | 1,376 | 2,752 | 2,752 | 2,752 | 2,752 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 688 | 1,376 | 2,752 | 2,752 | 2,752 | 2,752 |
| L_QCFS_PA_FTBC_ASNM_R0 | 688 | 1,376 | 2,752 | 2,752 | 2,752 | 2,752 |

## FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2,752 | 5,504 | 11,008 | 22,016 | 44,032 | 88,064 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 2,752 | 5,504 | 11,008 | 11,136 | 11,264 | 11,520 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 2,752 | 5,504 | 11,008 | 11,136 | 11,264 | 11,520 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2,752 | 5,504 | 11,008 | 11,136 | 11,264 | 11,520 |
| J_QCFS_PA_FTBC_R0 | 2,752 | 5,504 | 11,008 | 11,008 | 11,008 | 11,008 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 2,752 | 5,504 | 11,008 | 11,008 | 11,008 | 11,008 |
| L_QCFS_PA_FTBC_ASNM_R0 | 2,752 | 5,504 | 11,008 | 11,008 | 11,008 | 11,008 |

## Bias synthesis MACs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0 | 0 | 0 | 22,016 | 44,032 | 88,064 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 22,016 | 44,032 | 88,064 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0 | 0 | 0 | 22,016 | 44,032 | 88,064 |
| J_QCFS_PA_FTBC_R0 | 0 | 0 | 0 | 9,632 | 20,640 | 42,656 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 9,632 | 20,640 | 42,656 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0 | 0 | 0 | 9,632 | 20,640 | 42,656 |

## Full-teacher calibration elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B_QCFS_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C_QCFS_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| D_QCFS_FULL_FTBC_R0 | 1.253976 | 1.365962 | 2.542179 | 4.919728 | 9.610394 | 19.110932 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.253976 | 1.365962 | 2.542179 | 4.919728 | 9.610394 | 19.110932 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.253976 | 1.365962 | 2.542179 | 4.919728 | 9.610394 | 19.110932 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.253976 | 1.365962 | 2.542179 | 4.919728 | 9.610394 | 19.110932 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.253976 | 1.365962 | 2.542179 | 4.919728 | 9.610394 | 19.110932 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.253976 | 1.365962 | 2.542179 | 4.919728 | 9.610394 | 19.110932 |
| J_QCFS_PA_FTBC_R0 | 1.253976 | 1.365962 | 2.542179 | 4.919728 | 9.610394 | 19.110932 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 1.253976 | 1.365962 | 2.542179 | 4.919728 | 9.610394 | 19.110932 |
| L_QCFS_PA_FTBC_ASNM_R0 | 1.253976 | 1.365962 | 2.542179 | 4.919728 | 9.610394 | 19.110932 |

## Compression elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B_QCFS_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C_QCFS_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| D_QCFS_FULL_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.056858 | 0.027255 | 0.056836 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.056858 | 0.027255 | 0.056836 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.056858 | 0.027255 | 0.056836 |
| J_QCFS_PA_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.027956 | 0.008824 | 0.014431 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.027956 | 0.008824 | 0.014431 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.027956 | 0.008824 | 0.014431 |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.091405 | 1.264956 | 1.903542 | 3.143761 | 5.487775 | 10.356279 |
| B_QCFS_STANDARD_SNM_R0 | 1.193442 | 1.483974 | 2.051864 | 3.394350 | 6.037333 | 11.511112 |
| C_QCFS_ASNM_R0 | 1.091405 | 1.264956 | 2.051864 | 3.394350 | 6.037333 | 11.511112 |
| D_QCFS_FULL_FTBC_R0 | 1.105321 | 1.291015 | 1.827687 | 3.073067 | 5.389192 | 10.121878 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.130368 | 1.392173 | 1.956557 | 3.385621 | 5.886246 | 11.199218 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.105321 | 1.392173 | 1.956557 | 3.385621 | 5.886246 | 11.199218 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.069360 | 1.323230 | 1.824718 | 3.172229 | 5.483220 | 10.302709 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.142305 | 1.400577 | 1.983502 | 3.425688 | 6.026950 | 11.428470 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.069360 | 1.400577 | 1.983502 | 3.425688 | 6.026950 | 11.428470 |
| J_QCFS_PA_FTBC_R0 | 1.071284 | 1.270313 | 1.841019 | 3.191568 | 5.537139 | 10.313828 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 1.154369 | 1.365475 | 2.041823 | 3.419877 | 6.152275 | 11.382297 |
| L_QCFS_PA_FTBC_ASNM_R0 | 1.071284 | 1.365475 | 2.041823 | 3.419877 | 6.152275 | 11.382297 |

## Compression summary

| T | Full params | Temporal params | PA params | Temporal saving | PA saving | Temporal MACs | PA MACs | Temporal energy | PA energy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 688 | 688 | 688 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 2 | 1,376 | 1,376 | 1,376 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 4 | 2,752 | 2,752 | 2,752 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 8 | 5,504 | 2,784 | 2,752 | 49.42% | 50.00% | 22,016 | 9,632 | 0.842512 | 0.941325 |
| 16 | 11,008 | 2,816 | 2,752 | 74.42% | 75.00% | 44,032 | 20,640 | 0.693008 | 0.864102 |
| 32 | 22,016 | 2,880 | 2,752 | 86.92% | 87.50% | 88,064 | 42,656 | 0.586493 | 0.754117 |

## A-SNM selection

- QCFS SNM-on T: 4, 8, 16, 32; selection elapsed: 6.055215s.

### QCFS accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 4.70% | 4.70% | off |
| 2 | 6.40% | 6.10% | off |
| 4 | 10.10% | 12.60% | on |
| 8 | 21.40% | 30.10% | on |
| 16 | 39.20% | 45.70% | on |
| 32 | 49.30% | 51.30% | on |

- Full-FTBC SNM-on T: 2, 4, 8, 16, 32; selection elapsed: 5.624559s.

### Full-FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 6.20% | 6.20% | off |
| 2 | 11.00% | 11.10% | on |
| 4 | 20.90% | 23.40% | on |
| 8 | 36.60% | 39.70% | on |
| 16 | 47.10% | 48.20% | on |
| 32 | 51.20% | 52.40% | on |

- Temporal-LR FTBC SNM-on T: 2, 4, 8, 16, 32; selection elapsed: 5.677795s.

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 6.20% | 6.20% | off |
| 2 | 11.00% | 11.10% | on |
| 4 | 20.90% | 23.40% | on |
| 8 | 37.00% | 39.20% | on |
| 16 | 45.80% | 49.00% | on |
| 32 | 50.10% | 51.80% | on |

- Parity-Anchor FTBC SNM-on T: 2, 4, 8, 16, 32; selection elapsed: 5.782233s.

### Parity-Anchor FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 6.20% | 6.20% | off |
| 2 | 11.00% | 11.10% | on |
| 4 | 20.90% | 23.40% | on |
| 8 | 37.80% | 39.30% | on |
| 16 | 46.60% | 49.60% | on |
| 32 | 51.20% | 52.90% | on |

## Validation-selection generalization audit

| Family | T | Selected | Test off | Test on | Test-best | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 1 | off | 7.93% | 7.93% | off | yes |
| Full-FTBC | 1 | off | 14.66% | 14.66% | off | yes |
| Temporal-LR FTBC | 1 | off | 14.66% | 14.66% | off | yes |
| Parity-Anchor FTBC | 1 | off | 14.66% | 14.66% | off | yes |
| QCFS | 2 | off | 11.66% | 12.04% | on | no |
| Full-FTBC | 2 | on | 22.69% | 22.92% | on | yes |
| Temporal-LR FTBC | 2 | on | 22.69% | 22.92% | on | yes |
| Parity-Anchor FTBC | 2 | on | 22.69% | 22.92% | on | yes |
| QCFS | 4 | on | 22.42% | 25.41% | on | yes |
| Full-FTBC | 4 | on | 38.81% | 40.82% | on | yes |
| Temporal-LR FTBC | 4 | on | 38.81% | 40.82% | on | yes |
| Parity-Anchor FTBC | 4 | on | 38.81% | 40.82% | on | yes |
| QCFS | 8 | on | 46.37% | 57.27% | on | yes |
| Full-FTBC | 8 | on | 59.15% | 60.50% | on | yes |
| Temporal-LR FTBC | 8 | on | 58.94% | 60.38% | on | yes |
| Parity-Anchor FTBC | 8 | on | 59.26% | 61.22% | on | yes |
| QCFS | 16 | on | 64.03% | 66.52% | on | yes |
| Full-FTBC | 16 | on | 67.58% | 67.43% | off | no |
| Temporal-LR FTBC | 16 | on | 67.64% | 68.15% | on | yes |
| Parity-Anchor FTBC | 16 | on | 67.84% | 67.70% | off | no |
| QCFS | 32 | on | 68.78% | 69.00% | on | yes |
| Full-FTBC | 32 | on | 69.37% | 68.50% | off | no |
| Temporal-LR FTBC | 32 | on | 69.37% | 69.19% | off | no |
| Parity-Anchor FTBC | 32 | on | 69.31% | 68.58% | off | no |

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
| A-SNM cache | C_QCFS_ASNM_R0 | 2 | A_QCFS_R0 | yes |
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
| A-SNM cache | C_QCFS_ASNM_R0 | 32 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 32 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 32 | K_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |

## Per-layer Temporal-LR FTBC reconstruction

### T=1

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |

### T=2

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |

### T=4

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |

### T=8

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | temporal_low_rank | 16 | 0.00093673 | 0.49256712 | 0.11253321 |
| `conv2_x.0.residual_function.2` | temporal_low_rank | 16 | 0.00057738 | 0.67277956 | 0.07831758 |
| `conv2_x.0.act` | temporal_low_rank | 16 | 0.00032743 | 0.44917578 | 0.07480834 |
| `conv2_x.1.residual_function.2` | temporal_low_rank | 16 | 0.00009319 | 0.46652123 | 0.04443043 |
| `conv2_x.1.act` | temporal_low_rank | 16 | 0.00008441 | 0.31250408 | 0.03940286 |
| `conv2_x.2.residual_function.2` | temporal_low_rank | 16 | 0.00000392 | 0.15070027 | 0.01082194 |
| `conv2_x.2.act` | temporal_low_rank | 16 | 0.00022952 | 0.33775771 | 0.07100661 |
| `conv3_x.0.residual_function.2` | temporal_low_rank | 32 | 0.00002622 | 0.19337696 | 0.02052793 |
| `conv3_x.0.act` | temporal_low_rank | 32 | 0.00002831 | 0.20877540 | 0.01691367 |
| `conv3_x.1.residual_function.2` | temporal_low_rank | 32 | 0.00000440 | 0.11542708 | 0.00637735 |
| `conv3_x.1.act` | temporal_low_rank | 32 | 0.00003130 | 0.13607678 | 0.02185537 |
| `conv3_x.2.residual_function.2` | temporal_low_rank | 32 | 0.00000316 | 0.10459626 | 0.00731348 |
| `conv3_x.2.act` | temporal_low_rank | 32 | 0.00003864 | 0.12867767 | 0.03432176 |
| `conv4_x.0.residual_function.2` | temporal_low_rank | 64 | 0.00001215 | 0.14888190 | 0.01466646 |
| `conv4_x.0.act` | temporal_low_rank | 64 | 0.00002812 | 0.16198434 | 0.02333174 |
| `conv4_x.1.residual_function.2` | temporal_low_rank | 64 | 0.00000838 | 0.21447203 | 0.01437332 |
| `conv4_x.1.act` | temporal_low_rank | 64 | 0.00009728 | 0.20039891 | 0.04655536 |
| `conv4_x.2.residual_function.2` | temporal_low_rank | 64 | 0.00000842 | 0.20560233 | 0.01150482 |
| `conv4_x.2.act` | temporal_low_rank | 64 | 0.00101154 | 0.21221319 | 0.12012336 |

### T=16

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | temporal_low_rank | 16 | 0.00113295 | 0.60538507 | 0.15041569 |
| `conv2_x.0.residual_function.2` | temporal_low_rank | 16 | 0.00055741 | 0.75885528 | 0.08096220 |
| `conv2_x.0.act` | temporal_low_rank | 16 | 0.00036807 | 0.60349542 | 0.06698432 |
| `conv2_x.1.residual_function.2` | temporal_low_rank | 16 | 0.00010887 | 0.65843326 | 0.06455845 |
| `conv2_x.1.act` | temporal_low_rank | 16 | 0.00018096 | 0.60511863 | 0.05726363 |
| `conv2_x.2.residual_function.2` | temporal_low_rank | 16 | 0.00001119 | 0.35279039 | 0.01293348 |
| `conv2_x.2.act` | temporal_low_rank | 16 | 0.00026328 | 0.49120918 | 0.10317941 |
| `conv3_x.0.residual_function.2` | temporal_low_rank | 32 | 0.00010286 | 0.53304070 | 0.10712366 |
| `conv3_x.0.act` | temporal_low_rank | 32 | 0.00009435 | 0.52816373 | 0.07013326 |
| `conv3_x.1.residual_function.2` | temporal_low_rank | 32 | 0.00001314 | 0.28026262 | 0.02851943 |
| `conv3_x.1.act` | temporal_low_rank | 32 | 0.00005912 | 0.26330453 | 0.05735529 |
| `conv3_x.2.residual_function.2` | temporal_low_rank | 32 | 0.00001121 | 0.27708656 | 0.02442226 |
| `conv3_x.2.act` | temporal_low_rank | 32 | 0.00005897 | 0.22404687 | 0.05621725 |
| `conv4_x.0.residual_function.2` | temporal_low_rank | 64 | 0.00002635 | 0.30671102 | 0.03157689 |
| `conv4_x.0.act` | temporal_low_rank | 64 | 0.00006567 | 0.34452561 | 0.03276374 |
| `conv4_x.1.residual_function.2` | temporal_low_rank | 64 | 0.00002007 | 0.45731723 | 0.02071095 |
| `conv4_x.1.act` | temporal_low_rank | 64 | 0.00016356 | 0.35889569 | 0.07037936 |
| `conv4_x.2.residual_function.2` | temporal_low_rank | 64 | 0.00002512 | 0.48858708 | 0.02751582 |
| `conv4_x.2.act` | temporal_low_rank | 64 | 0.00147846 | 0.35309359 | 0.16363245 |

### T=32

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | temporal_low_rank | 16 | 0.00112117 | 0.66227394 | 0.13040124 |
| `conv2_x.0.residual_function.2` | temporal_low_rank | 16 | 0.00046255 | 0.74860847 | 0.07896733 |
| `conv2_x.0.act` | temporal_low_rank | 16 | 0.00038067 | 0.72118378 | 0.10313410 |
| `conv2_x.1.residual_function.2` | temporal_low_rank | 16 | 0.00012106 | 0.79549485 | 0.06662497 |
| `conv2_x.1.act` | temporal_low_rank | 16 | 0.00017963 | 0.72488701 | 0.07104879 |
| `conv2_x.2.residual_function.2` | temporal_low_rank | 16 | 0.00000941 | 0.44535396 | 0.01431968 |
| `conv2_x.2.act` | temporal_low_rank | 16 | 0.00021069 | 0.58999050 | 0.09549610 |
| `conv3_x.0.residual_function.2` | temporal_low_rank | 32 | 0.00008083 | 0.65082002 | 0.12423556 |
| `conv3_x.0.act` | temporal_low_rank | 32 | 0.00008271 | 0.67292136 | 0.08502857 |
| `conv3_x.1.residual_function.2` | temporal_low_rank | 32 | 0.00001054 | 0.35256928 | 0.03240393 |
| `conv3_x.1.act` | temporal_low_rank | 32 | 0.00005102 | 0.34419298 | 0.06624009 |
| `conv3_x.2.residual_function.2` | temporal_low_rank | 32 | 0.00000964 | 0.35966793 | 0.02727955 |
| `conv3_x.2.act` | temporal_low_rank | 32 | 0.00005028 | 0.29155639 | 0.05775257 |
| `conv4_x.0.residual_function.2` | temporal_low_rank | 64 | 0.00002346 | 0.40412691 | 0.03513613 |
| `conv4_x.0.act` | temporal_low_rank | 64 | 0.00005794 | 0.45105991 | 0.04005779 |
| `conv4_x.1.residual_function.2` | temporal_low_rank | 64 | 0.00001811 | 0.60165000 | 0.02425192 |
| `conv4_x.1.act` | temporal_low_rank | 64 | 0.00017842 | 0.51963776 | 0.08712445 |
| `conv4_x.2.residual_function.2` | temporal_low_rank | 64 | 0.00002268 | 0.64516366 | 0.03127158 |
| `conv4_x.2.act` | temporal_low_rank | 64 | 0.00196844 | 0.56322312 | 0.18404466 |

## Per-layer Parity-Anchor FTBC reconstruction

### T=1

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |

### T=2

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |

### T=4

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.0.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.1.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.residual_function.2` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv2_x.2.act` | full | 16 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.0.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.1.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.residual_function.2` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv3_x.2.act` | full | 32 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.0.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.1.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.residual_function.2` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |
| `conv4_x.2.act` | full | 64 | 0.00000000 | 0.00000000 | 0.00000000 |

### T=8

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | parity_anchor | 16 | 0.00106205 | 0.52448237 | 0.12757808 |
| `conv2_x.0.residual_function.2` | parity_anchor | 16 | 0.00060214 | 0.68705368 | 0.07470533 |
| `conv2_x.0.act` | parity_anchor | 16 | 0.00033894 | 0.45699966 | 0.07199962 |
| `conv2_x.1.residual_function.2` | parity_anchor | 16 | 0.00007913 | 0.42990687 | 0.02895692 |
| `conv2_x.1.act` | parity_anchor | 16 | 0.00007490 | 0.29436752 | 0.03589848 |
| `conv2_x.2.residual_function.2` | parity_anchor | 16 | 0.00000433 | 0.15825926 | 0.01034701 |
| `conv2_x.2.act` | parity_anchor | 16 | 0.00031622 | 0.39645669 | 0.09172133 |
| `conv3_x.0.residual_function.2` | parity_anchor | 32 | 0.00005053 | 0.26845357 | 0.04939508 |
| `conv3_x.0.act` | parity_anchor | 32 | 0.00003982 | 0.24759297 | 0.03247950 |
| `conv3_x.1.residual_function.2` | parity_anchor | 32 | 0.00000479 | 0.12049602 | 0.01347807 |
| `conv3_x.1.act` | parity_anchor | 32 | 0.00005336 | 0.17767791 | 0.03129493 |
| `conv3_x.2.residual_function.2` | parity_anchor | 32 | 0.00000332 | 0.10732184 | 0.00824043 |
| `conv3_x.2.act` | parity_anchor | 32 | 0.00007133 | 0.17482661 | 0.05348971 |
| `conv4_x.0.residual_function.2` | parity_anchor | 64 | 0.00001378 | 0.15855667 | 0.01750061 |
| `conv4_x.0.act` | parity_anchor | 64 | 0.00002769 | 0.16072747 | 0.03080345 |
| `conv4_x.1.residual_function.2` | parity_anchor | 64 | 0.00000729 | 0.20003851 | 0.01203356 |
| `conv4_x.1.act` | parity_anchor | 64 | 0.00008271 | 0.18478258 | 0.03004437 |
| `conv4_x.2.residual_function.2` | parity_anchor | 64 | 0.00001091 | 0.23405756 | 0.01474654 |
| `conv4_x.2.act` | parity_anchor | 64 | 0.00103007 | 0.21414866 | 0.13448668 |

### T=16

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | parity_anchor | 16 | 0.00174561 | 0.75144869 | 0.14386898 |
| `conv2_x.0.residual_function.2` | parity_anchor | 16 | 0.00069494 | 0.84731621 | 0.08587351 |
| `conv2_x.0.act` | parity_anchor | 16 | 0.00042334 | 0.64721590 | 0.08011271 |
| `conv2_x.1.residual_function.2` | parity_anchor | 16 | 0.00008405 | 0.57854468 | 0.04015774 |
| `conv2_x.1.act` | parity_anchor | 16 | 0.00012709 | 0.50711787 | 0.04242386 |
| `conv2_x.2.residual_function.2` | parity_anchor | 16 | 0.00000661 | 0.27100289 | 0.01162883 |
| `conv2_x.2.act` | parity_anchor | 16 | 0.00030395 | 0.52778417 | 0.10233067 |
| `conv3_x.0.residual_function.2` | parity_anchor | 32 | 0.00008857 | 0.49464330 | 0.09220473 |
| `conv3_x.0.act` | parity_anchor | 32 | 0.00006301 | 0.43159887 | 0.05808093 |
| `conv3_x.1.residual_function.2` | parity_anchor | 32 | 0.00000621 | 0.19261283 | 0.02092609 |
| `conv3_x.1.act` | parity_anchor | 32 | 0.00004896 | 0.23961347 | 0.04278341 |
| `conv3_x.2.residual_function.2` | parity_anchor | 32 | 0.00000576 | 0.19860737 | 0.01818691 |
| `conv3_x.2.act` | parity_anchor | 32 | 0.00006104 | 0.22796071 | 0.06251743 |
| `conv4_x.0.residual_function.2` | parity_anchor | 64 | 0.00002266 | 0.28443983 | 0.02122692 |
| `conv4_x.0.act` | parity_anchor | 64 | 0.00004868 | 0.29662514 | 0.04120139 |
| `conv4_x.1.residual_function.2` | parity_anchor | 64 | 0.00001458 | 0.38974965 | 0.01427149 |
| `conv4_x.1.act` | parity_anchor | 64 | 0.00013086 | 0.32102111 | 0.04710731 |
| `conv4_x.2.residual_function.2` | parity_anchor | 64 | 0.00001740 | 0.40671515 | 0.01738194 |
| `conv4_x.2.act` | parity_anchor | 64 | 0.00115167 | 0.31163740 | 0.15822034 |

### T=32

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | parity_anchor | 16 | 0.00221103 | 0.93003517 | 0.12986861 |
| `conv2_x.0.residual_function.2` | parity_anchor | 16 | 0.00069139 | 0.91524154 | 0.10287534 |
| `conv2_x.0.act` | parity_anchor | 16 | 0.00042990 | 0.76639581 | 0.08433926 |
| `conv2_x.1.residual_function.2` | parity_anchor | 16 | 0.00008141 | 0.65232801 | 0.04682724 |
| `conv2_x.1.act` | parity_anchor | 16 | 0.00012520 | 0.60517925 | 0.04753371 |
| `conv2_x.2.residual_function.2` | parity_anchor | 16 | 0.00000488 | 0.32075414 | 0.01171366 |
| `conv2_x.2.act` | parity_anchor | 16 | 0.00020616 | 0.58361423 | 0.11057956 |
| `conv3_x.0.residual_function.2` | parity_anchor | 32 | 0.00005995 | 0.56051558 | 0.10576607 |
| `conv3_x.0.act` | parity_anchor | 32 | 0.00004595 | 0.50153095 | 0.06852030 |
| `conv3_x.1.residual_function.2` | parity_anchor | 32 | 0.00000504 | 0.24392620 | 0.02477005 |
| `conv3_x.1.act` | parity_anchor | 32 | 0.00003174 | 0.27149910 | 0.05173846 |
| `conv3_x.2.residual_function.2` | parity_anchor | 32 | 0.00000482 | 0.25439703 | 0.02324148 |
| `conv3_x.2.act` | parity_anchor | 32 | 0.00003894 | 0.25659215 | 0.06676054 |
| `conv4_x.0.residual_function.2` | parity_anchor | 64 | 0.00002130 | 0.38503906 | 0.02457516 |
| `conv4_x.0.act` | parity_anchor | 64 | 0.00004700 | 0.40625399 | 0.04417882 |
| `conv4_x.1.residual_function.2` | parity_anchor | 64 | 0.00001511 | 0.54956508 | 0.02017480 |
| `conv4_x.1.act` | parity_anchor | 64 | 0.00012852 | 0.44103470 | 0.06693092 |
| `conv4_x.2.residual_function.2` | parity_anchor | 64 | 0.00001589 | 0.54003263 | 0.02304562 |
| `conv4_x.2.act` | parity_anchor | 64 | 0.00107766 | 0.41673583 | 0.16735208 |
