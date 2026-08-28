# ImageNet QCFS + Full/PA-FTBC + Standard/HA-SNM Ablation

Status: running

- Architecture: `resnet34`
- QCFS L: 8
- Checkpoint: `ImageNet-ResNet34-t8.pth`
- Checkpoint SHA256: `8f98b197a943aee0a1cb8971a04a7e1d1fed0cb80f5d32a0dd89c9bd6ece6bb2`
- Calibration tensor SHA256: `eafacb4fd9b2d65e2171a6f1e5aace1e9e1da4d5a281c60988ff829ef84c075b`
- ImageNet validation samples: 50,000
- Evaluation temporal batch budget: 32
- Protocol version: `imagenet-full-pa-ha-v1`
- Implementation SHA256: `016287623e9ed01ac468a619c39b7a90415f94f9ebb3ffda3b6658531606c3da`
- GPU: `NVIDIA GeForce RTX 5090`
- Total active elapsed: 109202.002s
- Published ANN reference: [ANN2SNN_SRP](https://github.com/hzc1208/ANN2SNN_SRP)
- All configurations use R0; HA-SNM is frozen at start=1.25, end=0.5, reference=8.
- Full-FTBC uses two fixed training images, 50 iterations, and alpha=0.5 in formal runs.
- PA-FTBC is constructed from the same Full-FTBC teacher and falls back to Full at T<=4.

## ANN reference

| Metric | Value |
|---|---:|
| Top-1 | 74.30% |
| Top-5 | 91.94% |
| Samples | 50,000 |
| Elapsed | 178.731s |

## Primary accuracy

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 2.41% | 4.95% | 12.92% | 40.13% | 64.83% | 71.95% | 32.87% |
| B_QCFS_STANDARD_SNM_R0 | 2.41% | 5.13% | 16.87% | 58.42% | 65.01% | 68.93% | 36.13% |
| C_QCFS_HA_SNM_R0 | 2.41% | 7.27% | 28.75% | 61.79% | 59.05% | 68.54% | 37.97% |
| D_QCFS_FULL_FTBC_R0 | 6.17% | 14.94% | 39.33% | 63.04% | 71.57% | - | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 6.17% | 15.25% | 42.94% | 67.29% | 72.47% | - | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 6.17% | 18.71% | 51.41% | 69.04% | 71.89% | - | - |
| G_QCFS_PA_FTBC_R0 | 6.17% | 14.94% | 39.33% | 62.42% | 70.99% | - | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 6.17% | 15.25% | 42.94% | 66.47% | 71.13% | - | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 6.17% | 18.71% | 51.41% | 68.05% | 69.84% | - | - |

## Top-5 accuracy

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 6.83% | 12.57% | 28.50% | 65.45% | 86.80% | 90.93% |
| B_QCFS_STANDARD_SNM_R0 | 6.83% | 13.13% | 35.30% | 82.16% | 85.92% | 88.57% |
| C_QCFS_HA_SNM_R0 | 6.83% | 17.89% | 52.79% | 83.82% | 80.98% | 88.20% |
| D_QCFS_FULL_FTBC_R0 | 16.82% | 33.28% | 65.05% | 85.36% | 90.64% | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 16.82% | 33.63% | 69.13% | 88.15% | 90.76% | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 16.82% | 39.33% | 76.63% | 88.98% | 90.51% | - |
| G_QCFS_PA_FTBC_R0 | 16.82% | 33.28% | 65.05% | 84.78% | 90.22% | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 16.82% | 33.63% | 69.13% | 87.47% | 90.19% | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 16.82% | 39.33% | 76.63% | 88.31% | 89.22% | - |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 9.04401471 | 8.56735024 | 6.89765876 | 3.93426032 | 1.62850454 | 0.58722800 |
| B_QCFS_STANDARD_SNM_R0 | 9.04401471 | 8.50621188 | 6.30126637 | 2.12437218 | 1.66830294 | 1.03908021 |
| C_QCFS_HA_SNM_R0 | 9.04401471 | 7.97475529 | 4.76073546 | 1.93189457 | 2.58403375 | 1.12744287 |
| D_QCFS_FULL_FTBC_R0 | 6.99784620 | 5.74973057 | 4.02836958 | 1.89432207 | 0.69976473 | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 6.99784620 | 5.73892105 | 3.81214515 | 1.25901471 | 0.45350218 | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 6.99784620 | 5.48974570 | 3.06505826 | 0.99930135 | 0.57211467 | - |
| G_QCFS_PA_FTBC_R0 | 6.99784620 | 5.74973057 | 4.02836958 | 1.96877799 | 0.82452564 | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 6.99784620 | 5.73892105 | 3.81214515 | 1.33700112 | 0.67024746 | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 6.99784620 | 5.48974570 | 3.06505826 | 1.11180595 | 0.93633246 | - |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 25.195410% | 24.897564% | 24.350222% | 24.002785% | 23.861763% | 23.797062% |
| B_QCFS_STANDARD_SNM_R0 | 25.195410% | 24.899432% | 24.445252% | 24.249412% | 24.192685% | 24.133654% |
| C_QCFS_HA_SNM_R0 | 25.195410% | 24.976303% | 24.619358% | 24.456968% | 24.265579% | 24.160524% |
| D_QCFS_FULL_FTBC_R0 | 23.486226% | 23.675256% | 23.770624% | 23.718436% | 23.729722% | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 23.486226% | 23.685860% | 23.846353% | 23.904532% | 23.973849% | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 23.486226% | 23.762808% | 23.981976% | 24.061594% | 24.026225% | - |
| G_QCFS_PA_FTBC_R0 | 23.486226% | 23.675256% | 23.770624% | 23.701016% | 23.724787% | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 23.486226% | 23.685860% | 23.846353% | 23.889611% | 23.992942% | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 23.486226% | 23.762808% | 23.981976% | 24.050057% | 24.053192% | - |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_STANDARD_SNM_R0 | 0.000000% | 0.084387% | 0.200930% | 0.322377% | 0.376843% | 0.369734% |
| C_QCFS_HA_SNM_R0 | 0.000000% | 0.371784% | 0.531292% | 0.628797% | 0.466313% | 0.399280% |
| D_QCFS_FULL_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.041598% | 0.128216% | 0.237629% | 0.287632% | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0.000000% | 0.212788% | 0.381935% | 0.482538% | 0.357042% | - |
| G_QCFS_PA_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.041598% | 0.128216% | 0.239000% | 0.300380% | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0.000000% | 0.212788% | 0.381935% | 0.484310% | 0.373507% | - |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 74.804590% | 75.102436% | 75.649778% | 75.997215% | 76.138237% | 76.202938% |
| B_QCFS_STANDARD_SNM_R0 | 74.804590% | 75.016181% | 75.353819% | 75.428211% | 75.430472% | 75.496612% |
| C_QCFS_HA_SNM_R0 | 74.804590% | 74.651913% | 74.849349% | 74.914235% | 75.268108% | 75.440196% |
| D_QCFS_FULL_FTBC_R0 | 76.513774% | 76.324744% | 76.229376% | 76.281564% | 76.270278% | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 76.513774% | 76.272541% | 76.025431% | 75.857839% | 75.738519% | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 76.513774% | 76.024404% | 75.636090% | 75.455867% | 75.616733% | - |
| G_QCFS_PA_FTBC_R0 | 76.513774% | 76.324744% | 76.229376% | 76.298984% | 76.275213% | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 76.513774% | 76.272541% | 76.025431% | 75.871389% | 75.706678% | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 76.513774% | 76.024404% | 75.636090% | 75.465633% | 75.573301% | - |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 668,256,217,881,880 | 1,333,801,030,135,608 | 2,602,178,696,109,136 | 5,095,071,238,389,304 | 10,098,489,621,275,664 | 20,122,128,147,130,504 |
| B_QCFS_STANDARD_SNM_R0 | 668,256,217,881,880 | 1,337,520,538,798,976 | 2,643,164,115,858,552 | 5,285,423,884,864,744 | 10,610,142,242,666,888 | 21,164,504,133,493,328 |
| C_QCFS_HA_SNM_R0 | 668,256,217,881,880 | 1,361,530,059,342,744 | 2,729,714,704,995,688 | 5,483,527,140,419,768 | 10,743,525,482,823,696 | 21,258,588,861,298,096 |
| D_QCFS_FULL_FTBC_R0 | 618,515,557,846,424 | 1,245,235,099,659,248 | 2,509,524,718,221,440 | 5,005,514,271,961,552 | 10,026,620,876,187,672 | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 618,515,557,846,424 | 1,247,759,556,916,272 | 2,539,021,730,402,224 | 5,149,585,090,459,768 | 10,398,902,781,063,728 | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 618,515,557,846,424 | 1,265,565,169,628,232 | 2,606,985,249,373,704 | 5,299,735,445,126,280 | 10,491,828,127,973,720 | - |
| G_QCFS_PA_FTBC_R0 | 618,515,557,846,424 | 1,245,235,099,659,248 | 2,509,524,718,221,440 | 4,999,359,072,613,048 | 10,023,909,963,301,320 | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 618,515,557,846,424 | 1,247,759,556,916,272 | 2,539,021,730,402,224 | 5,144,519,118,810,304 | 10,430,021,593,952,072 | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 618,515,557,846,424 | 1,265,565,169,628,232 | 2,606,985,249,373,704 | 5,296,844,222,331,360 | 10,534,604,299,679,128 | - |

## FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_HA_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 7,616 | 15,232 | 30,464 | 60,928 | 121,856 | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 7,616 | 15,232 | 30,464 | 60,928 | 121,856 | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 7,616 | 15,232 | 30,464 | 60,928 | 121,856 | - |
| G_QCFS_PA_FTBC_R0 | 7,616 | 15,232 | 30,464 | 30,464 | 30,464 | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 7,616 | 15,232 | 30,464 | 30,464 | 30,464 | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 7,616 | 15,232 | 30,464 | 30,464 | 30,464 | - |

## FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_HA_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 30,464 | 60,928 | 121,856 | 243,712 | 487,424 | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 30,464 | 60,928 | 121,856 | 243,712 | 487,424 | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 30,464 | 60,928 | 121,856 | 243,712 | 487,424 | - |
| G_QCFS_PA_FTBC_R0 | 30,464 | 60,928 | 121,856 | 121,856 | 121,856 | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 30,464 | 60,928 | 121,856 | 121,856 | 121,856 | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 30,464 | 60,928 | 121,856 | 121,856 | 121,856 | - |

## FTBC synthesis MACs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_HA_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 0 | 0 | 0 | 0 | 0 | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0 | 0 | 0 | 0 | 0 | - |
| G_QCFS_PA_FTBC_R0 | 0 | 0 | 0 | 106,624 | 228,480 | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 106,624 | 228,480 | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0 | 0 | 0 | 106,624 | 228,480 | - |

## Peak CUDA memory

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 15.508 GiB | 8.657 GiB | 5.398 GiB | 3.994 GiB | 3.282 GiB | 2.928 GiB |
| B_QCFS_STANDARD_SNM_R0 | 15.604 GiB | 8.704 GiB | 5.422 GiB | 3.988 GiB | 3.282 GiB | 2.928 GiB |
| C_QCFS_HA_SNM_R0 | 15.604 GiB | 8.704 GiB | 5.422 GiB | 3.988 GiB | 3.282 GiB | 2.928 GiB |
| D_QCFS_FULL_FTBC_R0 | 15.125 GiB | 8.466 GiB | 5.398 GiB | 3.989 GiB | 3.282 GiB | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 15.221 GiB | 8.513 GiB | 5.398 GiB | 3.987 GiB | 3.283 GiB | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 15.221 GiB | 8.513 GiB | 5.398 GiB | 3.987 GiB | 3.283 GiB | - |
| G_QCFS_PA_FTBC_R0 | 15.125 GiB | 8.466 GiB | 5.398 GiB | 3.988 GiB | 3.282 GiB | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 15.221 GiB | 8.513 GiB | 5.398 GiB | 3.986 GiB | 3.283 GiB | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 15.221 GiB | 8.513 GiB | 5.398 GiB | 3.986 GiB | 3.283 GiB | - |

## Evaluation elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 336.204s | 640.413s | 1169.445s | 2056.105s | 3940.386s | 11115.933s |
| B_QCFS_STANDARD_SNM_R0 | 356.734s | 678.664s | 1242.760s | 2205.698s | 4239.941s | 12657.695s |
| C_QCFS_HA_SNM_R0 | 356.661s | 679.051s | 1244.698s | 2213.586s | 4273.769s | 13112.810s |
| D_QCFS_FULL_FTBC_R0 | 313.540s | 608.557s | 1123.485s | 1989.765s | 3858.957s | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 333.436s | 647.309s | 1197.500s | 2144.638s | 4145.475s | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 333.355s | 647.744s | 1199.293s | 2152.407s | 4179.842s | - |
| G_QCFS_PA_FTBC_R0 | 313.540s | 608.557s | 1123.485s | 1999.347s | 3903.748s | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 333.436s | 647.309s | 1197.500s | 2152.782s | 4262.986s | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 333.355s | 647.744s | 1199.293s | 2160.789s | 4427.851s | - |

## Pure inference seconds/image

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.00595268 | 0.01051743 | 0.01911194 | 0.03310329 | 0.06206376 | 0.14014039 |
| B_QCFS_STANDARD_SNM_R0 | 0.00626288 | 0.01133161 | 0.02058975 | 0.03602976 | 0.06817486 | 0.16728802 |
| C_QCFS_HA_SNM_R0 | 0.00623181 | 0.01129074 | 0.02065664 | 0.03612496 | 0.06875209 | 0.17336367 |
| D_QCFS_FULL_FTBC_R0 | 0.00536982 | 0.00988853 | 0.01817298 | 0.03182469 | 0.06030648 | - |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.00577824 | 0.01069767 | 0.01969656 | 0.03486690 | 0.06642101 | - |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0.00570327 | 0.01070984 | 0.01961992 | 0.03497057 | 0.06709638 | - |
| G_QCFS_PA_FTBC_R0 | 0.00536982 | 0.00988853 | 0.01817298 | 0.03194795 | 0.06124336 | - |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.00577824 | 0.01069767 | 0.01969656 | 0.03499777 | 0.06735229 | - |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0.00570327 | 0.01070984 | 0.01961992 | 0.03524742 | 0.06794588 | - |

## Full-FTBC calibration

| T | Elapsed | Peak CUDA memory | Parameters | Bytes | Schedule SHA256 |
|---:|---:|---:|---:|---:|---|
| 1 | 25.373s | 1.328 GiB | 7,616 | 30,464 | `7430c854d54da34ef79dc9c21d75d03aea83687c04c338e6b726a5a14dbc20ab` |
| 2 | 41.978s | 1.409 GiB | 15,232 | 60,928 | `232613505170512516d1f9b78a4514a20be626d56f8e18a97bf52e4c1d5571ff` |
| 4 | 80.353s | 1.737 GiB | 30,464 | 121,856 | `10ea20742eda5d55e839e670116d1c29c0f3dd85039e9118189320437be461c2` |
| 8 | 138.927s | 2.532 GiB | 60,928 | 243,712 | `c418098c97415871e58708c9dd3a52ff1ab8a2bf4827cfdf78fdf529ae806bb2` |
| 16 | 269.369s | 4.072 GiB | 121,856 | 487,424 | `f2e68a82a0361d3633a7f02ae7881a4c86bea0abae444c2c4cccdd5e065d23cd` |
| 32 | 537.108s | 7.154 GiB | 243,712 | 974,848 | `86d9541dd06a6c4f5483260ebdc6974458c56b6009a5881cf7aadd25fcaa6d91` |

## PA-FTBC compression

| T | Representation | Full params | PA params | Storage saving | Explained energy | Compression elapsed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Full-FTBC fallback | 7,616 | 7,616 | 0.00% | 1.00000000 | 0.000s |
| 2 | Full-FTBC fallback | 15,232 | 15,232 | 0.00% | 1.00000000 | 0.000s |
| 4 | Full-FTBC fallback | 30,464 | 30,464 | 0.00% | 1.00000000 | 0.000s |
| 8 | t0 anchor + t1 anchor + tail mean + tail parity | 60,928 | 30,464 | 50.00% | 0.94282972 | 0.008s |
| 16 | t0 anchor + t1 anchor + tail mean + tail parity | 121,856 | 30,464 | 75.00% | 0.88011711 | 0.009s |
| 32 | - | - | - | - | - | - |

## Exact fallback checks

| T | Mode | Full config | PA config | Exact |
|---:|---|---|---|---|
| 1 | off | D_QCFS_FULL_FTBC_R0 | G_QCFS_PA_FTBC_R0 | yes |
| 1 | standard | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | H_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| 1 | ha | F_QCFS_FULL_FTBC_HA_SNM_R0 | I_QCFS_PA_FTBC_HA_SNM_R0 | yes |
| 2 | off | D_QCFS_FULL_FTBC_R0 | G_QCFS_PA_FTBC_R0 | yes |
| 2 | standard | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | H_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| 2 | ha | F_QCFS_FULL_FTBC_HA_SNM_R0 | I_QCFS_PA_FTBC_HA_SNM_R0 | yes |
| 4 | off | D_QCFS_FULL_FTBC_R0 | G_QCFS_PA_FTBC_R0 | yes |
| 4 | standard | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | H_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| 4 | ha | F_QCFS_FULL_FTBC_HA_SNM_R0 | I_QCFS_PA_FTBC_HA_SNM_R0 | yes |

## Completion checks

| Check | T | Expected | Passed |
|---|---:|---:|---|
