# ImageNet QCFS + Full/PA-FTBC + Standard/HA-SNM Ablation

Status: complete

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
- Total active elapsed: 32.123s
- Published ANN reference: [ANN2SNN_SRP](https://github.com/hzc1208/ANN2SNN_SRP)
- All configurations use R0; HA-SNM is frozen at start=1.25, end=0.5, reference=8.
- Full-FTBC uses two fixed training images, 50 iterations, and alpha=0.5 in formal runs.
- PA-FTBC is constructed from the same Full-FTBC teacher and falls back to Full at T<=4.

## ANN reference

| Metric | Value |
|---|---:|
| Top-1 | 87.50% |
| Top-5 | 96.88% |
| Samples | 64 |
| Elapsed | 1.120s |

## Primary accuracy

| Config | T=4 | T=8 | T=32 | Mean |
|---|---:|---:|---:|---:|
| A_QCFS_R0 | 18.75% | 62.50% | 50.00% | 43.75% |
| B_QCFS_STANDARD_SNM_R0 | 18.75% | 50.00% | 50.00% | 39.58% |
| C_QCFS_HA_SNM_R0 | 56.25% | 50.00% | 50.00% | 52.08% |
| D_QCFS_FULL_FTBC_R0 | 68.75% | 62.50% | 50.00% | 60.42% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 68.75% | 62.50% | 50.00% | 60.42% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 68.75% | 62.50% | 50.00% | 60.42% |
| G_QCFS_PA_FTBC_R0 | 68.75% | 62.50% | 50.00% | 60.42% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 68.75% | 62.50% | 50.00% | 60.42% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 68.75% | 62.50% | 50.00% | 60.42% |

## Top-5 accuracy

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 25.00% | 87.50% | 100.00% |
| B_QCFS_STANDARD_SNM_R0 | 56.25% | 100.00% | 100.00% |
| C_QCFS_HA_SNM_R0 | 75.00% | 87.50% | 100.00% |
| D_QCFS_FULL_FTBC_R0 | 75.00% | 100.00% | 100.00% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 81.25% | 100.00% | 100.00% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 93.75% | 100.00% | 100.00% |
| G_QCFS_PA_FTBC_R0 | 75.00% | 100.00% | 100.00% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 81.25% | 100.00% | 100.00% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 93.75% | 100.00% | 100.00% |

## ANN-SNN logit MSE

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 8.22378564 | 4.30505286 | 0.91611826 |
| B_QCFS_STANDARD_SNM_R0 | 7.61667285 | 2.22381006 | 0.82328467 |
| C_QCFS_HA_SNM_R0 | 5.81891211 | 1.76318573 | 0.81091943 |
| D_QCFS_FULL_FTBC_R0 | 3.73464026 | 2.04835742 | 0.40034985 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 3.47734607 | 1.35257227 | 0.40609464 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 2.63874048 | 1.09932166 | 0.38530907 |
| G_QCFS_PA_FTBC_R0 | 3.73464026 | 2.08678674 | 0.40266443 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 3.47734607 | 1.39436328 | 0.55176418 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 2.63874048 | 1.10007654 | 0.52883467 |

## Positive spike rate

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 24.734081% | 24.437689% | 23.969211% |
| B_QCFS_STANDARD_SNM_R0 | 24.817905% | 24.635919% | 24.233432% |
| C_QCFS_HA_SNM_R0 | 24.963328% | 24.813966% | 24.253985% |
| D_QCFS_FULL_FTBC_R0 | 24.495167% | 24.328870% | 23.958117% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 24.575716% | 24.491119% | 24.172275% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 24.704892% | 24.630277% | 24.186675% |
| G_QCFS_PA_FTBC_R0 | 24.495167% | 24.332217% | 23.957067% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 24.575716% | 24.495499% | 24.184974% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 24.704892% | 24.636373% | 24.200226% |

## Negative spike rate

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_STANDARD_SNM_R0 | 0.158480% | 0.260517% | 0.295900% |
| C_QCFS_HA_SNM_R0 | 0.445247% | 0.535573% | 0.319429% |
| D_QCFS_FULL_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.147363% | 0.221594% | 0.248959% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0.410332% | 0.450624% | 0.267560% |
| G_QCFS_PA_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.147363% | 0.222429% | 0.258473% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0.410332% | 0.452430% | 0.277899% |

## Overall spike sparsity

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 75.265919% | 75.562311% | 76.030789% |
| B_QCFS_STANDARD_SNM_R0 | 75.023615% | 75.103564% | 75.470668% |
| C_QCFS_HA_SNM_R0 | 74.591426% | 74.650461% | 75.426586% |
| D_QCFS_FULL_FTBC_R0 | 75.504833% | 75.671130% | 76.041883% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 75.276921% | 75.287287% | 75.578767% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 74.884776% | 74.919099% | 75.545765% |
| G_QCFS_PA_FTBC_R0 | 75.504833% | 75.667783% | 76.042933% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 75.276921% | 75.282073% | 75.556553% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 74.884776% | 74.911197% | 75.521874% |

## Input-driven SOPs

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 847,410,434,864 | 831,053,193,936 | 812,588,947,992 |
| B_QCFS_STANDARD_SNM_R0 | 859,181,161,136 | 857,697,310,536 | 846,411,056,640 |
| C_QCFS_HA_SNM_R0 | 884,196,129,992 | 887,168,517,208 | 849,346,477,424 |
| D_QCFS_FULL_FTBC_R0 | 833,318,702,576 | 823,826,059,184 | 811,876,135,704 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 844,079,317,208 | 845,201,809,464 | 837,987,536,968 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 866,417,686,256 | 868,150,346,968 | 839,992,918,120 |
| G_QCFS_PA_FTBC_R0 | 833,318,702,576 | 824,267,861,344 | 811,845,230,560 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 844,079,317,208 | 845,750,410,328 | 839,875,040,472 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 866,417,686,256 | 868,909,871,072 | 842,004,233,496 |

## FTBC parameters

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 |
| C_QCFS_HA_SNM_R0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 30,464 | 60,928 | 243,712 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 30,464 | 60,928 | 243,712 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 30,464 | 60,928 | 243,712 |
| G_QCFS_PA_FTBC_R0 | 30,464 | 30,464 | 30,464 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 30,464 | 30,464 | 30,464 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 30,464 | 30,464 | 30,464 |

## FTBC storage bytes

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 |
| C_QCFS_HA_SNM_R0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 121,856 | 243,712 | 974,848 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 121,856 | 243,712 | 974,848 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 121,856 | 243,712 | 974,848 |
| G_QCFS_PA_FTBC_R0 | 121,856 | 121,856 | 121,856 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 121,856 | 121,856 | 121,856 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 121,856 | 121,856 | 121,856 |

## FTBC synthesis MACs

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 |
| C_QCFS_HA_SNM_R0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 0 | 0 | 0 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0 | 0 | 0 |
| G_QCFS_PA_FTBC_R0 | 0 | 106,624 | 472,192 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0 | 106,624 | 472,192 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0 | 106,624 | 472,192 |

## Peak CUDA memory

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 5.398 GiB | 3.989 GiB | 2.927 GiB |
| B_QCFS_STANDARD_SNM_R0 | 5.422 GiB | 3.986 GiB | 2.926 GiB |
| C_QCFS_HA_SNM_R0 | 5.422 GiB | 3.986 GiB | 2.926 GiB |
| D_QCFS_FULL_FTBC_R0 | 5.398 GiB | 3.986 GiB | 2.928 GiB |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 5.398 GiB | 3.984 GiB | 2.928 GiB |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 5.398 GiB | 3.984 GiB | 2.928 GiB |
| G_QCFS_PA_FTBC_R0 | 5.398 GiB | 3.986 GiB | 2.927 GiB |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 5.398 GiB | 3.985 GiB | 2.927 GiB |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 5.398 GiB | 3.985 GiB | 2.927 GiB |

## Evaluation elapsed

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 0.659s | 0.514s | 0.733s |
| B_QCFS_STANDARD_SNM_R0 | 0.576s | 0.519s | 0.611s |
| C_QCFS_HA_SNM_R0 | 0.602s | 0.524s | 0.634s |
| D_QCFS_FULL_FTBC_R0 | 0.562s | 0.485s | 0.557s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.556s | 0.510s | 0.598s |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0.577s | 0.512s | 0.620s |
| G_QCFS_PA_FTBC_R0 | 0.562s | 0.480s | 0.589s |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.556s | 0.519s | 0.621s |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0.577s | 0.524s | 0.657s |

## Pure inference seconds/image

| Config | T=4 | T=8 | T=32 |
|---|---:|---:|---:|
| A_QCFS_R0 | 0.00000000 | 0.00000000 | 0.00000000 |
| B_QCFS_STANDARD_SNM_R0 | 0.00000000 | 0.00000000 | 0.00000000 |
| C_QCFS_HA_SNM_R0 | 0.00000000 | 0.00000000 | 0.00000000 |
| D_QCFS_FULL_FTBC_R0 | 0.00000000 | 0.00000000 | 0.00000000 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.00000000 | 0.00000000 | 0.00000000 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0.00000000 | 0.00000000 | 0.00000000 |
| G_QCFS_PA_FTBC_R0 | 0.00000000 | 0.00000000 | 0.00000000 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.00000000 | 0.00000000 | 0.00000000 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0.00000000 | 0.00000000 | 0.00000000 |

## Full-FTBC calibration

| T | Elapsed | Peak CUDA memory | Parameters | Bytes | Schedule SHA256 |
|---:|---:|---:|---:|---:|---|
| 4 | 1.673s | 1.735 GiB | 30,464 | 121,856 | `08a70a150c938ecb53e8575b4bac70b9c4c73b675291cde22fd89e451fa289f3` |
| 8 | 2.770s | 2.531 GiB | 60,928 | 243,712 | `4722b7b88c3c64f728fe9a3d9585dc07ffbe501b085806c36445201eddfa1cd5` |
| 32 | 10.556s | 7.154 GiB | 243,712 | 974,848 | `9d801d03a4e6b0fe5cd60dd0c1de5783dc3355dbfbd23da5a8b492c2e4c2be06` |

## PA-FTBC compression

| T | Representation | Full params | PA params | Storage saving | Explained energy | Compression elapsed |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Full-FTBC fallback | 30,464 | 30,464 | 0.00% | 1.00000000 | 0.000s |
| 8 | t0 anchor + t1 anchor + tail mean + tail parity | 60,928 | 30,464 | 50.00% | 0.87262060 | 0.008s |
| 32 | t0 anchor + t1 anchor + tail mean + tail parity | 243,712 | 30,464 | 87.50% | 0.61479295 | 0.008s |

## Exact fallback checks

| T | Mode | Full config | PA config | Exact |
|---:|---|---|---|---|
| 4 | off | D_QCFS_FULL_FTBC_R0 | G_QCFS_PA_FTBC_R0 | yes |
| 4 | standard | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | H_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| 4 | ha | F_QCFS_FULL_FTBC_HA_SNM_R0 | I_QCFS_PA_FTBC_HA_SNM_R0 | yes |

## Completion checks

| Check | T | Expected | Passed |
|---|---:|---:|---|
| ann_sample_count | - | 64 | yes |
| evaluation_sample_count | 4 | 16 | yes |
| pa_storage_and_fallback | 4 | - | yes |
| evaluation_sample_count | 8 | 8 | yes |
| pa_storage_and_fallback | 8 | - | yes |
| evaluation_sample_count | 32 | 2 | yes |
| pa_storage_and_fallback | 32 | - | yes |
| fallback_cache_count | - | 3 | yes |
