# ImageNet QCFS + Full/PA-FTBC + Standard/HA-SNM Ablation

Status: complete

- Architecture: `vgg16`
- QCFS L: 16
- Checkpoint: `ImageNet-VGG16-t16.pth`
- Checkpoint SHA256: `4027d8f06497dd34718fb0e2be910768a22c64d116e4e6af4c58a80a4b5422c6`
- Calibration tensor SHA256: `eafacb4fd9b2d65e2171a6f1e5aace1e9e1da4d5a281c60988ff829ef84c075b`
- ImageNet validation samples: 50,000
- Evaluation temporal batch budget: 32
- Protocol version: `imagenet-full-pa-ha-v1`
- Implementation SHA256: `016287623e9ed01ac468a619c39b7a90415f94f9ebb3ffda3b6658531606c3da`
- GPU: `NVIDIA GeForce RTX 5090`
- Total active elapsed: 70448.078s
- Published ANN reference: [ANN2SNN_SRP](https://github.com/hzc1208/ANN2SNN_SRP)
- All configurations use R0; HA-SNM is frozen at start=1.25, end=0.5, reference=8.
- Full-FTBC uses two fixed training images, 50 iterations, and alpha=0.5 in formal runs.
- PA-FTBC is constructed from the same Full-FTBC teacher and falls back to Full at T<=4.

## ANN reference

| Metric | Value |
|---|---:|
| Top-1 | 74.25% |
| Top-5 | 91.93% |
| Samples | 50,000 |
| Elapsed | 52.157s |

## Primary accuracy

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.69% | 1.50% | 5.35% | 33.12% | 66.96% | 73.05% | 30.11% |
| B_QCFS_STANDARD_SNM_R0 | 0.69% | 1.58% | 7.98% | 59.30% | 67.47% | 71.26% | 34.71% |
| C_QCFS_HA_SNM_R0 | 0.69% | 2.52% | 21.40% | 55.91% | 64.87% | 70.94% | 36.06% |
| D_QCFS_FULL_FTBC_R0 | 15.92% | 30.67% | 53.56% | 68.56% | 73.00% | 74.03% | 52.62% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 15.92% | 30.97% | 54.95% | 70.25% | 73.10% | 72.62% | 52.97% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 15.92% | 33.44% | 58.25% | 71.03% | 72.81% | 72.51% | 53.99% |
| G_QCFS_PA_FTBC_R0 | 15.92% | 30.67% | 53.56% | 68.91% | 73.10% | 74.00% | 52.69% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 15.92% | 30.97% | 54.95% | 70.33% | 73.13% | 72.75% | 53.01% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 15.92% | 33.44% | 58.25% | 71.11% | 72.85% | 72.73% | 54.05% |

## Top-5 accuracy

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 2.47% | 4.45% | 13.71% | 60.05% | 87.82% | 91.30% |
| B_QCFS_STANDARD_SNM_R0 | 2.47% | 4.80% | 20.05% | 82.79% | 87.00% | 89.38% |
| C_QCFS_HA_SNM_R0 | 2.47% | 7.02% | 44.59% | 77.30% | 84.75% | 89.06% |
| D_QCFS_FULL_FTBC_R0 | 32.34% | 52.97% | 77.40% | 88.46% | 91.18% | 91.83% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 32.34% | 53.46% | 78.64% | 89.54% | 91.05% | 90.17% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 32.34% | 56.72% | 81.54% | 89.89% | 90.85% | 90.08% |
| G_QCFS_PA_FTBC_R0 | 32.34% | 52.97% | 77.40% | 88.63% | 91.30% | 91.81% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 32.34% | 53.46% | 78.64% | 89.72% | 91.15% | 90.39% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 32.34% | 56.72% | 81.54% | 90.08% | 90.92% | 90.35% |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 2.66850828 | 2.10827508 | 1.77109800 | 1.05484392 | 0.37030192 | 0.10417200 |
| B_QCFS_STANDARD_SNM_R0 | 2.66850828 | 2.10052061 | 1.63770311 | 0.52912456 | 0.39452248 | 0.17939055 |
| C_QCFS_HA_SNM_R0 | 2.66850828 | 2.01656248 | 1.08466509 | 0.71463318 | 0.49751320 | 0.19239659 |
| D_QCFS_FULL_FTBC_R0 | 2.29150389 | 1.24491107 | 0.69777997 | 0.31508466 | 0.12400896 | 0.05816039 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2.29150389 | 1.23946912 | 0.65619771 | 0.23412607 | 0.10353738 | 0.09828803 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 2.29150389 | 1.17436125 | 0.56362199 | 0.20316077 | 0.11598352 | 0.10296220 |
| G_QCFS_PA_FTBC_R0 | 2.29150389 | 1.24491107 | 0.69777997 | 0.31043448 | 0.12203131 | 0.05750571 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 2.29150389 | 1.23946912 | 0.65619771 | 0.23313094 | 0.10261197 | 0.09567972 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 2.29150389 | 1.17436125 | 0.56362199 | 0.20207648 | 0.11433771 | 0.09876283 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 14.080592% | 16.350289% | 16.329020% | 16.129923% | 16.033499% | 15.981509% |
| B_QCFS_STANDARD_SNM_R0 | 14.080592% | 16.372886% | 16.417905% | 16.277994% | 16.175243% | 16.106683% |
| C_QCFS_HA_SNM_R0 | 14.080592% | 16.478814% | 16.564689% | 16.398138% | 16.205162% | 16.115530% |
| D_QCFS_FULL_FTBC_R0 | 15.694121% | 15.723408% | 15.804253% | 15.880210% | 15.902719% | 15.909067% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 15.694121% | 15.735997% | 15.849918% | 15.967653% | 16.005301% | 16.013572% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 15.694121% | 15.794004% | 15.936388% | 16.050876% | 16.029856% | 16.021667% |
| G_QCFS_PA_FTBC_R0 | 15.694121% | 15.723408% | 15.804253% | 15.871358% | 15.899711% | 15.908181% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 15.694121% | 15.735997% | 15.849918% | 15.957823% | 16.001975% | 16.012233% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 15.694121% | 15.794004% | 15.936388% | 16.039205% | 16.026275% | 16.019125% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_STANDARD_SNM_R0 | 0.000000% | 0.032899% | 0.107417% | 0.173132% | 0.165484% | 0.139264% |
| C_QCFS_HA_SNM_R0 | 0.000000% | 0.256123% | 0.340789% | 0.329514% | 0.199899% | 0.148859% |
| D_QCFS_FULL_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.015587% | 0.055003% | 0.098637% | 0.108561% | 0.106847% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0.000000% | 0.126117% | 0.191969% | 0.205844% | 0.135588% | 0.115428% |
| G_QCFS_PA_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.015587% | 0.055003% | 0.096530% | 0.106518% | 0.104734% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0.000000% | 0.126117% | 0.191969% | 0.199823% | 0.133281% | 0.112659% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 85.919408% | 83.649711% | 83.670980% | 83.870077% | 83.966501% | 84.018491% |
| B_QCFS_STANDARD_SNM_R0 | 85.919408% | 83.594214% | 83.474678% | 83.548874% | 83.659273% | 83.754054% |
| C_QCFS_HA_SNM_R0 | 85.919408% | 83.265064% | 83.094522% | 83.272348% | 83.594939% | 83.735611% |
| D_QCFS_FULL_FTBC_R0 | 84.305879% | 84.276592% | 84.195747% | 84.119790% | 84.097281% | 84.090933% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 84.305879% | 84.248415% | 84.095078% | 83.933709% | 83.886138% | 83.879581% |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 84.305879% | 84.079880% | 83.871643% | 83.743280% | 83.834556% | 83.862904% |
| G_QCFS_PA_FTBC_R0 | 84.305879% | 84.276592% | 84.195747% | 84.128642% | 84.100289% | 84.091819% |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 84.305879% | 84.248415% | 84.095078% | 83.945647% | 83.891506% | 83.883033% |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 84.305879% | 84.079880% | 83.871643% | 83.760972% | 83.840444% | 83.868216% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 131,155,249,608,128 | 290,760,361,352,768 | 567,988,951,151,232 | 1,107,119,316,313,792 | 2,187,978,863,135,680 | 4,348,185,344,767,168 |
| B_QCFS_STANDARD_SNM_R0 | 131,155,249,608,128 | 292,552,554,835,008 | 583,385,630,142,592 | 1,161,081,157,439,936 | 2,290,051,204,818,240 | 4,522,816,194,946,112 |
| C_QCFS_HA_SNM_R0 | 131,155,249,608,128 | 303,218,093,749,184 | 614,455,470,585,600 | 1,208,437,532,067,008 | 2,311,793,492,418,112 | 4,535,217,484,495,168 |
| D_QCFS_FULL_FTBC_R0 | 131,258,617,689,088 | 266,470,406,732,736 | 536,638,633,296,896 | 1,078,171,430,885,376 | 2,159,108,200,552,256 | 4,319,353,020,979,648 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 131,258,617,689,088 | 267,273,745,073,088 | 543,548,844,228,480 | 1,105,306,435,572,864 | 2,222,437,318,112,576 | 4,450,976,526,112,448 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 131,258,617,689,088 | 272,491,520,740,416 | 560,040,583,267,328 | 1,135,173,253,204,864 | 2,239,027,762,423,488 | 4,461,968,743,608,256 |
| G_QCFS_PA_FTBC_R0 | 131,258,617,689,088 | 266,470,406,732,736 | 536,638,633,296,896 | 1,077,057,517,096,064 | 2,157,993,266,628,288 | 4,318,513,088,491,648 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 131,258,617,689,088 | 267,273,745,073,088 | 543,548,844,228,480 | 1,103,839,376,672,128 | 2,220,760,659,735,744 | 4,448,798,330,771,072 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 131,258,617,689,088 | 272,491,520,740,416 | 560,040,583,267,328 | 1,132,970,311,363,968 | 2,237,192,906,650,944 | 4,458,179,015,082,752 |

## FTBC parameters

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_HA_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 12,416 | 24,832 | 49,664 | 99,328 | 198,656 | 397,312 |
| G_QCFS_PA_FTBC_R0 | 12,416 | 24,832 | 49,664 | 49,664 | 49,664 | 49,664 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 12,416 | 24,832 | 49,664 | 49,664 | 49,664 | 49,664 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 12,416 | 24,832 | 49,664 | 49,664 | 49,664 | 49,664 |

## FTBC storage bytes

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_HA_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 49,664 | 99,328 | 198,656 | 397,312 | 794,624 | 1,589,248 |
| G_QCFS_PA_FTBC_R0 | 49,664 | 99,328 | 198,656 | 198,656 | 198,656 | 198,656 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 49,664 | 99,328 | 198,656 | 198,656 | 198,656 | 198,656 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 49,664 | 99,328 | 198,656 | 198,656 | 198,656 | 198,656 |

## FTBC synthesis MACs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_QCFS_HA_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0 | 0 | 0 | 0 | 0 | 0 |
| G_QCFS_PA_FTBC_R0 | 0 | 0 | 0 | 173,824 | 372,480 | 769,792 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0 | 0 | 0 | 173,824 | 372,480 | 769,792 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0 | 0 | 0 | 173,824 | 372,480 | 769,792 |

## Peak CUDA memory

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 7.984 GiB | 5.164 GiB | 3.921 GiB | 3.518 GiB | 3.311 GiB | 3.211 GiB |
| B_QCFS_STANDARD_SNM_R0 | 8.080 GiB | 5.211 GiB | 3.945 GiB | 3.518 GiB | 3.312 GiB | 3.211 GiB |
| C_QCFS_HA_SNM_R0 | 8.080 GiB | 5.211 GiB | 3.945 GiB | 3.518 GiB | 3.312 GiB | 3.211 GiB |
| D_QCFS_FULL_FTBC_R0 | 7.601 GiB | 4.972 GiB | 3.921 GiB | 3.518 GiB | 3.312 GiB | 3.213 GiB |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 7.697 GiB | 5.019 GiB | 3.921 GiB | 3.517 GiB | 3.313 GiB | 3.213 GiB |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 7.697 GiB | 5.019 GiB | 3.921 GiB | 3.517 GiB | 3.313 GiB | 3.213 GiB |
| G_QCFS_PA_FTBC_R0 | 7.601 GiB | 4.972 GiB | 3.921 GiB | 3.518 GiB | 3.311 GiB | 3.211 GiB |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 7.697 GiB | 5.019 GiB | 3.921 GiB | 3.517 GiB | 3.312 GiB | 3.211 GiB |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 7.697 GiB | 5.019 GiB | 3.921 GiB | 3.517 GiB | 3.312 GiB | 3.211 GiB |

## Evaluation elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 96.426s | 184.087s | 341.204s | 622.919s | 1342.811s | 4135.299s |
| B_QCFS_STANDARD_SNM_R0 | 102.207s | 194.721s | 362.581s | 670.658s | 1504.572s | 4778.891s |
| C_QCFS_HA_SNM_R0 | 102.204s | 195.042s | 363.594s | 674.565s | 1549.356s | 5675.139s |
| D_QCFS_FULL_FTBC_R0 | 90.192s | 174.979s | 328.360s | 604.634s | 1309.149s | 4642.993s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 95.791s | 185.909s | 350.160s | 653.395s | 1468.411s | 5422.722s |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 95.764s | 186.122s | 351.000s | 656.225s | 1508.583s | 5217.000s |
| G_QCFS_PA_FTBC_R0 | 90.192s | 174.979s | 328.360s | 608.839s | 1368.389s | 4868.199s |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 95.791s | 185.909s | 350.160s | 657.278s | 1531.879s | 5734.329s |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 95.764s | 186.122s | 351.000s | 660.927s | 1577.217s | 6009.842s |

## Pure inference seconds/image

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.00199876 | 0.00327844 | 0.00571149 | 0.00994597 | 0.01915588 | 0.05021788 |
| B_QCFS_STANDARD_SNM_R0 | 0.00205688 | 0.00352024 | 0.00616687 | 0.01088894 | 0.02156166 | 0.06212469 |
| C_QCFS_HA_SNM_R0 | 0.00212691 | 0.00341385 | 0.00618424 | 0.01095983 | 0.02220174 | 0.07519059 |
| D_QCFS_FULL_FTBC_R0 | 0.00184299 | 0.00311051 | 0.00555206 | 0.00957642 | 0.01867330 | 0.05398923 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.00200694 | 0.00336622 | 0.00592830 | 0.01061113 | 0.02087461 | 0.06745053 |
| F_QCFS_FULL_FTBC_HA_SNM_R0 | 0.00201028 | 0.00334415 | 0.00591908 | 0.01063727 | 0.02159489 | 0.06739119 |
| G_QCFS_PA_FTBC_R0 | 0.00184299 | 0.00311051 | 0.00555206 | 0.00968770 | 0.01909093 | 0.05995496 |
| H_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.00200694 | 0.00336622 | 0.00592830 | 0.01071649 | 0.02181266 | 0.07507021 |
| I_QCFS_PA_FTBC_HA_SNM_R0 | 0.00201028 | 0.00334415 | 0.00591908 | 0.01068963 | 0.02259673 | 0.07982650 |

## Full-FTBC calibration

| T | Elapsed | Peak CUDA memory | Parameters | Bytes | Schedule SHA256 |
|---:|---:|---:|---:|---:|---|
| 1 | 5.455s | 2.127 GiB | 12,416 | 49,664 | `b8cca0e30e1207c7d976653b04ebd7f8794c9caf828f5a726f29860d06b25a21` |
| 2 | 7.553s | 2.154 GiB | 24,832 | 99,328 | `067b69a8f92f8f1482234bd1a2bbe5c849810b9c9b7dd0d519803df35e8037d8` |
| 4 | 12.207s | 2.346 GiB | 49,664 | 198,656 | `0abff7a3cc4df3daea9ed4502b082591821e48d21c1ecd669440f4ef3d0fc819` |
| 8 | 21.208s | 2.946 GiB | 99,328 | 397,312 | `45cf7ab4b8d522e4a7737b8e5637907d04512f0ee56e12d32b507498000d8562` |
| 16 | 42.942s | 4.104 GiB | 198,656 | 794,624 | `92af08a7b1f57494da3d6c4f7e815b85c7d5701d98ed4e7a842c082b9d98f168` |
| 32 | 88.359s | 6.419 GiB | 397,312 | 1,589,248 | `4732f55c5010070503322a29a9345287b169522ae5cb8559fb3d60266581f1ae` |

## PA-FTBC compression

| T | Representation | Full params | PA params | Storage saving | Explained energy | Compression elapsed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Full-FTBC fallback | 12,416 | 12,416 | 0.00% | 1.00000000 | 0.000s |
| 2 | Full-FTBC fallback | 24,832 | 24,832 | 0.00% | 1.00000000 | 0.000s |
| 4 | Full-FTBC fallback | 49,664 | 49,664 | 0.00% | 1.00000000 | 0.000s |
| 8 | t0 anchor + t1 anchor + tail mean + tail parity | 99,328 | 49,664 | 50.00% | 0.66262126 | 0.007s |
| 16 | t0 anchor + t1 anchor + tail mean + tail parity | 198,656 | 49,664 | 75.00% | 0.44639737 | 0.010s |
| 32 | t0 anchor + t1 anchor + tail mean + tail parity | 397,312 | 49,664 | 87.50% | 0.32237471 | 0.008s |

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
| ann_sample_count | - | 50000 | yes |
| evaluation_sample_count | 1 | 50000 | yes |
| pa_storage_and_fallback | 1 | - | yes |
| evaluation_sample_count | 2 | 50000 | yes |
| pa_storage_and_fallback | 2 | - | yes |
| evaluation_sample_count | 4 | 50000 | yes |
| pa_storage_and_fallback | 4 | - | yes |
| evaluation_sample_count | 8 | 50000 | yes |
| pa_storage_and_fallback | 8 | - | yes |
| evaluation_sample_count | 16 | 50000 | yes |
| pa_storage_and_fallback | 16 | - | yes |
| evaluation_sample_count | 32 | 50000 | yes |
| pa_storage_and_fallback | 32 | - | yes |
| t1_snm_equivalence | - | - | yes |
| fallback_cache_count | - | 9 | yes |
