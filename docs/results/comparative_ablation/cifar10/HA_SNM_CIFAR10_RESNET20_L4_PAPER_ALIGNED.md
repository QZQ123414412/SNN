# QCFS + Full/Temporal-LR/PA-FTBC + HA-SNM Ablation

Status: complete

- Dataset/architecture: CIFAR-10/resnet20
- QCFS L: 4
- ANN accuracy: 90.72%
- Checkpoint: `resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth`
- Checkpoint SHA256: `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3`
- Fit/validation SHA256: `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df` / `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c`
- Test samples: 10,000
- Evaluation profile: `paper_era`
- HA-SNM threshold schedule: start=1.25, end=0.5, reference horizon=8.0, linear.
- HA-SNM keeps the original transmitted-credit/R0 rule and changes only the negative-spike decision threshold.
- It uses the original -theta event amplitude, adds no dense neuron state, and has two global FP32 deployment constants plus one fixed reference horizon (12 bytes if stored).
- Full-FTBC is fitted independently at every T with SNM off; Temporal-LR and PA are compressed from that same teacher.
- Temporal-LR and PA fall back exactly to Full-FTBC at T<=4.
- Checkpoint note: paper-aligned retrained checkpoint selected by peak test accuracy; not a strict paper reproduction.

## Primary accuracy

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 66.87% | 76.91% | 85.48% | 89.95% | 91.14% | 91.52% | 83.64% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 66.87% | 77.23% | 86.43% | 90.32% | 91.28% | 91.50% | 83.94% |
| C_QCFS_FULL_FTBC_HA_R0 | 66.87% | 78.41% | 87.51% | 90.64% | 91.32% | 91.50% | 84.38% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 66.87% | 76.91% | 85.48% | 89.92% | 91.20% | 91.53% | 83.65% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 66.87% | 77.23% | 86.43% | 90.46% | 91.30% | 91.45% | 83.96% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 66.87% | 78.41% | 87.51% | 90.67% | 91.32% | 91.54% | 84.39% |
| G_QCFS_PA_FTBC_OFF_R0 | 66.87% | 76.91% | 85.48% | 89.84% | 91.31% | 91.47% | 83.65% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 66.87% | 77.23% | 86.43% | 90.52% | 91.34% | 91.51% | 83.98% |
| I_QCFS_PA_FTBC_HA_R0 | 66.87% | 78.41% | 87.51% | 90.50% | 91.35% | 91.43% | 84.34% |

## HA-SNM accuracy gain

| Family | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-FTBC: HA - standard | +0.00pp | +1.18pp | +1.08pp | +0.32pp | +0.04pp | +0.00pp | +0.437pp |
| Full-FTBC: HA - off | +0.00pp | +1.50pp | +2.03pp | +0.69pp | +0.18pp | -0.02pp | +0.730pp |
| Temporal-LR FTBC: HA - standard | +0.00pp | +1.18pp | +1.08pp | +0.21pp | +0.02pp | +0.09pp | +0.430pp |
| Temporal-LR FTBC: HA - off | +0.00pp | +1.50pp | +2.03pp | +0.75pp | +0.12pp | +0.01pp | +0.735pp |
| PA-FTBC: HA - standard | +0.00pp | +1.18pp | +1.08pp | -0.02pp | +0.01pp | -0.08pp | +0.362pp |
| PA-FTBC: HA - off | +0.00pp | +1.50pp | +2.03pp | +0.66pp | +0.04pp | -0.04pp | +0.698pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 6.03734207 | 4.09543720 | 2.28381756 | 1.01453309 | 0.47973536 | 0.33086622 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 6.03734207 | 4.04523697 | 2.06264020 | 0.78492031 | 0.40149023 | 0.32134079 |
| C_QCFS_FULL_FTBC_HA_R0 | 6.03734207 | 3.72907472 | 1.78492365 | 0.73109304 | 0.39975736 | 0.32164230 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 6.03734207 | 4.09543720 | 2.28381756 | 1.02688559 | 0.48762484 | 0.33292181 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 6.03734207 | 4.04523697 | 2.06264020 | 0.80063316 | 0.41028484 | 0.32860306 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 6.03734207 | 3.72907472 | 1.78492365 | 0.74109534 | 0.41180813 | 0.32997162 |
| G_QCFS_PA_FTBC_OFF_R0 | 6.03734207 | 4.09543720 | 2.28381756 | 1.01301171 | 0.46905996 | 0.32786180 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 6.03734207 | 4.04523697 | 2.06264020 | 0.78912397 | 0.39651619 | 0.32097304 |
| I_QCFS_PA_FTBC_HA_R0 | 6.03734207 | 3.72907472 | 1.78492365 | 0.73576800 | 0.39608895 | 0.32132240 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 27.975261% | 27.310085% | 27.150731% | 27.258196% | 27.372965% | 27.446133% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 27.975261% | 27.309916% | 27.189594% | 27.369341% | 27.539179% | 27.631510% |
| C_QCFS_FULL_FTBC_HA_R0 | 27.975261% | 27.317116% | 27.247119% | 27.457954% | 27.570799% | 27.642999% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 27.975261% | 27.310085% | 27.150731% | 27.234174% | 27.318202% | 27.402296% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 27.975261% | 27.309916% | 27.189594% | 27.345471% | 27.486041% | 27.585875% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 27.975261% | 27.317116% | 27.247119% | 27.434236% | 27.517225% | 27.597174% |
| G_QCFS_PA_FTBC_OFF_R0 | 27.975261% | 27.310085% | 27.150731% | 27.281778% | 27.397677% | 27.451380% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 27.975261% | 27.309916% | 27.189594% | 27.388161% | 27.550642% | 27.625301% |
| I_QCFS_PA_FTBC_HA_R0 | 27.975261% | 27.317116% | 27.247119% | 27.471187% | 27.580085% | 27.636582% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 0.000000% | 0.067523% | 0.169508% | 0.260592% | 0.290594% | 0.277240% |
| C_QCFS_FULL_FTBC_HA_R0 | 0.000000% | 0.324771% | 0.461164% | 0.510617% | 0.357728% | 0.297708% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 0.000000% | 0.067523% | 0.169508% | 0.261952% | 0.293959% | 0.278850% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 0.000000% | 0.324771% | 0.461164% | 0.513841% | 0.361236% | 0.299215% |
| G_QCFS_PA_FTBC_OFF_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 0.000000% | 0.067523% | 0.169508% | 0.258730% | 0.285412% | 0.271182% |
| I_QCFS_PA_FTBC_HA_R0 | 0.000000% | 0.324771% | 0.461164% | 0.507893% | 0.351424% | 0.291395% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 72.024739% | 72.689915% | 72.849269% | 72.741804% | 72.627035% | 72.553867% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 72.024739% | 72.622561% | 72.640898% | 72.370067% | 72.170227% | 72.091250% |
| C_QCFS_FULL_FTBC_HA_R0 | 72.024739% | 72.358113% | 72.291717% | 72.031429% | 72.071473% | 72.059293% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 72.024739% | 72.689915% | 72.849269% | 72.765826% | 72.681798% | 72.597704% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 72.024739% | 72.622561% | 72.640898% | 72.392577% | 72.220000% | 72.135275% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 72.024739% | 72.358113% | 72.291717% | 72.051923% | 72.121539% | 72.103611% |
| G_QCFS_PA_FTBC_OFF_R0 | 72.024739% | 72.689915% | 72.849269% | 72.718222% | 72.602323% | 72.548620% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 72.024739% | 72.622561% | 72.640898% | 72.353109% | 72.163946% | 72.103517% |
| I_QCFS_PA_FTBC_HA_R0 | 72.024739% | 72.358113% | 72.291717% | 72.020920% | 72.068491% | 72.072024% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 128,672,920,314 | 252,801,717,190 | 503,546,961,836 | 1,013,125,919,270 | 2,037,728,556,012 | 4,089,509,842,670 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,028,676,462,258 | 2,075,858,402,230 | 4,163,986,383,636 |
| C_QCFS_FULL_FTBC_HA_R0 | 128,672,920,314 | 255,697,150,476 | 515,314,140,166 | 1,043,761,065,792 | 2,084,379,579,162 | 4,169,232,565,068 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 128,672,920,314 | 252,801,717,190 | 503,546,961,836 | 1,012,092,293,274 | 2,032,750,443,196 | 4,081,734,199,412 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,027,739,437,692 | 2,071,486,886,460 | 4,156,510,964,090 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 128,672,920,314 | 255,697,150,476 | 515,314,140,166 | 1,042,967,560,318 | 2,080,019,985,910 | 4,161,733,206,044 |
| G_QCFS_PA_FTBC_OFF_R0 | 128,672,920,314 | 252,801,717,190 | 503,546,961,836 | 1,014,213,601,660 | 2,039,735,647,820 | 4,090,470,202,740 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,029,563,691,228 | 2,076,488,815,520 | 4,162,287,224,442 |
| I_QCFS_PA_FTBC_HA_R0 | 128,672,920,314 | 255,697,150,476 | 515,314,140,166 | 1,044,393,427,368 | 2,084,723,584,292 | 4,167,445,411,842 |

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
| A_QCFS_FULL_FTBC_OFF_R0 | 1.098118 | 1.259788 | 1.823813 | 3.022869 | 5.252901 | 9.805030 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 1.170210 | 1.342458 | 1.941210 | 3.275779 | 5.808331 | 10.990204 |
| C_QCFS_FULL_FTBC_HA_R0 | 1.170106 | 1.400911 | 1.985163 | 3.326309 | 5.889022 | 11.660363 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 1.104500 | 1.274051 | 1.822604 | 3.107549 | 5.391418 | 10.231412 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 1.200032 | 1.379540 | 1.937531 | 3.348536 | 5.981401 | 11.685138 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 1.171552 | 1.375845 | 1.979467 | 3.440762 | 6.083073 | 12.160342 |
| G_QCFS_PA_FTBC_OFF_R0 | 1.092118 | 1.237945 | 1.815070 | 3.070125 | 5.472802 | 10.493768 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 1.163466 | 1.372689 | 1.950973 | 3.343802 | 5.957401 | 11.774701 |
| I_QCFS_PA_FTBC_HA_R0 | 1.213064 | 1.381939 | 1.956188 | 3.438890 | 6.208495 | 12.093552 |

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
