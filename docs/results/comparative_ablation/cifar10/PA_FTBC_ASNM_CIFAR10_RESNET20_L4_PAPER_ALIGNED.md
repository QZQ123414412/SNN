# QCFS + Full-FTBC + Temporal-LR FTBC + Parity-Anchor FTBC + A-SNM Ablation

Status: complete

- Dataset/architecture: CIFAR-10/resnet20
- QCFS L: 4
- ANN accuracy: 90.72%
- Checkpoint: `resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth`
- Checkpoint SHA256: `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3`
- Fit/validation SHA256: `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df` / `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c`
- Test samples: 10,000
- Evaluation profile: `paper_era`
- Full-FTBC is fitted independently at every T with SNM off.
- Temporal-LR uses a shared learned rank-4 SVD basis with threshold normalization.
- PA-FTBC uses no SVD or stored basis: t=0/t=1 anchors plus tail mean and tail parity.
- Both compressed methods fall back exactly to Full-FTBC at T<=4.
- Every family freezes its own strict accuracy-gated A-SNM decisions before test inference.
- Checkpoint note: CIFAR-10/ResNet20 QCFS-L4 paper-aligned retrained checkpoint; selected by peak test accuracy during training and therefore subject to test-set model-selection bias; not a strict paper reproduction.

## Primary accuracy table

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | SNM-on T |
|---|---:|---:|---:|---:|---:|---:|---|
| A_QCFS_R0 | 63.89% | 73.29% | 83.38% | 89.82% | 91.32% | 91.69% | none |
| B_QCFS_STANDARD_SNM_R0 | 63.89% | 73.65% | 84.47% | 90.66% | 91.49% | 91.61% | 1, 2, 4, 8, 16, 32 |
| C_QCFS_ASNM_R0 | 63.89% | 73.65% | 84.47% | 90.66% | 91.49% | 91.69% | 2, 4, 8, 16 |
| D_QCFS_FULL_FTBC_R0 | 66.87% | 76.91% | 85.48% | 89.95% | 91.14% | 91.52% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 66.87% | 77.23% | 86.43% | 90.32% | 91.28% | 91.50% | 1, 2, 4, 8, 16, 32 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 66.87% | 77.23% | 86.43% | 90.32% | 91.14% | 91.50% | 2, 4, 8, 32 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 66.87% | 76.91% | 85.48% | 89.92% | 91.20% | 91.53% | none |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 66.87% | 77.23% | 86.43% | 90.46% | 91.30% | 91.45% | 1, 2, 4, 8, 16, 32 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 66.87% | 77.23% | 86.43% | 89.92% | 91.30% | 91.53% | 2, 4, 16 |
| J_QCFS_PA_FTBC_R0 | 66.87% | 76.91% | 85.48% | 89.84% | 91.31% | 91.47% | none |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 66.87% | 77.23% | 86.43% | 90.52% | 91.34% | 91.51% | 1, 2, 4, 8, 16, 32 |
| L_QCFS_PA_FTBC_ASNM_R0 | 66.87% | 77.23% | 86.43% | 90.52% | 91.34% | 91.51% | 2, 4, 8, 16, 32 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 82.23% |
| B_QCFS_STANDARD_SNM_R0 | 82.63% |
| C_QCFS_ASNM_R0 | 82.64% |
| D_QCFS_FULL_FTBC_R0 | 83.64% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 83.94% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 83.92% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 83.65% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 83.96% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 83.88% |
| J_QCFS_PA_FTBC_R0 | 83.65% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 83.98% |
| L_QCFS_PA_FTBC_ASNM_R0 | 83.98% |

## PA-FTBC accuracy comparisons

| T | PA off - Temporal off | PA standard - Temporal standard | PA A-SNM - Temporal A-SNM |
|---:|---:|---:|---:|
| 1 | +0.00pp | +0.00pp | +0.00pp |
| 2 | +0.00pp | +0.00pp | +0.00pp |
| 4 | +0.00pp | +0.00pp | +0.00pp |
| 8 | -0.08pp | +0.06pp | +0.60pp |
| 16 | +0.11pp | +0.04pp | +0.04pp |
| 32 | -0.06pp | +0.06pp | -0.02pp |
| Mean | -0.00pp | +0.03pp | +0.10pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 7.00063061 | 4.90796963 | 2.88301961 | 1.24661561 | 0.52432480 | 0.33648041 |
| B_QCFS_STANDARD_SNM_R0 | 7.00063061 | 4.83002728 | 2.53766746 | 0.81202291 | 0.37796354 | 0.31696458 |
| C_QCFS_ASNM_R0 | 7.00063061 | 4.83002728 | 2.53766746 | 0.81202291 | 0.37796354 | 0.33648041 |
| D_QCFS_FULL_FTBC_R0 | 6.03734207 | 4.09543720 | 2.28381756 | 1.01453309 | 0.47973536 | 0.33086622 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 6.03734207 | 4.04523697 | 2.06264020 | 0.78492031 | 0.40149023 | 0.32134079 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 6.03734207 | 4.04523697 | 2.06264020 | 0.78492031 | 0.47973536 | 0.32134079 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 6.03734207 | 4.09543720 | 2.28381756 | 1.02688559 | 0.48762484 | 0.33292181 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 6.03734207 | 4.04523697 | 2.06264020 | 0.80063316 | 0.41028484 | 0.32860306 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 6.03734207 | 4.04523697 | 2.06264020 | 1.02688559 | 0.41028484 | 0.33292181 |
| J_QCFS_PA_FTBC_R0 | 6.03734207 | 4.09543720 | 2.28381756 | 1.01301171 | 0.46905996 | 0.32786180 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 6.03734207 | 4.04523697 | 2.06264020 | 0.78912397 | 0.39651619 | 0.32097304 |
| L_QCFS_PA_FTBC_ASNM_R0 | 6.03734207 | 4.04523697 | 2.06264020 | 0.78912397 | 0.39651619 | 0.32097304 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 28.354949% | 28.555594% | 28.016150% | 27.685887% | 27.566007% | 27.503956% |
| B_QCFS_STANDARD_SNM_R0 | 28.354949% | 28.539306% | 28.034652% | 27.783526% | 27.711952% | 27.669914% |
| C_QCFS_ASNM_R0 | 28.354949% | 28.539306% | 28.034652% | 27.783526% | 27.711952% | 27.503956% |
| D_QCFS_FULL_FTBC_R0 | 27.975261% | 27.310085% | 27.150731% | 27.258196% | 27.372965% | 27.446133% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 27.975261% | 27.309916% | 27.189594% | 27.369341% | 27.539179% | 27.631510% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 27.975261% | 27.309916% | 27.189594% | 27.369341% | 27.372965% | 27.631510% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 27.975261% | 27.310085% | 27.150731% | 27.234174% | 27.318202% | 27.402296% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 27.975261% | 27.309916% | 27.189594% | 27.345471% | 27.486041% | 27.585875% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 27.975261% | 27.309916% | 27.189594% | 27.234174% | 27.486041% | 27.402296% |
| J_QCFS_PA_FTBC_R0 | 27.975261% | 27.310085% | 27.150731% | 27.281778% | 27.397677% | 27.451380% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 27.975261% | 27.309916% | 27.189594% | 27.388161% | 27.550642% | 27.625301% |
| L_QCFS_PA_FTBC_ASNM_R0 | 27.975261% | 27.309916% | 27.189594% | 27.388161% | 27.550642% | 27.625301% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| B_QCFS_STANDARD_SNM_R0 | 0.000000% | 0.076160% | 0.174030% | 0.267810% | 0.290966% | 0.270947% |
| C_QCFS_ASNM_R0 | 0.000000% | 0.076160% | 0.174030% | 0.267810% | 0.290966% | 0.000000% |
| D_QCFS_FULL_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.067523% | 0.169508% | 0.260592% | 0.290594% | 0.277240% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000000% | 0.067523% | 0.169508% | 0.260592% | 0.000000% | 0.277240% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.067523% | 0.169508% | 0.261952% | 0.293959% | 0.278850% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000000% | 0.067523% | 0.169508% | 0.000000% | 0.293959% | 0.000000% |
| J_QCFS_PA_FTBC_R0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000% | 0.067523% | 0.169508% | 0.258730% | 0.285412% | 0.271182% |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.000000% | 0.067523% | 0.169508% | 0.258730% | 0.285412% | 0.271182% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 71.645051% | 71.444406% | 71.983850% | 72.314113% | 72.433993% | 72.496044% |
| B_QCFS_STANDARD_SNM_R0 | 71.645051% | 71.384534% | 71.791317% | 71.948664% | 71.997082% | 72.059140% |
| C_QCFS_ASNM_R0 | 71.645051% | 71.384534% | 71.791317% | 71.948664% | 71.997082% | 72.496044% |
| D_QCFS_FULL_FTBC_R0 | 72.024739% | 72.689915% | 72.849269% | 72.741804% | 72.627035% | 72.553867% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 72.024739% | 72.622561% | 72.640898% | 72.370067% | 72.170227% | 72.091250% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 72.024739% | 72.622561% | 72.640898% | 72.370067% | 72.627035% | 72.091250% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 72.024739% | 72.689915% | 72.849269% | 72.765826% | 72.681798% | 72.597704% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 72.024739% | 72.622561% | 72.640898% | 72.392577% | 72.220000% | 72.135275% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 72.024739% | 72.622561% | 72.640898% | 72.765826% | 72.220000% | 72.597704% |
| J_QCFS_PA_FTBC_R0 | 72.024739% | 72.689915% | 72.849269% | 72.718222% | 72.602323% | 72.548620% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 72.024739% | 72.622561% | 72.640898% | 72.353109% | 72.163946% | 72.103517% |
| L_QCFS_PA_FTBC_ASNM_R0 | 72.024739% | 72.622561% | 72.640898% | 72.353109% | 72.163946% | 72.103517% |

## Input-driven SOPs

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 132,340,624,864 | 265,631,172,838 | 520,897,861,530 | 1,029,514,514,622 | 2,051,636,957,312 | 4,096,720,484,146 |
| B_QCFS_STANDARD_SNM_R0 | 132,340,624,864 | 265,995,411,938 | 524,840,201,812 | 1,045,867,621,090 | 2,089,739,057,842 | 4,168,858,567,532 |
| C_QCFS_ASNM_R0 | 132,340,624,864 | 265,995,411,938 | 524,840,201,812 | 1,045,867,621,090 | 2,089,739,057,842 | 4,096,720,484,146 |
| D_QCFS_FULL_FTBC_R0 | 128,672,920,314 | 252,801,717,190 | 503,546,961,836 | 1,013,125,919,270 | 2,037,728,556,012 | 4,089,509,842,670 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,028,676,462,258 | 2,075,858,402,230 | 4,163,986,383,636 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,028,676,462,258 | 2,037,728,556,012 | 4,163,986,383,636 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 128,672,920,314 | 252,801,717,190 | 503,546,961,836 | 1,012,092,293,274 | 2,032,750,443,196 | 4,081,734,199,412 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,027,739,437,692 | 2,071,486,886,460 | 4,156,510,964,090 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,012,092,293,274 | 2,071,486,886,460 | 4,081,734,199,412 |
| J_QCFS_PA_FTBC_R0 | 128,672,920,314 | 252,801,717,190 | 503,546,961,836 | 1,014,213,601,660 | 2,039,735,647,820 | 4,090,470,202,740 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,029,563,691,228 | 2,076,488,815,520 | 4,162,287,224,442 |
| L_QCFS_PA_FTBC_ASNM_R0 | 128,672,920,314 | 253,244,985,908 | 507,527,855,058 | 1,029,563,691,228 | 2,076,488,815,520 | 4,162,287,224,442 |

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
| D_QCFS_FULL_FTBC_R0 | 1.185600 | 1.601615 | 2.487149 | 4.930897 | 9.682466 | 19.089558 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.185600 | 1.601615 | 2.487149 | 4.930897 | 9.682466 | 19.089558 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.185600 | 1.601615 | 2.487149 | 4.930897 | 9.682466 | 19.089558 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.185600 | 1.601615 | 2.487149 | 4.930897 | 9.682466 | 19.089558 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.185600 | 1.601615 | 2.487149 | 4.930897 | 9.682466 | 19.089558 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.185600 | 1.601615 | 2.487149 | 4.930897 | 9.682466 | 19.089558 |
| J_QCFS_PA_FTBC_R0 | 1.185600 | 1.601615 | 2.487149 | 4.930897 | 9.682466 | 19.089558 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 1.185600 | 1.601615 | 2.487149 | 4.930897 | 9.682466 | 19.089558 |
| L_QCFS_PA_FTBC_ASNM_R0 | 1.185600 | 1.601615 | 2.487149 | 4.930897 | 9.682466 | 19.089558 |

## Compression elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B_QCFS_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| C_QCFS_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| D_QCFS_FULL_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.057606 | 0.024309 | 0.032712 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.057606 | 0.024309 | 0.032712 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.057606 | 0.024309 | 0.032712 |
| J_QCFS_PA_FTBC_R0 | 0.000000 | 0.000000 | 0.000000 | 0.030130 | 0.009229 | 0.011532 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.030130 | 0.009229 | 0.011532 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.000000 | 0.000000 | 0.000000 | 0.030130 | 0.009229 | 0.011532 |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.060156 | 1.254623 | 1.851225 | 3.108317 | 5.490141 | 10.273742 |
| B_QCFS_STANDARD_SNM_R0 | 1.130733 | 1.364485 | 1.987356 | 3.378161 | 6.010100 | 11.483887 |
| C_QCFS_ASNM_R0 | 1.060156 | 1.364485 | 1.987356 | 3.378161 | 6.010100 | 10.273742 |
| D_QCFS_FULL_FTBC_R0 | 1.078623 | 1.263234 | 1.806391 | 3.026914 | 5.406502 | 10.045405 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.159959 | 1.306028 | 1.909722 | 3.348393 | 5.904258 | 11.125852 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.078623 | 1.306028 | 1.909722 | 3.348393 | 5.406502 | 11.125852 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.083446 | 1.246310 | 1.785970 | 3.085570 | 5.498981 | 10.204912 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.137397 | 1.343299 | 1.932508 | 3.378228 | 6.088616 | 11.380112 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.083446 | 1.343299 | 1.932508 | 3.085570 | 6.088616 | 10.204912 |
| J_QCFS_PA_FTBC_R0 | 1.099542 | 1.252842 | 1.785577 | 3.119932 | 5.551427 | 10.386763 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 1.135468 | 1.346824 | 1.929653 | 3.361096 | 6.147562 | 11.786737 |
| L_QCFS_PA_FTBC_ASNM_R0 | 1.099542 | 1.346824 | 1.929653 | 3.361096 | 6.147562 | 11.786737 |

## Compression summary

| T | Full params | Temporal params | PA params | Temporal saving | PA saving | Temporal MACs | PA MACs | Temporal energy | PA energy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 688 | 688 | 688 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 2 | 1,376 | 1,376 | 1,376 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 4 | 2,752 | 2,752 | 2,752 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 8 | 5,504 | 2,784 | 2,752 | 49.42% | 50.00% | 22,016 | 9,632 | 0.904179 | 0.884148 |
| 16 | 11,008 | 2,816 | 2,752 | 74.42% | 75.00% | 44,032 | 20,640 | 0.775599 | 0.684333 |
| 32 | 22,016 | 2,880 | 2,752 | 86.92% | 87.50% | 88,064 | 42,656 | 0.657317 | 0.549004 |

## A-SNM selection

- QCFS SNM-on T: 2, 4, 8, 16; selection elapsed: 6.054116s.

### QCFS accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 37.70% | 37.70% | off |
| 2 | 45.80% | 46.60% | on |
| 4 | 56.70% | 58.80% | on |
| 8 | 70.20% | 75.80% | on |
| 16 | 81.60% | 83.40% | on |
| 32 | 84.60% | 84.40% | off |

- Full-FTBC SNM-on T: 2, 4, 8, 32; selection elapsed: 5.605980s.

### Full-FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 47.60% | 47.60% | off |
| 2 | 59.60% | 60.30% | on |
| 4 | 72.90% | 73.30% | on |
| 8 | 80.80% | 81.50% | on |
| 16 | 83.90% | 83.90% | off |
| 32 | 84.70% | 85.20% | on |

- Temporal-LR FTBC SNM-on T: 2, 4, 16; selection elapsed: 5.690113s.

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 47.60% | 47.60% | off |
| 2 | 59.60% | 60.30% | on |
| 4 | 72.90% | 73.30% | on |
| 8 | 81.60% | 81.10% | off |
| 16 | 81.30% | 83.30% | on |
| 32 | 85.10% | 84.90% | off |

- Parity-Anchor FTBC SNM-on T: 2, 4, 8, 16, 32; selection elapsed: 5.815823s.

### Parity-Anchor FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 1 | 47.60% | 47.60% | off |
| 2 | 59.60% | 60.30% | on |
| 4 | 72.90% | 73.30% | on |
| 8 | 80.60% | 81.10% | on |
| 16 | 82.30% | 83.10% | on |
| 32 | 84.80% | 85.20% | on |

## Validation-selection generalization audit

| Family | T | Selected | Test off | Test on | Test-best | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 1 | off | 63.89% | 63.89% | off | yes |
| Full-FTBC | 1 | off | 66.87% | 66.87% | off | yes |
| Temporal-LR FTBC | 1 | off | 66.87% | 66.87% | off | yes |
| Parity-Anchor FTBC | 1 | off | 66.87% | 66.87% | off | yes |
| QCFS | 2 | on | 73.29% | 73.65% | on | yes |
| Full-FTBC | 2 | on | 76.91% | 77.23% | on | yes |
| Temporal-LR FTBC | 2 | on | 76.91% | 77.23% | on | yes |
| Parity-Anchor FTBC | 2 | on | 76.91% | 77.23% | on | yes |
| QCFS | 4 | on | 83.38% | 84.47% | on | yes |
| Full-FTBC | 4 | on | 85.48% | 86.43% | on | yes |
| Temporal-LR FTBC | 4 | on | 85.48% | 86.43% | on | yes |
| Parity-Anchor FTBC | 4 | on | 85.48% | 86.43% | on | yes |
| QCFS | 8 | on | 89.82% | 90.66% | on | yes |
| Full-FTBC | 8 | on | 89.95% | 90.32% | on | yes |
| Temporal-LR FTBC | 8 | off | 89.92% | 90.46% | on | no |
| Parity-Anchor FTBC | 8 | on | 89.84% | 90.52% | on | yes |
| QCFS | 16 | on | 91.32% | 91.49% | on | yes |
| Full-FTBC | 16 | off | 91.14% | 91.28% | on | no |
| Temporal-LR FTBC | 16 | on | 91.20% | 91.30% | on | yes |
| Parity-Anchor FTBC | 16 | on | 91.31% | 91.34% | on | yes |
| QCFS | 32 | off | 91.69% | 91.61% | off | yes |
| Full-FTBC | 32 | on | 91.52% | 91.50% | off | no |
| Temporal-LR FTBC | 32 | off | 91.53% | 91.45% | off | yes |
| Parity-Anchor FTBC | 32 | on | 91.47% | 91.51% | on | yes |

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
| A-SNM cache | C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
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
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 8 | G_QCFS_TEMPORAL_LR_FTBC_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 8 | K_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 16 | D_QCFS_FULL_FTBC_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 16 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | L_QCFS_PA_FTBC_ASNM_R0 | 16 | K_QCFS_PA_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 32 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 32 | G_QCFS_TEMPORAL_LR_FTBC_R0 | yes |
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
| `conv1.2` | temporal_low_rank | 16 | 0.00221193 | 0.43339035 | 0.19999257 |
| `conv2_x.0.residual_function.2` | temporal_low_rank | 16 | 0.00054821 | 0.50940514 | 0.10658646 |
| `conv2_x.0.act` | temporal_low_rank | 16 | 0.00098599 | 0.31824768 | 0.12365936 |
| `conv2_x.1.residual_function.2` | temporal_low_rank | 16 | 0.00026599 | 0.25123206 | 0.07308058 |
| `conv2_x.1.act` | temporal_low_rank | 16 | 0.00036686 | 0.21380313 | 0.06857233 |
| `conv2_x.2.residual_function.2` | temporal_low_rank | 16 | 0.00003459 | 0.16305241 | 0.01899654 |
| `conv2_x.2.act` | temporal_low_rank | 16 | 0.00028908 | 0.28237739 | 0.06159171 |
| `conv3_x.0.residual_function.2` | temporal_low_rank | 32 | 0.00004892 | 0.23041390 | 0.02717712 |
| `conv3_x.0.act` | temporal_low_rank | 32 | 0.00005441 | 0.18093970 | 0.03657334 |
| `conv3_x.1.residual_function.2` | temporal_low_rank | 32 | 0.00003574 | 0.21125221 | 0.03678860 |
| `conv3_x.1.act` | temporal_low_rank | 32 | 0.00005671 | 0.17281631 | 0.03150228 |
| `conv3_x.2.residual_function.2` | temporal_low_rank | 32 | 0.00001101 | 0.16987500 | 0.01512011 |
| `conv3_x.2.act` | temporal_low_rank | 32 | 0.00012548 | 0.18703771 | 0.07029897 |
| `conv4_x.0.residual_function.2` | temporal_low_rank | 64 | 0.00001364 | 0.18258704 | 0.01319743 |
| `conv4_x.0.act` | temporal_low_rank | 64 | 0.00001317 | 0.26153427 | 0.00993067 |
| `conv4_x.1.residual_function.2` | temporal_low_rank | 64 | 0.00000755 | 0.30923423 | 0.00877268 |
| `conv4_x.1.act` | temporal_low_rank | 64 | 0.00001181 | 0.18982035 | 0.01426379 |
| `conv4_x.2.residual_function.2` | temporal_low_rank | 64 | 0.00000344 | 0.18305521 | 0.00637197 |
| `conv4_x.2.act` | temporal_low_rank | 64 | 0.00009847 | 0.19332823 | 0.03589006 |

### T=16

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | temporal_low_rank | 16 | 0.00310871 | 0.54853648 | 0.21521726 |
| `conv2_x.0.residual_function.2` | temporal_low_rank | 16 | 0.00064798 | 0.64655077 | 0.11869960 |
| `conv2_x.0.act` | temporal_low_rank | 16 | 0.00195760 | 0.52715015 | 0.14035490 |
| `conv2_x.1.residual_function.2` | temporal_low_rank | 16 | 0.00032149 | 0.35579699 | 0.07347021 |
| `conv2_x.1.act` | temporal_low_rank | 16 | 0.00080621 | 0.41635537 | 0.10446745 |
| `conv2_x.2.residual_function.2` | temporal_low_rank | 16 | 0.00005747 | 0.28393051 | 0.03670780 |
| `conv2_x.2.act` | temporal_low_rank | 16 | 0.00033537 | 0.40600711 | 0.06421970 |
| `conv3_x.0.residual_function.2` | temporal_low_rank | 32 | 0.00005619 | 0.33341163 | 0.03971597 |
| `conv3_x.0.act` | temporal_low_rank | 32 | 0.00008213 | 0.30778098 | 0.04692609 |
| `conv3_x.1.residual_function.2` | temporal_low_rank | 32 | 0.00004362 | 0.32708627 | 0.04276649 |
| `conv3_x.1.act` | temporal_low_rank | 32 | 0.00009819 | 0.31887260 | 0.05954671 |
| `conv3_x.2.residual_function.2` | temporal_low_rank | 32 | 0.00001857 | 0.30611709 | 0.01803241 |
| `conv3_x.2.act` | temporal_low_rank | 32 | 0.00020145 | 0.33311135 | 0.10433756 |
| `conv4_x.0.residual_function.2` | temporal_low_rank | 64 | 0.00002298 | 0.32955015 | 0.02723963 |
| `conv4_x.0.act` | temporal_low_rank | 64 | 0.00002514 | 0.49767873 | 0.02036980 |
| `conv4_x.1.residual_function.2` | temporal_low_rank | 64 | 0.00001604 | 0.61729234 | 0.02540574 |
| `conv4_x.1.act` | temporal_low_rank | 64 | 0.00002482 | 0.38358897 | 0.02485555 |
| `conv4_x.2.residual_function.2` | temporal_low_rank | 64 | 0.00000684 | 0.35801375 | 0.01483539 |
| `conv4_x.2.act` | temporal_low_rank | 64 | 0.00031721 | 0.48216426 | 0.07310715 |

### T=32

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | temporal_low_rank | 16 | 0.00380474 | 0.65787792 | 0.22434425 |
| `conv2_x.0.residual_function.2` | temporal_low_rank | 16 | 0.00065169 | 0.70627767 | 0.15384132 |
| `conv2_x.0.act` | temporal_low_rank | 16 | 0.00214475 | 0.64114052 | 0.16747321 |
| `conv2_x.1.residual_function.2` | temporal_low_rank | 16 | 0.00032523 | 0.44003850 | 0.08324236 |
| `conv2_x.1.act` | temporal_low_rank | 16 | 0.00074527 | 0.51890796 | 0.11178413 |
| `conv2_x.2.residual_function.2` | temporal_low_rank | 16 | 0.00006709 | 0.39242935 | 0.04020270 |
| `conv2_x.2.act` | temporal_low_rank | 16 | 0.00032551 | 0.51837558 | 0.07144532 |
| `conv3_x.0.residual_function.2` | temporal_low_rank | 32 | 0.00005279 | 0.42658892 | 0.04344925 |
| `conv3_x.0.act` | temporal_low_rank | 32 | 0.00007953 | 0.41684076 | 0.06804912 |
| `conv3_x.1.residual_function.2` | temporal_low_rank | 32 | 0.00003964 | 0.43266755 | 0.06805337 |
| `conv3_x.1.act` | temporal_low_rank | 32 | 0.00006573 | 0.36459878 | 0.06793946 |
| `conv3_x.2.residual_function.2` | temporal_low_rank | 32 | 0.00001259 | 0.34801158 | 0.01949354 |
| `conv3_x.2.act` | temporal_low_rank | 32 | 0.00013326 | 0.37996325 | 0.12196872 |
| `conv4_x.0.residual_function.2` | temporal_low_rank | 64 | 0.00001860 | 0.41053677 | 0.03099309 |
| `conv4_x.0.act` | temporal_low_rank | 64 | 0.00001731 | 0.56408376 | 0.02293454 |
| `conv4_x.1.residual_function.2` | temporal_low_rank | 64 | 0.00001042 | 0.67897016 | 0.02771562 |
| `conv4_x.1.act` | temporal_low_rank | 64 | 0.00001862 | 0.46023110 | 0.03131580 |
| `conv4_x.2.residual_function.2` | temporal_low_rank | 64 | 0.00000552 | 0.44764242 | 0.01513281 |
| `conv4_x.2.act` | temporal_low_rank | 64 | 0.00023921 | 0.57110530 | 0.07370950 |

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
| `conv1.2` | parity_anchor | 16 | 0.00276590 | 0.48463139 | 0.19726305 |
| `conv2_x.0.residual_function.2` | parity_anchor | 16 | 0.00063998 | 0.55039519 | 0.10352297 |
| `conv2_x.0.act` | parity_anchor | 16 | 0.00155077 | 0.39911845 | 0.18162930 |
| `conv2_x.1.residual_function.2` | parity_anchor | 16 | 0.00052098 | 0.35160658 | 0.10553609 |
| `conv2_x.1.act` | parity_anchor | 16 | 0.00077682 | 0.31111604 | 0.17033371 |
| `conv2_x.2.residual_function.2` | parity_anchor | 16 | 0.00010649 | 0.28609166 | 0.06169505 |
| `conv2_x.2.act` | parity_anchor | 16 | 0.00041755 | 0.33937278 | 0.07592806 |
| `conv3_x.0.residual_function.2` | parity_anchor | 32 | 0.00006417 | 0.26389986 | 0.05655909 |
| `conv3_x.0.act` | parity_anchor | 32 | 0.00014072 | 0.29099935 | 0.08995490 |
| `conv3_x.1.residual_function.2` | parity_anchor | 32 | 0.00006467 | 0.28415689 | 0.07993270 |
| `conv3_x.1.act` | parity_anchor | 32 | 0.00005987 | 0.17755592 | 0.04296350 |
| `conv3_x.2.residual_function.2` | parity_anchor | 32 | 0.00001542 | 0.20100339 | 0.01581183 |
| `conv3_x.2.act` | parity_anchor | 32 | 0.00018847 | 0.22922510 | 0.06931213 |
| `conv4_x.0.residual_function.2` | parity_anchor | 64 | 0.00002062 | 0.22450440 | 0.01668660 |
| `conv4_x.0.act` | parity_anchor | 64 | 0.00001060 | 0.23464635 | 0.01336018 |
| `conv4_x.1.residual_function.2` | parity_anchor | 64 | 0.00000355 | 0.21206877 | 0.00887105 |
| `conv4_x.1.act` | parity_anchor | 64 | 0.00000951 | 0.17035489 | 0.01542384 |
| `conv4_x.2.residual_function.2` | parity_anchor | 64 | 0.00000634 | 0.24858294 | 0.00681727 |
| `conv4_x.2.act` | parity_anchor | 64 | 0.00013935 | 0.22998087 | 0.04384890 |

### T=16

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | parity_anchor | 16 | 0.00633427 | 0.78300303 | 0.23793106 |
| `conv2_x.0.residual_function.2` | parity_anchor | 16 | 0.00074518 | 0.69335413 | 0.15137100 |
| `conv2_x.0.act` | parity_anchor | 16 | 0.00266866 | 0.61548752 | 0.23793924 |
| `conv2_x.1.residual_function.2` | parity_anchor | 16 | 0.00057200 | 0.47458792 | 0.16277933 |
| `conv2_x.1.act` | parity_anchor | 16 | 0.00100605 | 0.46510425 | 0.22412297 |
| `conv2_x.2.residual_function.2` | parity_anchor | 16 | 0.00011287 | 0.39791548 | 0.08510445 |
| `conv2_x.2.act` | parity_anchor | 16 | 0.00043302 | 0.46134377 | 0.11653250 |
| `conv3_x.0.residual_function.2` | parity_anchor | 32 | 0.00008205 | 0.40290058 | 0.07496078 |
| `conv3_x.0.act` | parity_anchor | 32 | 0.00017115 | 0.44429812 | 0.14529726 |
| `conv3_x.1.residual_function.2` | parity_anchor | 32 | 0.00006269 | 0.39212003 | 0.11750388 |
| `conv3_x.1.act` | parity_anchor | 32 | 0.00005449 | 0.23755598 | 0.04878895 |
| `conv3_x.2.residual_function.2` | parity_anchor | 32 | 0.00002086 | 0.32440552 | 0.02316989 |
| `conv3_x.2.act` | parity_anchor | 32 | 0.00016858 | 0.30472755 | 0.09670542 |
| `conv4_x.0.residual_function.2` | parity_anchor | 64 | 0.00003419 | 0.40201941 | 0.02783224 |
| `conv4_x.0.act` | parity_anchor | 64 | 0.00001978 | 0.44137853 | 0.02015498 |
| `conv4_x.1.residual_function.2` | parity_anchor | 64 | 0.00000869 | 0.45440704 | 0.01337362 |
| `conv4_x.1.act` | parity_anchor | 64 | 0.00001573 | 0.30538264 | 0.02099563 |
| `conv4_x.2.residual_function.2` | parity_anchor | 64 | 0.00001236 | 0.48137456 | 0.01334554 |
| `conv4_x.2.act` | parity_anchor | 64 | 0.00030330 | 0.47146985 | 0.08122643 |

### T=32

| Layer | Representation | Channels | MSE | NRMSE | Max abs error |
|---|---|---:|---:|---:|---:|
| `conv1.2` | parity_anchor | 16 | 0.00618894 | 0.83905602 | 0.24454466 |
| `conv2_x.0.residual_function.2` | parity_anchor | 16 | 0.00080627 | 0.78558815 | 0.12687272 |
| `conv2_x.0.act` | parity_anchor | 16 | 0.00274788 | 0.72571087 | 0.25377846 |
| `conv2_x.1.residual_function.2` | parity_anchor | 16 | 0.00047693 | 0.53287208 | 0.18851468 |
| `conv2_x.1.act` | parity_anchor | 16 | 0.00087384 | 0.56188750 | 0.25813651 |
| `conv2_x.2.residual_function.2` | parity_anchor | 16 | 0.00008691 | 0.44664347 | 0.09111863 |
| `conv2_x.2.act` | parity_anchor | 16 | 0.00035054 | 0.53793782 | 0.12677553 |
| `conv3_x.0.residual_function.2` | parity_anchor | 32 | 0.00007035 | 0.49243939 | 0.08487061 |
| `conv3_x.0.act` | parity_anchor | 32 | 0.00012985 | 0.53261936 | 0.17841811 |
| `conv3_x.1.residual_function.2` | parity_anchor | 32 | 0.00004306 | 0.45096004 | 0.13700163 |
| `conv3_x.1.act` | parity_anchor | 32 | 0.00004021 | 0.28516632 | 0.05023606 |
| `conv3_x.2.residual_function.2` | parity_anchor | 32 | 0.00001497 | 0.37950245 | 0.02925080 |
| `conv3_x.2.act` | parity_anchor | 32 | 0.00010575 | 0.33848131 | 0.11032426 |
| `conv4_x.0.residual_function.2` | parity_anchor | 64 | 0.00002705 | 0.49503830 | 0.03538723 |
| `conv4_x.0.act` | parity_anchor | 64 | 0.00001666 | 0.55344254 | 0.02614442 |
| `conv4_x.1.residual_function.2` | parity_anchor | 64 | 0.00000806 | 0.59698194 | 0.01527955 |
| `conv4_x.1.act` | parity_anchor | 64 | 0.00001353 | 0.39226836 | 0.02470726 |
| `conv4_x.2.residual_function.2` | parity_anchor | 64 | 0.00001031 | 0.61180043 | 0.01682332 |
| `conv4_x.2.act` | parity_anchor | 64 | 0.00028015 | 0.61804676 | 0.10758220 |
