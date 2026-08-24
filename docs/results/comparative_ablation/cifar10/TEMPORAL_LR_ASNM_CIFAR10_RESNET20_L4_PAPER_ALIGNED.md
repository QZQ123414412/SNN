# QCFS + Full-FTBC + Temporal-LR FTBC + A-SNM CIFAR-10 Ablation

- Status: complete
- Dataset: CIFAR-10
- Architecture: resnet20
- QCFS levels: L=4
- ResNet20 evaluation profile: paper_era
- Checkpoint: `resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth`
- Checkpoint SHA256: `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3`
- ANN accuracy on the 10,000-image test set: 90.72%
- Time steps: [1, 2, 4, 8, 16, 32]
- Full-FTBC fit: 5 x 200, alpha=0.4
- A-SNM validation: 5 x 200
- Temporal-LR: shared rank-4 basis, threshold-normalized, no exempt layer.
- Temporal-LR falls back to Full-FTBC at T<=4 and is active at T>4.
- Fit batch SHA256: `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df`
- Validation batch SHA256: `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-10 normalization, and Cutout(1,16).
- Test uses only ToTensor and normalization with shuffle=False.
- Every SNN uses QCFS L=4, rate coding/schedule, ratio=1.0, R0=True, SNM margin=0, FP32.
- Full-FTBC is independently fitted at every T with SNM off; Temporal-LR is compressed from that frozen teacher.
- Each family enables SNM only when SNM-on has strictly higher validation accuracy; ties select off.
- Test data is first accessed after all three A-SNM families are frozen.
- A-SNM guarantees validation-set selection only; test-set reversals are reported diagnostically and never retuned.
- Checkpoint-selection note: the checkpoint is selected by the highest accuracy observed on the 10,000-image CIFAR-10 test set during 300 training epochs; this creates model-selection bias.
- Checkpoint-interpretation note: the checkpoint is the CIFAR-10/ResNet20 QCFS-L4 paper-aligned retrained model evaluated with the paper_era profile; it is not a strict reproduction of the paper's reported accuracy.

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

## Temporal-LR accuracy comparisons

| Comparison | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| G-D | +0.00pp | +0.00pp | +0.00pp | -0.03pp | +0.06pp | +0.01pp | +0.01pp |
| H-E | +0.00pp | +0.00pp | +0.00pp | +0.14pp | +0.02pp | -0.05pp | +0.02pp |
| I-F | +0.00pp | +0.00pp | +0.00pp | -0.40pp | +0.16pp | +0.03pp | -0.03pp |

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 7.000631 | 4.907970 | 2.883020 | 1.246616 | 0.524325 | 0.336480 |
| B_QCFS_STANDARD_SNM_R0 | 7.000631 | 4.830027 | 2.537667 | 0.812023 | 0.377964 | 0.316965 |
| C_QCFS_ASNM_R0 | 7.000631 | 4.830027 | 2.537667 | 0.812023 | 0.377964 | 0.336480 |
| D_QCFS_FULL_FTBC_R0 | 6.037342 | 4.095437 | 2.283818 | 1.014533 | 0.479735 | 0.330866 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 6.037342 | 4.045237 | 2.062640 | 0.784920 | 0.401490 | 0.321341 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 6.037342 | 4.045237 | 2.062640 | 0.784920 | 0.479735 | 0.321341 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 6.037342 | 4.095437 | 2.283818 | 1.026886 | 0.487625 | 0.332922 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 6.037342 | 4.045237 | 2.062640 | 0.800633 | 0.410285 | 0.328603 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 6.037342 | 4.045237 | 2.062640 | 1.026886 | 0.410285 | 0.332922 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 28.3549% | 28.5556% | 28.0161% | 27.6859% | 27.5660% | 27.5040% |
| B_QCFS_STANDARD_SNM_R0 | 28.3549% | 28.5393% | 28.0347% | 27.7835% | 27.7120% | 27.6699% |
| C_QCFS_ASNM_R0 | 28.3549% | 28.5393% | 28.0347% | 27.7835% | 27.7120% | 27.5040% |
| D_QCFS_FULL_FTBC_R0 | 27.9753% | 27.3101% | 27.1507% | 27.2582% | 27.3730% | 27.4461% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 27.9753% | 27.3099% | 27.1896% | 27.3693% | 27.5392% | 27.6315% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 27.9753% | 27.3099% | 27.1896% | 27.3693% | 27.3730% | 27.6315% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 27.9753% | 27.3101% | 27.1507% | 27.2342% | 27.3182% | 27.4023% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 27.9753% | 27.3099% | 27.1896% | 27.3455% | 27.4860% | 27.5859% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 27.9753% | 27.3099% | 27.1896% | 27.2342% | 27.4860% | 27.4023% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.0762% | 0.1740% | 0.2678% | 0.2910% | 0.2709% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0762% | 0.1740% | 0.2678% | 0.2910% | 0.0000% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0675% | 0.1695% | 0.2606% | 0.2906% | 0.2772% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0675% | 0.1695% | 0.2606% | 0.0000% | 0.2772% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0675% | 0.1695% | 0.2620% | 0.2940% | 0.2788% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.0000% | 0.0675% | 0.1695% | 0.0000% | 0.2940% | 0.0000% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 71.6451% | 71.4444% | 71.9839% | 72.3141% | 72.4340% | 72.4960% |
| B_QCFS_STANDARD_SNM_R0 | 71.6451% | 71.3845% | 71.7913% | 71.9487% | 71.9971% | 72.0591% |
| C_QCFS_ASNM_R0 | 71.6451% | 71.3845% | 71.7913% | 71.9487% | 71.9971% | 72.4960% |
| D_QCFS_FULL_FTBC_R0 | 72.0247% | 72.6899% | 72.8493% | 72.7418% | 72.6270% | 72.5539% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 72.0247% | 72.6226% | 72.6409% | 72.3701% | 72.1702% | 72.0913% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 72.0247% | 72.6226% | 72.6409% | 72.3701% | 72.6270% | 72.0913% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 72.0247% | 72.6899% | 72.8493% | 72.7658% | 72.6818% | 72.5977% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 72.0247% | 72.6226% | 72.6409% | 72.3926% | 72.2200% | 72.1353% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 72.0247% | 72.6226% | 72.6409% | 72.7658% | 72.2200% | 72.5977% |

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

## Temporal bias synthesis MACs

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

## Full-teacher calibration elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 1.150s | 1.510s | 2.439s | 4.868s | 9.432s | 18.448s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.150s | 1.510s | 2.439s | 4.868s | 9.432s | 18.448s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.150s | 1.510s | 2.439s | 4.868s | 9.432s | 18.448s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.150s | 1.510s | 2.439s | 4.868s | 9.432s | 18.448s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.150s | 1.510s | 2.439s | 4.868s | 9.432s | 18.448s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.150s | 1.510s | 2.439s | 4.868s | 9.432s | 18.448s |

## Temporal compression elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000s | 0.000s | 0.000s | 0.060s | 0.023s | 0.035s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.060s | 0.023s | 0.035s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.060s | 0.023s | 0.035s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.083s | 1.232s | 1.762s | 2.966s | 5.422s | 10.018s |
| B_QCFS_STANDARD_SNM_R0 | 1.085s | 1.313s | 1.870s | 3.238s | 6.144s | 11.279s |
| C_QCFS_ASNM_R0 | 1.083s | 1.313s | 1.870s | 3.238s | 6.144s | 10.018s |
| D_QCFS_FULL_FTBC_R0 | 1.014s | 1.155s | 1.701s | 2.882s | 5.320s | 10.062s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.112s | 1.329s | 1.859s | 3.174s | 5.992s | 12.143s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.014s | 1.329s | 1.859s | 3.174s | 5.320s | 12.143s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.022s | 1.180s | 1.719s | 3.068s | 5.506s | 10.043s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.048s | 1.232s | 1.890s | 3.505s | 6.062s | 12.756s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.022s | 1.232s | 1.890s | 3.068s | 6.062s | 10.043s |

## Temporal-LR compression

| T | Effective mode | Effective rank | Explained energy | Full parameters | Temporal parameters | Storage ratio | Storage reduction | Synthesis MACs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | full fallback | 1 | 1.000000 | 688 | 688 | 1.000000 | 0.00% | 0 |
| 2 | full fallback | 2 | 1.000000 | 1,376 | 1,376 | 1.000000 | 0.00% | 0 |
| 4 | full fallback | 4 | 1.000000 | 2,752 | 2,752 | 1.000000 | 0.00% | 0 |
| 8 | temporal_low_rank | 4 | 0.904179 | 5,504 | 2,784 | 0.505814 | 49.42% | 22,016 |
| 16 | temporal_low_rank | 4 | 0.775599 | 11,008 | 2,816 | 0.255814 | 74.42% | 44,032 |
| 32 | temporal_low_rank | 4 | 0.657317 | 22,016 | 2,880 | 0.130814 | 86.92% | 88,064 |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 2, 4, 8, 16 | 5.891s |
| Full-FTBC | 2, 4, 8, 32 | 5.692s |
| Temporal-LR FTBC | 2, 4, 16 | 5.695s |

### QCFS accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 37.7000% | 37.7000% | +0.0000pp | off |
| 2 | 45.8000% | 46.6000% | +0.8000pp | on |
| 4 | 56.7000% | 58.8000% | +2.1000pp | on |
| 8 | 70.2000% | 75.8000% | +5.6000pp | on |
| 16 | 81.6000% | 83.4000% | +1.8000pp | on |
| 32 | 84.6000% | 84.4000% | -0.2000pp | off |

### Full-FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 47.6000% | 47.6000% | +0.0000pp | off |
| 2 | 59.6000% | 60.3000% | +0.7000pp | on |
| 4 | 72.9000% | 73.3000% | +0.4000pp | on |
| 8 | 80.8000% | 81.5000% | +0.7000pp | on |
| 16 | 83.9000% | 83.9000% | +0.0000pp | off |
| 32 | 84.7000% | 85.2000% | +0.5000pp | on |

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 47.6000% | 47.6000% | +0.0000pp | off |
| 2 | 59.6000% | 60.3000% | +0.7000pp | on |
| 4 | 72.9000% | 73.3000% | +0.4000pp | on |
| 8 | 81.6000% | 81.1000% | -0.5000pp | off |
| 16 | 81.3000% | 83.3000% | +2.0000pp | on |
| 32 | 85.1000% | 84.9000% | -0.2000pp | off |

## Validation-selection generalization audit

This table is diagnostic only. Test accuracy never changes a frozen A-SNM decision.

| Family | T | Selected from validation | Test off | Test on | Test-best mode | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 1 | off | 63.89% | 63.89% | off | yes |
| Full-FTBC | 1 | off | 66.87% | 66.87% | off | yes |
| Temporal-LR FTBC | 1 | off | 66.87% | 66.87% | off | yes |
| QCFS | 2 | on | 73.29% | 73.65% | on | yes |
| Full-FTBC | 2 | on | 76.91% | 77.23% | on | yes |
| Temporal-LR FTBC | 2 | on | 76.91% | 77.23% | on | yes |
| QCFS | 4 | on | 83.38% | 84.47% | on | yes |
| Full-FTBC | 4 | on | 85.48% | 86.43% | on | yes |
| Temporal-LR FTBC | 4 | on | 85.48% | 86.43% | on | yes |
| QCFS | 8 | on | 89.82% | 90.66% | on | yes |
| Full-FTBC | 8 | on | 89.95% | 90.32% | on | yes |
| Temporal-LR FTBC | 8 | off | 89.92% | 90.46% | on | no |
| QCFS | 16 | on | 91.32% | 91.49% | on | yes |
| Full-FTBC | 16 | off | 91.14% | 91.28% | on | no |
| Temporal-LR FTBC | 16 | on | 91.20% | 91.30% | on | yes |
| QCFS | 32 | off | 91.69% | 91.61% | off | yes |
| Full-FTBC | 32 | on | 91.52% | 91.50% | off | no |
| Temporal-LR FTBC | 32 | off | 91.53% | 91.45% | off | yes |

## Equivalence checks

| Kind | Config/family | T | Expected source | Exact |
|---|---|---:|---|---|
| validation fallback | off:full=temporal | 1 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 1 | Full-FTBC on | yes |
| validation fallback | off:full=temporal | 2 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 2 | Full-FTBC on | yes |
| validation fallback | off:full=temporal | 4 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 4 | Full-FTBC on | yes |
| gate fallback | full=temporal | 1 | identical validation metrics | yes |
| gate fallback | full=temporal | 2 | identical validation metrics | yes |
| gate fallback | full=temporal | 4 | identical validation metrics | yes |
| test fallback | off:full=temporal | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 1 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 1 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 1 | D_QCFS_FULL_FTBC_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1 | G_QCFS_TEMPORAL_LR_FTBC_R0 | yes |
| test fallback | off:full=temporal | 2 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 2 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| test fallback | off:full=temporal | 4 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 4 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 4 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 8 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 8 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 8 | G_QCFS_TEMPORAL_LR_FTBC_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 16 | D_QCFS_FULL_FTBC_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 16 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 32 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 32 | G_QCFS_TEMPORAL_LR_FTBC_R0 | yes |

## Per-layer Temporal-LR reconstruction

### T=1

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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
