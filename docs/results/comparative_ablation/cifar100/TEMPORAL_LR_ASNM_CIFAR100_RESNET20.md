# QCFS + Full-FTBC + Temporal-LR FTBC + A-SNM CIFAR-100 Ablation

- Status: complete
- Architecture: resnet20
- Checkpoint: `resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth`
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- ANN accuracy on the 10,000-image test set: 68.68%
- Time steps: [1, 2, 4, 8, 16, 32]
- Full-FTBC fit: 5 x 200, alpha=0.4
- A-SNM validation: 5 x 200
- Temporal-LR: shared rank-4 basis, threshold-normalized, no exempt layer.
- Temporal-LR falls back to Full-FTBC at T<=4 and is active at T>4.
- Fit batch SHA256: `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a`
- Validation batch SHA256: `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3`
- Train batches use RandomCrop, RandomHorizontalFlip, CIFAR10Policy, ToTensor, CIFAR-100 normalization, and Cutout(1,16).
- Test uses only ToTensor and normalization with shuffle=False.
- Every SNN uses QCFS L=8, rate coding/schedule, ratio=1.0, R0=True, SNM margin=0, FP32.
- Full-FTBC is independently fitted at every T with SNM off; Temporal-LR is compressed from that frozen teacher.
- Each family enables SNM only when SNM-on has strictly higher validation accuracy; ties select off.
- Test data is first accessed after all three A-SNM families are frozen.
- A-SNM guarantees validation-set selection only; test-set reversals are reported diagnostically and never retuned.

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

## ANN-SNN logit MSE

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 17.060971 | 13.857015 | 9.981356 | 5.256991 | 1.980072 | 0.713141 |
| B_QCFS_STANDARD_SNM_R0 | 17.060971 | 13.665451 | 8.934425 | 3.276014 | 1.072969 | 0.396984 |
| C_QCFS_ASNM_R0 | 17.060971 | 13.857015 | 8.934425 | 3.276014 | 1.072969 | 0.396984 |
| D_QCFS_FULL_FTBC_R0 | 11.704163 | 8.864907 | 5.987022 | 2.978636 | 1.153339 | 0.473841 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 11.704163 | 8.808012 | 5.633623 | 2.561259 | 0.864456 | 0.368019 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 11.704163 | 8.808012 | 5.633623 | 2.561259 | 0.864456 | 0.368019 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 11.704163 | 8.864907 | 5.987022 | 2.988893 | 1.188850 | 0.531669 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 11.704163 | 8.808012 | 5.633623 | 2.560220 | 0.794747 | 0.354499 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 11.704163 | 8.808012 | 5.633623 | 2.560220 | 0.794747 | 0.354499 |

## Positive spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 21.7705% | 22.3887% | 21.5543% | 20.9977% | 20.7175% | 20.5833% |
| B_QCFS_STANDARD_SNM_R0 | 21.7705% | 22.6098% | 21.9433% | 21.5252% | 21.2464% | 21.0660% |
| C_QCFS_ASNM_R0 | 21.7705% | 22.3887% | 21.9433% | 21.5252% | 21.2464% | 21.0660% |
| D_QCFS_FULL_FTBC_R0 | 21.1393% | 20.3062% | 20.1176% | 20.1557% | 20.3057% | 20.3607% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 21.1393% | 20.4988% | 20.5168% | 20.6697% | 20.8247% | 20.8403% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 21.1393% | 20.4988% | 20.5168% | 20.6697% | 20.8247% | 20.8403% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 21.1393% | 20.3062% | 20.1176% | 20.1573% | 20.4246% | 20.4157% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 21.1393% | 20.4988% | 20.5168% | 20.6698% | 20.9471% | 20.8925% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 21.1393% | 20.4988% | 20.5168% | 20.6698% | 20.9471% | 20.8925% |

## Negative spike rate

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| B_QCFS_STANDARD_SNM_R0 | 0.0000% | 0.1214% | 0.2756% | 0.4826% | 0.5466% | 0.5036% |
| C_QCFS_ASNM_R0 | 0.0000% | 0.0000% | 0.2756% | 0.4826% | 0.5466% | 0.5036% |
| D_QCFS_FULL_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0820% | 0.2394% | 0.3983% | 0.4626% | 0.4568% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.0000% | 0.0820% | 0.2394% | 0.3983% | 0.4626% | 0.4568% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.0000% | 0.0820% | 0.2394% | 0.3978% | 0.4719% | 0.4646% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.0000% | 0.0820% | 0.2394% | 0.3978% | 0.4719% | 0.4646% |

## Overall spike sparsity

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 78.2295% | 77.6113% | 78.4457% | 79.0023% | 79.2825% | 79.4167% |
| B_QCFS_STANDARD_SNM_R0 | 78.2295% | 77.2688% | 77.7811% | 77.9922% | 78.2070% | 78.4304% |
| C_QCFS_ASNM_R0 | 78.2295% | 77.6113% | 77.7811% | 77.9922% | 78.2070% | 78.4304% |
| D_QCFS_FULL_FTBC_R0 | 78.8607% | 79.6938% | 79.8824% | 79.8443% | 79.6943% | 79.6393% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 78.8607% | 79.4192% | 79.2439% | 78.9320% | 78.7127% | 78.7029% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 78.8607% | 79.4192% | 79.2439% | 78.9320% | 78.7127% | 78.7029% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 78.8607% | 79.6938% | 79.8824% | 79.8427% | 79.5754% | 79.5843% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 78.8607% | 79.4192% | 79.2439% | 78.9324% | 78.5809% | 78.6429% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 78.8607% | 79.4192% | 79.2439% | 78.9324% | 78.5809% | 78.6429% |

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
| D_QCFS_FULL_FTBC_R0 | 1.004s | 1.370s | 2.505s | 4.873s | 9.931s | 19.274s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.004s | 1.370s | 2.505s | 4.873s | 9.931s | 19.274s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.004s | 1.370s | 2.505s | 4.873s | 9.931s | 19.274s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.004s | 1.370s | 2.505s | 4.873s | 9.931s | 19.274s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.004s | 1.370s | 2.505s | 4.873s | 9.931s | 19.274s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.004s | 1.370s | 2.505s | 4.873s | 9.931s | 19.274s |

## Temporal compression elapsed

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| B_QCFS_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| C_QCFS_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| D_QCFS_FULL_FTBC_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s | 0.000s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000s | 0.000s | 0.000s | 0.055s | 0.021s | 0.037s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000s | 0.000s | 0.000s | 0.055s | 0.021s | 0.037s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000s | 0.000s | 0.000s | 0.055s | 0.021s | 0.037s |

## Inference elapsed (statistics disabled)

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 1.113s | 1.332s | 1.933s | 3.223s | 6.113s | 11.380s |
| B_QCFS_STANDARD_SNM_R0 | 1.171s | 1.438s | 2.122s | 3.576s | 6.521s | 12.685s |
| C_QCFS_ASNM_R0 | 1.113s | 1.332s | 2.122s | 3.576s | 6.521s | 12.685s |
| D_QCFS_FULL_FTBC_R0 | 1.119s | 1.292s | 1.918s | 3.132s | 5.534s | 9.909s |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 1.185s | 1.441s | 2.082s | 3.586s | 6.193s | 11.032s |
| F_QCFS_FULL_FTBC_ASNM_R0 | 1.119s | 1.441s | 2.082s | 3.586s | 6.193s | 11.032s |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 1.112s | 1.263s | 1.888s | 3.467s | 7.515s | 10.115s |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 1.167s | 1.390s | 2.092s | 3.696s | 6.673s | 11.342s |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 1.112s | 1.390s | 2.092s | 3.696s | 6.673s | 11.342s |

## Temporal-LR compression

| T | Effective mode | Effective rank | Explained energy | Full parameters | Temporal parameters | Storage ratio | Storage reduction | Synthesis MACs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | full fallback | 1 | 1.000000 | 688 | 688 | 1.000000 | 0.00% | 0 |
| 2 | full fallback | 2 | 1.000000 | 1,376 | 1,376 | 1.000000 | 0.00% | 0 |
| 4 | full fallback | 4 | 1.000000 | 2,752 | 2,752 | 1.000000 | 0.00% | 0 |
| 8 | temporal_low_rank | 4 | 0.842512 | 5,504 | 2,784 | 0.505814 | 49.42% | 22,016 |
| 16 | temporal_low_rank | 4 | 0.693008 | 11,008 | 2,816 | 0.255814 | 74.42% | 44,032 |
| 32 | temporal_low_rank | 4 | 0.586493 | 22,016 | 2,880 | 0.130814 | 86.92% | 88,064 |

## A-SNM selection

| Family | SNM-on T | Validation inference + selection |
|---|---|---:|
| QCFS | 4, 8, 16, 32 | 6.051s |
| Full-FTBC | 2, 4, 8, 16, 32 | 5.686s |
| Temporal-LR FTBC | 2, 4, 8, 16, 32 | 5.844s |

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

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off val. acc. | SNM-on val. acc. | On-off | Selected |
|---:|---:|---:|---:|---|
| 1 | 6.2000% | 6.2000% | +0.0000pp | off |
| 2 | 11.0000% | 11.1000% | +0.1000pp | on |
| 4 | 20.9000% | 23.4000% | +2.5000pp | on |
| 8 | 37.0000% | 39.2000% | +2.2000pp | on |
| 16 | 45.8000% | 49.0000% | +3.2000pp | on |
| 32 | 50.1000% | 51.8000% | +1.7000pp | on |

## Validation-selection generalization audit

This table is diagnostic only. Test accuracy never changes a frozen A-SNM decision.

| Family | T | Selected from validation | Test off | Test on | Test-best mode | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 1 | off | 7.93% | 7.93% | off | yes |
| Full-FTBC | 1 | off | 14.66% | 14.66% | off | yes |
| Temporal-LR FTBC | 1 | off | 14.66% | 14.66% | off | yes |
| QCFS | 2 | off | 11.66% | 12.04% | on | no |
| Full-FTBC | 2 | on | 22.69% | 22.92% | on | yes |
| Temporal-LR FTBC | 2 | on | 22.69% | 22.92% | on | yes |
| QCFS | 4 | on | 22.42% | 25.41% | on | yes |
| Full-FTBC | 4 | on | 38.81% | 40.82% | on | yes |
| Temporal-LR FTBC | 4 | on | 38.81% | 40.82% | on | yes |
| QCFS | 8 | on | 46.37% | 57.27% | on | yes |
| Full-FTBC | 8 | on | 59.15% | 60.50% | on | yes |
| Temporal-LR FTBC | 8 | on | 58.94% | 60.38% | on | yes |
| QCFS | 16 | on | 64.03% | 66.52% | on | yes |
| Full-FTBC | 16 | on | 67.58% | 67.43% | off | no |
| Temporal-LR FTBC | 16 | on | 67.64% | 68.15% | on | yes |
| QCFS | 32 | on | 68.78% | 69.00% | on | yes |
| Full-FTBC | 32 | on | 69.37% | 68.50% | off | no |
| Temporal-LR FTBC | 32 | on | 69.37% | 69.19% | off | no |

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
| A-SNM cache | C_QCFS_ASNM_R0 | 2 | A_QCFS_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 2 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| test fallback | off:full=temporal | 4 | D_QCFS_FULL_FTBC_R0 | yes |
| test fallback | on:full=temporal | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 4 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 4 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 4 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 8 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 8 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 8 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 16 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 16 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 16 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | C_QCFS_ASNM_R0 | 32 | B_QCFS_STANDARD_SNM_R0 | yes |
| A-SNM cache | F_QCFS_FULL_FTBC_ASNM_R0 | 32 | E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | yes |
| A-SNM cache | I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 32 | H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | yes |

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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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

| Layer | Representation | Channels | MSE | NRMSE | Max abs. error |
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
