# CIFAR-100 / ResNet20 QCFS Six-way Ablation

- Status: **COMPLETE**
- Run kind: smoke
- CSRR: disabled in every configuration
- Dataset / architecture: CIFAR-100 / ResNet20
- QCFS activation levels: L=8
- Selected checkpoint training-log accuracy: 68.78%
- QCFS ANN accuracy re-evaluated here: 68.68%
- QCFS training profile: paper_era
- Pre-report A_QCFS_R0 T=32 gate: 68.78% (gap=-0.10pp)
- Weight origin: official_implementation_retrained
- Checkpoint: resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth
- Checkpoint size: 1,180,823 bytes
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- Official implementation: https://github.com/putshua/ANN_SNN_QCFS (commit `eca136bd085087567013240ee14fb6159a2b6da7`)
- Official checkpoint folder checked: https://drive.google.com/drive/folders/1P-2egAraWtsQYNzp8lcJvZVEG_KLVV5Q
- Time steps: [8]
- Calibration: batches=5, batch_size=200, alpha=0.4, ridge=0.001, coefficient_clip=0.25, w_under=1.0, w_over=2.5
- Calibration data SHA256: `3856b5e03966e94e502b5472736e0269cd313288a07b3cbe0895d02dc80d0e18`
- Seed: 42
- Coding is rate, ratio=1, R0=legacy_clamp in every group; ScaleOps are therefore expected to be zero.
- The over/under weights apply only to state-low-rank regression. Full FTBC retains the preceding per-timestep mean-bias solver.
- SOPs are input-driven; positive and negative spikes both count as events. Raw image input is not counted as a spike source.

## Configuration Matrix

| Config | QCFS | SNM | R0 | Full FTBC | State-LR FTBC | CSRR |
|---|---|---|---|---|---|---|
| F_QCFS_SNM_R0_STATE_LR | Yes | Yes | Yes | No | Yes | No |

## Accuracy

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 0.49% |

## Input-driven SOPs

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 667,998,113,300 |

## Time-scale operations

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 0 |

## Positive spike rate

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 17.3972% |

## Negative spike rate

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 0.7295% |

## Overall spike sparsity

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 81.8733% |

## FTBC parameters

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 2,064 |

## FTBC storage bytes

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 8,256 |

## Calibration elapsed

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 8.5s |

## Inference elapsed (statistics disabled)

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | 3.6s |

## Effective FTBC Mode

| Config | T=8 |
|---|---|
| F_QCFS_SNM_R0_STATE_LR | state_low_rank |

State-LR has three channel-wise coefficients. At T=1 and T=2 it falls back to full FTBC; full-vs-low-rank comparisons are interpretable from T>=4.

## Published QCFS Reference (CIFAR-100 / ResNet20)

| ANN | T=2 | T=4 | T=8 | T=16 | T=32 |
|---:|---:|---:|---:|---:|---:|
| 69.94% | 19.96% | 34.14% | 55.37% | 67.33% | 69.82% |

## Per-layer Detail

### F_QCFS_SNM_R0_STATE_LR, T=8

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| conv1.2 | 30.4801% | 0.0000% | 69.5199% | 0 | 0 | 0 |
| conv2_x.0.residual_function.2 | 23.5638% | 0.7219% | 75.7143% | 399,508,624 | 57,529,241,856 | 0 |
| conv2_x.0.act | 23.3848% | 0.2169% | 76.3983% | 318,317,508 | 45,837,721,152 | 0 |
| conv2_x.1.residual_function.2 | 5.5766% | 0.9524% | 93.4710% | 309,352,228 | 44,546,720,832 | 0 |
| conv2_x.1.act | 23.6829% | 0.3819% | 75.9352% | 85,576,676 | 12,323,041,344 | 0 |
| conv2_x.2.residual_function.2 | 7.8893% | 0.8463% | 91.2644% | 315,421,769 | 45,420,734,736 | 0 |
| conv2_x.2.act | 28.0417% | 0.1718% | 71.7865% | 114,499,662 | 16,487,951,328 | 0 |
| conv3_x.0.residual_function.2 | 12.4796% | 0.5108% | 87.0096% | 369,799,682 | 106,502,308,416 | 0 |
| conv3_x.0.act | 20.1862% | 0.3639% | 79.4499% | 454,933,678 | 36,352,180,672 | 0 |
| conv3_x.1.residual_function.2 | 4.6823% | 0.9714% | 94.3463% | 134,677,065 | 38,786,994,720 | 0 |
| conv3_x.1.act | 21.2158% | 0.3461% | 78.4381% | 37,052,011 | 10,670,979,168 | 0 |
| conv3_x.2.residual_function.2 | 4.1629% | 1.0748% | 94.7623% | 141,308,091 | 40,696,730,208 | 0 |
| conv3_x.2.act | 21.9612% | 0.3038% | 77.7350% | 34,325,906 | 9,885,860,928 | 0 |
| conv4_x.0.residual_function.2 | 8.2670% | 1.2638% | 90.4692% | 145,915,849 | 84,047,529,024 | 0 |
| conv4_x.0.act | 9.3450% | 1.5136% | 89.1414% | 177,146,214 | 27,327,304,576 | 0 |
| conv4_x.1.residual_function.2 | 5.3469% | 1.6474% | 93.0057% | 35,581,528 | 20,494,960,128 | 0 |
| conv4_x.1.act | 14.4283% | 2.6590% | 82.9127% | 22,918,854 | 13,201,259,904 | 0 |
| conv4_x.2.residual_function.2 | 7.4870% | 2.8143% | 89.6987% | 55,991,725 | 32,251,233,600 | 0 |
| conv4_x.2.act | 15.5463% | 3.3514% | 81.1023% | 33,755,158 | 19,442,971,008 | 0 |
| fc | - | - | - | 61,923,897 | 6,192,389,700 | 0 |
