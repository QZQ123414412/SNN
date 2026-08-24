# QCFS + Full/Temporal-LR/PA-FTBC + HA-SNM Ablation

Status: complete

- Dataset/architecture: CIFAR-10/resnet20
- QCFS L: 4
- ANN accuracy: 89.50%
- Checkpoint: `resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth`
- Checkpoint SHA256: `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3`
- Fit/validation SHA256: `91909d93eb8fb74d65302d1548e3d026296c22235f21255153411bff6ad97905` / `2680aa74ee4792d98247b8f0fe96b3727ae6f1e43319d93759f379e940ba4405`
- Test samples: 200
- Evaluation profile: `paper_era`
- HA-SNM threshold schedule: start=1.5, end=0.5, linear.
- HA-SNM keeps the original transmitted-credit/R0 rule and changes only the negative-spike decision threshold.
- It uses the original -theta event amplitude, adds no dense neuron state, and has two global FP32 deployment constants (8 bytes).
- Full-FTBC is fitted independently at every T with SNM off; Temporal-LR and PA are compressed from that same teacher.
- Temporal-LR and PA fall back exactly to Full-FTBC at T<=4.
- Checkpoint note: paper-aligned retrained checkpoint selected by peak test accuracy; not a strict paper reproduction.

## Primary accuracy

| Config | T=4 | T=8 | Mean |
|---|---:|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 84.00% | 88.00% | 86.00% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 86.00% | 91.50% | 88.75% |
| C_QCFS_FULL_FTBC_HA_R0 | 89.00% | 91.00% | 90.00% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 84.00% | 90.00% | 87.00% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 86.00% | 91.50% | 88.75% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 89.00% | 90.00% | 89.50% |
| G_QCFS_PA_FTBC_OFF_R0 | 84.00% | 89.00% | 86.50% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 86.00% | 91.00% | 88.50% |
| I_QCFS_PA_FTBC_HA_R0 | 89.00% | 90.00% | 89.50% |

## HA-SNM accuracy gain

| Family | T=4 | T=8 | Mean |
|---|---:|---:|---:|
| Full-FTBC: HA - standard | +3.00pp | -0.50pp | +1.250pp |
| Full-FTBC: HA - off | +5.00pp | +3.00pp | +4.000pp |
| Temporal-LR FTBC: HA - standard | +3.00pp | -1.50pp | +0.750pp |
| Temporal-LR FTBC: HA - off | +5.00pp | +0.00pp | +2.500pp |
| PA-FTBC: HA - standard | +3.00pp | -1.00pp | +1.000pp |
| PA-FTBC: HA - off | +5.00pp | +1.00pp | +3.000pp |

## ANN-SNN logit MSE

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 2.43739551 | 1.04979565 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 2.22103198 | 0.81218958 |
| C_QCFS_FULL_FTBC_HA_R0 | 1.88761865 | 0.76884985 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 2.43739551 | 1.01729687 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 2.22103198 | 0.78239435 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 1.88761865 | 0.73470380 |
| G_QCFS_PA_FTBC_OFF_R0 | 2.43739551 | 1.05111572 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 2.22103198 | 0.80012720 |
| I_QCFS_PA_FTBC_HA_R0 | 1.88761865 | 0.74923871 |

## Positive spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 27.716289% | 27.542939% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 27.743605% | 27.642264% |
| C_QCFS_FULL_FTBC_HA_R0 | 27.791818% | 27.704885% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 27.716289% | 27.522162% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 27.743605% | 27.618433% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 27.791818% | 27.684539% |
| G_QCFS_PA_FTBC_OFF_R0 | 27.716289% | 27.545578% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 27.743605% | 27.642797% |
| I_QCFS_PA_FTBC_HA_R0 | 27.791818% | 27.707364% |

## Negative spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 0.000000% | 0.000000% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 0.176818% | 0.268052% |
| C_QCFS_FULL_FTBC_HA_R0 | 0.468361% | 0.497771% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 0.000000% | 0.000000% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 0.176818% | 0.267350% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 0.468361% | 0.496551% |
| G_QCFS_PA_FTBC_OFF_R0 | 0.000000% | 0.000000% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 0.176818% | 0.268410% |
| I_QCFS_PA_FTBC_HA_R0 | 0.468361% | 0.498448% |

## Overall spike sparsity

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 72.283711% | 72.457061% |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 72.079578% | 72.089684% |
| C_QCFS_FULL_FTBC_HA_R0 | 71.739820% | 71.797343% |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 72.283711% | 72.477838% |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 72.079578% | 72.114217% |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 71.739820% | 71.818910% |
| G_QCFS_PA_FTBC_OFF_R0 | 72.283711% | 72.454422% |
| H_QCFS_PA_FTBC_STANDARD_R0 | 72.079578% | 72.088792% |
| I_QCFS_PA_FTBC_HA_R0 | 71.739820% | 71.794188% |

## Input-driven SOPs

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 10,274,686,026 | 20,454,749,958 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 10,356,663,346 | 20,775,820,030 |
| C_QCFS_FULL_FTBC_HA_R0 | 10,515,430,892 | 21,050,082,094 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 10,274,686,026 | 20,442,720,938 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 10,356,663,346 | 20,758,772,146 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 10,515,430,892 | 21,037,332,308 |
| G_QCFS_PA_FTBC_OFF_R0 | 10,274,686,026 | 20,459,226,352 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 10,356,663,346 | 20,777,793,756 |
| I_QCFS_PA_FTBC_HA_R0 | 10,515,430,892 | 21,055,251,384 |

## FTBC parameters

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 2,752 | 5,504 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 2,752 | 5,504 |
| C_QCFS_FULL_FTBC_HA_R0 | 2,752 | 5,504 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 2,752 | 2,784 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 2,752 | 2,784 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 2,752 | 2,784 |
| G_QCFS_PA_FTBC_OFF_R0 | 2,752 | 2,752 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 2,752 | 2,752 |
| I_QCFS_PA_FTBC_HA_R0 | 2,752 | 2,752 |

## FTBC storage bytes

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 11,008 | 22,016 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 11,008 | 22,016 |
| C_QCFS_FULL_FTBC_HA_R0 | 11,008 | 22,016 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 11,008 | 11,136 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 11,008 | 11,136 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 11,008 | 11,136 |
| G_QCFS_PA_FTBC_OFF_R0 | 11,008 | 11,008 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 11,008 | 11,008 |
| I_QCFS_PA_FTBC_HA_R0 | 11,008 | 11,008 |

## Bias synthesis MACs

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 0 | 0 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 0 | 0 |
| C_QCFS_FULL_FTBC_HA_R0 | 0 | 0 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 0 | 22,016 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 0 | 22,016 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 0 | 22,016 |
| G_QCFS_PA_FTBC_OFF_R0 | 0 | 9,632 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 0 | 9,632 |
| I_QCFS_PA_FTBC_HA_R0 | 0 | 9,632 |

## Inference elapsed

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_FULL_FTBC_OFF_R0 | 0.023225 | 0.051028 |
| B_QCFS_FULL_FTBC_STANDARD_R0 | 0.029134 | 0.052348 |
| C_QCFS_FULL_FTBC_HA_R0 | 0.029210 | 0.049835 |
| D_QCFS_TEMPORAL_FTBC_OFF_R0 | 0.024375 | 0.046339 |
| E_QCFS_TEMPORAL_FTBC_STANDARD_R0 | 0.025504 | 0.054668 |
| F_QCFS_TEMPORAL_FTBC_HA_R0 | 0.025201 | 0.052075 |
| G_QCFS_PA_FTBC_OFF_R0 | 0.025969 | 0.046798 |
| H_QCFS_PA_FTBC_STANDARD_R0 | 0.025101 | 0.056073 |
| I_QCFS_PA_FTBC_HA_R0 | 0.025741 | 0.055402 |

## HA-SNM overhead

| Item | Value |
|---|---:|
| Additional dense per-neuron state | 0 bytes |
| Global FP32 constants | 2 (8 bytes) |
| SignedIF layers | 19 |
| Per layer/time decision overhead | one scalar threshold interpolation and the existing comparison |

## Exact fallback checks

| T | Mode | Full=Temporal | Full=PA |
|---:|---|---|---|
| 4 | SNM-off | yes | yes |
| 4 | standard SNM | yes | yes |
| 4 | HA-SNM | yes | yes |
