# QCFS + Full-FTBC + Temporal-LR FTBC + Parity-Anchor FTBC + A-SNM Ablation

Status: complete

- Dataset/architecture: CIFAR-10/resnet20
- QCFS L: 4
- ANN accuracy: 89.50%
- Checkpoint: `resnet20_L[4]_paper_L4_bs300_seed42_testbest.pth`
- Checkpoint SHA256: `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3`
- Fit/validation SHA256: `91909d93eb8fb74d65302d1548e3d026296c22235f21255153411bff6ad97905` / `2680aa74ee4792d98247b8f0fe96b3727ae6f1e43319d93759f379e940ba4405`
- Test samples: 200
- Evaluation profile: `paper_era`
- Full-FTBC is fitted independently at every T with SNM off.
- Temporal-LR uses a shared learned rank-4 SVD basis with threshold normalization.
- PA-FTBC uses no SVD or stored basis: t=0/t=1 anchors plus tail mean and tail parity.
- Both compressed methods fall back exactly to Full-FTBC at T<=4.
- Every family freezes its own strict accuracy-gated A-SNM decisions before test inference.
- Checkpoint note: CIFAR-10/ResNet20 QCFS-L4 paper-aligned retrained checkpoint; selected by peak test accuracy during training and therefore subject to test-set model-selection bias; not a strict paper reproduction.

## Primary accuracy table

| Config | T=4 | T=8 | SNM-on T |
|---|---:|---:|---|
| A_QCFS_R0 | 83.00% | 89.50% | none |
| B_QCFS_STANDARD_SNM_R0 | 88.50% | 89.50% | 4, 8 |
| C_QCFS_ASNM_R0 | 88.50% | 89.50% | 4, 8 |
| D_QCFS_FULL_FTBC_R0 | 84.00% | 88.00% | none |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 86.00% | 91.50% | 4, 8 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 86.00% | 91.50% | 4, 8 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 84.00% | 90.00% | none |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 86.00% | 91.50% | 4, 8 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 86.00% | 91.50% | 4, 8 |
| J_QCFS_PA_FTBC_R0 | 84.00% | 89.00% | none |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 86.00% | 91.00% | 4, 8 |
| L_QCFS_PA_FTBC_ASNM_R0 | 86.00% | 91.00% | 4, 8 |

## Mean accuracy over evaluated time steps

| Config | Mean accuracy |
|---|---:|
| A_QCFS_R0 | 86.25% |
| B_QCFS_STANDARD_SNM_R0 | 89.00% |
| C_QCFS_ASNM_R0 | 89.00% |
| D_QCFS_FULL_FTBC_R0 | 86.00% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 88.75% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 88.75% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 87.00% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 88.75% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 88.75% |
| J_QCFS_PA_FTBC_R0 | 86.50% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 88.50% |
| L_QCFS_PA_FTBC_ASNM_R0 | 88.50% |

## PA-FTBC accuracy comparisons

| T | PA off - Temporal off | PA standard - Temporal standard | PA A-SNM - Temporal A-SNM |
|---:|---:|---:|---:|
| 4 | +0.00pp | +0.00pp | +0.00pp |
| 8 | -1.00pp | -0.50pp | -0.50pp |
| Mean | -0.50pp | -0.25pp | -0.25pp |

## ANN-SNN logit MSE

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 2.70476660 | 1.15644946 |
| B_QCFS_STANDARD_SNM_R0 | 2.39909473 | 0.79287976 |
| C_QCFS_ASNM_R0 | 2.39909473 | 0.79287976 |
| D_QCFS_FULL_FTBC_R0 | 2.43739551 | 1.04979565 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2.22103198 | 0.81218958 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2.22103198 | 0.81218958 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 2.43739551 | 1.01729687 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 2.22103198 | 0.78239435 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2.22103198 | 0.78239435 |
| J_QCFS_PA_FTBC_R0 | 2.43739551 | 1.05111572 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 2.22103198 | 0.80012720 |
| L_QCFS_PA_FTBC_ASNM_R0 | 2.22103198 | 0.80012720 |

## Positive spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 28.050109% | 27.714498% |
| B_QCFS_STANDARD_SNM_R0 | 28.063371% | 27.810334% |
| C_QCFS_ASNM_R0 | 28.063371% | 27.810334% |
| D_QCFS_FULL_FTBC_R0 | 27.716289% | 27.542939% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 27.743605% | 27.642264% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 27.743605% | 27.642264% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 27.716289% | 27.522162% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 27.743605% | 27.618433% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 27.743605% | 27.618433% |
| J_QCFS_PA_FTBC_R0 | 27.716289% | 27.545578% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 27.743605% | 27.642797% |
| L_QCFS_PA_FTBC_ASNM_R0 | 27.743605% | 27.642797% |

## Negative spike rate

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.000000% | 0.000000% |
| B_QCFS_STANDARD_SNM_R0 | 0.174302% | 0.267639% |
| C_QCFS_ASNM_R0 | 0.174302% | 0.267639% |
| D_QCFS_FULL_FTBC_R0 | 0.000000% | 0.000000% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.176818% | 0.268052% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.176818% | 0.268052% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000% | 0.000000% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.176818% | 0.267350% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.176818% | 0.267350% |
| J_QCFS_PA_FTBC_R0 | 0.000000% | 0.000000% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.176818% | 0.268410% |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.176818% | 0.268410% |

## Overall spike sparsity

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 71.949891% | 72.285502% |
| B_QCFS_STANDARD_SNM_R0 | 71.762326% | 71.922027% |
| C_QCFS_ASNM_R0 | 71.762326% | 71.922027% |
| D_QCFS_FULL_FTBC_R0 | 72.283711% | 72.457061% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 72.079578% | 72.089684% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 72.079578% | 72.089684% |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 72.283711% | 72.477838% |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 72.079578% | 72.114217% |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 72.079578% | 72.114217% |
| J_QCFS_PA_FTBC_R0 | 72.283711% | 72.454422% |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 72.079578% | 72.088792% |
| L_QCFS_PA_FTBC_ASNM_R0 | 72.079578% | 72.088792% |

## Input-driven SOPs

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 10,426,035,882 | 20,603,191,794 |
| B_QCFS_STANDARD_SNM_R0 | 10,500,810,602 | 20,926,707,668 |
| C_QCFS_ASNM_R0 | 10,500,810,602 | 20,926,707,668 |
| D_QCFS_FULL_FTBC_R0 | 10,274,686,026 | 20,454,749,958 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 10,356,663,346 | 20,775,820,030 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 10,356,663,346 | 20,775,820,030 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 10,274,686,026 | 20,442,720,938 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 10,356,663,346 | 20,758,772,146 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 10,356,663,346 | 20,758,772,146 |
| J_QCFS_PA_FTBC_R0 | 10,274,686,026 | 20,459,226,352 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 10,356,663,346 | 20,777,793,756 |
| L_QCFS_PA_FTBC_ASNM_R0 | 10,356,663,346 | 20,777,793,756 |

## FTBC parameters

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 2,752 | 5,504 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 2,752 | 5,504 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 2,752 | 5,504 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 2,752 | 2,784 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 2,752 | 2,784 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 2,752 | 2,784 |
| J_QCFS_PA_FTBC_R0 | 2,752 | 2,752 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 2,752 | 2,752 |
| L_QCFS_PA_FTBC_ASNM_R0 | 2,752 | 2,752 |

## FTBC storage bytes

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 11,008 | 22,016 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 11,008 | 22,016 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 11,008 | 22,016 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 11,008 | 11,136 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 11,008 | 11,136 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 11,008 | 11,136 |
| J_QCFS_PA_FTBC_R0 | 11,008 | 11,008 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 11,008 | 11,008 |
| L_QCFS_PA_FTBC_ASNM_R0 | 11,008 | 11,008 |

## Bias synthesis MACs

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0 | 0 |
| B_QCFS_STANDARD_SNM_R0 | 0 | 0 |
| C_QCFS_ASNM_R0 | 0 | 0 |
| D_QCFS_FULL_FTBC_R0 | 0 | 0 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0 | 0 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0 | 0 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0 | 22,016 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0 | 22,016 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0 | 22,016 |
| J_QCFS_PA_FTBC_R0 | 0 | 9,632 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0 | 9,632 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0 | 9,632 |

## Full-teacher calibration elapsed

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.000000 | 0.000000 |
| B_QCFS_STANDARD_SNM_R0 | 0.000000 | 0.000000 |
| C_QCFS_ASNM_R0 | 0.000000 | 0.000000 |
| D_QCFS_FULL_FTBC_R0 | 0.546132 | 1.009540 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.546132 | 1.009540 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.546132 | 1.009540 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.546132 | 1.009540 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.546132 | 1.009540 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.546132 | 1.009540 |
| J_QCFS_PA_FTBC_R0 | 0.546132 | 1.009540 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.546132 | 1.009540 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.546132 | 1.009540 |

## Compression elapsed

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.000000 | 0.000000 |
| B_QCFS_STANDARD_SNM_R0 | 0.000000 | 0.000000 |
| C_QCFS_ASNM_R0 | 0.000000 | 0.000000 |
| D_QCFS_FULL_FTBC_R0 | 0.000000 | 0.000000 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.000000 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.000000 | 0.000000 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.000000 | 0.068337 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.068337 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.000000 | 0.068337 |
| J_QCFS_PA_FTBC_R0 | 0.000000 | 0.041777 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.000000 | 0.041777 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.000000 | 0.041777 |

## Inference elapsed (statistics disabled)

| Config | T=4 | T=8 |
|---|---:|---:|
| A_QCFS_R0 | 0.023591 | 0.048909 |
| B_QCFS_STANDARD_SNM_R0 | 0.029006 | 0.052550 |
| C_QCFS_ASNM_R0 | 0.029006 | 0.052550 |
| D_QCFS_FULL_FTBC_R0 | 0.023453 | 0.049707 |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 0.025926 | 0.055155 |
| F_QCFS_FULL_FTBC_ASNM_R0 | 0.025926 | 0.055155 |
| G_QCFS_TEMPORAL_LR_FTBC_R0 | 0.021887 | 0.046912 |
| H_QCFS_TEMPORAL_LR_FTBC_STANDARD_SNM_R0 | 0.025106 | 0.057140 |
| I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0 | 0.025106 | 0.057140 |
| J_QCFS_PA_FTBC_R0 | 0.020518 | 0.049422 |
| K_QCFS_PA_FTBC_STANDARD_SNM_R0 | 0.026078 | 0.053891 |
| L_QCFS_PA_FTBC_ASNM_R0 | 0.026078 | 0.053891 |

## Compression summary

| T | Full params | Temporal params | PA params | Temporal saving | PA saving | Temporal MACs | PA MACs | Temporal energy | PA energy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2,752 | 2,752 | 2,752 | 0.00% | 0.00% | 0 | 0 | 1.000000 | 1.000000 |
| 8 | 5,504 | 2,784 | 2,752 | 49.42% | 50.00% | 22,016 | 9,632 | 0.855774 | 0.849295 |

## A-SNM selection

- QCFS SNM-on T: 4, 8; selection elapsed: 0.481305s.

### QCFS accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 4 | 59.00% | 62.00% | on |
| 8 | 68.50% | 81.00% | on |

- Full-FTBC SNM-on T: 4, 8; selection elapsed: 0.272908s.

### Full-FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 4 | 70.50% | 72.00% | on |
| 8 | 80.50% | 83.50% | on |

- Temporal-LR FTBC SNM-on T: 4, 8; selection elapsed: 0.293752s.

### Temporal-LR FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 4 | 70.50% | 72.00% | on |
| 8 | 79.00% | 83.00% | on |

- Parity-Anchor FTBC SNM-on T: 4, 8; selection elapsed: 0.273499s.

### Parity-Anchor FTBC accuracy-gate trace

| T | SNM-off validation accuracy | SNM-on validation accuracy | Selected |
|---:|---:|---:|---|
| 4 | 70.50% | 72.00% | on |
| 8 | 80.50% | 84.00% | on |

## Validation-selection generalization audit

| Family | T | Selected | Test off | Test on | Test-best | Match |
|---|---:|---|---:|---:|---|---|
| QCFS | 4 | on | 83.00% | 88.50% | on | yes |
| Full-FTBC | 4 | on | 84.00% | 86.00% | on | yes |
| Temporal-LR FTBC | 4 | on | 84.00% | 86.00% | on | yes |
| Parity-Anchor FTBC | 4 | on | 84.00% | 86.00% | on | yes |
| QCFS | 8 | on | 89.50% | 89.50% | off | no |
| Full-FTBC | 8 | on | 88.00% | 91.50% | on | yes |
| Temporal-LR FTBC | 8 | on | 90.00% | 91.50% | on | yes |
| Parity-Anchor FTBC | 8 | on | 89.00% | 91.00% | on | yes |

## Equivalence checks

| Kind | Name | T | Source | Exact |
|---|---|---:|---|---|
| validation fallback | off:full=temporal | 4 | Full-FTBC off | yes |
| validation fallback | on:full=temporal | 4 | Full-FTBC on | yes |
| validation fallback | off:full=pa | 4 | Full-FTBC off | yes |
| validation fallback | on:full=pa | 4 | Full-FTBC on | yes |
| gate fallback | full=temporal | 4 | identical validation metrics | yes |
| gate fallback | full=pa | 4 | identical validation metrics | yes |
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

## Per-layer Temporal-LR FTBC reconstruction

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
| `conv1.2` | temporal_low_rank | 16 | 0.00047703 | 0.49429998 | 0.07374726 |
| `conv2_x.0.residual_function.2` | temporal_low_rank | 16 | 0.00010506 | 0.58651346 | 0.04153423 |
| `conv2_x.0.act` | temporal_low_rank | 16 | 0.00031850 | 0.44397634 | 0.05688540 |
| `conv2_x.1.residual_function.2` | temporal_low_rank | 16 | 0.00006852 | 0.39816552 | 0.02945636 |
| `conv2_x.1.act` | temporal_low_rank | 16 | 0.00019084 | 0.33927485 | 0.03919053 |
| `conv2_x.2.residual_function.2` | temporal_low_rank | 16 | 0.00000985 | 0.25561839 | 0.01458037 |
| `conv2_x.2.act` | temporal_low_rank | 16 | 0.00007914 | 0.27495101 | 0.03397037 |
| `conv3_x.0.residual_function.2` | temporal_low_rank | 32 | 0.00000976 | 0.24571653 | 0.01210847 |
| `conv3_x.0.act` | temporal_low_rank | 32 | 0.00001442 | 0.21947806 | 0.02108851 |
| `conv3_x.1.residual_function.2` | temporal_low_rank | 32 | 0.00000495 | 0.24685015 | 0.01725496 |
| `conv3_x.1.act` | temporal_low_rank | 32 | 0.00001822 | 0.23235802 | 0.01985191 |
| `conv3_x.2.residual_function.2` | temporal_low_rank | 32 | 0.00000102 | 0.17096809 | 0.00659266 |
| `conv3_x.2.act` | temporal_low_rank | 32 | 0.00001955 | 0.21455874 | 0.02020727 |
| `conv4_x.0.residual_function.2` | temporal_low_rank | 64 | 0.00000196 | 0.16111486 | 0.00581663 |
| `conv4_x.0.act` | temporal_low_rank | 64 | 0.00000188 | 0.21685952 | 0.00588441 |
| `conv4_x.1.residual_function.2` | temporal_low_rank | 64 | 0.00000093 | 0.25207958 | 0.00407085 |
| `conv4_x.1.act` | temporal_low_rank | 64 | 0.00000233 | 0.21167934 | 0.00556504 |
| `conv4_x.2.residual_function.2` | temporal_low_rank | 64 | 0.00000060 | 0.23401162 | 0.00329312 |
| `conv4_x.2.act` | temporal_low_rank | 64 | 0.00002905 | 0.32776144 | 0.02029308 |

## Per-layer Parity-Anchor FTBC reconstruction

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
| `conv1.2` | parity_anchor | 16 | 0.00054082 | 0.52631199 | 0.07683923 |
| `conv2_x.0.residual_function.2` | parity_anchor | 16 | 0.00012219 | 0.63250697 | 0.03815830 |
| `conv2_x.0.act` | parity_anchor | 16 | 0.00038598 | 0.48875287 | 0.06554987 |
| `conv2_x.1.residual_function.2` | parity_anchor | 16 | 0.00009337 | 0.46480963 | 0.03565676 |
| `conv2_x.1.act` | parity_anchor | 16 | 0.00023744 | 0.37844229 | 0.04912890 |
| `conv2_x.2.residual_function.2` | parity_anchor | 16 | 0.00001446 | 0.30976391 | 0.02096065 |
| `conv2_x.2.act` | parity_anchor | 16 | 0.00010147 | 0.31131867 | 0.03821164 |
| `conv3_x.0.residual_function.2` | parity_anchor | 32 | 0.00000916 | 0.23807521 | 0.01288225 |
| `conv3_x.0.act` | parity_anchor | 32 | 0.00001546 | 0.22730948 | 0.02085305 |
| `conv3_x.1.residual_function.2` | parity_anchor | 32 | 0.00000560 | 0.26252761 | 0.02185043 |
| `conv3_x.1.act` | parity_anchor | 32 | 0.00002037 | 0.24564958 | 0.02650032 |
| `conv3_x.2.residual_function.2` | parity_anchor | 32 | 0.00000087 | 0.15753032 | 0.00674477 |
| `conv3_x.2.act` | parity_anchor | 32 | 0.00001875 | 0.21009189 | 0.02129970 |
| `conv4_x.0.residual_function.2` | parity_anchor | 64 | 0.00000159 | 0.14543165 | 0.00533583 |
| `conv4_x.0.act` | parity_anchor | 64 | 0.00000201 | 0.22456157 | 0.00715112 |
| `conv4_x.1.residual_function.2` | parity_anchor | 64 | 0.00000098 | 0.25892243 | 0.00457356 |
| `conv4_x.1.act` | parity_anchor | 64 | 0.00000250 | 0.21946256 | 0.00640738 |
| `conv4_x.2.residual_function.2` | parity_anchor | 64 | 0.00000052 | 0.21923511 | 0.00448197 |
| `conv4_x.2.act` | parity_anchor | 64 | 0.00002386 | 0.29708007 | 0.02428957 |
