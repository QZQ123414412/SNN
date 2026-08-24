# CIFAR-10 versus CIFAR-100 Full-FTBC + A-SNM Comparison

- Status: complete
- Source: script-generated from four formal progress JSON files.
- Raw accuracy levels across datasets are descriptive, not a controlled measure of dataset difficulty.
- ResNet20 is the cleaner protocol comparison; CIFAR-10/VGG16 has uncertain training L provenance.

## resnet20

| Config | CIFAR-10 mean | CIFAR-100 mean | C10-C100 |
|---|---:|---:|---:|
| A_QCFS_R0 | 73.55% | 36.87% | +36.68pp |
| B_QCFS_STANDARD_SNM_R0 | 74.70% | 39.70% | +35.01pp |
| C_QCFS_ASNM_R0 | 74.70% | 39.63% | +35.07pp |
| D_QCFS_FULL_FTBC_R0 | 81.52% | 45.38% | +36.14pp |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 82.02% | 45.81% | +36.22pp |
| F_QCFS_FULL_FTBC_ASNM_R0 | 81.77% | 45.81% | +35.96pp |

- CIFAR-10 QCFS gate: 2, 4, 8, 16
- CIFAR-100 QCFS gate: 4, 8, 16, 32
- CIFAR-10 Full-FTBC gate: 4, 32
- CIFAR-100 Full-FTBC gate: 2, 4, 8, 16, 32

## vgg16

| Config | CIFAR-10 mean | CIFAR-100 mean | C10-C100 |
|---|---:|---:|---:|
| A_QCFS_R0 | 93.17% | 70.81% | +22.36pp |
| B_QCFS_STANDARD_SNM_R0 | 93.32% | 71.28% | +22.03pp |
| C_QCFS_ASNM_R0 | 93.25% | 71.31% | +21.94pp |
| D_QCFS_FULL_FTBC_R0 | 93.72% | 72.35% | +21.38pp |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 93.83% | 72.64% | +21.19pp |
| F_QCFS_FULL_FTBC_ASNM_R0 | 93.82% | 72.64% | +21.18pp |

- CIFAR-10 QCFS gate: 2, 4, 16
- CIFAR-100 QCFS gate: 2, 4, 8, 16
- CIFAR-10 Full-FTBC gate: 2, 4, 8
- CIFAR-100 Full-FTBC gate: 2, 4, 8, 16, 32
