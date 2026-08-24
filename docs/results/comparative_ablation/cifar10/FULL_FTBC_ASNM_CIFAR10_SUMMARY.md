# CIFAR-10 QCFS + Full-FTBC + A-SNM Summary

- Status: complete
- Source: script-generated from the two formal progress JSON files.
- A-SNM uses validation accuracy only; test-oracle results are diagnostic.
- ResNet20 uses a test-best checkpoint, so its reported accuracy includes model-selection bias.
- VGG16 is a legacy checkpoint probably trained with L=4 and evaluated post-hoc with L=8.

## Sources

- resnet20: `docs/results/comparative_ablation/cifar10/FULL_FTBC_ASNM_CIFAR10_resnet20.progress.json`
- vgg16: `docs/results/comparative_ablation/cifar10/FULL_FTBC_ASNM_CIFAR10_vgg16.progress.json`

## resnet20

- Checkpoint: `resnet20_L[8]_bs128_fixed_repo_seed42_testbest.pth`
- SHA256: `eb8301ebda8ae91e52f2f273306befa5d349931c05b829a9440dafa05df70631`
- ANN accuracy: 92.79%
- QCFS A-SNM SNM-on T: 2, 4, 8, 16
- Full-FTBC A-SNM SNM-on T: 4, 32

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 40.96% | 53.49% | 73.11% | 88.50% | 92.32% | 92.92% | 73.55% |
| B_QCFS_STANDARD_SNM_R0 | 40.96% | 54.30% | 76.66% | 90.82% | 92.57% | 92.90% | 74.70% |
| C_QCFS_ASNM_R0 | 40.96% | 54.30% | 76.66% | 90.82% | 92.57% | 92.92% | 74.70% |
| D_QCFS_FULL_FTBC_R0 | 58.84% | 70.87% | 83.59% | 90.39% | 92.50% | 92.93% | 81.52% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 58.84% | 71.17% | 84.90% | 91.38% | 92.76% | 93.09% | 82.02% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 58.84% | 70.87% | 84.90% | 90.39% | 92.50% | 93.09% | 81.77% |

| Comparison | Mean accuracy change |
|---|---:|
| C-A | +1.16pp |
| C-B | +0.00pp |
| F-D | +0.25pp |
| F-E | -0.26pp |

## vgg16

- Checkpoint: `cifar10-vgg16-example.pth`
- SHA256: `093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84`
- ANN accuracy: 95.51%
- QCFS A-SNM SNM-on T: 2, 4, 16
- Full-FTBC A-SNM SNM-on T: 2, 4, 8

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_QCFS_R0 | 88.26% | 91.06% | 93.81% | 95.00% | 95.42% | 95.48% | 93.17% |
| B_QCFS_STANDARD_SNM_R0 | 88.26% | 91.11% | 94.11% | 95.28% | 95.56% | 95.58% | 93.32% |
| C_QCFS_ASNM_R0 | 88.26% | 91.11% | 94.11% | 95.00% | 95.56% | 95.48% | 93.25% |
| D_QCFS_FULL_FTBC_R0 | 89.87% | 91.98% | 94.27% | 95.24% | 95.51% | 95.47% | 93.72% |
| E_QCFS_FULL_FTBC_STANDARD_SNM_R0 | 89.87% | 92.05% | 94.54% | 95.48% | 95.51% | 95.53% | 93.83% |
| F_QCFS_FULL_FTBC_ASNM_R0 | 89.87% | 92.05% | 94.54% | 95.48% | 95.51% | 95.47% | 93.82% |

| Comparison | Mean accuracy change |
|---|---:|
| C-A | +0.08pp |
| C-B | -0.06pp |
| F-D | +0.10pp |
| F-E | -0.01pp |
