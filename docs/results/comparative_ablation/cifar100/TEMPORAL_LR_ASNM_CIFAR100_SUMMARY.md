# QCFS + Temporal-LR FTBC + A-SNM CIFAR-100 Summary

- Status: complete
- The final method is `I_QCFS_TEMPORAL_LR_FTBC_ASNM_R0`.
- Rank-4 Temporal-LR falls back to Full-FTBC at T<=4.
- A-SNM is selected independently on the fixed 1,000-image augmented validation set.

## Final-method accuracy

| Architecture | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | SNM-on T |
|---|---:|---:|---:|---:|---:|---:|---|
| resnet20 | 14.66% | 22.92% | 40.82% | 60.38% | 68.15% | 69.19% | 2, 4, 8, 16, 32 |
| vgg16 | 61.72% | 67.68% | 74.01% | 77.12% | 77.65% | 77.57% | 2, 4, 8, 16, 32 |

## Temporal-LR storage reduction versus Full-FTBC

| Architecture | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| resnet20 | 0.00% | 0.00% | 0.00% | 49.42% | 74.42% | 86.92% |
| vgg16 | 0.00% | 0.00% | 0.00% | 49.97% | 74.97% | 87.47% |

## Source reports

- `TEMPORAL_LR_ASNM_CIFAR100_RESNET20.md`
- `TEMPORAL_LR_ASNM_CIFAR100_VGG16.md`
