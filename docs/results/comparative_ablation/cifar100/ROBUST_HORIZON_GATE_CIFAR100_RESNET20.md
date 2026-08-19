# Robust Horizon-Gated SNM Validation

- Architecture: resnet20
- Checkpoint: D:\master_degree_paper\workspace\QCFS\cifar100-checkpoints\resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- ANN accuracy: 68.68%
- Time steps: [4, 8, 16, 32]
- Fit set: 5 x 200, original calibration augmentation.
- Validation: 3 disjoint subsets x 5 x 200, mild crop/flip without AutoAugment or Cutout.
- Fit SHA256: `b352e0b83efebb196eb4b88fbb8566a986439b40dc48a3b5c2de3a1683565069`
- Validation SHA256: ['13662de6f9255de37fc3cf41e8e70a5f7b2f41105510aa7dd1b8c7cd217317ed', '5f48f8d819de762ef87838a94cc65089675edb7d56208f6fe8be7d09b2dd3d75', 'f41d1bb0dd09578b76215ee80b8fce2aaa7f8cd7f9549f670e5c273e7d3ef27b']
- Test set is evaluated only after the horizon mode is selected.
- Robust accuracy = validation mean - 0.5 x subset standard deviation.
- Within the accuracy tolerance, ANN-SNN logit MSE is minimized before event overhead.

## Validation-selected horizon mode

| Family | T | Selected | Off val. | Standard val. | Stage val. | Stage margins |
|---|---:|---|---:|---:|---:|---|
| temporal | 4 | stage_gated | 41.27+/-0.78 | 45.60+/-1.24 | 45.43+/-0.59 | early=1, middle=0, late=0, final=0 |
| temporal | 8 | stage_gated | 68.93+/-1.29 | 71.70+/-0.22 | 72.17+/-0.33 | early=0, middle=2, late=0, final=0 |
| temporal | 16 | stage_gated | 78.60+/-0.29 | 80.57+/-0.90 | 80.67+/-0.87 | early=0, middle=0, late=0, final=0.25 |
| temporal | 32 | stage_gated | 83.07+/-0.74 | 84.10+/-1.20 | 84.00+/-0.78 | early=0.5, middle=0, late=0, final=0 |
| hybrid | 4 | stage_gated | 41.27+/-0.78 | 45.60+/-1.24 | 45.43+/-0.59 | early=1, middle=0, late=0, final=0 |
| hybrid | 8 | stage_gated | 68.07+/-0.48 | 71.13+/-0.76 | 71.47+/-0.76 | early=0, middle=2, late=0, final=0 |
| hybrid | 16 | stage_gated | 79.00+/-0.22 | 80.50+/-0.99 | 80.40+/-0.86 | early=0, middle=0, late=0, final=0.25 |
| hybrid | 32 | standard | 83.13+/-0.87 | 83.87+/-0.90 | 83.87+/-0.90 | early=0, middle=0, late=0, final=0 |

## Test accuracy

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 39.62% | 59.74% | 67.27% | 69.51% |
| temporal/standard | 41.98% | 61.73% | 68.30% | 69.53% |
| temporal/stage_gated | 41.23% | 62.26% | 68.29% | 69.09% |
| hybrid/off | 39.62% | 60.10% | 67.47% | 69.38% |
| hybrid/standard | 41.98% | 61.90% | 68.16% | 69.02% |
| hybrid/stage_gated | 41.23% | 62.30% | 68.17% | 69.02% |

## Negative spike rate

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| temporal/standard | 0.2361% | 0.3954% | 0.4672% | 0.4611% |
| temporal/stage_gated | 0.1325% | 0.2983% | 0.4628% | 0.3518% |
| hybrid/off | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| hybrid/standard | 0.2361% | 0.3945% | 0.4654% | 0.4600% |
| hybrid/stage_gated | 0.1325% | 0.2974% | 0.4610% | 0.4600% |

## Input-driven SOPs

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 375,768,878,856 | 756,464,556,660 | 1,543,663,988,720 | 3,083,144,369,020 |
| temporal/standard | 387,210,741,216 | 792,939,954,624 | 1,623,033,787,520 | 3,230,974,093,092 |
| temporal/stage_gated | 388,261,425,756 | 786,182,971,976 | 1,622,846,074,020 | 3,226,117,703,080 |
| hybrid/off | 375,768,878,856 | 756,246,407,172 | 1,540,029,995,404 | 3,079,476,109,932 |
| hybrid/standard | 387,210,741,216 | 792,639,237,296 | 1,619,036,538,388 | 3,226,919,689,468 |
| hybrid/stage_gated | 388,261,425,756 | 785,935,618,720 | 1,618,851,051,688 | 3,226,919,689,468 |

## FTBC storage bytes

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 11,008 | 11,136 | 11,264 | 11,520 |
| temporal/standard | 11,008 | 11,136 | 11,264 | 11,520 |
| temporal/stage_gated | 11,008 | 11,136 | 11,264 | 11,520 |
| hybrid/off | 11,008 | 12,160 | 14,336 | 18,688 |
| hybrid/standard | 11,008 | 12,160 | 14,336 | 18,688 |
| hybrid/stage_gated | 11,008 | 12,160 | 14,336 | 18,688 |

## Inference elapsed

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 1.6s | 2.7s | 4.8s | 9.1s |
| temporal/standard | 1.7s | 2.9s | 5.3s | 10.3s |
| temporal/stage_gated | 1.7s | 3.0s | 5.3s | 10.4s |
| hybrid/off | 1.6s | 2.7s | 4.8s | 9.1s |
| hybrid/standard | 1.7s | 2.9s | 5.3s | 10.3s |
| hybrid/stage_gated | 1.7s | 3.0s | 5.3s | 10.3s |
