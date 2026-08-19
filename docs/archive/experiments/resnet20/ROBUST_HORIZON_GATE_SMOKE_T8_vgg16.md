# Robust Horizon-Gated SNM Validation

- Architecture: vgg16
- Checkpoint: D:\master_degree_paper\workspace\QCFS\cifar100-checkpoints\cifar100-vgg16-l8-example.pth
- Checkpoint SHA256: `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339`
- ANN accuracy: 75.50%
- Time steps: [8]
- Fit set: 1 x 200, original calibration augmentation.
- Validation: 2 disjoint subsets x 1 x 200, test preprocessing without AutoAugment.
- Fit SHA256: `94da172b8586ec7a99ba3bf0149f5f41102253fd6a0507c0f497d70dc3fe37c4`
- Validation SHA256: ['a3b600315f94e0b7ce5166593cc2bf78fdf3f25acdb8f8c7415b4b85dc5a250a', 'e89e5d9fbef7104247eafe905b614438cd77758038c981ba7a358e5006a3fd06']
- Test set is evaluated only after the horizon mode is selected.
- Robust accuracy = validation mean - 0.5 x subset standard deviation.

## Validation-selected horizon mode

| Family | T | Selected | Off val. | Standard val. | Stage val. | Stage margins |
|---|---:|---|---:|---:|---:|---|
| temporal | 8 | off | 100.00+/-0.00 | 100.00+/-0.00 | 100.00+/-0.00 | early=1, middle=2, late=0, final=2 |

## Test accuracy

| Family / mode | T=8 |
|---|---:|
| temporal/off | 77.50% |
| temporal/standard | 79.50% |
| temporal/stage_gated | 79.00% |

## Negative spike rate

| Family / mode | T=8 |
|---|---:|
| temporal/off | 0.0000% |
| temporal/standard | 0.0422% |
| temporal/stage_gated | 0.0218% |

## Input-driven SOPs

| Family / mode | T=8 |
|---|---:|
| temporal/off | 103,020,321,088 |
| temporal/standard | 103,997,622,976 |
| temporal/stage_gated | 103,613,609,408 |

## FTBC storage bytes

| Family / mode | T=8 |
|---|---:|
| temporal/off | 198,784 |
| temporal/standard | 198,784 |
| temporal/stage_gated | 198,784 |

## Inference elapsed

| Family / mode | T=8 |
|---|---:|
| temporal/off | 0.1s |
| temporal/standard | 0.1s |
| temporal/stage_gated | 0.1s |
