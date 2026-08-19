# Robust Horizon-Gated SNM Validation

- Architecture: vgg16
- Checkpoint: D:\master_degree_paper\workspace\QCFS\cifar100-checkpoints\cifar100-vgg16-l8-example.pth
- Checkpoint SHA256: `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339`
- ANN accuracy: 75.50%
- Time steps: [8]
- Fit set: 1 x 200, original calibration augmentation.
- Validation: 2 disjoint subsets x 1 x 200, mild crop/flip without AutoAugment or Cutout.
- Fit SHA256: `94da172b8586ec7a99ba3bf0149f5f41102253fd6a0507c0f497d70dc3fe37c4`
- Validation SHA256: ['244e01c8e5480dbd186006da8f1672628f9efe6bf27f621494ae5066f933f987', '5c2c9c2772de67951c860a22b6b56c4bb1763c558ed1d6cf30dd24fc57abf5f5']
- Test set is evaluated only after the horizon mode is selected.
- Robust accuracy = validation mean - 0.5 x subset standard deviation.

## Validation-selected horizon mode

| Family | T | Selected | Off val. | Standard val. | Stage val. | Stage margins |
|---|---:|---|---:|---:|---:|---|
| temporal | 8 | stage_gated | 99.00+/-0.00 | 100.00+/-0.00 | 100.00+/-0.00 | early=2, middle=0, late=1, final=2 |

## Test accuracy

| Family / mode | T=8 |
|---|---:|
| temporal/off | 77.50% |
| temporal/standard | 79.50% |
| temporal/stage_gated | 78.50% |

## Negative spike rate

| Family / mode | T=8 |
|---|---:|
| temporal/off | 0.0000% |
| temporal/standard | 0.0422% |
| temporal/stage_gated | 0.0201% |

## Input-driven SOPs

| Family / mode | T=8 |
|---|---:|
| temporal/off | 103,020,321,088 |
| temporal/standard | 103,997,622,976 |
| temporal/stage_gated | 103,759,076,928 |

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
