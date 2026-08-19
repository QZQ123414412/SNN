# Robust Horizon-Gated SNM Validation

- Architecture: resnet20
- Checkpoint: D:\master_degree_paper\workspace\QCFS\cifar100-checkpoints\resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth
- Checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- ANN accuracy: 68.00%
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
| temporal | 8 | stage_gated | 68.75+/-1.75 | 70.00+/-2.00 | 71.25+/-1.75 | early=0, middle=2, late=0, final=1 |
| hybrid | 8 | stage_gated | 66.00+/-2.50 | 72.50+/-1.50 | 72.25+/-0.75 | early=0, middle=0, late=0, final=1 |

## Test accuracy

| Family / mode | T=8 |
|---|---:|
| temporal/off | 59.50% |
| temporal/standard | 62.50% |
| temporal/stage_gated | 62.50% |
| hybrid/off | 58.50% |
| hybrid/standard | 63.00% |
| hybrid/stage_gated | 62.50% |

## Negative spike rate

| Family / mode | T=8 |
|---|---:|
| temporal/off | 0.0000% |
| temporal/standard | 0.4278% |
| temporal/stage_gated | 0.3045% |
| hybrid/off | 0.0000% |
| hybrid/standard | 0.4274% |
| hybrid/stage_gated | 0.4157% |

## Input-driven SOPs

| Family / mode | T=8 |
|---|---:|
| temporal/off | 15,563,681,176 |
| temporal/standard | 16,319,687,092 |
| temporal/stage_gated | 16,173,841,840 |
| hybrid/off | 15,564,737,832 |
| hybrid/standard | 16,320,998,016 |
| hybrid/stage_gated | 16,317,344,716 |

## FTBC storage bytes

| Family / mode | T=8 |
|---|---:|
| temporal/off | 11,136 |
| temporal/standard | 11,136 |
| temporal/stage_gated | 11,136 |
| hybrid/off | 12,160 |
| hybrid/standard | 12,160 |
| hybrid/stage_gated | 12,160 |

## Inference elapsed

| Family / mode | T=8 |
|---|---:|
| temporal/off | 0.0s |
| temporal/standard | 0.0s |
| temporal/stage_gated | 0.0s |
| hybrid/off | 0.0s |
| hybrid/standard | 0.0s |
| hybrid/stage_gated | 0.0s |
