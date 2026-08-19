# Robust Horizon-Gated SNM Validation

- Architecture: vgg16
- Checkpoint: D:\master_degree_paper\workspace\QCFS\cifar100-checkpoints\cifar100-vgg16-l8-example.pth
- Checkpoint SHA256: `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339`
- ANN accuracy: 77.35%
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
| temporal | 4 | stage_gated | 97.80+/-0.57 | 98.47+/-0.54 | 98.67+/-0.46 | early=0.25, middle=0.5, late=0, final=0 |
| temporal | 8 | standard | 99.47+/-0.26 | 99.70+/-0.22 | 99.70+/-0.22 | early=0, middle=0, late=0, final=0 |
| temporal | 16 | standard | 99.77+/-0.12 | 99.80+/-0.08 | 99.80+/-0.08 | early=0, middle=0, late=0, final=0 |
| temporal | 32 | stage_gated | 99.83+/-0.09 | 99.83+/-0.09 | 99.83+/-0.09 | early=0, middle=0, late=0, final=2 |

## Test accuracy

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 73.34% | 76.27% | 77.34% | 77.64% |
| temporal/standard | 73.81% | 77.18% | 77.47% | 77.57% |
| temporal/stage_gated | 73.70% | 77.18% | 77.47% | 77.58% |

## Negative spike rate

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| temporal/standard | 0.0287% | 0.0400% | 0.0349% | 0.0268% |
| temporal/stage_gated | 0.0190% | 0.0400% | 0.0349% | 0.0267% |

## Input-driven SOPs

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 2,547,168,945,920 | 5,107,982,767,488 | 10,210,794,959,168 | 20,397,974,131,584 |
| temporal/standard | 2,563,191,510,016 | 5,154,980,072,704 | 10,287,869,168,064 | 20,513,138,496,768 |
| temporal/stage_gated | 2,558,565,673,856 | 5,154,980,072,704 | 10,287,869,168,064 | 20,513,073,116,416 |

## FTBC storage bytes

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 198,656 | 198,784 | 198,912 | 199,168 |
| temporal/standard | 198,656 | 198,784 | 198,912 | 199,168 |
| temporal/stage_gated | 198,656 | 198,784 | 198,912 | 199,168 |

## Inference elapsed

| Family / mode | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|
| temporal/off | 2.9s | 5.1s | 10.0s | 19.9s |
| temporal/standard | 3.1s | 5.6s | 10.4s | 114.5s |
| temporal/stage_gated | 3.1s | 5.8s | 10.5s | 39.7s |
