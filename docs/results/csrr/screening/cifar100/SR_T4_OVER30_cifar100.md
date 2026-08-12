# Signed Successive-Refinement Ablation

- Dataset: cifar100
- Model: VGG16
- Checkpoint: cifar100-vgg16-l8-example
- Time steps: [4]
- Global geometric ratios: [1.0]
- Positive hysteresis margins: [0.52]
- Negative hysteresis margins: [1.45]
- Calibration: batches=5, alpha=0.4, ridge=0.001, coefficient_clip=0.25
- SOPs are input-driven event operations. ScaleOps are reported separately and are not added to SOPs.

## Accuracy

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 74.69% |

## Input-driven SOPs

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 2,146,819,927,104 |

## Time-scale operations

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 0 |

## Positive spike rate

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 13.3020% |

## Negative spike rate

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 0.0491% |

## Overall spike sparsity

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 86.6489% |

## FTBC parameters

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 37,248 |

## FTBC storage bytes

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 148,992 |

## Calibration elapsed

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 7.9s |

## Inference elapsed (statistics disabled)

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 4.6s |

## Configuration Detail

| Config | Coding | Schedule | Ratio | PosMargin | NegMargin | R0 mode | Effective FTBC by T |
|---|---|---|---:|---:|---:|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | successive_refinement | geometric | 1 | 0.52 | 1.45 | credit_only | T=4:state_low_rank |

## Per-layer Detail

### M_SR_GEOM_STATE_LR_R1_P0.52_N1.45, T=4

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| layer1.2 | 24.4225% | 0.0000% | 75.5775% | 0 | 0 | 0 |
| layer1.6 | 12.6644% | 0.0266% | 87.3091% | 640,220,965 | 368,767,275,840 | 0 |
| layer2.2 | 11.9378% | 0.0142% | 88.0480% | 332,685,486 | 383,253,679,872 | 0 |
| layer2.6 | 7.3618% | 0.0584% | 92.5799% | 156,656,740 | 180,468,564,480 | 0 |
| layer3.2 | 8.8507% | 0.0451% | 91.1042% | 97,257,093 | 224,080,342,272 | 0 |
| layer3.6 | 6.7949% | 0.0857% | 93.1194% | 58,299,244 | 134,321,458,176 | 0 |
| layer3.10 | 4.7712% | 0.0972% | 95.1315% | 45,092,707 | 103,893,596,928 | 0 |
| layer4.2 | 4.4712% | 0.0766% | 95.4523% | 31,905,883 | 147,022,308,864 | 0 |
| layer4.6 | 2.5143% | 0.1604% | 97.3253% | 14,901,950 | 68,668,185,600 | 0 |
| layer4.10 | 2.9245% | 0.3235% | 96.7520% | 8,764,442 | 40,386,548,736 | 0 |
| layer5.2 | 21.9241% | 0.6031% | 77.4728% | 10,642,929 | 49,042,616,832 | 0 |
| layer5.6 | 38.0716% | 0.1420% | 61.7863% | 18,454,278 | 85,037,313,024 | 0 |
| layer5.10 | 26.0089% | 0.0050% | 73.9861% | 31,304,624 | 144,251,707,392 | 0 |
| classifier.2 | 19.4219% | 0.0000% | 80.5781% | 21,310,613 | 87,288,270,848 | 0 |
| classifier.5 | 12.7918% | 0.0007% | 87.2075% | 31,820,815 | 130,338,058,240 | 0 |

