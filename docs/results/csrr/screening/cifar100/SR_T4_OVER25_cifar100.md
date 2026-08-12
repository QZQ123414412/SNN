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
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 74.95% |

## Input-driven SOPs

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 2,196,668,349,504 |

## Time-scale operations

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 0 |

## Positive spike rate

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 13.5452% |

## Negative spike rate

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 0.0522% |

## Overall spike sparsity

| Config | T=4 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 86.4026% |

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
| layer1.2 | 24.6752% | 0.0000% | 75.3248% | 0 | 0 | 0 |
| layer1.6 | 12.9336% | 0.0285% | 87.0379% | 646,845,701 | 372,583,123,776 | 0 |
| layer2.2 | 12.2100% | 0.0158% | 87.7743% | 339,793,546 | 391,442,164,992 | 0 |
| layer2.6 | 7.5587% | 0.0646% | 92.3768% | 160,245,182 | 184,602,449,664 | 0 |
| layer3.2 | 9.0908% | 0.0476% | 90.8616% | 99,919,442 | 230,214,394,368 | 0 |
| layer3.6 | 7.0035% | 0.0868% | 92.9097% | 59,889,634 | 137,985,716,736 | 0 |
| layer3.10 | 4.9470% | 0.1024% | 94.9507% | 46,467,305 | 107,060,670,720 | 0 |
| layer4.2 | 4.6662% | 0.0791% | 95.2547% | 33,091,332 | 152,484,857,856 | 0 |
| layer4.6 | 2.6235% | 0.1686% | 97.2079% | 15,549,447 | 71,651,851,776 | 0 |
| layer4.10 | 3.0814% | 0.3480% | 96.5706% | 9,149,218 | 42,159,596,544 | 0 |
| layer5.2 | 22.5262% | 0.6395% | 76.8343% | 11,237,521 | 51,782,496,768 | 0 |
| layer5.6 | 38.6747% | 0.1362% | 61.1891% | 18,977,312 | 87,447,453,696 | 0 |
| layer5.10 | 26.2829% | 0.0052% | 73.7119% | 31,793,902 | 146,506,300,416 | 0 |
| classifier.2 | 19.7499% | 0.0000% | 80.2501% | 21,535,218 | 88,208,252,928 | 0 |
| classifier.5 | 13.1039% | 0.0007% | 86.8955% | 32,358,159 | 132,539,019,264 | 0 |

