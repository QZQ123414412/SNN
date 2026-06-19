# Signed Successive-Refinement Ablation

- Dataset: cifar100
- Model: VGG16
- Checkpoint: cifar100-vgg16-l8-example
- Time steps: [2]
- Global geometric ratios: [1.0]
- Positive hysteresis margins: [0.52]
- Negative hysteresis margins: [1.45]
- Calibration: batches=5, alpha=0.4, ridge=0.001, coefficient_clip=0.25
- SOPs are input-driven event operations. ScaleOps are reported separately and are not added to SOPs.

## Accuracy

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 68.04% |

## Input-driven SOPs

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 1,295,680,467,968 |

## Time-scale operations

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 0 |

## Positive spike rate

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 15.2068% |

## Negative spike rate

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 0.0102% |

## Overall spike sparsity

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 84.7830% |

## FTBC parameters

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 24,832 |

## FTBC storage bytes

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 99,328 |

## Calibration elapsed

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 3.7s |

## Inference elapsed (statistics disabled)

| Config | T=2 |
|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | 2.5s |

## Configuration Detail

| Config | Coding | Schedule | Ratio | PosMargin | NegMargin | R0 mode | Effective FTBC by T |
|---|---|---|---:|---:|---:|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.52_N1.45 | successive_refinement | geometric | 1 | 0.52 | 1.45 | credit_only | T=2:full |

## Per-layer Detail

### M_SR_GEOM_STATE_LR_R1_P0.52_N1.45, T=2

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| layer1.2 | 25.4370% | 0.0000% | 74.5630% | 0 | 0 | 0 |
| layer1.6 | 14.4726% | 0.0209% | 85.5065% | 333,407,408 | 192,042,667,008 | 0 |
| layer2.2 | 14.2550% | 0.0072% | 85.7378% | 189,969,570 | 218,844,944,640 | 0 |
| layer2.6 | 9.3671% | 0.0111% | 90.6218% | 93,468,752 | 107,676,002,304 | 0 |
| layer3.2 | 11.7798% | 0.0043% | 88.2159% | 61,461,259 | 141,606,740,736 | 0 |
| layer3.6 | 8.6886% | 0.0056% | 91.3058% | 38,614,022 | 88,966,706,688 | 0 |
| layer3.10 | 6.6894% | 0.0031% | 93.3076% | 28,489,126 | 65,638,946,304 | 0 |
| layer4.2 | 6.6215% | 0.0035% | 93.3750% | 21,929,733 | 101,052,209,664 | 0 |
| layer4.6 | 3.7973% | 0.0163% | 96.1864% | 10,854,383 | 50,016,996,864 | 0 |
| layer4.10 | 4.7428% | 0.0484% | 95.2088% | 6,248,201 | 28,791,710,208 | 0 |
| layer5.2 | 27.4252% | 0.0773% | 72.4975% | 7,849,832 | 36,172,025,856 | 0 |
| layer5.6 | 44.0254% | 0.0006% | 55.9740% | 11,265,030 | 51,909,258,240 | 0 |
| layer5.10 | 30.4991% | 0.0000% | 69.5009% | 18,033,057 | 83,096,326,656 | 0 |
| classifier.2 | 23.4535% | 0.0000% | 76.5465% | 12,492,421 | 51,168,956,416 | 0 |
| classifier.5 | 15.9600% | 0.0000% | 84.0400% | 19,213,129 | 78,696,976,384 | 0 |

