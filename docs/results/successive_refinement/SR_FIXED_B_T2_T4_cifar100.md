# Signed Successive-Refinement Ablation

- Dataset: cifar100
- Model: VGG16
- Checkpoint: cifar100-vgg16-l8-example
- Time steps: [2, 4]
- Global geometric ratios: [1.0]
- Positive hysteresis margins: [0.55]
- Negative hysteresis margins: [1.3]
- Calibration: batches=5, alpha=0.4, ridge=0.001, coefficient_clip=0.25
- SOPs are input-driven event operations. ScaleOps are reported separately and are not added to SOPs.

## Accuracy

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 68.31% | 75.26% |

## Input-driven SOPs

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 1,280,789,567,360 | 2,239,573,181,120 |

## Time-scale operations

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 0 | 0 |

## Positive spike rate

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 15.0755% | 13.6699% |

## Negative spike rate

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 0.0164% | 0.0578% |

## Overall spike sparsity

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 84.9081% | 86.2723% |

## FTBC parameters

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 24,832 | 37,248 |

## FTBC storage bytes

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 99,328 | 148,992 |

## Calibration elapsed

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 3.7s | 7.7s |

## Inference elapsed (statistics disabled)

| Config | T=2 | T=4 |
|---|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | 2.5s | 4.6s |

## Configuration Detail

| Config | Coding | Schedule | Ratio | PosMargin | NegMargin | R0 mode | Effective FTBC by T |
|---|---|---|---:|---:|---:|---|---|
| M_SR_GEOM_STATE_LR_R1_P0.55_N1.3 | successive_refinement | geometric | 1 | 0.55 | 1.3 | credit_only | T=2:full, T=4:state_low_rank |

## Per-layer Detail

### M_SR_GEOM_STATE_LR_R1_P0.55_N1.3, T=2

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| layer1.2 | 25.4104% | 0.0000% | 74.5896% | 0 | 0 | 0 |
| layer1.6 | 14.2196% | 0.0313% | 85.7491% | 333,059,820 | 191,842,456,320 | 0 |
| layer2.2 | 14.2047% | 0.0114% | 85.7839% | 186,789,556 | 215,181,568,512 | 0 |
| layer2.6 | 9.1292% | 0.0219% | 90.8489% | 93,166,939 | 107,328,313,728 | 0 |
| layer3.2 | 11.5900% | 0.0089% | 88.4011% | 59,972,886 | 138,177,529,344 | 0 |
| layer3.6 | 8.4832% | 0.0127% | 91.5040% | 38,007,312 | 87,568,846,848 | 0 |
| layer3.10 | 6.4457% | 0.0064% | 93.5479% | 27,839,605 | 64,142,449,920 | 0 |
| layer4.2 | 6.3680% | 0.0065% | 93.6255% | 21,142,141 | 97,422,985,728 | 0 |
| layer4.6 | 3.6043% | 0.0257% | 96.3700% | 10,444,035 | 48,126,113,280 | 0 |
| layer4.10 | 4.4663% | 0.0681% | 95.4656% | 5,947,457 | 27,405,881,856 | 0 |
| layer5.2 | 27.4016% | 0.1208% | 72.4775% | 7,429,142 | 34,233,486,336 | 0 |
| layer5.6 | 44.4652% | 0.0024% | 55.5323% | 11,273,196 | 51,946,887,168 | 0 |
| layer5.10 | 31.3389% | 0.0000% | 68.6611% | 18,213,964 | 83,929,946,112 | 0 |
| classifier.2 | 24.1116% | 0.0000% | 75.8884% | 12,836,417 | 52,577,964,032 | 0 |
| classifier.5 | 16.3249% | 0.0000% | 83.6751% | 19,752,231 | 80,905,138,176 | 0 |

### M_SR_GEOM_STATE_LR_R1_P0.55_N1.3, T=4

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs | ScaleOps |
|---|---:|---:|---:|---:|---:|---:|
| layer1.2 | 24.6635% | 0.0000% | 75.3365% | 0 | 0 | 0 |
| layer1.6 | 13.0155% | 0.0371% | 86.9474% | 646,537,647 | 372,405,684,672 | 0 |
| layer2.2 | 12.4041% | 0.0206% | 87.5753% | 342,165,568 | 394,174,734,336 | 0 |
| layer2.6 | 7.6622% | 0.0778% | 92.2600% | 162,852,470 | 187,606,045,440 | 0 |
| layer3.2 | 9.3879% | 0.0583% | 90.5538% | 101,449,382 | 233,739,376,128 | 0 |
| layer3.6 | 7.1388% | 0.1057% | 92.7555% | 61,906,384 | 142,632,308,736 | 0 |
| layer3.10 | 5.1285% | 0.1059% | 94.7656% | 47,477,634 | 109,388,468,736 | 0 |
| layer4.2 | 4.9161% | 0.0833% | 95.0006% | 34,304,225 | 158,073,868,800 | 0 |
| layer4.6 | 2.7480% | 0.1602% | 97.0917% | 16,382,186 | 75,489,113,088 | 0 |
| layer4.10 | 3.2806% | 0.3032% | 96.4162% | 9,529,799 | 43,913,313,792 | 0 |
| layer5.2 | 23.9189% | 0.7045% | 75.3766% | 11,743,392 | 54,113,550,336 | 0 |
| layer5.6 | 40.0615% | 0.2083% | 59.7302% | 20,171,495 | 92,950,248,960 | 0 |
| layer5.10 | 27.2272% | 0.0141% | 72.7587% | 32,988,990 | 152,013,265,920 | 0 |
| classifier.2 | 19.6198% | 0.0000% | 80.3802% | 22,316,076 | 91,406,647,296 | 0 |
| classifier.5 | 12.9317% | 0.0020% | 87.0663% | 32,145,155 | 131,666,554,880 | 0 |

