# State-conditioned Low-rank FTBC Ablation

- Dataset: cifar100
- Model: VGG16
- Time steps: [4]
- Calibration: batches=5, alpha=0.4, ridge=0.001, coefficient_clip=0.25
- All configurations reuse the same materialized calibration batches.

## Accuracy

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 75.05% |

## Input-driven SOPs

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 2,250,806,621,632 |

## Positive spike rate

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 13.7727% |

## Negative spike rate

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 0.0624% |

## Overall spike sparsity

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 86.1649% |

## FTBC parameters

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 37,248 |

## FTBC storage bytes

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 148,992 |

## Calibration elapsed

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 7.6s |

## Inference elapsed (statistics disabled)

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 11.8s |

## Effective FTBC Mode

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | state_low_rank |

## Per-layer Detail

### H_STATE_LR_SOPS, T=4

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs |
|---|---:|---:|---:|---:|---:|
| layer1.2 | 24.8543% | 0.0000% | 75.1457% | 0 | 0 |
| layer1.6 | 13.1971% | 0.0343% | 86.7686% | 651,541,215 | 375,287,739,840 |
| layer2.2 | 12.4186% | 0.0197% | 87.5618% | 346,851,935 | 399,573,429,120 |
| layer2.6 | 7.7706% | 0.0770% | 92.1524% | 163,030,613 | 187,811,266,176 |
| layer3.2 | 9.3178% | 0.0555% | 90.6266% | 102,860,159 | 236,989,806,336 |
| layer3.6 | 7.1991% | 0.0899% | 92.7111% | 61,429,420 | 141,533,383,680 |
| layer3.10 | 5.1077% | 0.1161% | 94.7762% | 47,768,733 | 110,059,160,832 |
| layer4.2 | 4.8115% | 0.0902% | 95.0982% | 34,234,629 | 157,753,170,432 |
| layer4.6 | 2.7604% | 0.1958% | 97.0438% | 16,062,134 | 74,014,313,472 |
| layer4.10 | 3.4309% | 0.4290% | 96.1400% | 9,686,807 | 44,636,806,656 |
| layer5.2 | 23.6861% | 0.8566% | 75.4573% | 12,648,288 | 58,283,311,104 |
| layer5.6 | 39.0797% | 0.2089% | 60.7114% | 20,105,388 | 92,645,627,904 |
| layer5.10 | 26.3304% | 0.0074% | 73.6622% | 32,185,241 | 148,309,590,528 |
| classifier.2 | 20.1961% | 0.0000% | 79.8039% | 21,575,960 | 88,375,132,160 |
| classifier.5 | 13.6202% | 0.0007% | 86.3790% | 33,089,327 | 135,533,883,392 |
