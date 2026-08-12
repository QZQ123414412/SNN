# State-conditioned Low-rank FTBC Ablation

- Dataset: cifar100
- Model: VGG16
- Time steps: [4]
- Calibration: batches=5, alpha=0.4, ridge=0.001, coefficient_clip=0.25
- All configurations reuse the same materialized calibration batches.

## Accuracy

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 75.27% |

## Input-driven SOPs

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 2,251,013,012,160 |

## Positive spike rate

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 13.7734% |

## Negative spike rate

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 0.0624% |

## Overall spike sparsity

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 86.1642% |

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
| H_STATE_LR_SOPS | 7.7s |

## Inference elapsed (statistics disabled)

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | 11.9s |

## Effective FTBC Mode

| Config | T=4 |
|---|---|
| H_STATE_LR_SOPS | state_low_rank |

## Per-layer Detail

### H_STATE_LR_SOPS, T=4

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs |
|---|---:|---:|---:|---:|---:|
| layer1.2 | 24.8543% | 0.0000% | 75.1457% | 0 | 0 |
| layer1.6 | 13.1971% | 0.0343% | 86.7686% | 651,541,211 | 375,287,737,536 |
| layer2.2 | 12.4186% | 0.0197% | 87.5618% | 346,851,940 | 399,573,434,880 |
| layer2.6 | 7.7706% | 0.0770% | 92.1524% | 163,030,610 | 187,811,262,720 |
| layer3.2 | 9.3178% | 0.0555% | 90.6266% | 102,860,146 | 236,989,776,384 |
| layer3.6 | 7.1990% | 0.0899% | 92.7111% | 61,429,299 | 141,533,104,896 |
| layer3.10 | 5.1078% | 0.1160% | 94.7762% | 47,768,254 | 110,058,057,216 |
| layer4.2 | 4.8119% | 0.0903% | 95.0978% | 34,234,857 | 157,754,221,056 |
| layer4.6 | 2.7608% | 0.1957% | 97.0435% | 16,063,626 | 74,021,188,608 |
| layer4.10 | 3.4336% | 0.4290% | 96.1374% | 9,687,939 | 44,642,022,912 |
| layer5.2 | 23.6812% | 0.8560% | 75.4628% | 12,657,063 | 58,323,746,304 |
| layer5.6 | 39.0801% | 0.2084% | 60.7115% | 20,100,890 | 92,624,901,120 |
| layer5.10 | 26.3407% | 0.0071% | 73.6522% | 32,185,169 | 148,309,258,752 |
| classifier.2 | 20.2173% | 0.0000% | 79.7827% | 21,584,119 | 88,408,551,424 |
| classifier.5 | 13.6366% | 0.0007% | 86.3627% | 33,123,962 | 135,675,748,352 |
