# State-conditioned Low-rank FTBC Ablation

- Dataset: cifar100
- Model: VGG16
- Time steps: [32]
- Calibration: batches=5, alpha=0.4, ridge=0.001, coefficient_clip=0.25
- All configurations reuse the same materialized calibration batches.

## Accuracy

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | 77.61% |

## Input-driven SOPs

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | 19,980,507,105,408 |

## Positive spike rate

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | 14.7320% |

## Negative spike rate

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | 0.0314% |

## Overall spike sparsity

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | 85.2366% |

## FTBC parameters

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | 37,248 |

## FTBC storage bytes

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | 148,992 |

## Calibration elapsed

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | 143.4s |

## Inference elapsed (statistics disabled)

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | 92.0s |

## Effective FTBC Mode

| Config | T=32 |
|---|---|
| H_STATE_LR_SOPS | state_low_rank |

## Per-layer Detail

### H_STATE_LR_SOPS, T=32

| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs |
|---|---:|---:|---:|---:|---:|
| layer1.2 | 25.2655% | 0.0000% | 74.7345% | 0 | 0 |
| layer1.6 | 13.8985% | 0.0236% | 86.0779% | 5,298,549,838 | 3,051,964,706,688 |
| layer2.2 | 13.7356% | 0.0108% | 86.2536% | 2,919,680,970 | 3,363,472,477,440 |
| layer2.6 | 8.5547% | 0.0391% | 91.4062% | 1,441,413,712 | 1,660,508,596,224 |
| layer3.2 | 10.8913% | 0.0264% | 89.0822% | 901,123,466 | 2,076,188,465,664 |
| layer3.6 | 8.3416% | 0.0444% | 91.6140% | 572,404,854 | 1,318,820,783,616 |
| layer3.10 | 6.4116% | 0.0428% | 93.5456% | 439,669,038 | 1,012,997,463,552 |
| layer4.2 | 6.4262% | 0.0432% | 93.5306% | 338,396,068 | 1,559,329,081,344 |
| layer4.6 | 3.4799% | 0.0808% | 96.4393% | 169,592,628 | 781,482,829,824 |
| layer4.10 | 4.5184% | 0.1808% | 95.3008% | 93,342,088 | 430,120,341,504 |
| layer5.2 | 27.6354% | 0.4718% | 71.8928% | 123,186,138 | 567,641,723,904 |
| layer5.6 | 44.1729% | 0.1544% | 55.6727% | 184,203,106 | 848,807,912,448 |
| layer5.10 | 30.4736% | 0.0346% | 69.4919% | 290,503,431 | 1,338,639,810,048 |
| classifier.2 | 21.4483% | 0.0016% | 78.5500% | 199,938,176 | 818,946,768,896 |
| classifier.5 | 14.9099% | 0.0119% | 85.0781% | 281,148,961 | 1,151,586,144,256 |
