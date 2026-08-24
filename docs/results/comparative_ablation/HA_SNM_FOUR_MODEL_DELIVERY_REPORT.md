# HA-SNM Four-Model Delivery Report

Status: complete

HA-SNM uses a horizon-aware negative decision threshold while retaining standard SNM's transmitted-credit memory, R0 rule and -theta event amplitude. The globally frozen schedule is start=1.25, end=0.50 and reference horizon=8.

## Protocol audit

| Model | ANN | Checkpoint SHA256 | Fit hash | Validation hash | Test samples |
|---|---:|---|---|---|---:|
| CIFAR-10/ResNet20 L4 | 90.72% | `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3` | `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df` | `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c` | 10,000 |
| CIFAR-10/VGG16 L8 | 95.51% | `093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84` | `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df` | `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c` | 10,000 |
| CIFAR-100/ResNet20 L8 | 68.68% | `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2` | `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a` | `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3` | 10,000 |
| CIFAR-100/VGG16 L8 | 77.35% | `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339` | `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a` | `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3` | 10,000 |

## Accuracy: HA-SNM versus standard SNM

Each cell is `HA-SNM - standard SNM` in percentage points.

| Model / FTBC | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10/ResNet20 L4 / Full-FTBC | +0.00 | +1.18 | +1.08 | +0.32 | +0.04 | +0.00 | +0.437 |
| CIFAR-10/ResNet20 L4 / Temporal-LR FTBC | +0.00 | +1.18 | +1.08 | +0.21 | +0.02 | +0.09 | +0.430 |
| CIFAR-10/ResNet20 L4 / PA-FTBC | +0.00 | +1.18 | +1.08 | -0.02 | +0.01 | -0.08 | +0.362 |
| CIFAR-10/VGG16 L8 / Full-FTBC | +0.00 | +0.30 | +0.18 | +0.06 | +0.05 | +0.05 | +0.107 |
| CIFAR-10/VGG16 L8 / Temporal-LR FTBC | +0.00 | +0.30 | +0.18 | +0.06 | +0.02 | -0.02 | +0.090 |
| CIFAR-10/VGG16 L8 / PA-FTBC | +0.00 | +0.30 | +0.18 | +0.04 | +0.01 | -0.05 | +0.080 |
| CIFAR-100/ResNet20 L8 / Full-FTBC | +0.00 | +2.11 | +3.73 | +0.55 | -0.18 | +0.00 | +1.035 |
| CIFAR-100/ResNet20 L8 / Temporal-LR FTBC | +0.00 | +2.11 | +3.73 | +1.22 | -0.07 | +0.00 | +1.165 |
| CIFAR-100/ResNet20 L8 / PA-FTBC | +0.00 | +2.11 | +3.73 | +0.52 | -0.12 | -0.04 | +1.033 |
| CIFAR-100/VGG16 L8 / Full-FTBC | +0.00 | +0.84 | +0.68 | +0.18 | -0.04 | -0.01 | +0.275 |
| CIFAR-100/VGG16 L8 / Temporal-LR FTBC | +0.00 | +0.84 | +0.68 | -0.04 | -0.11 | +0.01 | +0.230 |
| CIFAR-100/VGG16 L8 / PA-FTBC | +0.00 | +0.84 | +0.68 | +0.66 | +0.00 | +0.08 | +0.377 |

## Six-step mean accuracy

| Model / FTBC | SNM-off | Standard SNM | HA-SNM | HA-standard | HA-off |
|---|---:|---:|---:|---:|---:|
| CIFAR-10/ResNet20 L4 / Full-FTBC | 83.645% | 83.938% | 84.375% | +0.437pp | +0.730pp |
| CIFAR-10/ResNet20 L4 / Temporal-LR FTBC | 83.652% | 83.957% | 84.387% | +0.430pp | +0.735pp |
| CIFAR-10/ResNet20 L4 / PA-FTBC | 83.647% | 83.983% | 84.345% | +0.362pp | +0.698pp |
| CIFAR-10/VGG16 L8 / Full-FTBC | 93.723% | 93.830% | 93.937% | +0.107pp | +0.213pp |
| CIFAR-10/VGG16 L8 / Temporal-LR FTBC | 93.723% | 93.803% | 93.893% | +0.090pp | +0.170pp |
| CIFAR-10/VGG16 L8 / PA-FTBC | 93.697% | 93.828% | 93.908% | +0.080pp | +0.212pp |
| CIFAR-100/ResNet20 L8 / Full-FTBC | 45.377% | 45.805% | 46.840% | +1.035pp | +1.463pp |
| CIFAR-100/ResNet20 L8 / Temporal-LR FTBC | 45.352% | 46.020% | 47.185% | +1.165pp | +1.833pp |
| CIFAR-100/ResNet20 L8 / PA-FTBC | 45.428% | 45.983% | 47.017% | +1.033pp | +1.588pp |
| CIFAR-100/VGG16 L8 / Full-FTBC | 72.348% | 72.637% | 72.912% | +0.275pp | +0.563pp |
| CIFAR-100/VGG16 L8 / Temporal-LR FTBC | 72.405% | 72.625% | 72.855% | +0.230pp | +0.450pp |
| CIFAR-100/VGG16 L8 / PA-FTBC | 72.338% | 72.570% | 72.947% | +0.377pp | +0.608pp |

## Four-model macro accuracy effects

| FTBC / comparison | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | Six-step mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-FTBC / HA - standard | +0.000 | +1.107 | +1.417 | +0.278 | -0.033 | +0.010 | +0.463 |
| Full-FTBC / HA - off | +0.000 | +1.287 | +2.380 | +0.940 | +0.025 | -0.178 | +0.742 |
| Full-FTBC / standard - off | +0.000 | +0.180 | +0.963 | +0.662 | +0.058 | -0.188 | +0.279 |
| Temporal-LR FTBC / HA - standard | +0.000 | +1.107 | +1.417 | +0.363 | -0.035 | +0.020 | +0.479 |
| Temporal-LR FTBC / HA - off | +0.000 | +1.287 | +2.380 | +1.055 | +0.135 | -0.075 | +0.797 |
| Temporal-LR FTBC / standard - off | +0.000 | +0.180 | +0.963 | +0.692 | +0.170 | -0.095 | +0.318 |
| PA-FTBC / HA - standard | +0.000 | +1.107 | +1.417 | +0.300 | -0.025 | -0.022 | +0.463 |
| PA-FTBC / HA - off | +0.000 | +1.287 | +2.380 | +1.205 | -0.038 | -0.175 | +0.777 |
| PA-FTBC / standard - off | +0.000 | +0.180 | +0.963 | +0.905 | -0.012 | -0.152 | +0.314 |

## Aggregate quality and cost

Ratios pool all four models and six time steps. Accuracy is an equal-weight mean difference.

| FTBC | Accuracy gain | Logit-MSE ratio | Standard neg. rate | HA neg. rate | Negative-rate ratio | SOP ratio | Timed-inference ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-FTBC | +0.463pp | 0.962434 | 0.125405% | 0.239613% | 1.910709 | 1.003528 | 1.015218 |
| Temporal-LR FTBC | +0.479pp | 0.962301 | 0.126586% | 0.241224% | 1.905616 | 1.003549 | 1.016124 |
| PA-FTBC | +0.463pp | 0.962461 | 0.124871% | 0.239165% | 1.915294 | 1.003518 | 1.012812 |

## Robustness and limitations

- Across 72 model/FTBC/time cells, HA-SNM has 44 wins, 16 ties and 12 losses versus standard SNM.
- T=2 and T=4 provide the main gain. At T=16/32 the macro difference from standard SNM is near zero, but individual cells can still be slightly lower.
- HA-SNM improves standard SNM; it does not guarantee that negative spikes outperform SNM-off at every long horizon. In particular, T=32 remains mildly negative versus off on macro average.
- The CIFAR-10/ResNet20 checkpoint retains its documented test-set model-selection bias from training. HA-SNM parameters and all screen rankings use calibration validation data only.
- Wall-clock ratios are descriptive PyTorch measurements on one GPU, not a neuromorphic energy claim.

## Validation-only schedule screen

| Rank | Start:end | Mean validation accuracy | Logit MSE | Negative-spike ratio | SOP ratio |
|---:|---|---:|---:|---:|---:|
| 1 | 1.25:0.5 | 71.8633% | 1.27212298 | 2.910622 | 1.014931 |
| 2 | 1.5:0.5 | 71.7550% | 1.27184913 | 2.857801 | 1.013732 |
| 3 | 2:0.5 | 71.6667% | 1.27716958 | 2.810805 | 1.012646 |
| 4 | 1.5:0.75 | 71.3800% | 1.33984828 | 1.596695 | 1.004030 |
| 5 | 1.25:0.75 | 71.2750% | 1.33530026 | 1.629752 | 1.004746 |
| 6 | 2:0.75 | 71.2400% | 1.34561218 | 1.556234 | 1.003052 |
| 7 | 1:1 | 71.0600% | 1.36543618 | 1.000000 | 1.000000 |
| 8 | 1.5:1 | 70.9650% | 1.37437780 | 0.934688 | 0.998622 |
| 9 | 2:1 | 70.9550% | 1.37790397 | 0.901105 | 0.997769 |

## Reproducibility audit

- Existing off/standard result regression: 1152 metric cells checked, 0 mismatches.
- Full fallback equality at T<=4: 72/72 pair checks exact.
- Four formal progress files have status `complete`, exact checkpoint hashes and 10,000 test samples.
- The four screening payloads record `test_images_evaluated=0`.

## Deliverables

- Method: `docs/methodology/HORIZON_AWARE_SNM.md`
- Experiment: `scripts/experiments/run_ha_snm_ablation.py`
- Validation screen: `scripts/experiments/screen_ha_snm.py`
- Formal per-model reports and `.progress.json` files are under the CIFAR-10/100 comparative-ablation directories.
