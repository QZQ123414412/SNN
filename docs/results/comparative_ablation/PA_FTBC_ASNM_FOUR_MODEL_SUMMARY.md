# Parity-Anchor FTBC Four-Model Ablation Summary

Status: complete

PA-FTBC replaces Temporal-LR's learned shared SVD basis with four fixed structured coefficients: t=0 anchor, t=1 anchor, tail mean and tail parity. It uses no SVD, no cross-layer concatenation, no threshold normalization and no stored time basis.

- Formal reports: 4
- Equivalence checks: 216/216 exact
- Existing-result regression cells: 1386, mismatches: 0
- Test set is used only after all four family-specific A-SNM gates are frozen.

## Protocol audit

| Model | ANN | Checkpoint SHA256 | Fit hash | Validation hash | PA SNM-on T | Reversals |
|---|---:|---|---|---|---|---:|
| CIFAR-10/ResNet20 L4 | 90.72% | `851e5475413440193a9e26aa6b6400cd23dcd8ef4794c60bd0e08728d2f409c3` | `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df` | `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c` | 2,4,8,16,32 | 0 |
| CIFAR-10/VGG16 L8 | 95.51% | `093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84` | `053b54e7f6ab19341f804fb5a7bc4ce3f28203aa8400a2bf5370ffdf368898df` | `237b49807bad9667caab5d8b8346eaa98c680283dbd9072856566c9517fc672c` | 2,4,8,16 | 1 |
| CIFAR-100/ResNet20 L8 | 68.68% | `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2` | `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a` | `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3` | 2,4,8,16,32 | 2 |
| CIFAR-100/VGG16 L8 | 77.35% | `8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339` | `9c12d1f2bc2972cbb843d46a275d776ef09b820815e721ab3b0a117e0d0f263a` | `d0fe62738ca5d259b0a912cdc71376fb284d0d74f031679f434ef359a6cd70c3` | 2,4,8,16 | 1 |

## Six-time-step mean accuracy

| Model | QCFS+A-SNM | Full+A-SNM | Temporal+A-SNM | PA+A-SNM | PA-Temporal | PA-Full |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10/ResNet20 L4 | 82.64% | 83.92% | 83.88% | 83.98% | +0.10pp | +0.07pp |
| CIFAR-10/VGG16 L8 | 93.25% | 93.82% | 93.79% | 93.80% | +0.02pp | -0.02pp |
| CIFAR-100/ResNet20 L8 | 39.63% | 45.81% | 46.02% | 45.98% | -0.04pp | +0.18pp |
| CIFAR-100/VGG16 L8 | 71.31% | 72.64% | 72.63% | 72.58% | -0.04pp | -0.05pp |
| Four-model macro mean |  |  |  |  | +0.010pp |  |

## PA versus Temporal by SNM mode

| Model | Off mean delta | Standard-SNM mean delta | A-SNM mean delta |
|---|---:|---:|---:|
| CIFAR-10/ResNet20 L4 | -0.00pp | +0.03pp | +0.10pp |
| CIFAR-10/VGG16 L8 | -0.03pp | +0.02pp | +0.02pp |
| CIFAR-100/ResNet20 L8 | +0.08pp | -0.04pp | -0.04pp |
| CIFAR-100/VGG16 L8 | -0.07pp | -0.06pp | -0.04pp |

## A-SNM aggregate metric comparison

Equal-weight means are taken over all six time steps. SOP is shown as PA/Temporal; rate and sparsity columns are PA minus Temporal.

| Model | Accuracy delta | Logit-MSE delta | Positive-rate delta | Negative-rate delta | Sparsity delta | SOP ratio |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10/ResNet20 L4 | +0.103pp | -0.04391317 | +0.073599pp | +0.086894pp | -0.160493pp | 1.012791 |
| CIFAR-10/VGG16 L8 | +0.017pp | -0.00206407 | -0.021611pp | +0.006661pp | +0.014950pp | 0.999191 |
| CIFAR-100/ResNet20 L8 | -0.037pp | +0.00114542 | -0.026662pp | -0.002742pp | +0.029404pp | 0.997180 |
| CIFAR-100/VGG16 L8 | -0.043pp | +0.00294925 | -0.029724pp | -0.004698pp | +0.034422pp | 0.993744 |

## Storage and bias-synthesis cost

| Model | T | Full params | Temporal params | PA params | PA saving vs Full | PA params vs Temporal | PA MACs vs Temporal |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10/ResNet20 L4 | 8 | 5,504 | 2,784 | 2,752 | 50.00% | 98.85% | 43.75% |
| CIFAR-10/ResNet20 L4 | 16 | 11,008 | 2,816 | 2,752 | 75.00% | 97.73% | 46.88% |
| CIFAR-10/ResNet20 L4 | 32 | 22,016 | 2,880 | 2,752 | 87.50% | 95.56% | 48.44% |
| CIFAR-10/VGG16 L8 | 8 | 99,328 | 49,696 | 49,664 | 50.00% | 99.94% | 43.75% |
| CIFAR-10/VGG16 L8 | 16 | 198,656 | 49,728 | 49,664 | 75.00% | 99.87% | 46.88% |
| CIFAR-10/VGG16 L8 | 32 | 397,312 | 49,792 | 49,664 | 87.50% | 99.74% | 48.44% |
| CIFAR-100/ResNet20 L8 | 8 | 5,504 | 2,784 | 2,752 | 50.00% | 98.85% | 43.75% |
| CIFAR-100/ResNet20 L8 | 16 | 11,008 | 2,816 | 2,752 | 75.00% | 97.73% | 46.88% |
| CIFAR-100/ResNet20 L8 | 32 | 22,016 | 2,880 | 2,752 | 87.50% | 95.56% | 48.44% |
| CIFAR-100/VGG16 L8 | 8 | 99,328 | 49,696 | 49,664 | 50.00% | 99.94% | 43.75% |
| CIFAR-100/VGG16 L8 | 16 | 198,656 | 49,728 | 49,664 | 75.00% | 99.87% | 46.88% |
| CIFAR-100/VGG16 L8 | 32 | 397,312 | 49,792 | 49,664 | 87.50% | 99.74% | 48.44% |

## Conclusion

Across all four formal models, the maximum absolute six-step mean-accuracy difference between PA-FTBC+A-SNM and Temporal-LR+A-SNM is 0.103pp, while the four-model macro-mean change is +0.010pp. PA removes SVD and the learned/stored temporal basis, uses slightly fewer parameters than Temporal-LR, and reduces bias-synthesis MAC equivalents by 51.56%-56.25% at T=8/16/32. The result supports PA-FTBC as a simpler accuracy-equivalent replacement under the tested protocols.

The CIFAR-10/ResNet20 L4 checkpoint retains the documented test-set model-selection bias from training; no test result was used to select PA structure or A-SNM gates.
