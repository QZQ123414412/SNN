# CIFAR-100 / ResNet20 Paper-era QCFS-gradient Diagnostic

- Status: completed but rejected; not a formal ablation result
- Architecture / dataset: ResNet20 / CIFAR-100
- QCFS activation: L=8, paper-era quantize-then-clamp order
- Seed: 42
- Initial learning rate: 0.02
- Epochs: 300, completed as an exact 75-epoch chunked run
- Batch size: 128
- Augmentation profile: paper-era public CIFAR-100 loader (no AutoAugment)
- Best test accuracy: 67.87%
- Required gate (current unified acceptance): 69.94%

The paper-era quantize-then-clamp and fixed-repository clamp-then-quantize
implementations have equal forward values but different surrogate gradients
near the clipping boundary. Switching only to the paper-era gradient reduced
the best accuracy from 68.78% to 67.87%. This checkpoint is therefore rejected.

Together with the batch-300 (67.17%), batch-128 fixed-repository (68.21%), and
paper-era data-pipeline (68.78%) runs, this exhausts the training differences
that can be established from the two public QCFS repositories and the paper.
The exact CIFAR-100/ResNet20 author checkpoint remains unavailable publicly.
