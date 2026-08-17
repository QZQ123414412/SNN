# CIFAR-100 / ResNet20 QCFS Batch-128 AutoAugment Diagnostic

- Status: completed but rejected; not a formal ablation result
- Architecture / dataset: ResNet20 / CIFAR-100
- QCFS activation: L=8
- Seed: 42
- Initial learning rate: 0.02
- Epochs: 300, completed with exact resumable state
- Batch size: 128
- Augmentation profile: fixed repository (AutoAugment and Cutout-16)
- Best test accuracy: 68.21%
- Required gate (current unified acceptance): 69.94%

This run changed only the batch size from the rejected batch-300 reproduction
and improved the best accuracy from 67.17% to 68.21%, but still failed the
predeclared ANN gate. The paper-era public CIFAR-100 loader differs from the
fixed repository: it uses crop, horizontal flip, and Cutout-16 without
AutoAugment. The next controlled reproduction keeps batch size 128, L=8,
learning rate 0.02, weight decay 5e-4, cosine schedule, seed 42, and 300 epochs,
and changes only the augmentation profile to the paper-era public loader.
