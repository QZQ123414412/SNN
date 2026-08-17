# CIFAR-100 / ResNet20 QCFS Paper-era Data-pipeline Diagnostic

- Status: completed but rejected; not a formal ablation result
- Architecture / dataset: ResNet20 / CIFAR-100
- QCFS activation: L=8, fixed-repository clamp/quantize order
- Seed: 42
- Initial learning rate: 0.02
- Epochs: 300, completed as an exact 75-epoch chunked run
- Batch size: 128
- Augmentation profile: paper-era public CIFAR-100 loader (no AutoAugment)
- Best test accuracy: 68.78%
- Required gate (current unified acceptance): 69.94%

Removing only AutoAugment improved the best accuracy from 68.21% to 68.78%,
but the checkpoint still failed the predeclared ANN gate. The paper-era QCFS
module also quantizes before clamping, whereas the fixed repository clamps
before quantization. These orders have identical forward values but different
surrogate gradients for values just outside the clipping interval. The next
controlled reproduction changes only this QCFS training-gradient profile.
