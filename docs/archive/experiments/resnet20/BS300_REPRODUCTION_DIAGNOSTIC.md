# CIFAR-100 / ResNet20 QCFS Batch-300 Reproduction Diagnostic

- Status: completed but rejected; not a formal ablation result
- Architecture / dataset: ResNet20 / CIFAR-100
- QCFS activation: L=8
- Seed: 42
- Initial learning rate: 0.02
- Epochs: 300, completed as an exact 150+150 resume
- Batch size: 300
- AutoAugment / Cutout: enabled / 16
- Best test accuracy: 67.17%
- Required gate (current unified acceptance): 69.94%

The midpoint state was epoch 149, cosine-scheduler step 150, learning rate
0.01. Epoch 150 after resume exactly reproduced the uninterrupted diagnostic
run (`loss=303.91367`, train accuracy 51.136%, test accuracy 58.58%), proving
optimizer, scheduler, shuffle, augmentation, and RNG continuity.

The source QCFS paper does not state a batch size. The paper-era public entry
used 128, while the fixed repository changed first to 200 and later to 300 for
the uploaded CIFAR-100/VGG16 example. Because the otherwise faithful batch-300
run failed the ResNet20 gate, the next controlled reproduction changes only
the training batch size to 128.
