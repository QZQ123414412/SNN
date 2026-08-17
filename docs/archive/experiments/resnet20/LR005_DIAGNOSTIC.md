# CIFAR-100 / ResNet20 QCFS LR=0.05 Diagnostic

- Status: stopped; not a formal result
- Date: 2026-08-12
- Architecture / dataset: ResNet20 / CIFAR-100
- QCFS activation: L=8
- Seed: 42
- Initial learning rate: 0.05
- Stopped after epoch 100/300
- Best observed test accuracy: 53.99%

This run followed the `0.05 for cifar100` comment in the fixed repository's
training script. It was stopped after checking Appendix A.1 of the original
QCFS paper, which specifies learning rate 0.02 for CIFAR-100 and L=8
specifically for CIFAR-100 / ResNet20. The 0.05 run is retained only as a
diagnostic and its checkpoint must not be used for the ablation.

The training log and interim checkpoint remain under `cifar100-checkpoints/`
and are intentionally excluded from version control.
