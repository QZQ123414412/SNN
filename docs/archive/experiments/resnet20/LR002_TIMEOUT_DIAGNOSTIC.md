# CIFAR-100 / ResNet20 QCFS LR=0.02 Timeout Diagnostic

- Status: stopped by the command execution limit; not a formal result
- Date: 2026-08-12
- Architecture / dataset: ResNet20 / CIFAR-100
- QCFS activation: L=8
- Seed: 42
- Initial learning rate: 0.02
- Last completed epoch: 194/300
- Best observed test accuracy: 62.67%

The original author training entry stored only the best model weights, not the
optimizer, cosine scheduler, epoch, and RNG states. The 60-minute command limit
therefore made an exact continuation impossible. The partial run is retained
as a diagnostic and must not be treated as a 300-epoch reproduction.

The training entry now atomically stores a full state after every epoch,
including Python, NumPy, Torch CPU, and Torch CUDA RNG states. A real two-part
smoke run verified continuous epoch numbering, cosine scheduler state, best
accuracy, and safe state deserialization. The formal retraining starts from
epoch 0 under a distinct `official_retrain_seed42_lr002_resumable` suffix and
runs as two exact 150-epoch invocations.
