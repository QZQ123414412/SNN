# CIFAR-100 / ResNet20 QCFS Fixed-budget Fine-tuning Diagnostic

- Status: completed but rejected; not a formal baseline or ablation result
- Source checkpoint accuracy: 68.78%
- Source checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- Architecture / dataset: ResNet20 / CIFAR-100
- QCFS activation: `L=8`, fixed-repository clamp/quantize order
- Data pipeline: paper-era CIFAR-100 loader without AutoAugment
- Batch size / seed: 128 / 42
- Optimizer: SGD, momentum 0.9, weight decay 0.0005
- Schedule: three independent 50-epoch cosine-annealing trajectories
- Target: at least 69.94% Top-1 accuracy

## Results

Each trajectory started from the same 68.78% checkpoint and the same seeded
data-order state. No trajectory continued from another trajectory.

| Trajectory | Initial LR | Best epoch (zero-based) | Best accuracy |
|---|---:|---:|---:|
| FT_LR005 | 0.005 | 42 | 68.69% |
| FT_LR002 | 0.002 | 49 | 68.68% |
| FT_LR001 | 0.001 | 26 | 68.66% |

The highest fine-tuned accuracy was 68.69%, which is 0.09 percentage points
below the source checkpoint and 1.25 percentage points below the 69.94%
target. Consequently, no `target_ge69_94` checkpoint was created and none of
these fine-tuned checkpoints qualifies for the formal SNN ablation.

The complete machine-readable histories, resumable states, checkpoint hashes,
and verification summary are stored under the git-ignored directory
`cifar100-checkpoints/resnet20_qcfs_finetune_68_78_to_69_94/`. The best
fine-tuned checkpoint was independently reloaded, strictly converted to the
19-layer SignedIF ResNet20, and re-evaluated at 68.69%.

This result shows that restarting conventional SGD with the tested learning
rates initially disrupts the already-converged solution and does not improve
the source ANN within the predeclared 150-epoch budget. Further training
changes require a separately declared second-stage protocol.
