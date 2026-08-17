# CIFAR-100 / ResNet20 QCFS DIST Fine-tuning Diagnostic

- Status: completed but rejected; not a formal baseline or ablation result
- Source checkpoint accuracy: 68.78%
- Source checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- Teacher: public CIFAR-100 ResNet56 from `chenyaofo/pytorch-cifar-models`
- Teacher checkpoint SHA256: `f2eff4c8461ca1e0d39af83a65f7243bf7d29ec421efee3064bcee93a3caaa73`
- Verified teacher accuracy: 72.62% with its published batch-256 validation protocol
- Student architecture / dataset: ResNet20 / CIFAR-100
- Student activation: QCFS, `L=8`, `fixed_repo` training semantics
- Student data pipeline: `paper_era`, Cutout length 16, no AutoAugment
- Student batch size / seed: 128 / 42
- Loss: cross entropy + 2 x DIST, DIST temperature 4, beta 1, gamma 1
- Optimizer: SGD, momentum 0.9, weight decay 0.0005
- Schedule: two independent 100-epoch cosine trajectories
- Target: at least 69.94% Top-1 accuracy

## Numerical and input validation

The source checkpoint reproduces 68.78% only with the same cuDNN TF32
semantics used when it was selected. Explicitly disabling cuDNN TF32 changes
17 of the 10,000 predictions and gives 68.61%. The formal run therefore fixes
`cudnn_allow_tf32=true` and `cuda_matmul_allow_tf32=false` in its machine-
readable experiment signature.

The teacher uses its published CIFAR-100 normalization, while the student
retains the QCFS repository normalization. A fixed affine input adapter maps
the student-normalized augmented tensor to the teacher normalization without
changing the represented image. Both normalization tuples are recorded in
`experiment_config.json`.

## Results

Both trajectories restarted from the same 68.78% source checkpoint and the
same seeded random state. Neither trajectory inherited optimizer, scheduler,
model, or RNG state from the other.

| Trajectory | Weight LR | Threshold LR | Best epoch (zero-based) | Final accuracy | Best accuracy |
|---|---:|---:|---:|---:|---:|
| `KD_FT_WLR1E4_TLR1E5` | 0.0001 | 0.00001 | 86 | 68.32% | 68.66% |
| `KD_FT_WLR5E5_TLR5E6` | 0.00005 | 0.000005 | 99 | 68.46% | 68.46% |

The global best is the first trajectory's 68.66% checkpoint, with SHA256
`2ce524a35adc556e064f2bc9b7de3f365f03a6557fce1144357273103787b0b6` for
the separately copied global-best file. It is 0.12 percentage points below
the source checkpoint and 1.28 points below the 69.94% target.

Every trajectory contains exactly 100 history records and has status
`complete`. Each trajectory-best checkpoint and the copied global-best
checkpoint were strictly reloaded, re-evaluated on all 10,000 test images,
and converted to the 19-layer SignedIF ResNet20 without key or shape errors.

No `target_ge69_94` checkpoint was created. Under the predeclared protocol,
the T=32 conversion gate is evaluated only after the ANN gate passes, so it
was not run for this rejected result.

The complete checkpoints, training states, histories, configuration, and
summary are stored under the git-ignored directory
`cifar100-checkpoints/resnet20_qcfs_distill_finetune_68_78_to_69_94/`.

## Conclusion

This first-stage DIST recipe does not improve the already converged QCFS
source. Both learning rates initially move the model well below the source;
the cosine tail recovers most, but not all, of the lost accuracy. A subsequent
stage must be declared as a new protocol rather than extending or relabelling
either failed trajectory.
