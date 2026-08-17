# CIFAR-100 / ResNet20 QCFS Six-way Ablation Protocol

## Scope

This experiment evaluates QCFS, SNM, R0, full FTBC, and state-conditioned
low-rank FTBC on CIFAR-100 / ResNet20. CSRR is disabled in every group. All
groups use rate coding, temporal ratio 1, and the legacy-clamp R0 rule.

## QCFS Weight Requirement

The source ANN must be trained with the QCFS stair-step activation, not ReLU.
The activation level is fixed at `L=8`. A conventional ReLU ResNet20
checkpoint is not a valid substitute.

The fixed official repository is:

- <https://github.com/putshua/ANN_SNN_QCFS>
- verified source commit: `eca136bd085087567013240ee14fb6159a2b6da7`

The official public model folder was checked on 2026-08-12:

- <https://drive.google.com/drive/folders/1P-2egAraWtsQYNzp8lcJvZVEG_KLVV5Q>

It contained CIFAR-10/VGG16, CIFAR-100/VGG16, ImageNet/ResNet34, and
ImageNet/VGG16 files, but no CIFAR-100/ResNet20 checkpoint. Therefore the
required checkpoint is retrained with the fixed official implementation and
must be labelled `official_implementation_retrained`, not
`author_pretrained`.

The retraining configuration follows Appendix A.1 of the QCFS paper for the
reported CIFAR-100 / ResNet20 result:

- local source: `papers_for_reference/QCFS.pdf`
- source SHA256: `e23e1e9ae5dc6193b7908275c681cab371d2167208a5951f3867fc66580b9b07`
- training setup: PDF page 15
- CIFAR-100 / ResNet20 accuracy table: PDF page 20

- architecture: ResNet20
- dataset: CIFAR-100
- QCFS activation levels: `L=8`
- seed: 42
- batch size: 128 (the paper-era public training entry default)
- SGD: learning rate 0.02, momentum 0.9, weight decay 0.0005
- scheduler: cosine annealing over 300 epochs
- augmentation: random crop, horizontal flip, Cutout 16; the current
  controlled retry uses the paper-era public CIFAR-100 loader without
  AutoAugment

The paper itself does not state a batch size. Repository history shows 128 in
the paper-era entry, then 200 in the initial fixed-repository upload, and 300
in a later CIFAR-100/VGG16 example update. A completed 300-batch reproduction
reached 67.17%. Changing only the batch size to the paper-era value 128
improved the result to 68.21%. Using the paper-era public CIFAR-100
augmentation profile reached 68.78%, while the paper-era quantize-then-clamp
surrogate-gradient retry reached 67.87%. Per the explicit experiment decision,
the six-way ablation uses the 68.78% checkpoint:

- file: `resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth`
- training-log best accuracy: 68.78% at epoch 289
- SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`
- weight origin: `official_implementation_retrained`

The exact file re-evaluates at 68.68% on the current RTX 5080/PyTorch stack
(68.65% on CPU), a 0.10--0.13 percentage-point numerical difference from the
training log. Both the logged and locally re-evaluated values are recorded in
the result report rather than conflated.

## Weight Acceptance Gates

The experiment runner applies the following checks before an ablation point:

1. Load the checkpoint into the original IF/QCFS ResNet20 with `strict=True`.
2. Match every SignedIF activation to the original IF activation by module
   name and copy its positive threshold exactly.
3. Initialize only the additional negative threshold as
   `neg_thresh = -thresh`, then load the SignedIF network with `strict=True`.
4. Record filename, byte size, SHA256, source category, dataset,
   architecture, and number of QCFS activation layers.
5. For a formal run, require the exact selected checkpoint SHA256 shown above.
6. Require QCFS ANN accuracy of at least 68.63% on the full 10,000-image test
   set. This is the logged 68.78% minus a pre-declared 0.15-point environment
   tolerance.
7. Require the uncalibrated `A_QCFS_R0` result at T=32 to be no more than
   2.0 percentage points below its QCFS ANN accuracy.

Both accuracy gates run before any smoke or formal report is created, so a
rejected checkpoint cannot leave a failed result in a formal-results path.

The published QCFS reference curve for CIFAR-100 / ResNet20 is ANN 69.94%,
with 19.96%, 34.14%, 55.37%, 67.33%, and 69.82% at T=2, 4, 8, 16, and 32.
These values are a reproduction diagnostic, not values copied into the local
result cells.

## Six Configurations

| Config | SNM | R0 | FTBC | CSRR |
|---|---|---|---|---|
| A_QCFS_R0 | Off | On | None | Off |
| B_QCFS_SNM_R0 | On | On | None | Off |
| C_QCFS_R0_FULL_FTBC | Off | On | Full | Off |
| D_QCFS_SNM_R0_FULL_FTBC | On | On | Full | Off |
| E_QCFS_R0_STATE_LR | Off | On | State low-rank | Off |
| F_QCFS_SNM_R0_STATE_LR | On | On | State low-rank | Off |

The design is a symmetric 2-by-3 matrix: SNM is on/off, while FTBC is none,
full, or state low-rank. R0 is held on. The intended comparisons are:

- SNM: A-B, C-D, and E-F.
- Full FTBC: A-C and B-D.
- State low-rank FTBC: A-E and B-F.
- Full versus state low-rank: C-E and D-F.

State low-rank uses three per-channel coefficients. At T=1 and T=2 it falls
back to full FTBC because there are too few temporal observations; the report
always records the effective FTBC mode. Full-versus-low-rank conclusions use
T>=4.

## Fixed Formal Protocol

- time steps: `T=1,2,4,8,16,32`
- evaluation/calibration batch size: 200
- calibration data: the same materialized 5 batches (1,000 images) for every
  calibrated group
- seed: 42
- FTBC alpha: 0.4
- ridge: 0.001
- coefficient clip: 0.25 thresholds
- under-estimation weight: 1.0
- over-estimation weight: 2.5

The over/under weights apply only to the state-low-rank weighted regression.
Full FTBC preserves the preceding per-timestep mean-bias solver. Consequently,
the C-E and D-F comparisons measure the practical full-versus-compressed
calibration designs used in the preceding experiments, including this solver
difference; they are not a pure rank-only comparison.

The runner rejects a `formal` invocation that changes the six-group order,
time steps, batch size, calibration batch count, seed, or QCFS L.

## Metrics

The metrics are unchanged from the preceding ablations:

- top-1 accuracy
- input-driven SOPs, counting positive and negative spike events
- time-scale operations, expected to be zero for rate coding
- positive and negative spike rates
- overall and per-layer spike sparsity
- per-layer input spikes and SOPs
- FTBC parameters and storage bytes
- calibration time
- inference time with spike statistics disabled
- effective FTBC mode

The raw image input before the first spiking layer is not modelled as a spike
source, so the first spiking layer has zero input-driven SOPs. The final
100-class linear readout has no spike output, but its input-driven SOPs are
included as a compute-only row and excluded from spike-rate/sparsity
denominators. Projection-shortcut and residual-branch convolutions are counted
separately according to the ResNet20 graph.

## Execution

Run the smoke test before the formal experiment. Result overwriting is disabled
by default. Both commands below use the selected 68.78% QCFS checkpoint.

```powershell
$env:QCFS_CIFAR100_ROOT='D:\master_degree_paper\qcfs_cifar100_data_v2'
$env:QCFS_NUM_WORKERS='0'
D:\Anaconda\envs\ann2snn\python.exe -B scripts\experiments\run_resnet20_qcfs_ablation.py `
  --checkpoint 'cifar100-checkpoints\resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth' `
  --run_kind smoke --time_steps 2 4 --cali_batches 1 `
  --output 'docs\archive\experiments\resnet20\RESNET20_SIX_WAY_SMOKE_v1.md'
```

After the smoke test passes, run the locked formal protocol:

```powershell
$env:QCFS_CIFAR100_ROOT='D:\master_degree_paper\qcfs_cifar100_data_v2'
$env:QCFS_NUM_WORKERS='0'
D:\Anaconda\envs\ann2snn\python.exe -B scripts\experiments\run_resnet20_qcfs_ablation.py `
  --checkpoint 'cifar100-checkpoints\resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth' `
  --run_kind formal
```

The formal report is written incrementally to
`docs/results/comparative_ablation/cifar100/RESNET20_SIX_WAY_ABLATION.md` and
is marked `INCOMPLETE` until all 36 points have been recorded.

Each completed point is also saved atomically to a sibling
`RESNET20_SIX_WAY_ABLATION.progress.json` file. An interrupted run can be
continued with the same formal command plus `--resume`. Resume verifies the
checkpoint SHA256 and every fixed protocol field before accepting earlier
points. The signature also includes the full six-group switch matrix, weight
origin, gate thresholds, and a SHA256 of the materialized calibration tensors;
`--resume` and `--overwrite` cannot be combined.
