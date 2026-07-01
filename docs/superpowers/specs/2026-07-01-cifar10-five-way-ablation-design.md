# CIFAR-10 Five-Way Ablation Design

## Goal

Run a controlled CIFAR-10/VGG16 ablation with the same time steps, calibration
settings, metric definitions, timing method, and report structure as the
completed CIFAR-100 three-way experiment.

The five configurations isolate QCFS, signed negative spikes, full FTBC, and
state-conditioned low-rank FTBC while keeping R0 enabled in every group.

## Implementation Approach

Extend `scripts/experiments/run_successive_refinement_ablation.py`. This runner
already provides shared materialized calibration batches, incremental report
writing, aggregate spike/SOP statistics, storage accounting, statistics-free
inference timing, and per-layer details. Do not modify existing configuration
definitions or historical reports.

Add dedicated rate-coding configurations:

| Name | Signed/SNM | R0 | FTBC |
|---|---:|---:|---|
| `A_QCFS_R0` | disabled | enabled | none |
| `B_QCFS_SNM_R0` | enabled | enabled | none |
| `C_QCFS_R0_FULL_FTBC` | disabled | enabled | full |
| `D_QCFS_SNM_R0_FULL_FTBC` | enabled | enabled | full |
| `E_QCFS_SNM_R0_STATE_LR` | enabled | enabled | state-conditioned low-rank |

All configurations use rate coding, ratio 1.0, symmetric rate-mode thresholds,
and `legacy_clamp` R0. Configuration E uses `over_weight=2.5` and
`under_weight=1.0`, matching the completed controlled CIFAR-100 experiment.
The full-FTBC configurations accept the shared CLI value but the full FTBC
solver does not use low-rank over/under weighting.

At T=1 and T=2, configuration E intentionally falls back to full FTBC because
the three-coefficient low-rank representation would not reduce storage. The
report must expose the effective FTBC mode for each time step.

## Controlled Parameters

- Dataset: CIFAR-10.
- Model: VGG16.
- Checkpoint: `cifar10-vgg16-example.pth`.
- Dataset root: `D:\master_degree_paper\datasets` via `QCFS_CIFAR10_ROOT`.
- Time steps: 1, 2, 4, 8, 16, and 32.
- Seed: 42.
- Batch size: 200.
- Calibration batches: 5.
- FTBC alpha: 0.4.
- Low-rank ridge: 0.001.
- Low-rank coefficient clip: 0.25.
- Low-rank over-firing weight: 2.5.
- Low-rank under-firing weight: 1.0.

Every configuration must run in one process with the same checkpoint,
materialized calibration samples, test loader, random seed, and metric code.

## Metrics and Output

Write a new report to
`docs/results/ablation/CIFAR10_FIVE_WAY_ABLATION.md`. Do not overwrite any
existing CIFAR-10, state-FTBC, or successive-refinement report.

For every configuration and time step, report:

- Accuracy.
- Input-driven SOPs.
- Time-scale operations.
- Positive spike rate.
- Negative spike rate.
- Overall spike sparsity.
- FTBC parameter count.
- FTBC storage bytes.
- Calibration elapsed time.
- Inference elapsed time with statistics disabled.
- Effective FTBC mode.
- Per-layer spike rates, sparsity, input spikes, SOPs, and ScaleOps.

The report is rewritten after every completed configuration/time-step pair so
partial progress survives interruption.

## Testing and Verification

Before editing the runner, add a focused test that specifies all five
configuration switches and the E-group over-firing weight. Verify the test
fails because the new names are absent, then add the minimal configuration
entries and verify it passes.

Run the full unit-test suite before the experiment and again after report
generation. Verify the final report contains all five configuration names, all
six time-step columns, all required aggregate sections, configuration details,
and per-layer details with no incomplete cells.

## Interpretation Boundary

- A versus B measures the effect of SNM with R0 held enabled.
- A versus C measures full FTBC without SNM.
- B versus D measures full FTBC with SNM.
- D versus E compares full FTBC against the final low-rank calibration method;
  this comparison includes E's asymmetric low-rank calibration weight.
- Software elapsed time is descriptive and is not a hardware latency or energy
  measurement.
