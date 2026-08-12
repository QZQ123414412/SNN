# CSRR Three-Way Ablation Design

## Goal

Run a controlled CIFAR-100/VGG16 ablation comparing:

1. QCFS + SNM + R0 + full FTBC.
2. QCFS + SNM + R0 + state-conditioned low-rank FTBC.
3. QCFS + state-conditioned low-rank FTBC + CSRR.

All configurations must run in one process with the same checkpoint,
materialized calibration batches, test loader, random seed, and metric
implementation.

## Implementation Approach

Extend `scripts/experiments/run_successive_refinement_ablation.py` rather than
combining reports from separate runners. Preserve all existing configuration
names and historical result files. Add dedicated three-way configurations so
the controlled experiment does not change the meaning of prior experiments.

The new configurations are:

- `F_RATE_FULL_FTBC`: rate coding, signed negative spikes enabled, legacy R0,
  and full FTBC.
- `H_RATE_STATE_LR_MATCHED`: rate coding, signed negative spikes enabled,
  legacy R0, state-conditioned low-rank FTBC, and matched low-rank calibration
  weights.
- The existing expanded CSRR state-low-rank configuration based on
  `M_SR_GEOM_STATE_LR`, using the final CSRR margins and the same low-rank
  calibration weights as `H_RATE_STATE_LR_MATCHED`.

At T=1 and T=2, the state-low-rank implementation intentionally falls back to
full FTBC because a three-coefficient representation would not reduce storage.
The report must expose the effective FTBC mode for every time step.

## Controlled Parameters

- Dataset: CIFAR-100.
- Model: VGG16.
- Checkpoint: `cifar100-vgg16-l8-example.pth`.
- Time steps: 1, 2, 4, 8, 16, and 32.
- Seed: 42.
- Calibration batches: 5.
- FTBC alpha: 0.4.
- Low-rank ridge: 0.001.
- Low-rank coefficient clip: 0.25.
- Low-rank over-firing weight: 2.5.
- Low-rank under-firing weight: 1.0.
- CSRR time-scale ratio: 1.0.
- CSRR positive margin: 0.55.
- CSRR negative margin: 1.30.
- CSRR R0 mode: `credit_only`.

Full FTBC uses its existing step-wise calibration and therefore has no
low-rank over/under weighting parameter. The two low-rank configurations use
identical weights so their difference isolates the CSRR neuron dynamics.

## Metrics and Output

Write an independent report to
`docs/results/comparative_ablation/cifar100/SR_THREE_WAY_ABLATION_cifar100.md`.
Do not overwrite historical reports.

The report must contain, for every configuration and time step:

- Accuracy.
- Input-driven SOPs.
- Time-scale operations.
- Positive spike rate.
- Negative spike rate.
- Overall spike sparsity.
- FTBC parameter count.
- FTBC storage in bytes.
- Calibration elapsed time.
- Inference elapsed time with statistics disabled.
- Effective FTBC mode.
- Per-layer spike, sparsity, SOP, and ScaleOps details.

The experiment writes the report after each completed configuration/time-step
pair so partial progress survives interruption.

## Testing and Verification

Add a focused regression test before changing the runner. The test must verify
that the new full-FTBC and matched state-low-rank configurations have the
expected coding mode, signed/R0 switches, FTBC mode, and calibration weight.
It must fail before the configuration is implemented and pass afterward.

Run the focused test, then the complete test suite. Before reporting results,
verify that the generated report contains all three configurations, all six
time steps, every required aggregate metric section, configuration details,
and per-layer details.

## Result Interpretation Boundary

The full-FTBC versus low-rank comparison measures the complete low-rank method,
including its state-conditioned parameterization and asymmetric calibration.
The matched low-rank versus CSRR comparison is the stricter CSRR ablation
because both configurations use the same low-rank calibration weights.
Software elapsed time is descriptive and must not be presented as a hardware
latency or energy result.
