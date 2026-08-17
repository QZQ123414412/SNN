# 2026-08-17 CIFAR-100 / ResNet20 QCFS and State-LR Update

## Repository baseline

- Source branch: `feature/credit-aware-signed-refinement`
- Previous commit: `948606e` (`docs: organize experiment results and references`)
- Dataset / architecture: CIFAR-100 / ResNet20
- QCFS level: `L=8`
- Selected checkpoint training-log accuracy: `68.78%`
- Selected checkpoint SHA256: `1fbf69f69b56b0c258c1f4a47574aecd5c37b48e842a90e0e3ab49e8aff00dc2`

## Code changes

1. Added a native `resnet20_signed` construction path and model-level controls
   for SNM, R0, FTBC mode, coding mode, and the state-conditioned bias term.
2. Matched ANN and SNN calibration layers by stable module names instead of
   relying on positional `named_modules()` traversal.
3. Added a switch that disables only the State-LR `bias_state` term, enabling
   a controlled causal diagnostic without changing its base and slope terms.
4. Added checkpoint provenance and strict ANN/SNN loading helpers for the
   selected QCFS checkpoint.
5. Added resumable QCFS training with optimizer, scheduler, RNG, and complete
   training-state persistence, together with explicit paper-era/fixed-repo
   augmentation and quantization profiles.
6. Added graph-aware ResNet20 spike and SOP accounting, including projection
   shortcuts and the final linear readout.
7. Added formal six-way ablation and State-LR/SNM causal-diagnostic runners,
   including resumable progress records and protocol signatures.

## Experiment records

- Six-way ablation configurations: QCFS, QCFS+SNM, Full-FTBC with and without
  SNM, and State-LR with and without SNM at `T={1,2,4,8,16,32}`.
- State-LR causal switches at `T={4,8,16,32}`:
  unsigned-calibrated coefficients with SNM enabled at inference, removal of
  the state term, and final accumulated-coefficient clipping.
- The causal diagnostic shows that calibrating State-LR on the unsigned path,
  freezing its coefficients, and enabling SNM only for inference avoids the
  severe low-timestep collapse of jointly calibrated State-LR and SNM.
- Training and failed reproduction attempts are retained under
  `docs/archive/experiments/resnet20/` for provenance rather than presented as
  formal results.

## Validation

- Command: `D:\Anaconda\envs\ann2snn\python.exe -m unittest discover tests`
- Result: `89` tests passed on 2026-08-17.

## Exclusions

- Model checkpoints and datasets remain excluded by `.gitignore`.
- Generated `__pycache__` / `.pyc` artifacts and local `AGENTS.md` instructions
  are not part of this source update.

## Next experimental branch

The next method branch starts from this recorded baseline and targets the
decoupled pipeline:

`QCFS -> unsigned teacher FTBC -> Temporal-LR compression -> frozen/gated SNM`
