# Monotonic Signed Successive Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement direction one as a conversion-only monotonic signed successive-refinement code, together with complete ablations and cost reporting.

**Architecture:** A shared temporal schedule produces positive, normalized, non-increasing weights `w_t`. Each `SignedIF` emits a signed event with quantum `T*w_t*threshold`, so the existing temporal mean decodes `sum(w_t*s_t)`. The first spiking layer consolidates the repeated static image drive at the first step; later layers propagate weighted events normally. Negative events are limited by accumulated positive credit, and FTBC replay calls the same step function as deployment.

**Tech Stack:** Python, PyTorch, unittest, existing VGG16/QCFS conversion and FTBC utilities.

---

### Task 1: Temporal schedule

**Files:**
- Create: `models/temporal_coding.py`
- Create: `tests/test_monotonic_refinement.py`

- [ ] Add failing tests for uniform, binary/geometric and custom schedules.
- [ ] Verify schedules are positive, non-increasing and sum to one.
- [ ] Implement `make_time_weights()` and `make_event_scales()`.
- [ ] Run `python -m unittest tests.test_monotonic_refinement`.

### Task 2: SignedIF refinement dynamics

**Files:**
- Modify: `models/layer.py`
- Modify: `tests/test_monotonic_refinement.py`

- [ ] Add failing tests for weighted positive events, credit-limited negative events, `T=1` fallback and first-layer input consolidation.
- [ ] Add coding configuration, schedule cache, per-step event quantum and a reusable `refinement_step()`.
- [ ] Preserve the existing rate path exactly when refinement is disabled.
- [ ] Run focused and existing SignedIF tests.

### Task 3: Model propagation and readout ablation

**Files:**
- Modify: `models/VGG.py`
- Modify: `tests/test_monotonic_refinement.py`

- [ ] Add failing tests for model-wide coding configuration and first-layer marking.
- [ ] Add a weighted-readout-only mode for the required counterexample ablation.
- [ ] Keep all-layer refinement decoded by the existing temporal mean because event amplitudes already contain `T*w_t`.
- [ ] Run focused tests.

### Task 4: FTBC replay consistency

**Files:**
- Modify: `calibration.py`
- Modify: `tests/test_monotonic_refinement.py`

- [ ] Add a failing replay-equivalence test.
- [ ] Make full and state-low-rank FTBC replay call the same rate/refinement step logic used by `SignedIF.forward()`.
- [ ] Decode cumulative weighted transmission consistently when constructing calibration deviations.
- [ ] Run calibration and refinement tests.

### Task 5: Statistics and cost accounting

**Files:**
- Modify: `spike_stats.py`
- Modify: `tests/test_spike_stats.py`

- [ ] Add failing tests for per-time event counts and `ScaleOps`.
- [ ] Report input-driven SOPs independently from weighted-event scale operations.
- [ ] Include per-layer time distribution, positive/negative rates and sparsity.
- [ ] Run statistics tests.

### Task 6: Complete ablation runner

**Files:**
- Create: `scripts/experiments/run_monotonic_refinement_ablation.py`
- Modify: `docs/design/FINAL_INNOVATION_PIPELINE_PLAN.md`
- Create: `tests/test_monotonic_ablation.py`

- [ ] Define rate QCFS, rate SNM+R0, uniform refinement, readout-only weighting, fixed binary refinement, calibration-selected refinement, full FTBC and state-low-rank FTBC configurations.
- [ ] Select a global geometric ratio using calibration data only; choose best accuracy and use SOPs as the tie-breaker.
- [ ] Support `T=1,2,4,8,16,32`, incremental Markdown output, per-layer detail and limited-batch smoke runs.
- [ ] Report accuracy, SOPs, ScaleOps, spike rates, sparsity, FTBC storage, selected ratio, calibration time and pure inference time.
- [ ] Run CLI/configuration tests.

### Task 7: Verification

**Files:**
- Create: `docs/results/monotonic_refinement/` reports through the runner.

- [ ] Run the complete unittest suite.
- [ ] Run `git diff --check` and compile all changed Python files.
- [ ] Run a small `T=2/4` CIFAR-100 smoke ablation with limited batches.
- [ ] If the smoke run is valid, record the command for the full `T=1,2,4,8,16,32` experiment.
