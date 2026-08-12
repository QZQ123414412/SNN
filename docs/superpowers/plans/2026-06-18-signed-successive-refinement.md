# Signed Successive Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add monotonic signed successive-refinement coding to the existing QCFS/SNM/R0/FTBC pipeline and provide a reproducible CIFAR-100/VGG16 ablation.

**Architecture:** Spikes transmitted between layers remain binary signed events with amplitude `±threshold`. In refinement mode, every `SignedIF` applies the same normalized monotonic time scale `alpha_t` to its input current and membrane reset, so its effective represented value is `alpha_t * spike`; the model applies the same scales to temporal logits before averaging. The rate mode remains byte-for-byte behavior compatible, while FTBC calibration and statistics receive explicit refinement-aware decoding and scale-operation accounting.

**Tech Stack:** Python 3.9, PyTorch, unittest, existing VGG16/QCFS checkpoints and experiment utilities.

---

### Task 1: Monotonic Time-Scale Generator

**Files:**
- Create: `models/temporal_coding.py`
- Test: `tests/test_successive_refinement.py`

- [ ] **Step 1: Write failing tests for rate, geometric and invalid schedules**

Test that rate scales are all ones, geometric scales are positive and non-increasing, every schedule has mean one, `T=1` returns one, and ratios below one raise `ValueError`.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_successive_refinement
```

Expected: import failure for `models.temporal_coding`.

- [ ] **Step 3: Implement the minimal schedule API**

Add:

```python
def make_time_scales(time_steps, mode="rate", ratio=2.0, device=None, dtype=None):
    """Return positive scales with mean one."""
```

Supported modes:

```text
rate       alpha_t = 1
geometric  alpha_t proportional to ratio^(-t)
```

Normalize with `scales *= T / scales.sum()` and validate finite, positive, non-increasing output.

- [ ] **Step 4: Run the focused test**

Expected: all schedule tests pass.

### Task 2: Refinement-Aware SignedIF Dynamics

**Files:**
- Modify: `models/layer.py`
- Test: `tests/test_successive_refinement.py`

- [ ] **Step 1: Add failing neuron tests**

Cover:

```text
rate mode reproduces the current forward output;
T=1 refinement equals rate output;
geometric scales permit positive and later negative correction;
weighted transmitted state never becomes materially negative;
decoded partial output divides by cumulative time scale;
R0 clears negative membrane state when weighted transmitted is zero.
```

- [ ] **Step 2: Run tests and verify behavioral failures**

Run the focused unittest module and confirm missing coding-mode methods.

- [ ] **Step 3: Add refinement state and helper methods**

Add to `SignedIF`:

```python
self.coding_mode = "rate"
self.refinement_schedule = "geometric"
self.refinement_ratio = 2.0
self.time_scales = None
self.scale_operation_count = 0

def set_coding_mode(self, mode, schedule="geometric", ratio=2.0): ...
def get_time_scales(self, reference): ...
def get_time_scale(self, t, reference): ...
def decode_transmitted(self, transmitted, t): ...
```

Changing `T`, mode, schedule or ratio must invalidate cached scales.

- [ ] **Step 4: Implement refinement forward dynamics**

Keep the existing rate branch unchanged. In refinement mode:

```text
q_t = alpha_t * positive_threshold
mem starts at zero
mem = mem - alpha_t * FTBC_bias + alpha_t * input_t
positive event if mem >= q_t / 2
negative event if mem <= -q_t / 2 and transmitted >= q_t
mem = mem - sign(event) * q_t
transmitted = transmitted + sign(event) * q_t
R0 clamps membrane when transmitted <= numerical tolerance
emitted event remains sign(event) * positive_threshold
```

Use a tolerance derived from threshold and dtype epsilon; clamp near-zero transmitted values to exactly zero.

- [ ] **Step 5: Run neuron tests and existing suites**

Run:

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_successive_refinement
D:\Anaconda\envs\ann2snn\python.exe -m unittest discover tests
```

Expected: all tests pass and existing rate behavior remains unchanged.

### Task 3: Model-Level Configuration and Weighted Readout

**Files:**
- Modify: `models/VGG.py`
- Test: `tests/test_successive_refinement.py`

- [ ] **Step 1: Add failing model tests**

Verify `set_coding_mode()` updates every `SignedIF`, `set_T()` invalidates scales, and refinement temporal logits are multiplied by the same mean-one scale before `utils.val()` averages them.

- [ ] **Step 2: Add VGG configuration methods**

Add:

```python
def set_coding_mode(self, mode, schedule="geometric", ratio=2.0):
    ...

def get_temporal_readout_scales(self, reference):
    ...
```

- [ ] **Step 3: Apply refinement scales to final temporal logits**

After `out = self.expand(out)`, multiply only in refinement mode:

```python
out = out * scales.view(T, *([1] * (out.dim() - 1)))
```

Rate output must remain unchanged.

- [ ] **Step 4: Run focused and full tests**

Expected: model tests and all existing tests pass.

### Task 4: Refinement-Aware FTBC Calibration

**Files:**
- Modify: `calibration.py`
- Test: `tests/test_successive_refinement.py`

- [ ] **Step 1: Add a failing local-replay equivalence test**

For a small deterministic tensor, compare `SignedIF.forward()` state/output with a calibration replay helper under geometric refinement.

- [ ] **Step 2: Extract one-step SignedIF replay primitives**

Use the neuron’s public time-scale and decode helpers instead of duplicating schedule formulas in calibration.

- [ ] **Step 3: Update full and state-low-rank calibration**

For refinement mode:

```text
mem correction and input are multiplied by alpha_t;
reset uses q_t = alpha_t * threshold;
negative correction requires transmitted >= q_t;
deviation uses decode_transmitted(transmitted, t) - ANN activation.
```

For rate mode retain existing formulas.

- [ ] **Step 4: Run calibration and full tests**

Expected: replay equivalence passes and legacy FTBC tests remain green.

### Task 5: Scale-Operation Statistics

**Files:**
- Modify: `models/layer.py`
- Modify: `spike_stats.py`
- Modify: `scripts/experiments/run_stats_ablation.py`
- Test: `tests/test_spike_stats.py`

- [ ] **Step 1: Add failing statistics tests**

Verify rate mode reports zero scale operations and refinement mode reports input-driven scaled synaptic operations separately from SOPs.

- [ ] **Step 2: Record per-time-step event counts**

Store positive and negative event counts per time step in `SignedIF` while retaining aggregate counters.

- [ ] **Step 3: Extend `SpikeLayerStats`**

Add:

```python
input_spikes_by_time: tuple = ()
time_scales: tuple = ()
scale_operations: int = 0
```

Count scale operations as input-driven synaptic operations at time steps whose scale is not one. Do not add these operations to SOPs.

- [ ] **Step 4: Add report columns and summaries**

Report total scale operations and preserve the existing SOPs convention.

- [ ] **Step 5: Run focused and full tests**

Expected: all statistics formulas and legacy tests pass.

### Task 6: Direction-One Ablation Runner

**Files:**
- Create: `scripts/experiments/run_successive_refinement_ablation.py`
- Test: `tests/test_successive_refinement.py`

- [ ] **Step 1: Define explicit configurations**

Implement:

```text
C_RATE_SNM_R0
H_RATE_STATE_LR
I_SR_UNIFORM_SNM_R0
J_SR_GEOM_UNSIGNED
K_SR_GEOM_SNM
L_SR_GEOM_SNM_R0
M_SR_GEOM_STATE_LR
```

Allow `--ratios` or one `--ratio` so mild monotonic schedules and the binary ratio `2.0` can be compared globally without per-layer search.

- [ ] **Step 2: Reuse common checkpoint, calibration batches and metrics**

Every configuration must use the same checkpoint, fixed test loader, materialized calibration batches, random seed and time steps.

- [ ] **Step 3: Write incremental Markdown results**

Include:

```text
Accuracy
Input-driven SOPs
Scale operations
Positive/negative spike rates
Spike sparsity
FTBC parameters and bytes
Calibration time
Pure inference time
Per-layer statistics
```

- [ ] **Step 4: Add CLI/configuration tests**

Test configuration expansion and effective FTBC fallback for `T=1/2` without loading CIFAR data.

- [ ] **Step 5: Run all tests and compile checks**

Run:

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest discover tests
D:\Anaconda\envs\ann2snn\python.exe -c "import pathlib; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in pathlib.Path('.').rglob('*.py') if '__pycache__' not in p.parts]; print('compile ok')"
```

Expected: all tests pass and all Python files compile.

### Task 7: CIFAR-100/VGG16 Feasibility and Full Ablation

**Files:**
- Create: `docs/results/csrr/diagnostics/cifar100/SR_FEASIBILITY_cifar100_T2_T4.md`
- Create: `docs/results/csrr/final/cifar100/SR_ABLATION_cifar100.md`

- [ ] **Step 1: Run `T=2/4` screening**

Compare rate, uniform refinement, mild geometric ratios and binary ratio. Use the existing checkpoint:

```powershell
D:\Anaconda\envs\ann2snn\python.exe scripts\experiments\run_successive_refinement_ablation.py `
  -data cifar100 -id cifar100-vgg16-l8-example -dev 0 `
  --time_steps 2 4 --ratios 1.05 1.10 1.25 1.50 2.00 `
  --output docs/results/csrr/diagnostics/cifar100/SR_FEASIBILITY_cifar100_T2_T4.md
```

- [ ] **Step 2: Select the schedule by Pareto rule**

Select the smallest-SOPs candidate whose accuracy is not lower than the rate baseline by more than `0.10` percentage point. If no candidate satisfies that bound, select the highest-accuracy refinement candidate and explicitly mark direction one as not yet Pareto-superior.

- [ ] **Step 3: Run the selected full ablation**

Run `T=1,2,4,8,16,32` for the rate baseline, selected direction-one configuration, direction two, and their combination.

- [ ] **Step 4: Compare against success criteria**

Direction one succeeds only if at least two of `T=1/2/4/8` improve the accuracy-SOPs trade-off, `T=16/32` do not materially regress, and scale-operation overhead is reported separately.

- [ ] **Step 5: Run final verification**

Run the full unit-test suite, inspect `git diff --check`, and record exact commands and limitations in the result document.
