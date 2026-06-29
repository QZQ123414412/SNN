# CSRR Three-Way Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing CSRR runner and produce one controlled CIFAR-100/VGG16 report comparing full FTBC, matched state-low-rank FTBC, and matched state-low-rank FTBC with CSRR at T=1/2/4/8/16/32.

**Architecture:** Add two dedicated rate-coding configurations to the existing ordered configuration registry while preserving all historical entries. Reuse the runner's existing expansion, calibration, evaluation, statistics, incremental report writing, and per-layer reporting for all three groups in one process.

**Tech Stack:** Python 3.9, PyTorch, `unittest`, existing QCFS calibration and spike-statistics modules.

---

## File Structure

- Modify `tests/test_successive_refinement.py`: specify the two controlled rate configurations.
- Modify `scripts/experiments/run_successive_refinement_ablation.py`: register full-FTBC and matched state-low-rank configurations.
- Create `docs/results/successive_refinement/SR_THREE_WAY_ABLATION_cifar100.md`: generated aggregate and per-layer metrics.

### Task 1: Specify Controlled Configuration Semantics

**Files:**
- Modify: `tests/test_successive_refinement.py:15-22,316-405`
- Test: `tests/test_successive_refinement.py`

- [ ] **Step 1: Write the failing configuration test**

Import `BASE_CONFIGS` and add:

```python
def test_three_way_rate_configs_are_controlled(self):
    full = BASE_CONFIGS["F_RATE_FULL_FTBC"]
    low_rank = BASE_CONFIGS["H_RATE_STATE_LR_MATCHED"]

    for config in (full, low_rank):
        self.assertEqual(config["coding_mode"], "rate")
        self.assertTrue(config["signed"])
        self.assertTrue(config["r0"])
        self.assertEqual(config["r0_mode"], "legacy_clamp")
        self.assertFalse(config["expand_ratios"])

    self.assertEqual(full["ftbc_mode"], "full")
    self.assertIsNone(full["over_weight"])
    self.assertEqual(low_rank["ftbc_mode"], "state_low_rank")
    self.assertEqual(low_rank["over_weight"], FINAL_OVER_WEIGHT)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_successive_refinement.SuccessiveRefinementAblationConfigTest.test_three_way_rate_configs_are_controlled
```

Expected: FAIL with `KeyError: 'F_RATE_FULL_FTBC'` because the new configuration is absent.

- [ ] **Step 3: Commit the failing test**

```powershell
git add tests/test_successive_refinement.py
git commit -m "test: specify controlled CSRR three-way configs"
```

### Task 2: Add Full and Matched Low-Rank Configurations

**Files:**
- Modify: `scripts/experiments/run_successive_refinement_ablation.py:43-76`
- Test: `tests/test_successive_refinement.py`

- [ ] **Step 1: Add the minimal configuration entries**

Insert these entries in `BASE_CONFIGS` without changing existing entries:

```python
(
    "F_RATE_FULL_FTBC",
    dict(
        coding_mode="rate",
        schedule="rate",
        ratio=1.0,
        signed=True,
        r0=True,
        ftbc_mode="full",
        expand_ratios=False,
        positive_margin=0.5,
        negative_margin=0.5,
        r0_mode="legacy_clamp",
        over_weight=None,
    ),
),
(
    "H_RATE_STATE_LR_MATCHED",
    dict(
        coding_mode="rate",
        schedule="rate",
        ratio=1.0,
        signed=True,
        r0=True,
        ftbc_mode="state_low_rank",
        expand_ratios=False,
        positive_margin=0.5,
        negative_margin=0.5,
        r0_mode="legacy_clamp",
        over_weight=FINAL_OVER_WEIGHT,
    ),
),
```

- [ ] **Step 2: Run the focused test and verify GREEN**

Run the focused command from Task 1. Expected: `Ran 1 test` and `OK`.

- [ ] **Step 3: Run all successive-refinement tests**

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_successive_refinement
```

Expected: all tests pass.

- [ ] **Step 4: Run the complete test suite**

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest discover tests
```

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 5: Commit the implementation**

```powershell
git add scripts/experiments/run_successive_refinement_ablation.py
git commit -m "feat: add controlled CSRR three-way ablation configs"
```

### Task 3: Run the CIFAR-100 Three-Way Ablation

**Files:**
- Create: `docs/results/successive_refinement/SR_THREE_WAY_ABLATION_cifar100.md`

- [ ] **Step 1: Run all three configurations and six time steps**

```powershell
D:\Anaconda\envs\ann2snn\python.exe scripts\experiments\run_successive_refinement_ablation.py -data cifar100 -id cifar100-vgg16-l8-example --configs F_RATE_FULL_FTBC H_RATE_STATE_LR_MATCHED M_SR_GEOM_STATE_LR --time_steps 1 2 4 8 16 32 --ratios 1.0 --positive_margins 0.55 --negative_margins 1.3 --over_weight 2.5 --under_weight 1.0 --cali_batches 5 --seed 42 --output docs/results/successive_refinement/SR_THREE_WAY_ABLATION_cifar100.md
```

Expected: 18 completed summaries and exit code 0. The M configuration expands to `M_SR_GEOM_STATE_LR_R1_P0.55_N1.3`.

- [ ] **Step 2: Verify report structure and completeness**

Confirm the report contains all three names, six time-step columns, ten aggregate sections, configuration details, and per-layer details. Expected modes: full FTBC remains `full`; both low-rank configurations are `full` at T=1/2 and `state_low_rank` at T=4/8/16/32.

- [ ] **Step 3: Commit the generated report**

```powershell
git add docs/results/successive_refinement/SR_THREE_WAY_ABLATION_cifar100.md
git commit -m "results: record controlled CSRR three-way ablation"
```

### Task 4: Final Verification and Interpretation

**Files:**
- Verify: `docs/results/successive_refinement/SR_THREE_WAY_ABLATION_cifar100.md`

- [ ] **Step 1: Re-run the complete test suite**

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest discover tests
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Inspect repository changes**

```powershell
git status --short --branch
git log -4 --oneline --decorate
```

Expected: only pre-existing generated `__pycache__` modifications remain uncommitted; design, plan, test, implementation, and result commits appear in history.

- [ ] **Step 3: Summarize without overstating latency evidence**

Report accuracy, SOPs, positive/negative spike rates, sparsity, and FTBC storage trends. Treat calibration and PyTorch inference elapsed times as descriptive measurements, not hardware energy or deployment latency.
