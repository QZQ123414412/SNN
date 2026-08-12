# CIFAR-10 Five-Way Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five controlled rate-coding configurations and generate a complete CIFAR-10/VGG16 ablation report at T=1/2/4/8/16/32.

**Architecture:** Extend the ordered configuration registry in the existing successive-refinement runner without changing historical configurations. Reuse its shared calibration batches, FTBC modes, aggregate metrics, timing, incremental report writing, and per-layer statistics.

**Tech Stack:** Python 3.9, PyTorch, torchvision, `unittest`, existing QCFS calibration and spike-statistics modules.

---

## File Structure

- Modify `tests/test_successive_refinement.py`: specify the five configuration contracts.
- Modify `scripts/experiments/run_successive_refinement_ablation.py`: register five dedicated rate configurations.
- Create `docs/results/comparative_ablation/cifar10/CIFAR10_FIVE_WAY_ABLATION.md`: generated aggregate and per-layer results.

### Task 1: Specify the Five Configuration Contracts

**Files:**
- Modify: `tests/test_successive_refinement.py:15-22,316-430`
- Test: `tests/test_successive_refinement.py`

- [ ] **Step 1: Add a failing test**

Import `BASE_CONFIGS` if it is not already imported and add:

```python
def test_cifar10_five_way_configs_match_design(self):
    expected = {
        "A_QCFS_R0": (False, True, "none", None),
        "B_QCFS_SNM_R0": (True, True, "none", None),
        "C_QCFS_R0_FULL_FTBC": (False, True, "full", None),
        "D_QCFS_SNM_R0_FULL_FTBC": (True, True, "full", None),
        "E_QCFS_SNM_R0_STATE_LR": (
            True,
            True,
            "state_low_rank",
            FINAL_OVER_WEIGHT,
        ),
    }

    for name, (signed, r0, ftbc_mode, over_weight) in expected.items():
        config = BASE_CONFIGS[name]
        self.assertEqual(config["coding_mode"], "rate")
        self.assertEqual(config["schedule"], "rate")
        self.assertEqual(config["ratio"], 1.0)
        self.assertEqual(config["signed"], signed)
        self.assertEqual(config["r0"], r0)
        self.assertEqual(config["ftbc_mode"], ftbc_mode)
        self.assertEqual(config["r0_mode"], "legacy_clamp")
        self.assertEqual(config["over_weight"], over_weight)
        self.assertFalse(config["expand_ratios"])
```

- [ ] **Step 2: Verify the test fails for the missing names**

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_successive_refinement.SuccessiveRefinementAblationConfigTest.test_cifar10_five_way_configs_match_design
```

Expected: failure with `KeyError: 'A_QCFS_R0'`.

- [ ] **Step 3: Commit the failing test**

```powershell
git add tests/test_successive_refinement.py
git commit -m "test: specify CIFAR-10 five-way ablation configs"
```

### Task 2: Register the Five Configurations

**Files:**
- Modify: `scripts/experiments/run_successive_refinement_ablation.py:43-190`
- Test: `tests/test_successive_refinement.py`

- [ ] **Step 1: Add the five entries without changing existing entries**

```python
("A_QCFS_R0", dict(coding_mode="rate", schedule="rate", ratio=1.0,
    signed=False, r0=True, ftbc_mode="none", expand_ratios=False,
    positive_margin=0.5, negative_margin=0.5,
    r0_mode="legacy_clamp", over_weight=None)),
("B_QCFS_SNM_R0", dict(coding_mode="rate", schedule="rate", ratio=1.0,
    signed=True, r0=True, ftbc_mode="none", expand_ratios=False,
    positive_margin=0.5, negative_margin=0.5,
    r0_mode="legacy_clamp", over_weight=None)),
("C_QCFS_R0_FULL_FTBC", dict(coding_mode="rate", schedule="rate", ratio=1.0,
    signed=False, r0=True, ftbc_mode="full", expand_ratios=False,
    positive_margin=0.5, negative_margin=0.5,
    r0_mode="legacy_clamp", over_weight=None)),
("D_QCFS_SNM_R0_FULL_FTBC", dict(coding_mode="rate", schedule="rate",
    ratio=1.0, signed=True, r0=True, ftbc_mode="full",
    expand_ratios=False, positive_margin=0.5, negative_margin=0.5,
    r0_mode="legacy_clamp", over_weight=None)),
("E_QCFS_SNM_R0_STATE_LR", dict(coding_mode="rate", schedule="rate",
    ratio=1.0, signed=True, r0=True, ftbc_mode="state_low_rank",
    expand_ratios=False, positive_margin=0.5, negative_margin=0.5,
    r0_mode="legacy_clamp", over_weight=FINAL_OVER_WEIGHT)),
```

- [ ] **Step 2: Verify the focused test passes**

Run the focused command from Task 1. Expected: `Ran 1 test` and `OK`.

- [ ] **Step 3: Run the refinement test module**

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest tests.test_successive_refinement
```

Expected: all refinement tests pass.

- [ ] **Step 4: Run the complete test suite**

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest discover tests
```

Expected: zero failures and zero errors.

- [ ] **Step 5: Commit the implementation**

```powershell
git add scripts/experiments/run_successive_refinement_ablation.py
git commit -m "feat: add CIFAR-10 five-way ablation configs"
```

### Task 3: Run the CIFAR-10 Experiment

**Files:**
- Create: `docs/results/comparative_ablation/cifar10/CIFAR10_FIVE_WAY_ABLATION.md`

- [ ] **Step 1: Execute all 30 combinations**

```powershell
$env:QCFS_CIFAR10_ROOT='D:\master_degree_paper\datasets'
D:\Anaconda\envs\ann2snn\python.exe -u scripts\experiments\run_successive_refinement_ablation.py -data cifar10 -id cifar10-vgg16-example --configs A_QCFS_R0 B_QCFS_SNM_R0 C_QCFS_R0_FULL_FTBC D_QCFS_SNM_R0_FULL_FTBC E_QCFS_SNM_R0_STATE_LR --time_steps 1 2 4 8 16 32 --ratios 1.0 --positive_margins 0.5 --negative_margins 0.5 --over_weight 2.5 --under_weight 1.0 --cali_batches 5 --seed 42 --output docs/results/comparative_ablation/cifar10/CIFAR10_FIVE_WAY_ABLATION.md
```

Expected: 30 completed summaries and exit code 0.

- [ ] **Step 2: Verify report completeness**

Confirm all five names and all six time-step columns appear. Confirm the ten aggregate metric sections, configuration detail, and per-layer detail are present with no `-` cells. Confirm E uses `full` at T=1/2 and `state_low_rank` at T=4/8/16/32.

- [ ] **Step 3: Commit the report**

```powershell
git add docs/results/comparative_ablation/cifar10/CIFAR10_FIVE_WAY_ABLATION.md
git commit -m "results: record CIFAR-10 five-way ablation"
```

### Task 4: Final Verification and Summary

**Files:**
- Verify: `docs/results/comparative_ablation/cifar10/CIFAR10_FIVE_WAY_ABLATION.md`

- [ ] **Step 1: Re-run the complete test suite**

```powershell
D:\Anaconda\envs\ann2snn\python.exe -m unittest discover tests
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Inspect branch state**

```powershell
git status --short --branch
git log -6 --oneline --decorate
```

Expected: only the pre-existing tracked `__pycache__` changes remain uncommitted.

- [ ] **Step 3: Summarize the controlled comparisons**

Report A versus B, A versus C, B versus D, and D versus E for accuracy, SOPs,
spike rates, sparsity, storage, and software elapsed time. State that D versus
E includes E's asymmetric low-rank calibration weight and that software timing
is not a hardware energy result.
