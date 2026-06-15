# Repository Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the QCFS repository so source code, runnable scripts, formal results, methodology documents, figures, and development archives have clear locations without deleting experiment records.

**Architecture:** Keep reusable runtime modules (`models`, `preprocess`, `calibration.py`, `spike_stats.py`, `utils.py`) at the repository root to minimize behavioral risk. Move command-line entry points under `scripts`, group documents under `docs/methodology`, `docs/results`, `docs/design`, and `docs/archive`, and rename `figure` to `figures`. Add lightweight path bootstrapping to moved scripts so direct invocation from the repository root remains supported.

**Tech Stack:** Python 3.9, PyTorch, PowerShell, Git, Markdown.

---

### Task 1: Create the target directory skeleton

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/train/__init__.py`
- Create: `scripts/evaluate/__init__.py`
- Create: `scripts/experiments/__init__.py`
- Create: `docs/methodology/.gitkeep`
- Create: `docs/results/.gitkeep`
- Create: `docs/design/.gitkeep`
- Create: `docs/archive/.gitkeep`

- [ ] Create explicit directories without deleting any existing path.
- [ ] Add package marker files for the script directories.

### Task 2: Move runnable entry points

**Files:**
- Move: `main_train.py` to `scripts/train/main_train.py`
- Move: `main_test.py` to `scripts/evaluate/main_test.py`
- Move: `main_test_signed.py` to `scripts/evaluate/main_test_signed.py`
- Move: `run_ablation.py` to `scripts/experiments/run_ablation.py`
- Move: `run_stats_ablation.py` to `scripts/experiments/run_stats_ablation.py`
- Move: `run_state_ftbc_ablation.py` to `scripts/experiments/run_state_ftbc_ablation.py`

- [ ] Move each file with `git mv`.
- [ ] Add repository-root bootstrapping before project imports.
- [ ] Update the state-FTBC runner import to `scripts.experiments.run_stats_ablation`.
- [ ] Preserve all command-line arguments and experiment behavior.

### Task 3: Classify documents and experiment artifacts

**Files:**
- Move methodology documents to `docs/methodology/`.
- Move regular results to `docs/results/regular/`.
- Move spike-statistics results to `docs/results/spike_stats/`.
- Move final state-FTBC results to `docs/results/state_ftbc/`.
- Move smoke, validation, optimization, and failed-attempt reports to `docs/archive/state_ftbc/`.
- Move project design/history documents to `docs/design/` or `docs/archive/project_history/`.

- [ ] Preserve every Markdown file; do not delete historical results.
- [ ] Keep `STATE_LOW_RANK_FTBC_FINAL.md` and the full CIFAR-100 report together under formal results.
- [ ] Place `STATE_LOW_RANK_FTBC_addcmul_T4.md` under `docs/archive/state_ftbc/failed_attempts/`.

### Task 4: Organize figures and update output paths

**Files:**
- Move: `figure/` to `figures/`
- Modify: `scripts/experiments/run_stats_ablation.py`
- Modify: `scripts/experiments/run_state_ftbc_ablation.py`

- [ ] Move tracked figure files with explicit `git mv` operations.
- [ ] Update default report output paths to `docs/results/...`.
- [ ] Update any hard-coded `figure` references to `figures`.

### Task 5: Refresh repository documentation

**Files:**
- Modify: `README.md`
- Modify: `QUICK_START.md`
- Create: `docs/README.md`
- Create: `scripts/README.md`

- [ ] Document the new directory tree and the responsibility of each area.
- [ ] Replace old root-level commands with direct script paths.
- [ ] Explain which results are formal and which are archived development records.

### Task 6: Verify compatibility

**Files:**
- Test: `tests/test_calibration.py`
- Test: `tests/test_spike_stats.py`
- Test: `tests/test_state_low_rank_ftbc.py`

- [ ] Run `python -m unittest discover tests` and require all tests to pass.
- [ ] Compile every moved Python entry point.
- [ ] Run `--help` for each moved CLI script.
- [ ] Run `git diff --check`.
- [ ] Confirm no tracked file was deleted rather than moved and no checkpoint was added.
