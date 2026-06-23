# Minimal Refinement-aware QCFS Fine-tuning Validation

Date: 2026-06-24

This is a small incremental validation run, not a final paper-scale result. It checks whether the new `refinement_finetune.py` loop can produce checkpoints and whether those checkpoints can be fed back into the current monotonic refinement pipeline.

## Setup

- Dataset/model: CIFAR-100 / VGG16
- Source checkpoint: `cifar100-checkpoints/cifar100-vgg16-l8-example.pth`
- QCFS level: `L=8`
- Fine-tuning data: training split only
- Validation for model selection: training-set split with evaluation transform
- Test-set usage: only in downstream pipeline evaluation
- Fine-tuning budget:
  - Stage A: 1 epoch
  - Stage B: 1 epoch
  - Max train batches per epoch: 20
  - Max validation batches per epoch: 5
  - Batch size: 100
  - Time steps sampled during fine-tuning: `T={2,4}`, probabilities `{0.5,0.5}`
- Pipeline evaluation:
  - Time steps: `T=2,4`
  - Configs: `C_UNIFORM_REFINEMENT`, `F_CALIBRATED_REFINEMENT`
  - For `F_CALIBRATED_REFINEMENT`, candidate set was restricted to `ratio=1.1`, `positive_margin=0.5`, `negative_margin=0.5`
  - Calibration batches: 2
  - Max test batches: 20

## Fine-tuning records

| Checkpoint | Schedule | Ratio | Best validation refinement avg | Best stage |
|---|---:|---:|---:|---|
| `vgg16_cifar100_L[8]_uniform_ratio[1]_minimal_uniform_ft.pth` | uniform | 1.0 | 99.10% | A |
| `vgg16_cifar100_L[8]_geometric_ratio[1.1]_minimal_geometric_ft.pth` | geometric | 1.1 | 99.00% | B |

Detailed incremental logs:

- `docs/results/refinement_finetune/minimal_uniform_ft.jsonl`
- `docs/results/refinement_finetune/minimal_geometric_ft.jsonl`

## Pipeline evaluation summary

Accuracy on the first 20 test batches:

| Source checkpoint | Config | T=2 | T=4 |
|---|---|---:|---:|
| Original QCFS | C uniform refinement | 73.40% | 77.25% |
| Original QCFS | F forced geometric ratio=1.1 | 70.40% | 76.05% |
| Uniform-FT | C uniform refinement | 72.75% | 76.85% |
| Uniform-FT | F forced geometric ratio=1.1 | 70.40% | 75.70% |
| Geometric-FT | C uniform refinement | 72.60% | 76.55% |
| Geometric-FT | F forced geometric ratio=1.1 | 73.00% | 75.30% |

SOPs / ScaleOps:

| Source checkpoint | Config | T=2 SOPs | T=2 ScaleOps | T=4 SOPs | T=4 ScaleOps |
|---|---|---:|---:|---:|---:|
| Original QCFS | C uniform refinement | 276,631,801,536 | 0 | 577,633,875,776 | 0 |
| Original QCFS | F forced geometric ratio=1.1 | 281,236,422,848 | 181,685,834 | 594,034,369,856 | 383,961,823 |
| Uniform-FT | C uniform refinement | 281,990,623,360 | 0 | 587,014,143,616 | 0 |
| Uniform-FT | F forced geometric ratio=1.1 | 286,362,567,680 | 183,466,892 | 603,181,780,928 | 387,086,462 |
| Geometric-FT | C uniform refinement | 271,567,583,744 | 0 | 567,758,986,496 | 0 |
| Geometric-FT | F forced geometric ratio=1.1 | 275,985,328,000 | 180,782,388 | 584,280,171,776 | 382,128,303 |

Detailed pipeline reports:

- `docs/results/refinement_finetune/minimal_original_pipeline.md`
- `docs/results/refinement_finetune/minimal_uniform_ft_pipeline.md`
- `docs/results/refinement_finetune/minimal_geometric_ft_pipeline.md`

## Immediate interpretation

This minimal run does not yet prove the decreasing refinement fine-tuning hypothesis.

Observed:

- At `T=2`, geometric-FT with forced ratio `1.1` improved over original forced geometric by `+2.60 pp` and over uniform-FT forced geometric by `+2.60 pp`.
- At `T=4`, geometric-FT with forced ratio `1.1` was worse than original forced geometric by `-0.75 pp` and worse than uniform-FT forced geometric by `-0.40 pp`.
- Geometric-FT reduced SOPs compared with original and uniform-FT in both uniform and forced geometric pipeline configs.
- Uniform-FT did not improve the tested pipeline metrics in this small run.

Conclusion for this run:

```text
There is a useful T=2 signal, especially with lower SOPs,
but the effect is not consistent across T=2 and T=4.
```

Recommended next step:

```text
Run a stronger but still controlled second pass:
stage A = 3 epochs
stage B = 5-10 epochs
T = {2,4}
compare uniform-FT vs geometric-FT ratio=1.05 and ratio=1.1
use full test set only after validation selects candidates
```

