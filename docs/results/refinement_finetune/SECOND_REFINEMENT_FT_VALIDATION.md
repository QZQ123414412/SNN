# Second-pass Refinement-aware QCFS Fine-tuning Validation

Date: 2026-06-24

This run expands the minimal validation with a larger but still bounded training budget. It is intended as a controlled signal check before running full CIFAR-100 experiments.

## Setup

- Dataset/model: CIFAR-100 / VGG16
- Source checkpoint: `cifar100-checkpoints/cifar100-vgg16-l8-example.pth`
- QCFS level: `L=8`
- Fine-tuning schedules:
  - uniform, ratio `1.0`
  - geometric, ratio `1.05`
  - geometric, ratio `1.1`
- Fine-tuning time steps: `T={2,4}`, sampled with probabilities `{0.5,0.5}`
- Fine-tuning budget per checkpoint:
  - Stage A: 3 epochs
  - Stage B: 5 epochs
  - Max train batches per epoch: 100
  - Max validation batches per epoch: 20
  - Batch size: 100
- Pipeline evaluation:
  - Time steps: `T=2,4`
  - Configs: `C_UNIFORM_REFINEMENT`, `F_CALIBRATED_REFINEMENT`
  - `F` ratio candidates: `{1.05, 1.1}`
  - Margins fixed to `positive_margin=0.5`, `negative_margin=0.5`
  - Calibration batches: 2
  - Max test batches: 20

## Fine-tuning validation records

| Checkpoint | Schedule | Ratio | Best validation refinement avg |
|---|---:|---:|---:|
| `vgg16_cifar100_L[8]_uniform_ratio[1]_second_uniform_ft.pth` | uniform | 1.0 | 98.72% |
| `vgg16_cifar100_L[8]_geometric_ratio[1.05]_second_geometric105_ft.pth` | geometric | 1.05 | 98.38% |
| `vgg16_cifar100_L[8]_geometric_ratio[1.1]_second_geometric11_ft.pth` | geometric | 1.1 | 98.65% |

Incremental logs:

- `docs/results/refinement_finetune/second_uniform_ft.jsonl`
- `docs/results/refinement_finetune/second_geometric105_ft.jsonl`
- `docs/results/refinement_finetune/second_geometric11_ft.jsonl`

## Pipeline accuracy summary

Accuracy on the first 20 test batches:

| Source checkpoint | Config | T=2 | T=4 |
|---|---|---:|---:|
| Original QCFS | C uniform refinement | 73.40% | 77.25% |
| Original QCFS | F calibrated refinement | 72.70% | 77.20% |
| Uniform-FT | C uniform refinement | 71.50% | 75.90% |
| Uniform-FT | F calibrated refinement | 72.45% | 75.35% |
| Geometric-FT 1.05 | C uniform refinement | 73.05% | 77.00% |
| Geometric-FT 1.05 | F calibrated refinement | 72.55% | 76.85% |
| Geometric-FT 1.1 | C uniform refinement | 74.15% | 76.45% |
| Geometric-FT 1.1 | F calibrated refinement | 72.45% | 77.35% |

## SOPs and ScaleOps

| Source checkpoint | Config | T=2 SOPs | T=2 ScaleOps | T=4 SOPs | T=4 ScaleOps |
|---|---|---:|---:|---:|---:|
| Original QCFS | C uniform refinement | 276,631,801,536 | 0 | 577,633,875,776 | 0 |
| Original QCFS | F calibrated refinement | 278,883,945,472 | 179,844,729 | 585,985,161,408 | 376,487,592 |
| Uniform-FT | C uniform refinement | 280,341,903,040 | 0 | 584,770,654,976 | 0 |
| Uniform-FT | F calibrated refinement | 282,553,145,920 | 180,725,063 | 593,049,422,720 | 377,842,306 |
| Geometric-FT 1.05 | C uniform refinement | 285,312,684,416 | 0 | 591,874,900,992 | 0 |
| Geometric-FT 1.05 | F calibrated refinement | 287,325,272,704 | 183,033,916 | 600,009,277,952 | 381,015,447 |
| Geometric-FT 1.1 | C uniform refinement | 273,654,234,880 | 0 | 570,992,012,992 | 0 |
| Geometric-FT 1.1 | F calibrated refinement | 278,019,216,000 | 181,995,077 | 579,269,937,792 | 376,427,271 |

Detailed pipeline reports:

- `docs/results/refinement_finetune/second_original_pipeline.md`
- `docs/results/refinement_finetune/second_uniform_ft_pipeline.md`
- `docs/results/refinement_finetune/second_geometric105_ft_pipeline.md`
- `docs/results/refinement_finetune/second_geometric11_ft_pipeline.md`

## Interpretation

This run gives a stronger but still mixed signal.

Positive signal:

- `Geometric-FT 1.1` evaluated with `C_UNIFORM_REFINEMENT` gives the best `T=2` accuracy:

```text
Original C, T=2:      73.40%
Geometric-FT 1.1 C:   74.15%
Delta:                +0.75 pp
```

- The same configuration also reduces SOPs:

```text
Original C, T=2 SOPs:      276.632B
Geometric-FT 1.1 C SOPs:   273.654B
Delta:                    -1.08%
```

- At `T=4`, `Geometric-FT 1.1` with calibrated refinement slightly improves over original calibrated refinement:

```text
Original F, T=4:      77.20%
Geometric-FT 1.1 F:   77.35%
Delta:                +0.15 pp
```

Negative / inconclusive signal:

- `Uniform-FT` degrades both `C` and `F` results.
- `Geometric-FT 1.05` does not improve the pipeline.
- The clearest gain from `Geometric-FT 1.1` appears under `C_UNIFORM_REFINEMENT`, not under forced/calibrated decreasing deployment at `T=2`.
- Therefore this still does not prove that decreasing temporal weights themselves are the direct source of the gain.

Current conclusion:

```text
Refinement-aware fine-tuning is worth continuing.
Geometric ratio=1.1 is the best candidate so far.
However, the current evidence supports "fine-tuning improves low-T refinement behavior" more strongly than it proves "decreasing deployment weights are superior."
```

Recommended next experiment:

```text
Run Geometric-FT 1.1 and Original on full test set for T={2,4}.
Use C_UNIFORM_REFINEMENT and F_CALIBRATED_REFINEMENT.
Keep ratio candidates {1.05, 1.1}.
If the T=2 C gain and T=4 F gain survive full-test evaluation, then run longer training and add T=8.
```

