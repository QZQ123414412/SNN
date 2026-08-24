# ImageNet Full/PA-FTBC + Standard/HA-SNM Cloud Runbook

This runbook is for the single-GPU cloud execution of the frozen ImageNet
protocol. Local development and tests do not require ImageNet and do not
produce formal results.

## 1. Required cloud layout

Run every command from the repository root. The two checkpoints must be
uploaded separately because `*.pth` is ignored by Git:

```text
ImageNet-checkpoints/ImageNet-ResNet34-t8.pth
ImageNet-checkpoints/ImageNet-VGG16-t16.pth
```

The ImageNet root must contain class-organized `train/` and `val/`
directories. Set it once for the shell session:

```bash
export QCFS_IMAGENET_ROOT=/absolute/path/to/imagenet
```

The entry point rejects a checkpoint with a non-protocol SHA256, an
incomplete ImageNet tree, an output collision, or a non-single-GPU device
argument.

## 2. Preflight

Preflight loads each checkpoint strictly, verifies CUDA, checks 1,000 class
mappings and exact train/validation counts, materializes the fixed two-image
calibration tensor, and reports its SHA256 without starting an experiment.

```bash
python -B scripts/experiments/run_imagenet_ftbc_pa_ha_ablation.py \
  --architecture resnet34 \
  --device 0 \
  --preflight-only

python -B scripts/experiments/run_imagenet_ftbc_pa_ha_ablation.py \
  --architecture vgg16 \
  --device 0 \
  --preflight-only
```

The default temporal batch budget is 32, giving batches
`32,16,8,4,2,1` at `T=1,2,4,8,16,32`.

The formal Full-FTBC calibration setting—two images, batch size two, 50
iterations, and alpha 0.5—follows the
[FTBC supplementary material](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08702-supp.pdf).

## 3. Real-data smoke runs

Smoke runs are protocol-locked to `T=4,8,32`, one Full-FTBC calibration
iteration, and two validation batches. They write only below
`docs/archive/experiments/imagenet/`, with caches isolated from formal runs.

```bash
python -B scripts/experiments/run_imagenet_ftbc_pa_ha_ablation.py \
  --architecture resnet34 \
  --device 0 \
  --smoke

python -B scripts/experiments/run_imagenet_ftbc_pa_ha_ablation.py \
  --architecture vgg16 \
  --device 0 \
  --smoke
```

Check the generated reports before continuing:

- T=4 PA rows must be exact copies of the matched Full rows;
- T=8 must use four-coefficient PA compression;
- T=32 must fit in memory;
- all nine configuration rows must be present.

If the smoke result justifies a larger temporal budget, pass the same explicit
value to both preflight and formal commands. Once a formal run starts, that
value is part of the resume signature and cannot be changed.

## 4. Formal runs

Run the models sequentially, ResNet34 first. These commands evaluate all nine
configurations at six time steps on all 50,000 validation images. The ANN
reference gate stops the run if Top-1 differs from 74.32%/74.29% by more than
0.2 percentage points. These reference values are from
[ANN2SNN_SRP](https://github.com/hzc1208/ANN2SNN_SRP).

```bash
python -B scripts/experiments/run_imagenet_ftbc_pa_ha_ablation.py \
  --architecture resnet34 \
  --device 0 \
  --eval-temporal-batch-budget 32

python -B scripts/experiments/run_imagenet_ftbc_pa_ha_ablation.py \
  --architecture vgg16 \
  --device 0 \
  --eval-temporal-batch-budget 32
```

Every completed model/T/config is atomically recorded. Full-FTBC schedules and
the one-time ANN-logit cache are stored under ignored
`runtime_cache/imagenet/formal/<architecture>/` directories.

Resume an interrupted run with the original command plus `--resume`:

```bash
python -B scripts/experiments/run_imagenet_ftbc_pa_ha_ablation.py \
  --architecture resnet34 \
  --device 0 \
  --eval-temporal-batch-budget 32 \
  --resume
```

Resume verifies the protocol version, implementation-source hashes,
checkpoint hash, dataset signature, fixed calibration tensor hash, and all
result-affecting arguments before reusing any cache.

## 5. Two-model summary

After both progress files report `complete`, generate the combined report:

```bash
python -B scripts/experiments/summarize_imagenet_ftbc_pa_ha.py
```

The formal artifacts to sync back are:

```text
docs/results/comparative_ablation/imagenet/IMAGENET_RESNET34_L8_FULL_PA_HA_SNM.md
docs/results/comparative_ablation/imagenet/IMAGENET_RESNET34_L8_FULL_PA_HA_SNM.progress.json
docs/results/comparative_ablation/imagenet/IMAGENET_VGG16_L16_FULL_PA_HA_SNM.md
docs/results/comparative_ablation/imagenet/IMAGENET_VGG16_L16_FULL_PA_HA_SNM.progress.json
docs/results/comparative_ablation/imagenet/IMAGENET_FULL_PA_HA_SNM_TWO_MODEL_SUMMARY.md
```

Do not sync `runtime_cache/`; it contains recoverable ANN logits and Full-FTBC
schedules. Register the three Markdown reports and two progress files in
`docs/results/MANIFEST.md` only after the cloud runs complete.
