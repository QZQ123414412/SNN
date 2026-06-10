# Ablation Study: SNM++ + FTBC on CIFAR10 / VGG16

- Calibration: alpha=0.4, batches=5
- Seed: 42

| Config | T=64 |
|---|---|
| A_QCFS | **95.54%** |
| B_QCFS+SNM | **95.58%** |
| C_QCFS+SNM+R0 | **95.57%** |
| D_QCFS+FTBC | **95.46%** |
| E_QCFS+SNM+FTBC | **95.57%** |
| F_QCFS+SNM+R0+FTBC | **95.57%** |

## Config Legend

| Flag | Meaning |
|---|---|
| QCFS | Baseline positive-only IF with v(0)=θ/2 |
| +SNM | Signed spike + memory gate (neg spike only after pos) |
| +R0 | No-debt rule: if m(t)=0, v(t)←max(v(t),0) |
| +FTBC | Forward Temporal Bias Correction (per-timestep calibration) |
