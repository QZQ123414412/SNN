# Ablation Study: SNM++ + FTBC on CIFAR100 / VGG16

- Calibration: alpha=0.4, batches=5
- Seed: 42

| Config | T=64 |
|---|---|
| A_QCFS | **77.69%** |
| B_QCFS+SNM | **77.69%** |
| C_QCFS+SNM+R0 | **77.59%** |
| D_QCFS+FTBC | **77.73%** |
| E_QCFS+SNM+FTBC | **77.77%** |
| F_QCFS+SNM+R0+FTBC | **77.72%** |

## Config Legend

| Flag | Meaning |
|---|---|
| QCFS | Baseline positive-only IF with v(0)=θ/2 |
| +SNM | Signed spike + memory gate (neg spike only after pos) |
| +R0 | No-debt rule: if m(t)=0, v(t)←max(v(t),0) |
| +FTBC | Forward Temporal Bias Correction (per-timestep calibration) |
