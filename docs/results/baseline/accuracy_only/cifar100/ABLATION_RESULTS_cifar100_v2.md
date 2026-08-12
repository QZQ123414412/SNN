# Ablation Study: SNM++ + FTBC on CIFAR100 / VGG16

- Calibration: alpha=0.4, batches=5
- Seed: 42

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|---|
| A_QCFS | **58.81%** | **64.85%** | **70.50%** | **74.63%** | **76.62%** | **77.60%** |
| B_QCFS+SNM | **58.81%** | **65.18%** | **71.79%** | **75.69%** | **77.49%** | **77.65%** |
| C_QCFS+SNM+R0 | **58.81%** | **65.03%** | **72.33%** | **76.63%** | **77.51%** | **77.47%** |
| D_QCFS+FTBC | **62.08%** | **68.04%** | **72.66%** | **76.16%** | **77.58%** | **77.80%** |
| E_QCFS+SNM+FTBC | **61.71%** | **68.34%** | **73.53%** | **76.74%** | **77.83%** | **77.70%** |
| F_QCFS+SNM+R0+FTBC | **62.14%** | **68.43%** | **73.88%** | **77.22%** | **77.65%** | **77.81%** |

## Config Legend

| Flag | Meaning |
|---|---|
| QCFS | Baseline positive-only IF with v(0)=θ/2 |
| +SNM | Signed spike + memory gate (neg spike only after pos) |
| +R0 | No-debt rule: if m(t)=0, v(t)←max(v(t),0) |
| +FTBC | Forward Temporal Bias Correction (per-timestep calibration) |
