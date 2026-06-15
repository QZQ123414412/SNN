# Ablation Study: SNM++ + FTBC on CIFAR-100 / VGG16

- Calibration: alpha=0.4, batches=5
- Seed: 42

| Config | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|
| A_QCFS | **64.85%** | **70.50%** | **74.63%** | **76.62%** | **77.60%** |
| B_QCFS+SNM | **65.18%** | **71.79%** | **75.69%** | **77.49%** | **77.65%** |
| C_QCFS+SNM+R0 | **65.03%** | **72.33%** | **76.63%** | **77.51%** | **77.47%** |
| D_QCFS+FTBC | **67.67%** | **72.59%** | **75.94%** | **77.50%** | **77.93%** |
| E_QCFS+SNM+FTBC | **67.95%** | **73.38%** | **76.75%** | **77.68%** | **77.68%** |
| F_QCFS+SNM+R0+FTBC | **68.07%** | **74.05%** | **77.41%** | **77.64%** | **77.65%** |

## Config Legend

| Flag | Meaning |
|---|---|
| QCFS | Baseline positive-only IF with v(0)=θ/2 |
| +SNM | Signed spike + memory gate (neg spike only after pos) |
| +R0 | No-debt rule: if m(t)=0, v(t)←max(v(t),0) |
| +FTBC | Forward Temporal Bias Correction (per-timestep calibration) |
