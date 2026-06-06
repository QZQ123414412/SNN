# Ablation Study: SNM++ + FTBC on CIFAR10 / VGG16

- Calibration: alpha=0.4, batches=5
- Seed: 42

| Config | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|
| A_QCFS | **91.03%** | **93.74%** | **94.87%** | **95.33%** | **95.52%** |
| B_QCFS+SNM | **90.97%** | **93.89%** | **95.15%** | **95.51%** | **95.57%** |
| C_QCFS+SNM+R0 | **91.11%** | **94.12%** | **95.30%** | **95.55%** | **95.57%** |
| D_QCFS+FTBC | **91.97%** | **94.14%** | **95.15%** | **95.46%** | **95.41%** |
| E_QCFS+SNM+FTBC | **91.96%** | **94.34%** | **95.29%** | **95.55%** | **95.53%** |
| F_QCFS+SNM+R0+FTBC | **91.80%** | **94.65%** | **95.17%** | **95.55%** | **95.55%** |

## Config Legend

| Flag | Meaning |
|---|---|
| QCFS | Baseline positive-only IF with v(0)=θ/2 |
| +SNM | Signed spike + memory gate (neg spike only after pos) |
| +R0 | No-debt rule: if m(t)=0, v(t)←max(v(t),0) |
| +FTBC | Forward Temporal Bias Correction (per-timestep calibration) |
