# Ablation Study: SNM++ + FTBC on CIFAR10 / VGG16

- Calibration: alpha=0.4, batches=5
- Seed: 42

| Config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|---|
| A_QCFS | **88.24%** | **91.03%** | **93.74%** | **94.87%** | **95.33%** | **95.52%** |
| B_QCFS+SNM | **88.24%** | **90.97%** | **93.89%** | **95.15%** | **95.51%** | **95.57%** |
| C_QCFS+SNM+R0 | **88.24%** | **91.11%** | **94.12%** | **95.30%** | **95.55%** | **95.57%** |
| D_QCFS+FTBC | **89.47%** | **91.96%** | **94.29%** | **95.28%** | **95.46%** | **95.52%** |
| E_QCFS+SNM+FTBC | **90.01%** | **92.37%** | **94.30%** | **95.20%** | **95.53%** | **95.53%** |
| F_QCFS+SNM+R0+FTBC | **89.82%** | **91.89%** | **94.39%** | **95.35%** | **95.56%** | **95.64%** |

## Config Legend

| Flag | Meaning |
|---|---|
| QCFS | Baseline positive-only IF with v(0)=θ/2 |
| +SNM | Signed spike + memory gate (neg spike only after pos) |
| +R0 | No-debt rule: if m(t)=0, v(t)←max(v(t),0) |
| +FTBC | Forward Temporal Bias Correction (per-timestep calibration) |
