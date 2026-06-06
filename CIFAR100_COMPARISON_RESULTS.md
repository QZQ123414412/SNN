# CIFAR-100 IF vs SignedIF (SNM) 性能对比测试结果

## 测试信息

- **数据集**: CIFAR100
- **模型架构**: vgg16 / vgg16_signed
- **权重文件**: cifar100-vgg16-l8-example
- **测试时间**: 2026-02-12 15:56:37
- **测试脚本**: 
  - `main_test.py` (原始IF模型)
  - `main_test_signed.py` (SignedIF/SNM模型)

---

## 测试结果对比表

| 时间步 (T) | IF (原始) | SignedIF (SNM) | 提升 | 提升率 | 最佳模型 |
|-----------|----------|---------------|------|--------|---------|
| T=1 | **58.81%** | **58.81%** | +0.00% | +0.00% | 平局 |
| T=2 | **64.85%** | **65.18%** | +0.33% | +0.51% | SignedIF 🏆 |
| T=4 | **70.50%** | **71.79%** | +1.29% | +1.83% | SignedIF 🏆 |
| T=8 | **74.63%** | **75.69%** | +1.06% | +1.42% | SignedIF 🏆 |
| T=16 | **76.62%** | **77.49%** | +0.87% | +1.14% | SignedIF 🏆 |
| T=32 | **77.60%** | ❌ 失败 | - | - | IF |

---

## 性能趋势分析

### IF (原始) 模型性能趋势

| 时间步对比 | 精度提升 | 提升率 |
|-----------|---------|--------|
| T=1 → T=2 | +6.04% | +10.27% |
| T=2 → T=4 | +5.65% | +8.71% |
| T=4 → T=8 | +4.13% | +5.86% |
| T=8 → T=16 | +1.99% | +2.67% |
| T=16 → T=32 | +0.98% | +1.28% |

### SignedIF (SNM) 模型性能趋势

| 时间步对比 | 精度提升 | 提升率 |
|-----------|---------|--------|
| T=1 → T=2 | +6.37% | +10.83% |
| T=2 → T=4 | +6.61% | +10.14% |
| T=4 → T=8 | +3.90% | +5.43% |
| T=8 → T=16 | +1.80% | +2.38% |

---

## 详细输出

### IF (原始) 模型详细输出

#### T=1

```
Files already downloaded and verified
Files already downloaded and verified
58.81

```

#### T=2

```
Files already downloaded and verified
Files already downloaded and verified
64.85

```

#### T=4

```
Files already downloaded and verified
Files already downloaded and verified
70.5

```

#### T=8

```
Files already downloaded and verified
Files already downloaded and verified
74.63

```

#### T=16

```
Files already downloaded and verified
Files already downloaded and verified
76.62

```

#### T=32

```
Files already downloaded and verified
Files already downloaded and verified
77.6

```

### SignedIF (SNM) 模型详细输出

#### T=1

```
Files already downloaded and verified
Files already downloaded and verified

Loaded thresholds:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Thresholds after set_T:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Testing with Signed Spike + Memory neurons
Time steps: 1
Test Accuracy: 58.81%

```

#### T=2

```
Files already downloaded and verified
Files already downloaded and verified

Loaded thresholds:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Thresholds after set_T:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Testing with Signed Spike + Memory neurons
Time steps: 2
Test Accuracy: 65.18%

```

#### T=4

```
Files already downloaded and verified
Files already downloaded and verified

Loaded thresholds:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Thresholds after set_T:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Testing with Signed Spike + Memory neurons
Time steps: 4
Test Accuracy: 71.79%

```

#### T=8

```
Files already downloaded and verified
Files already downloaded and verified

Loaded thresholds:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Thresholds after set_T:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Testing with Signed Spike + Memory neurons
Time steps: 8
Test Accuracy: 75.69%

```

#### T=16

```
Files already downloaded and verified
Files already downloaded and verified

Loaded thresholds:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Thresholds after set_T:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Testing with Signed Spike + Memory neurons
Time steps: 16
Test Accuracy: 77.49%

```

#### T=32

```
Files already downloaded and verified
Files already downloaded and verified

Loaded thresholds:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Thresholds after set_T:
  layer1.2: pos_thresh=0.2222, neg_thresh=-0.2222

Testing with Signed Spike + Memory neurons
Time steps: 32
Traceback (most recent call last):
  File "main_test_signed.py", line 110, in <module>
    main()
  File "main_test_signed.py", line 104, in main
    acc = val(model, test_loader, device, args.time)
  File "/root/autodl-tmp/QCFS/utils.py", line 70, in val
    outputs = model(inputs).mean(0)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/autodl-tmp/QCFS/models/VGG.py", line 284, in forward
    out = self.layer1(x)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/container.py", line 141, in forward
    input = module(input)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 447, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 443, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution

```

---

## 关键发现

### 最佳性能

- **IF (原始)**: T=32 时达到 **77.60%**
- **SignedIF (SNM)**: T=16 时达到 **77.49%**
- **整体最佳**: IF (原始) 在 T=32 时达到 **77.60%**，比SignedIF提升 **0.11%**

### 各时间步对比分析

- **SignedIF优势时间步** (4个): T=2, 4, 8, 16
- **性能相同时间步** (1个): T=1

### 平均性能提升

- **平均精度提升**: +0.71% (SignedIF相比IF)
- **最大提升**: +1.29% (在T=4)
- **最小提升**: +0.00% (在T=1)

---

## 结论

### 性能总结

**低时间步 (T≤4)**:
- T=1: IF=58.81%, SignedIF=58.81% (+0.00%)
- T=2: IF=64.85%, SignedIF=65.18% (+0.33%)
- T=4: IF=70.50%, SignedIF=71.79% (+1.29%)

**中时间步 (T=8-16)**:
- T=8: IF=74.63%, SignedIF=75.69% (+1.06%)
- T=16: IF=76.62%, SignedIF=77.49% (+0.87%)


*报告生成时间: 2026-02-12 15:56:37*
