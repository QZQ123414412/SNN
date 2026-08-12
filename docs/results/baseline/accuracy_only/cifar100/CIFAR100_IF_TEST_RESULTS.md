# CIFAR-100 IF模型不同时间步测试结果

## 测试信息

- **数据集**: CIFAR100
- **模型架构**: vgg16
- **权重文件**: cifar100-vgg16-l8-example
- **测试时间**: 2026-02-12 15:42:42
- **测试脚本**: `main_test.py` (原始IF模型)

---

## 测试结果

| 时间步 (T) | 测试精度 (%) | 状态 |
|-----------|-------------|------|
| T=1 | **58.81%** | ✅ 成功 |
| T=2 | **64.85%** | ✅ 成功 |
| T=4 | **70.50%** | ✅ 成功 |
| T=8 | **74.63%** | ✅ 成功 |
| T=16 | **76.62%** | ✅ 成功 |

---

## 详细输出

### T=1

```
Files already downloaded and verified
Files already downloaded and verified
58.81

```

### T=2

```
Files already downloaded and verified
Files already downloaded and verified
64.85

```

### T=4

```
Files already downloaded and verified
Files already downloaded and verified
70.5

```

### T=8

```
Files already downloaded and verified
Files already downloaded and verified
74.63

```

### T=16

```
Files already downloaded and verified
Files already downloaded and verified
76.62

```

---

## 性能趋势分析

| 时间步对比 | 精度提升 | 提升率 |
|-----------|---------|--------|
| T=1 → T=2 | +6.04% | +10.27% |
| T=2 → T=4 | +5.65% | +8.71% |
| T=4 → T=8 | +4.13% | +5.86% |
| T=8 → T=16 | +1.99% | +2.67% |

---

## 结论

- **最佳性能**: T=16 时达到 **76.62%**
- **性能范围**: 58.81% (T=1) ~ 76.62% (T=16)
- **性能提升**: 17.81% (从T=1到T=16)

*报告生成时间: 2026-02-12 15:42:42*
