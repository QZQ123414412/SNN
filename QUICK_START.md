# SNM Signed Spike + Memory 快速开始指南

## 🚀 快速测试

### 测试SNM模型（推荐）
```bash
# 高精度模式 (T=32, 精度~77%)
python main_test_signed.py -data=cifar100 -arch=vgg16_signed -id=cifar100-vgg16-l8-example -T=32 -dev=0

# 平衡模式 (T=16, 精度~72%)
python main_test_signed.py -data=cifar100 -arch=vgg16_signed -id=cifar100-vgg16-l8-example -T=16 -dev=0
```

### 测试原始IF模型（对比）
```bash
python main_test.py -data=cifar100 -arch=vgg16 -id=cifar100-vgg16-l8-example -T=4 -dev=0
```

## 📊 性能对比

| 模型 | 时间步 | 精度 | 速度 |
|------|--------|------|------|
| IF (原始) | T=4 | 70.55% | ⚡⚡⚡ 快 |
| SignedIF (SNM) | T=16 | **72.22%** | ⚡⚡ 中等 |
| SignedIF (SNM) | T=32 | **77.03%** | ⚡ 较慢 |

## ⚙️ 参数说明

```bash
python main_test_signed.py \
    -data=<dataset>       # cifar10/cifar100/imagenet
    -arch=vgg16_signed    # 使用SignedIF模型
    -id=<checkpoint>      # checkpoint文件名（不含.pth）
    -T=<timesteps>        # 推荐16或32
    -dev=<gpu_id>         # GPU编号，如0
```

## ✅ 核心优势

1. **更高精度**: T=32时比IF提升6.48% (70.55% → 77.03%)
2. **双向脉冲**: 支持正负脉冲，信息表达更精确
3. **Memory机制**: 累积发送记录，确保合理的脉冲模式
4. **兼容性好**: 可直接加载IF模型的checkpoint

## ⚠️ 重要提示

1. **时间步要求**: 建议T≥16，T<8精度会很低
2. **不要覆盖阈值**: 使用checkpoint中加载的阈值，不要设置`--thresh`参数
3. **计算成本**: 更多时间步=更高精度但更慢

## 📁 修改的文件

- `models/layer.py` - 新增SignedIF类
- `models/VGG.py` - 新增VGG_Signed类和vgg16_signed()函数
- `models/__init__.py` - 添加vgg16_signed模型选项
- `main_test_signed.py` - SNM专用测试脚本
- `utils.py` - 导入SignedIF

## 🔬 技术细节

### SNM神经元动力学
```python
for each timestep:
    mem += input                           # 累积输入
    pos_spike = (mem >= thresh) * thresh   # 正脉冲
    neg_spike = (mem <= -thresh) * (-thresh) * (transmitted > 0)  # 负脉冲
    spike = pos_spike + neg_spike          # 合并
    mem -= spike                           # 重置
    transmitted += spike                   # 更新记忆
```

### 关键特性
- ✅ Signed Spike: 正负双向脉冲
- ✅ Memory: transmitted变量跟踪已发脉冲
- ✅ 限制机制: 负脉冲仅在发出正脉冲后才能发放
- ✅ GPU加速: 完全支持CUDA

## 📚 更多信息

- 详细测试结果: `TEST_RESULTS.md`
- 集成说明: `SNM_Integration_README.md`
- 修改总结: `MODIFICATIONS_SUMMARY.md`

## 💡 常见问题

**Q: 为什么T=4精度很低？**
A: SignedIF需要更多时间步来发挥双向脉冲的优势，建议T≥16

**Q: 能直接用IF的checkpoint吗？**
A: 可以！会自动转换thresh参数并生成neg_thresh

**Q: 如何选择时间步？**
A: 
- T=16: 平衡精度和速度 (72.22%)
- T=32: 最高精度 (77.03%)
- T≥64: 可能进一步提升，但更慢

**Q: 需要重新训练吗？**
A: 不需要！可直接测试。但重新训练可能获得更好效果。

---

🎉 **开始使用SNM的Signed Spike + Memory机制吧！**
