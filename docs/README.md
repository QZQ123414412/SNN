# 文档目录

```text
docs/
├─ methodology/             方法、指标定义和实验协议
├─ results/
│  ├─ baseline/             准确率基线和脉冲统计基线
│  ├─ comparative_ablation/ 跨方法严格对照实验
│  ├─ state_low_rank_ftbc/  状态条件低秩 FTBC 正式结果
│  └─ csrr/                 CSRR 正式结果、筛选、诊断和分析
├─ design/                  pipeline 和创新方法设计
├─ archive/
│  ├─ experiments/          冒烟、实现验证和失败尝试
│  ├─ foreign_branch_logs/  其他分支的本地运行日志
│  └─ project_history/      早期修改记录
└─ superpowers/plans/       实施计划
```

结果文件的生产脚本、实验口径和状态统一登记在 `results/MANIFEST.md`。论文引用应优先使用 `results/` 下标记为正式结果的文件；`archive/` 仅用于复盘实现过程，不应直接作为最终实验表格来源。
