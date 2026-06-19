# P6M Config Update Report

## Files Updated

| File | Change |
|------|--------|
| `docs/DECISIONS.md` | +D044-D051 (8 new decisions) |
| `memory.md` | +0.5 Baseline-Lite 路线 |
| `models/baselines/__init__.py` | NEW — baseline-lite 包 |
| `models/baselines/base_baseline.py` | NEW — 统一基类 |
| `reports/P6M_baseline_lite/baseline_table_policy.md` | NEW — 表格口径 |
| `docs/EXTERNAL_BASELINE_REFERENCES.md` | NEW — 外部参考清单 |

## New Decisions

| ID | Summary |
|----|---------|
| D044 | Baseline-Lite 路线取代逐个复现外部仓库 |
| D045 | CC/Codex 可直接实现 baseline-lite |
| D046 | 2025 baseline = MLCL-lite + DLF-lite |
| D047 | 经典 baseline = TFN/LMF/MulT/MISA/SelfMM/MMIM-lite |
| D048 | CASP 不入普通 baseline (TTA) |
| D049 | 论文表格分三类: 主模型 / baseline-lite / 原论文报告值 |
| D050 | 原论文报告值标注规则 |
| D051 | 教学演示/模拟不得入论文正式表 |

## Baseline-Lite Directory

```
models/baselines/           ← 已创建 __init__.py + base_baseline.py
configs/models/baselines/   ← 已创建目录 (待填充 yaml)
configs/experiments/p6m_baseline_lite/ ← 已创建目录
scripts/train_baseline_lite.py ← 待创建
```

## Baselines Fixed This Round

| Baseline | Priority | Implementation |
|----------|:--------:|:--------------:|
| TFN-lite | P1 | Pending |
| LMF-lite | P1 | Pending |
| MulT-lite | P0 | Pending |
| MISA-lite | P1 | Pending |
| SelfMM-lite | P0 | Pending |
| MMIM-lite | P1 | Pending |
| MLCL-lite | P0 | Pending |
| DLF-lite | P0 | Pending |

## Not In This Round

DPDF-LQ, DashFusion, R3DG, CASP (TTA separate table)

## Next Steps

1. 实现 TFN-lite → unit test → 1ep smoke → 10ep run → s42 full
2. 同理推进 MulT-lite, SelfMM-lite
3. 所有结果通过 `utils/metrics.py` 统一复算
