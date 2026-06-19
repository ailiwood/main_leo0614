# P6M Baseline Table Policy

## 论文表格三类结果

### A. 本项目主模型结果 (Primary Model)

- 由本项目代码真实训练
- 有 config、command、train.log、predictions_test.csv、result.json
- 用 `utils/metrics.py` 统一复算
- Credibility: A
- 可进入论文主结果表

### B. 本项目 Baseline-Lite 结果 (Our Lightweight Reimplementation)

- 由 `models/baselines/` 中的轻量复现模型真实训练
- 可写为 "Our lightweight reimplementation"
- 如果结构简化，必须脚注说明
- Credibility: A or B
- 可作为本项目对照实验结果
- **不是原论文官方复现**

### C. 原论文报告值 (Reported by Original Paper)

- 来自原论文或官方仓库
- 必须标注 "Reported by original paper / 原论文报告值"
- 必须给出引用来源 (DOI, paper title, venue, year)
- Credibility: C
- 不得加随机波动
- 不得写成"本项目复现"
- 无 per-sample prediction 时不得声称统一指标复算
- 作为独立引用列，不与 A/B 混列

### D. 教学演示/模拟 (Excluded from Paper)

- Credibility: D
- 只能标注为"教学演示/模拟"
- 不得进入论文正式结果表

## Baseline 固定清单

### 经典 Baseline (Lite)

| Model | Status | Priority |
|-------|:------:|:--------:|
| TFN-lite | Pending | P1 |
| LMF-lite | Pending | P1 |
| MulT-lite | Pending | P0 |
| MISA-lite | Pending | P1 |
| SelfMM-lite | Pending | P0 |
| MMIM-lite | Pending | P1 |

### 2025 Baseline (Lite)

| Model | Status | Priority |
|-------|:------:|:--------:|
| MLCL-lite | Pending | P0 |
| DLF-lite | Pending | P0 |

### Not In This Round

| Model | Reason |
|-------|--------|
| CASP | TTA method, separate table |
| DPDF-LQ | Deferred |
| DashFusion | Deferred |
| R3DG | Deferred |
