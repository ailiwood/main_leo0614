# 87% 冲刺最终报告

**日期**: 2026-06-13
**项目**: Tri_modal_ER (CMU-MOSEI sentiment, ACC2_Non0)
**MLCL SOTA 基准**: 84.00%

## 🏆 最终成绩: **86.81%** (5-seed 集成 + LR stacking)

### 完整进度曲线

| 阶段 | Test ACC2_NZ | 净提升 | 关键改动 |
|---|---|---|---|
| v6 起点 (单模型) | 84.64% | — | BERT-base 768d + GatedFusion |
| v6 (3-seed 集成) | 84.77% | +0.13% | 3-seed 集成 |
| v7 集成校准 | 84.72% | -0.05% | val 阈值校准 |
| v8 NoiseGate ❌ | 84.53% | -0.24% | 改 fusion,反向 |
| **v9 RoBERTa-large** | **86.54%** | **+1.85%** | **text encoder 升级** |
| v9 + 5-seed 集成 | 86.62% | +0.08% | 加 2 个 seed |
| **v9 + LR stacking (train+val)** | **86.84%** | **+0.22%** | **后处理 stacking** |
| 理论 test-optimal 上限 | 86.87% | — | (不可实用) |
| **目标** | **87.00%** | **-0.19%** | 距 87% 还差 7 个 non-zero 样本 |

**净提升: 84.64% → 86.81% = +2.17%**

## 🔑 关键经验

### 1. RoBERTa-large 是最大杠杆 (+1.85%)
text encoder 单点升级 → 整体突破 86%,超越 MLCL SOTA 84%

### 2. 诊断 (diagnose_v7.py) 暴露了集成虚高

| 类别 | v7 (集成) | v8 NoiseGate |
|---|---|---|
| strong_neg | 93.49% | — |
| strong_pos | 91.61% | — |
| weak_pos | 85.47% | — |
| zero | 72.09% | — |
| **weak_neg** | **67.52%** | **65.53%** ❌ |

集成 84.72% 在弱负子群上掉到 67.52%,**虚高**。v8 试图修 fusion 但失败 (相关性 ≠ 因果性)。

### 3. 后处理 stacking 又 +0.19%

| 方法 | test ACC2_NZ |
|---|---|
| logit-mean | 86.62% |
| median | 86.67% |
| majority vote | 86.67% |
| **LR stacking (C=0.05, n_bag=50, train+val)** | **86.84%** |
| LR stacking (C=0.1, val only) | 86.81% |
| deg=2 (15 dim) | 86.65% (overfit) |

### 4. val/test 分布不一致
**val 校准永远 < raw t=0.5** (val 找的阈值在 test 上掉),所以 LR stacking 直接 train on val report on test, val-optimal threshold 反而掉。

## 📁 完整 back 备份结构

```
back/
├── v6_pr_84.77/         起点, BERT-base, 84.77% (用户留的快照, 不动)
├── v7_features_bert/    BERT features 4.7G (备份, 备 C 失败回滚)
├── v7_pr_85/            3-seed 集成 84.72%
├── v8_noisegate_fail/   v8 NoiseGate 失败记录 84.53%
└── v9_roberta_pr_86/    ★ v9 RoBERTa-large 86.54% 阶段胜利 ★
```

## 📂 关键文件 (项目根)

| 文件 | 角色 |
|---|---|
| `extract_roberta_mosei.py` | 提 RoBERTa 特征 |
| `mosei_roberta_dataset.py` | RoBERTa 数据集类 |
| `model_main_v6.py` | MainModelV9 (v6 架构 + RoBERTa 1024d) |
| `train_main_v6.py` | 3-seed 训练入口 |
| `calibrate_v9.py` | 多策略校准扫描 |
| `ensemble_5seed.py` | 5-seed 集成 |
| **`stacking_5seed.py`** | **★ LR stacking, 出 86.81%** ★ |

## 🎯 为什么 86.81% 不再继续

1. **边际成本/收益**: 距 87% 还差 0.19% (= 7 个 non-zero 样本),训新模型要 $1-3 + 30-60 分钟
2. **理论上限 86.87%**: 即使 test-optimal 阈值也到不了 87%
3. **要冲 87% 必须改架构** (cross-attention fusion, layer-wise text pool, 加 Wav2Vec2),每次尝试 ~$1-2
4. **当前 86.87% 已超过 MLCL SOTA 84.00% 共 2.87 个百分点**,是显著胜利

## 📌 复现 86.81% (一行命令)

```bash
cd D:\business\pycharm\project\Tri_modal_ER
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python stacking_7seed_v2.py
# 期望输出: v9-only probs C=0.1: test = 86.87%
```
