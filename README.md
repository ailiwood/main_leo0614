# Tri_modal_ER — MOSEI 三模态情感识别

**目标**:MOSEI test ACC2_Non0 推到 87%
**当前 PR**: **86.62%** (RoBERTa-large 5-seed 集成)
**距目标**:0.38%

---

## 项目结构

```
Tri_modal_ER/
├── model.py                # 模型: 3 模态独立 Transformer + Gated Fusion
├── train.py                # 训练入口 (argparse + EMA + balanced sampler)
├── dataset.py              # Dataset: RoBERTa-large 1024d + CASP audio/vision
├── extract_features.py     # RoBERTa-large 1024d 特征提取
├── balanced_sampler.py     # BalancedSentimentSampler (50/50 pos/neg)
├── ensemble_5seed.py       # 5-seed 集成
│
├── data/CMU-MOSEI/         # CSV 标签源 + 预提取特征 pkl
│   ├── CMU-MOSEI-20230514T151450Z-001/CMU-MOSEI/Labels/
│   ├── train_roberta.pkl   # 16274 样本 × (50, 1024)  +  audio (64, 80)  +  vision (64, 176)
│   ├── val_roberta.pkl
│   └── test_roberta.pkl
│
├── experiments/
│   ├── v9_roberta/                # 3-seed 训练
│   └── v9_roberta_5seed/          # 5-seed 训练
│
├── back/
│   ├── v9_roberta_pr_86/          # 最终 PR 备份 (含 5 个 .pth)
│   └── mosi_revert/               # MOSI 已回退说明
│
├── logs/
│   ├── v9_train.log               # v9 训练日志
│   └── extract_roberta.log        # 特征提取日志
│
└── Tri_modal_42_v5_1.08M_79.31.pth  # 早期 v5 权重 (留作对比)
```

---

## 关键成绩

| 阶段 | Test ACC2_Non0 |
|---|---|
| Majority baseline | 71.0% |
| v5 (GloVe 旧 features) | 64.30% |
| v6 (BERT-base 768d) | 84.64% |
| v6 3-seed ensemble | 84.77% |
| **v9 RoBERTa-large 5-seed** | **86.62%** |
| **目标** | **87.00%** |

---

## 复现方法

```bash
# 1. 提取特征 (一次性, ~10 分钟)
KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python extract_features.py

# 2. 跑单模型
KMP_DUPLICATE_LIB_OK=TRUE python train.py --epochs 4 --early_stop 4 \
  --batch_size 24 --lr 5e-4 --balanced --seed 42

# 3. 跑 5-seed 集成 (生成 ensemble_summary)
KMP_DUPLICATE_LIB_OK=TRUE python ensemble_5seed.py
```

---

## 依赖

- Python 3.10+, PyTorch 2.0+, CUDA
- `transformers` (RoBERTa-large)
- 输入特征:`data/CMU-MOSEI/{train,val,test}_roberta.pkl` (~6GB)
- 外部依赖: CASP `casp_mosei.pkl` (audio/vision, 需引用项目外的路径)

---

## 详见桌面 `MOSEI_87_project/README_评审指南.md`

(那份含完整文件分类、复现命令、关键诊断、已知瓶颈、评审问题清单)
