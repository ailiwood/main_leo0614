# P0 初始文件结构记录

> 生成时间：2026-06-16  
> 用途：记录 P0 阶段的完整文件结构，为 P5 文件重构和目录重构作准备  
> 状态：只记录，不移动、不删除、不修改

---

## 一、当前目录结构总览

```
new_code/
├── .git/                           # Git 仓库
├── .gitignore                      # Git ignore 规则
├── .idea/                          # PyCharm IDE 配置
│
├── [配置文件 — Markdown]
│   ├── claude.md                   # ★ 最高执行文件
│   ├── workflow.md                 # ★ 工作流程 P0-P10
│   ├── 实验设计.md                  # ★ 实验设计
│   ├── 主模型实现规划.md             # ★ 主模型实现技术依据
│   ├── 代码重构建议.md              # ★ 代码重构规范
│   ├── 最新学术进展与代码推荐.md     # 文献与代码推荐
│   ├── memory.md                   # 阶段记忆
│   ├── 经验总结.md                  # 可复用经验
│   ├── project_instruction.md      # 网页版 AI 项目提示词
│   ├── CC启动提示词_两阶段版.md     # CC 两阶段启动提示词
│   └── 0613实验FINAL_REPORT.md     # ⚠ 旧报告 (v9 RoBERTa 86.81%)
│
├── [核心代码 — V9 RoBERTa 路线]
│   ├── model.py                    # MainModelV9: Transformer+GatedFusion
│   ├── train.py                    # train_main_v6.py: V9 训练入口
│   ├── dataset.py                  # MOSEIRobertaDataset: RoBERTa pkl 加载
│   ├── extract_features.py         # RoBERTa-large 特征提取脚本
│   ├── balanced_sampler.py         # 平衡正负采样器
│   └── ensemble_5seed.py           # 5-seed 集成脚本
│
├── [历史模型实现 — utils_models/]
│   ├── lstm_v.py                   # ⚠ NX-AI mLSTM, AGPL-3.0
│   ├── lstm_v_utils.py             # lstm_v.py 工具函数
│   ├── ours_model.py               # rob_d2v_MATF: 旧主模型
│   ├── DEConv.py                   # DEConv_2: 动态卷积
│   ├── attention_encoder.py        # vision_xLSTM, CMELayer
│   └── transformer.py              # selfTransformer
│
├── [工具模块 — utils_tools/]
│   ├── metricsTop.py               # ⚠ 不完备的指标模块
│   ├── data_loader.py              # 旧数据加载器
│   └── tools.py                    # 工具函数
│
├── [训练引擎 — utils_train/]
│   └── en_train.py                 # 旧模型训练引擎
│
├── [特征提取链路 — tmdc_adapter/]
│   ├── 00_verify_inputs.py         # 输入验证
│   ├── 10_extract_audio_wav2vec.py # wav2vec 音频特征提取
│   ├── 10b_extract_audio_wav2vec_lv60.py
│   ├── 11_extract_text_deberta.py  # DeBERTa 文本特征提取
│   ├── 11b_extract_text_deberta_v3.py
│   ├── 12_emit_mosi_vision_from_frames.py
│   ├── 12b_extract_visual_clip.py  # CLIP 视觉特征提取
│   ├── 20_build_mosi_pkl.py        # MOSI pkl 构建
│   ├── 30_emit_to_gcnet_datasets.py
│   ├── features/                   # 特征缓存
│   │   ├── mosi/                   # MOSI TMDC 特征 (已提取)
│   │   │   ├── deberta-large-4-UTT/
│   │   │   ├── manet_UTT/
│   │   │   └── wav2vec-large-c-UTT/
│   │   ├── mosei/                  # MOSEI TMDC 特征 (目录为空⚠)
│   │   │   ├── deberta-large-4-UTT/
│   │   │   ├── manet_UTT/
│   │   │   └── wav2vec-large-c-UTT/
│   │   ├── mosei_pkls/             # (空目录)
│   │   └── mosi_pkls/              # CMUMOSI_features_raw_2way.pkl
│   ├── hf_cache/                   # HuggingFace 模型缓存
│   │   └── hub/models--.../        # wav2vec2, deberta, clip 缓存
│   └── logs/                       # TMDC 提取日志
│
├── [实验产物 — experiments/]
│   ├── v9_roberta/                 # V9 3-seed 权重 (seeds 42/100/2024)
│   │   ├── best_seed42.pth         # 23MB
│   │   ├── best_seed100.pth        # 23MB
│   │   ├── best_seed2024.pth       # 23MB
│   │   ├── calibration_results.json
│   │   └── ensemble_summary.json
│   └── v9_roberta_5seed/           # V9 追加 2-seed 权重 (seeds 7/1234)
│       ├── best_seed7.pth          # 23MB
│       ├── best_seed1234.pth       # 23MB
│       ├── ensemble_5seed_summary.json
│       └── ensemble_summary.json
│
├── [数据 — data/]
│   ├── mosi/                       # CMU-MOSI 原始数据
│   │   ├── label.csv               # 标签文件
│   │   ├── Frames/                 # 93 个视频帧目录 (JPG)
│   │   ├── Raw/                    # 93 个原始视频目录 (MP4)
│   │   └── wav/                    # 93 个音频目录 (WAV)
│   └── CMU-MOSEI/                  # CMU-MOSEI 原始数据
│       └── CMU-MOSEI-.../CMU-MOSEI/
│           ├── Audio_chunk/        # 音频分块 (Train/Val/Test)
│           ├── Labels/             # 标签 CSV (train/val/test)
│           ├── Test_original/      # 原始测试集
│           └── Val_original/       # 原始验证集
│
├── [日志 — logs/]
│   ├── extract_roberta.log         # RoBERTa 特征提取日志
│   └── v9_train.log                # V9 训练日志
│
├── [旧结果 — results_20260307/]
│   ├── 接口测试报告.md
│   ├── 数据集格式分析报告.md
│   ├── MOSI_MOSEI_消融实验汇总.md
│   ├── 代码修改说明和结果总结.md
│   ├── ablation_results_comparison.pdf/png
│   ├── xlstm_contribution.pdf/png
│   ├── ablations_mosei_v3_results.json
│   └── ablations_mosi_results.json
│
├── [论文相关]
│   ├── 000论文初稿03181845.docx     # ★ 论文初稿 (3MB, 2025-03-18)
│   ├── ~$0论文初稿03181845.docx    # Word 临时文件
│   └── 创新点对应论文/              # 参考论文 PDF
│       ├── 创新点参考论文-1.pdf     # 26.9MB
│       └── 创新点参考论文xlstm.pdf  # 1.8MB
│
├── [新建目录 — P0 产物]
│   ├── reports/                    # P0 报告 (本次新建)
│   └── docs/                       # 文档 (本次新建)
│
└── [缺失的目标目录 — 待 P2-P5 创建]
    (configs/, models/, engine/, utils/, scripts/, outputs/, external/, third_party/, baselines/, archives/, env/)
```

---

## 二、文件分类统计

| 类别 | 数量 | 说明 |
|------|------|------|
| Markdown 配置文件 | 11 | 含 0613旧报告 |
| Python 代码文件 | ~25 | 含 tmdc_adapter 脚本 |
| 模型权重 (.pth) | 5 | 每文件 ~23MB，总计 ~115MB |
| JSON 结果文件 | 5 | 实验配置与指标 |
| 原始数据 (音频/视频/帧) | 数千 | MOSI 93视频 + MOSEI Audio_chunk |
| 标签 CSV | 6 | MOSI 1个 + MOSEI 5个 |
| TMDC 特征 npy | 上千 | MOSI 已提取，MOSEI 未提取 |
| PDF 参考论文 | 2 | 创新点参考论文 |
| 论文初稿 docx | 1 | 3MB |
| 旧结果图像 | 4 | PNG/PDF |

---

## 三、可疑重复/过时文件

| 路径 | 问题 | 建议动作 |
|------|------|----------|
| `0613实验FINAL_REPORT.md` | 旧 v9 报告，不可入论文 | P10 归档 |
| `results_20260307/` | 第一轮旧结果 | P10 归档 |
| `model.py` | V9 路线，非 xLSTM-Fusion | 保留为参考，P2 重写 |
| `train.py` | V9 训练入口，硬编码旧路径 | 保留为参考，P2 重写 |
| `dataset.py` | V9 数据集，硬编码旧路径 | 保留为参考，P2 重写 |
| `~$0论文初稿03181845.docx` | Word 临时锁文件 | 可安全忽略 |
| `experiments/v9_roberta*/` | V9 权重，非主模型 | 保留为参考 |
| `logs/extract_roberta.log` | V9 特征提取日志 | 保留为记录 |

---

## 四、不应直接采用的旧产物

| 产物 | 原因 | 状态 |
|------|------|------|
| 旧主模型架构图 | CLAUDE.md 明确作废 | 已作废 |
| 旧 HANDOFF | 已删除 | 不存在 |
| V9 实验结果 (86.81%) | 不含 xLSTM/AWAF/CME/DEConv | 仅参考，不入论文主表 |
| results_20260307 消融结果 | 第一轮产物，指标口径未统一 | 仅参考，不入论文主表 |

---

## 五、缺失的关键组件（需 P2-P5 创建）

- [ ] `utils/metrics.py` — 统一指标实现 (ACC2_Non0/F1_Non0/MAE/Corr/ACC7/Has0)
- [ ] `models/encoders/slstm.py` — sLSTM 时序编码器
- [ ] `models/fusion/awaf.py` — AWAF 融合模块
- [ ] `models/heads.py` — 多任务预测头
- [ ] `models/ours_xlstm_fusion.py` — 主模型
- [ ] `engine/trainer.py` — 训练器
- [ ] `engine/evaluator.py` — 评估器
- [ ] `engine/registry.py` — 模型注册
- [ ] `scripts/train_main.py` — 统一训练入口
- [ ] `scripts/eval_main.py` — 统一评估入口
- [ ] `configs/` — 配置目录
- [ ] `data/dataset.py` — 统一数据集
- [ ] `data/README_data.md` — 数据说明
- [ ] `docs/FILE_STRUCTURE.md` — 文件结构说明
- [ ] `reports/experiment_registry.csv` — 实验登记表
- [ ] `env/` — 环境配置文件
