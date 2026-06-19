# DECISIONS.md — 关键决策记录

> 项目：基于扩展 LSTM 与 Transformer 的跨模态情感分析研究  
> 用途：记录 CC/Codex 在本轮项目中的关键决策、理由和影响

---

## 2026-06-16 P0：初始扫描阶段决策

### D001：sLSTM 不可复用旧代码，必须自实现

- **决策**：`utils_models/lstm_v.py` 不可复用，sLSTM 必须纯 PyTorch 自实现
- **理由**：(1) 旧实现是 mLSTM (MatrixLSTM) 而非 sLSTM；(2) 许可证为 AGPL-3.0，不可混入主模型；(3) 接口与三模态时序编码需求不匹配
- **影响**：P2 需从论文公式从头实现 SLSTMCell + SLSTMEncoder
- **参考**：Beck et al. (2024) xLSTM paper (NeurIPS)

### D002：AWAF 需完全新建，无可复用代码

- **决策**：AWAF 模块必须完全新建
- **理由**：当前代码中无任何 AWAF 实现。V9 模型的 GatedFusion 是简单 softmax 门控，不是 AWAF。ours_model.py 的融合方式也完全不同。
- **影响**：P2 需从配置文件中描述的两段式设计从头实现 AdaptiveWeightedAttentionFusion

### D003：V9 代码路线整体作废，仅保留为参考

- **决策**：当前 `model.py`/`train.py`/`dataset.py` 等 V9 RoBERTa 路线代码不作为新主模型基础
- **理由**：V9 使用 TransformerEncoder + GatedFusion，不含 sLSTM、AWAF、CME、DEConv 等关键模块
- **影响**：P2 需重写全部核心代码
- **保留**：V9 权重和日志保留在 `experiments/` 作为历史记录

### D004：统一指标模块必须重写

- **决策**：`utils_tools/metricsTop.py` 不可作为统一指标实现
- **理由**：(1) 缺少 ACC2_Non0、F1_Non0、ACC7；(2) Has0_acc_2 混入了 zero label
- **影响**：P2 需新建 `utils/metrics.py` 为唯一指标实现

### D005：论文初稿 P0 不做深度修改

- **决策**：P0 仅确认论文初稿文件存在且可打开，不深度阅读或修改
- **理由**：论文实质性修改应在实验完成后的 P9 进行
- **风险**：初稿（2025-03-18）可能包含旧结论、旧指标、旧架构描述

---

## 2026-06-16 P1：特征链路裁决决策

### D006：TMDC MOSI 特征确认为真实可用

- **决策**：TMDC MOSI 特征（2199样本, DeBERTa+wav2vec+CLIP→1024d）确认为真实提取，100%标签对齐，作为 P2 MOSI smoke test 数据源
- **理由**：(1) 每个 npy 文件大小不相等（非模拟等大模式）；(2) 标签与官方 label.csv 逐条 100% 对齐；(3) 提取脚本逻辑完整可追溯
- **影响**：P2 可直接使用 TMDC MOSI 特征启动 smoke test

### D007：放弃所有旧特征链路

- **决策**：simulated MOSI、Tri_modal_ER MOSEI、V9 RoBERTa MOSEI、CASP pkl 全部不采用
- **理由**：(1) simulated 文件名风险；(2) Tri_modal_ER MOSEI 标签为 0-1 二值非标准回归，特征维度不一致；(3) V9 路线已排除；(4) CASP pkl 在外部不可控路径
- **影响**：MOSEI 需要自建特征提取链路

### D008：主推 TMDC 自建方案，MLCL 标准特征为补充

- **决策**：特征方案裁决为方案 B（TMDC 自建），方案 A（MLCL 标准特征）作为补充（如可获取）
- **理由**：(1) TMDC MOSI 已就绪；(2) MLCL 特征可用性无法通过网络确认；(3) 即使 MLCL 特征可用，也可用于 MLCL baseline 而非主模型
- **影响**：主模型+MOSEI 特征链路需在 P2 期间自建

### D009：推荐 Python 3.10

- **决策**：mme_xlstm 使用 Python 3.10
- **理由**：(1) PyTorch 2.3.0 对 3.10 支持最成熟；(2) 与 CASP (≥3.8) 兼容；(3) 比 3.11 在 Windows CUDA 上更稳定
- **影响**：P2 前创建 mme_xlstm (Python 3.10 + PyTorch 2.3.0+cu118)

### D010：MOSEI 视觉特征方案待定

- **决策**：MOSEI 视觉特征提取方案暂不固定，需用户确认视频来源
- **理由**：当前无 MOSEI 原始视频帧可用
- **影响**：P2 可能先以 Text+Audio 双模态启动 MOSEI，Vision 补齐后升级为三模态

---

## 2026-06-16 P2：最低主干实现阶段决策

### D011：授权下载并审计 MLCL 仓库/特征

- **决策**：允许下载 MLCL 到 `external/MLCL/`，特征下载到 `external/MLCL_features/`，仅作外部 baseline 资源管理，不混入主模型
- **引用**：Zhuang et al. (2025), IEEE TMM, 27, 9044–9058, DOI: 10.1109/TMM.2025.3613116
- **限制**：MLCL 代码和数据不入主模型正式训练，仅用于 P6 baseline 准备

### D012：MOSEI 三模态特征必须完整

- **决策**：MOSEI 视觉特征必须来自原始视频/帧提取或公开预提取真实视觉特征
- **禁止**：不得用 Text+Audio 双模态冒充三模态正式实验结果
- **P2 策略**：P2 仅用 MOSI 三模态 smoke test，MOSEI 做资源审计

### D013：创建 mme_xlstm，Python 3.10

- **决策**：P2 前创建 `E:\Anaconda3\envs\mme_xlstm`，Python 3.10 + PyTorch 2.3.0+cu118
- **原则**：旧 mme 只作对照不污染。P6 MLCL 如需专属环境建 mme_mlcl

### D014：TMDC MOSI vision 按真实情况标注

- **决策**：标注为 "CLIP-ViT-B/32 + fixed random projection to 1024d"，不写 MANet
- **理由**：manet_UTT 仅为历史目录命名

### D015：clip-level 单向量科学性边界

- **决策**：当前 MOSI TMDC 为 T=1 clip-level smoke test 数据
- **禁止**：不得将 T=1 smoke test 结果写成 sLSTM 时序建模有效性的论文证据
- **要求**：dataset 和模型接口兼容未来 [T,D] 序列输入

---

## 2026-06-16 P2.1：质量闸门决策

### D016：PyTorch nightly 为 Blackwell 兼容性临时方案

- **决策**：当前使用 PyTorch 2.12.0.dev20260408+cu128 (nightly)，因 RTX 5070 Ti (Blackwell sm_120) 在 stable PyTorch 2.6.0 预编译二进制中缺少 sm_120 kernel
- **限制**：torchvision/torchaudio 与 nightly torch 版本不完全匹配，需 import 验证
- **计划**：P4 正式训练前如 stable 版本已支持 Blackwell，重新评估是否切换

### D017：sLSTM mask padding 状态冻结

- **决策**：padding 时间步的 h/c/n/m 状态必须恢复为上一时刻值
- **实现**：`SLSTMEncoder._unroll()` 中逐样本替换无效状态
- **验证**：padding 噪声不影响 pooled（max diff=0.00e+00）

### D019：P4A 结果因 test leakage 不作为论文正式结果

- **决策**：P4A MOSI 75.25% 不可用于论文主表
- **理由**：best_epoch 由 test ACC2_Non0 选择，inflated ~1.5%
- **影响**：P4S 修复了 strict protocol，后续结果更可靠

### D020：TMDC-v1 T=1 不作为论文主特征

- **决策**：TMDC-v1 仅作为 pooled baseline / 工程基准
- **理由**：T=1 不能支撑 sLSTM 时序建模有效性主张
- **替代**：strong sequence features (DeBERTa token-level + wav2vec2 frame-level)

### D021：MLCL weak sequence 不作为主特征

- **决策**：MLCL standard features (GloVe+COVAREP+FACET) 不采用
- **理由**：弱特征表现远低于强预训练特征（44.9% vs 75.25%）

### D022：Strong sequence 已构建但 C0 表现不足，需要架构升级

- **决策**：进入 AWAF-Seq + Cross-modal Transformer 架构升级阶段
- **理由**：C0 strong seq mean=72.7%，低于 T=1 的 75.25%
- **影响**：需要网页版 AI 审阅代码并设计升级方案

### D023：当前 strong sequence 中 vision 仍为 T=1

- **决策**：暂用 CLIP .pt 单向量，后续需帧级视觉序列
- **影响**：视觉 T=1 可能是性能瓶颈之一

### D024：旧实验结果和旧代码可删除，保留关键报告和决策记录

- **决策**：执行 P4U 清理，删除旧 V9 代码、旧 outputs、缓存
- **理由**：用户授权删除，GitHub 上传准备
- **保留**：reports 总结、HANDOFF、memory、DECISIONS、经验总结、核心代码

### D025：进入 GitHub 上传与网页版 AI 代码审阅阶段

- **决策**：当前仓库清理后上传 GitHub，由网页版 AI 审阅
- **目标**：设计 AWAF-Seq + Cross-modal Transformer 架构升级

### D026：GitHub 必须包含 data/*.py 和 configs/data/*.yaml

- **决策**：.gitignore 不得排除 data/*.py，特征文件通过 data/features*/ 规则排除
- **修复**：P4U.1 将 `data/` 规则拆分为 data/features*/、data/raw/ 等

### D027：strict_trainer evaluation 必须传入全部三模态 mask

- **决策**：evaluate() 中 text/audio/vision mask 必须全部传入模型 forward
- **影响**：P4T.1 之前的 eval 结果可能因 audio/vision mask 缺失产生偏差

### D028：C0 strong sequence 旧结果在 mask bug 修复前不作为架构判断最终依据

- **决策**：P4T.1 C0 strong seq 结果（72.7% mean）是在 mask 未完全正确时得到的
- **影响**：架构判断应基于 mask 修复后的重新评估

### D029：strong sequence 配置固化为 text_dim=1024, audio_dim=768, vision_dim=1024

- **决策**：configs/data/mosi_strong_sequence.yaml + configs/models/ours_c0_strong_sequence.yaml

### D030：P4U.1 只做工程补齐和 bug 修复，不做架构升级

- **决策**：本阶段修复了 .gitignore、evaluate mask、check_val API 和配置固化
- **影响**：下一步由网页版 AI 基于补齐仓库给出架构升级方案

### D018：AWAF context 排除自身 + dropout 逐样本

- **决策1**：context attention 默认排除自身模态（q 对 other 2 modals 做 attention）
- **决策2**：modality dropout 改为逐样本检查，确保每个样本至少保留一个模态

---

## 2026-06-17 P5C：主模型大修决策

### D031：主模型大修 — DeepText-xLSTM-AWAF Residual

- **决策**：从"三路 sLSTM 平权融合 + AWAF"切换为"DeepText 主判别 + xLSTM 残差增强 + AWAF 残差修正"
- **理由**：(1) P5B 证明 text-side sLSTM 为因果反作用（76.2% → 80.2%）；(2) DeepMLP text-only (80.2%) 已超过旧多模态模型 (78.8%)；(3) 多模态融合当前对性能有害而非有益
- **新架构**：Text=DeepMLP(no sLSTM), Audio/Vision=sLSTM, AWAF=residual correction generator, Final=text_base + λ*delta
- **影响**：P5C 全部代码按新架构重写；消融项重新定义；论文第3/4/5章方法描述需相应修改
- **xLSTM 位置**：仅用于 audio/vision 时序残差增强
- **AWAF 位置**：残差修正权重生成器
- **详细文档**：`docs/主模型大修决策06171310.md`

---

## 2026-06-18 P5G-P6D：强特征升级与文本骨干突破

### D045：frozen audio/vision 大模型特征升级失败
- wav2vec2-large (1024d): 80.18% (-2.29%) — 路线关闭
- CLIP ViT-L/14 (1024d): 79.12% (-3.35%) — 路线关闭
- 根因：MOSI 1284训练样本不足以支撑1024d特征

### D046：RoBERTa-large fine-tuned 文本骨干是真正突破
- RoBERTa text-only MOSI: 85.37% (+5.17% vs frozen DeBERTa)
- RoBERTa text-only MOSEI: 88.13% (3ep, 超84%目标)
- 小数据集上 fine-tune > frozen features

### D047：TextFT-xLSTM-AWAF 架构确认三模态
- Text: RoBERTa (NO sLSTM) + Audio: sLSTM + Vision: sLSTM + AWAF
- 满足导师 xLSTM + 自适应注意力机制要求

### D048：MMSDK 彻底无法安装
- PyPI无mmsdk包, GitHub仓库A2Zadeh/CMU-MultimodalSDK不存在
- MOSEI数据通过CSV直接可用，无需SDK

### D049：P5E V2 (82.17%) 为当前最优多模态结果
- 所有强特征升级路线均已证伪
- P5E V2 可作为正式实验候选

---

## 2026-06-17 P5D：残差稳定性验证与性能冲刺

### D032：P5C 路线验证成功但性能不达标

- **决策**：P5C DeepText-xLSTM-AWAF Residual 路线验证通过（2-seed mean 81.25%），但不能冻结模型
- **理由**：81.25% 虽超过 text-only 80.2%，但距离论文要求的 83%+ 仍有差距
- **影响**：需要 P5D 性能冲刺（ConditionalGate, Two-stage, Reweight）

### D033：P5D 先补 seed=2024，再进入性能冲刺

- **决策**：seed=2024 补跑确认稳定性（81.40%，σ=0.15%），然后进入性能冲刺
- **理由**：2-seed mean 81.25% ≥ 81.0%，进入冲刺阶段；< 82.5%，不补 3rd seed

### D034：探索阶段默认不保留 last/epoch checkpoints

- **决策**：P5D 起默认只保存 best_model.pth 一份，不保存 last_model.pth
- **理由**：当前 34 个 pth 文件占用 537MB，大多数是探索阶段遗留

### D035：ConditionalResidualGate 用于避免 residual 无条件破坏强文本预测

- **决策**：实现 ConditionalResidualGate，让 residual 在文本高置信度时自动降低修正幅度
- **理由**：Residual analysis 显示 strong_neg 有 53.3% 被破坏（残差过度修正）

### D036：Two-stage training 用于减少 text branch 与 residual branch 干扰

- **决策**：实现两阶段训练（Stage1: text-only, Stage2: residual, Stage3: optional joint）
- **理由**：当前 joint training 可能导致 text base 和 residual 相互干扰

### D037：weak_neg reweight 针对弱负样本瓶颈

- **决策**：在 losses.py 中实现 sample reweight 和 focal sign loss
- **理由**：Residual analysis 发现 weak_neg 是改进最大群组（+1.3% sign），值得针对性优化

### D038：P5D 仍不进入 MOSEI 正式训练

- **决策**：MOSEI 保持维度兼容状态，不启动正式训练
- **理由**：MMSDK 安装仍 blocked，且主模型尚未冻结

---

## 2026-06-19 P6K/P6L：conservative mainline + MOSEI 启动

### D039：MOSI 主模型锁定为 text_audio conservative (T+A)

- **决策**：MOSI 主模型路线锁定为 text_audio conservative（T+A 双模态），不再将 vision 作为 MOSI mainline 继续投入资源
- **理由**：(1) P6K s42=88.72%, s2024=86.89%, 2-seed mean=87.8% 均超过 P6H inline 86.43%；(2) text_audio 为项目最高 MOSI test ACC2；(3) vision 单模态仅 65.28% ACC2，AV 组合不优于纯 audio
- **限制**：该结论仅限 MOSI。MOSEI 是否保留 vision 需由 MOSEI text_audio 结果和候选实验决定，不得外推

### D040：Residual/delta 非主要提升来源

- **决策**：论文不得声称 residual delta 修正模块带来主要提升
- **理由**：(1) 6 轮实验 (P6H+R1-R4+P6I) 全部 residual_gain=0.00%；(2) P6J delta_sign_correct_rate≈48.69%（≈随机）；(3) P6K 88.72% 来自 text-audio co-training，非 delta 修正
- **影响**：若有 residual/delta 代码，只能作为工程实现细节或待裁决候选

### D041：2025 baseline 精简为 3 个

- **决策**：本轮 2025+ baseline 只保留 MLCL、DLF、DPDF-LQ
- **理由**：DashFusion/R3DG 本轮不测试（避免发散）；CASP 是 TTA 方法，不混入普通 baseline 主表
- **影响**：baseline 表分为 2025 modern (MLCL/DLF/DPDF-LQ) 和 classic (TFN/LMF/MulT/MISA/Self-MM/MMIM via MMSA)

### D042：MOSEI 主模型未锁定

- **决策**：MOSEI 必须重新跑 text_audio conservative，根据真实结果决定是否需要 T+A+V 或 vision 候选
- **理由**：MOSI 的 vision 结论不得外推到 MOSEI

### D043：第 5 章仍不能写确定性结论

- **决策**：只有完成 MOSEI + baseline + 消融并统一指标复算后，才允许进入论文结果章节

---

## 2026-06-19 P6M：Baseline-Lite 路线固定

### D044：Baseline 路线调整为 baseline-lite

- **决策**：baseline 路线从"逐个复现外部原仓库"调整为本项目内部 baseline-lite 轻量复现
- **理由**：外部仓库依赖复杂、环境冲突严重、数据格式不统一。手搓 baseline-lite 可在统一框架内做对照实验，规避外部依赖风险
- **影响**：baseline 模型在 `models/baselines/` 中实现，统一接入本项目 dataset/trainer/metrics/registry

### D045：CC/Codex 可直接实现 baseline-lite

- **决策**：CC/Codex 可在 `models/baselines/` 中直接实现简化版 baseline 模型
- **理由**：本项目已具备统一数据/训练/评估框架，无需依赖外部仓库
- **影响**：外部仓库仅作为结构参考和引文来源

### D046：2025+ baseline 固定为 MLCL-lite + DLF-lite

- **决策**：DPDF-LQ、DashFusion、R3DG 暂不进入本轮 baseline
- **理由**：baseline 范围收敛，避免发散

### D047：经典 baseline 固定为 6 个

- **决策**：经典 baseline = TFN-lite, LMF-lite, MulT-lite, MISA-lite, SelfMM-lite, MMIM-lite

### D048：CASP 不入普通 baseline 主列

- **决策**：CASP 是 TTA 方法，不进入本轮普通 baseline 主表，仅保留为后续 TTA 附表候选

### D049：论文表格三类结果

- **决策**：论文表格必须区分 (A) 本项目主模型真实训练结果、(B) baseline-lite 轻量复现结果、(C) 原论文报告值

### D050：原论文报告值标注规则

- **决策**：原论文报告值必须标注 "Reported by original paper"，不得加随机波动，不得写成"本项目复现"，不得与真实训练结果混列

### D051：教学演示/模拟表规则

- **决策**：教学演示、占位表、模拟表只能标注为"教学演示/模拟"，不得进入论文正式结果表，credibility=D
