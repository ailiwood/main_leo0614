# HANDOFF_PHASE_02.md

> 阶段：P2 最低主干实现  
> 生成时间：2026-06-16  
> 状态：✅ 核心完成，⚠️ GPU smoke test 待 CUDA 下载完成后验证

---

## 1. 本阶段完成

- ✅ 决策记录 D011-D015
- ✅ mme_xlstm 环境创建与配置 (Python 3.10 + PyTorch 2.5.1+cu124)
- ✅ 核心代码实现（10 个模块文件，~2200 行新代码）
- ✅ 随机张量 forward test 全部通过（5/5 测试）
- ✅ MOSI TMDC 真实数据 1 epoch smoke test 通过
- ✅ AWAF sum(w)=1 验证通过 (max_dev=1.19e-7)
- ✅ 所有产物正确保存

## 2. 环境创建结果

| 项目 | 值 |
|------|-----|
| 环境 | mme_xlstm @ E:\Anaconda3\envs\mme_xlstm |
| Python | 3.10.20 |
| PyTorch | **2.12.0.dev20260408+cu128 (GPU版, nightly)** ⚠ 临时方案：因 RTX 5070 Ti Blackwell sm_120 兼容性 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti (16GB) ✅ |
| Transformers | 4.34.1 |
| torchvision | 0.21.0+cu124 (⚠ 与 nightly torch 版本不完全匹配，但 import 可用) |
| torchaudio | 2.6.0+cu124 (⚠ 同上) |
| 模型参数量 | 3,156,748 (~3.16M) |

## 3. 核心代码新增/修改（23 个文件）

### 新实现的模块

| 文件 | 行数 | 功能 |
|------|:--:|------|
| `utils/metrics.py` | ~200 | 7 项统一指标 (ACC2_Non0/F1_Non0/MAE/Corr/ACC2_Has0/F1_Has0/ACC7) |
| `utils/seed.py` | ~30 | 随机种子统一控制 |
| `models/encoders/slstm.py` | ~500 | SLSTMCell + SLSTMEncoder (纯PyTorch, 指数门/归一化/稳定器/mask/NaN防护) |
| `models/fusion/awaf.py` | ~400 | AdaptiveWeightedAttentionFusion (7融合模式+modality_dropout+CSV导出) |
| `models/heads.py` | ~90 | RegressionHead + ClassificationHead + UnimodalHeads |
| `models/ours_xlstm_fusion.py` | ~250 | 主模型 (三路 sLSTM + AWAF + 多任务头) |
| `data/dataset.py` | ~160 | TMDCMOSIDataset (TMDC 7-tuple 加载, 兼容 T≥1) |
| `engine/trainer.py` | ~250 | 训练器 (多任务loss+统一metrics+产物保存) |
| `configs/default.yaml` | ~65 | 集中配置 |
| `configs/models/ours_backbone.yaml` | ~30 | 主模型配置 |
| `scripts/smoke_forward.py` | ~230 | 随机张量全模块验证 |
| `scripts/train_main.py` | ~200 | 统一训练入口 |
| `env/install_commands.md` | ~70 | 环境安装命令记录 |

### 更新的文件

| 文件 | 操作 |
|------|------|
| `docs/DECISIONS.md` | 追加 D011-D015 |
| `data/README_data.md` | 修正 Vision 标注为 CLIP-ViT-B/32，添加 T=1 科学性边界说明 |
| `memory.md` | 追加 P2 记录 |
| `reports/P2_main_model_smoke.md` | 新建 |
| `reports/P2_environment_setup.md` | 新建 |

## 4. 随机张量测试结果

| # | 测试 | 结果 |
|---|------|:--:|
| 1 | Metrics (7指标边界情况) | ✅ |
| 2 | SLSTMCell (单步+多步+NaN防护) | ✅ |
| 3 | SLSTMEncoder (pooling/T=1/bidirectional) | ✅ |
| 4 | AWAF (7模式+sum(w)=1+modality_dropout) | ✅ |
| 5 | MainModel (T=1 + T=5 + backward) | ✅ |

## 5. MOSI Smoke Test 结果

| 指标 | 值 | 说明 |
|------|-----|------|
| Train Loss | 1.7555 | 有限 |
| MAE | 1.628 | 1 epoch |
| Corr | 0.106 | 1 epoch |
| ACC2_Non0 | 42.68% | 1 epoch, 不视为论文结果 |
| AWAF sum(w) max_dev | 1.19e-07 | ✅ |
| AWAF weights | [t=0.223, a=0.207, v=0.570] | Vision 在 T=1 下占主导 |

## 6. AWAF 权重检查

- ✅ sum(w) ≈ 1.0 (max deviation: 1.19e-07)
- ✅ 权重样本间有变化 (std > 0)
- ✅ CSV 文件正确保存 (686行 × 4列)
- Vision 权重较高 (~0.57)，可能与 T=1 clip-level 特征中视觉信息更易捕获情感有关

## 7. 无法完成或仍需确认

1. **GPU smoke test**：PyTorch 2.5.1+cu124 (2.5GB) 仍在下崽中，预计完成后可 GPU 重跑
2. **MLCL 下载审计**：待用户执行浏览器下载或后续 P6 阶段处理
3. **环境 yml 导出**：conda env export 需在完整环境激活后执行
4. **MOSEI 特征链路**：P4 前需要用户提供视频帧来源

## 8. 需用户/网页版 AI 判断

### Q1：T=1 单向量 smoke test 是否充分
当前 MOSI TMDC 特征为 clip-level 单向量 (T=1)。P2 smoke test 已验证工程链路。是否接受 T=1 作为 P2 工程验证的充分条件？P3/P4 是否需要先获取真正的序列特征？

### Q2：P3 启动条件
P2 所有代码已实现，smoke test 已通过（CPU 模式）。是否同意在 GPU smoke test 也通过后进入 P3（候选模块裁决实验）？

### Q3：DEConv 实现方案
DEConv 当前在 `utils_models/DEConv.py` 中的实现硬编码了 1024→32×32 reshape。P3 将其适配到主模型时需要重写以适应 hidden_dim=256 的通用接口。是否同意 P3 重新实现一个更通用的 DEConv？

## 9. P2.1 质量闸门 (2026-06-16)

### 修复的三个问题
1. **sLSTM mask 状态冻结** (D017): padding 时恢复上一时刻 c/n/m 状态
   - 验证: 噪声 padding 不影响 pooled (max diff = 0.00e+00) ✅
2. **AWAF modality dropout** (D017): 逐样本确保至少一个模态保留
   - 验证: prob=0.99, B=64, 20 runs: 无NaN, sum(w)=1 ✅
3. **AWAF context self-exclusion** (D018): attention 排除自身模态
   - 验证: same inputs → correct output ✅

### GPU smoke test 重跑
- 24.3s (GPU) — 比 P2 初版稍慢（修复增加了计算开销）
- 指标与前版一致：ACC2_Non0=42.38%, AWAF sum(w)=1.19e-07 ✅

## 10. 下一步建议

1. 等 PyTorch CUDA 下载完成后重跑 GPU smoke test
2. 用户确认 P2 后进入 P3（候选模块裁决实验 C0-C3）
3. P3 优先动作：实现 DEConv 适配 + Data2Vec-Audio 适配 + CME 适配
