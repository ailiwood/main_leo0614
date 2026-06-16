# P2.1 质量闸门报告

> 生成时间：2026-06-16  
> 阶段：P2.1 质量闸门与小修补  
> 状态：✅ 完成

---

## 一、修复的问题

### 修复 1: sLSTM mask 状态冻结 (D017)

**文件**: `models/encoders/slstm.py` → `SLSTMEncoder._unroll()`

**问题**: mask=0 的 padding 时间步，h_t 被清零但 state (c_t/n_t/m_t) 已被 cell 更新，导致 padding 位置的噪声输入污染后续时间步的状态。

**修复**: 在 cell 更新后，对 padding 样本逐样本恢复 h_t/c_t/n_t/m_t 为上一时刻状态：
- 有效时间步: 使用新状态
- padding 时间步: 恢复上一时刻完整状态 + h_t 置零

**验证**: 构造相同前 3 步 + 后 2 步 100x 噪声 padding 的两个序列，两序列 `masked_mean` 和 `last_valid` 的 pooled 输出差异均为 `0.00e+00` ✅

### 修复 2: AWAF modality dropout 逐样本保留 (D017)

**文件**: `models/fusion/awaf.py` → `_modality_dropout()`

**问题**: 只检查整个 batch 是否三个模态全被 dropout，不能保证每个样本至少保留一个模态。

**修复**: 
1. 逐样本计算 `n_kept = mask.sum(dim=1)`
2. 对 `n_kept == 0` 的样本，随机选择一个模态恢复：`rescue_mod = randint(0, 3)`
3. 将对应 mask 设为 1.0

**验证**: `prob=0.99, batch_size=64`，20 次运行：无 NaN，所有样本 `sum(w)=1` ✅

### 修复 3: AWAF context 排除自身模态 (D018)

**文件**: `models/fusion/awaf.py` → `_context_enhance()`

**问题**: context attention 的 query 对全部 3 个模态计算（包括自身），不符合论文定义"以自身为 query，其他模态为 key/value"。

**修复**: 构建 `actual_mask`（3×3，对角线为 -inf），在 attention score 上施加 mask 使自身位置权重≈0。

**验证**: 三个模态输入相同时，AWAF 仍正常输出 sum(w)=1，无 NaN ✅

---

## 二、环境记录统一

更新文件:
- `reports/P2_environment_setup.md` — 更新 PyTorch 版本为 2.12.0.dev+cu128，标注 nightly 临时方案
- `HANDOFF_PHASE_02.md` — 同步更新
- `env/pip_freeze_mme_xlstm.txt` — 已重新导出
- `memory.md` — 追加 P2.1 记录

当前环境:
| 项目 | 值 |
|------|-----|
| Python | 3.10.20 |
| PyTorch | 2.12.0.dev20260408+cu128 (GPU, nightly) |
| CUDA | 12.8 |
| GPU | RTX 5070 Ti (Blackwell sm_120) ✅ |
| Transformers | 4.34.1 |
| torchvision | 0.21.0+cu124 (⚠ 与 nightly torch 版本不完全匹配) |
| torchaudio | 2.6.0+cu124 (⚠ 同上) |

---

## 三、单元测试结果

| 测试 | 结果 |
|------|:--:|
| Metrics (7指标+边界) | ✅ |
| SLSTMCell (单步+多步+NaN防护) | ✅ |
| SLSTMEncoder (pooling/T=1/bidirectional) | ✅ |
| sLSTM mask 状态冻结 (padding 噪声不影响 pooled) | ✅ |
| AWAF (7融合模式+sum(w)=1) | ✅ |
| AWAF modality dropout per-sample (prob=0.99, n=20) | ✅ |
| AWAF context self-exclusion | ✅ |
| MainModel (T=1+T=5+backward+7 fusion modes) | ✅ |
| **MOSI GPU 1 epoch smoke test** | ✅ |

## 四、MOSI GPU Smoke Test (P2.1 修复后)

| 指标 | 值 |
|------|-----|
| Train Loss | 1.7759 |
| MAE | 1.604 |
| Corr | 0.107 |
| ACC2_Non0 | 42.38% |
| AWAF sum(w) max_dev | **1.19e-07** ✅ |
| AWAF weights | [text=0.482, audio=0.215, vision=0.303] |
| 训练耗时 | 24.3s (GPU) |

---

## 五、新增/修改文件

| 文件 | 操作 |
|------|:--:|
| `models/encoders/slstm.py` | 修改 — mask 状态冻结逻辑 |
| `models/fusion/awaf.py` | 修改 — dropout 逐样本 + context self-exclusion |
| `reports/P2_1_quality_gate.md` | 新建 |
| `reports/P2_environment_setup.md` | 更新 — PyTorch nightly 版本 |
| `env/pip_freeze_mme_xlstm.txt` | 更新 |
| `HANDOFF_PHASE_02.md` | 更新 |
| `memory.md` | 追加 P2.1 记录 |
| `docs/DECISIONS.md` | 追加 D016-D018 |
