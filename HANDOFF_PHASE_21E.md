# HANDOFF_PHASE_21E.md — P6E TextFT Mainline & Baseline Engineering

> 日期: 2026-06-18
> 阶段: P6E — 工程硬化、双数据集多模态与 baseline runner

---

## 一、本阶段完成

### 1. MMSDK 安装 ✅
- PyPI包 `cmu-multimodal-sdk` v0.0.6 安装成功
- `from mmsdk import mmdatasdk` 导入正常
- **此前错误**：搜索了不存在的 `mmsdk` PyPI包和错误的 `A2Zadeh/CMU-MultimodalSDK` 仓库
- **正确入口**：`CMU-MultiComp-Lab/CMU-MultimodalSDK` GitHub + `cmu-multimodal-sdk` PyPI

### 2. Zero Fallback 修复 ✅
- `data/textft_multimodal_dataset.py` 新增 `formal_mode` 参数（默认 True）
- AV 特征缺失时直接 `raise FileNotFoundError`
- 禁止静默生成全零 AV → 杜绝伪多模态

### 3. 工程风险审计 ✅
- 所有关键风险点已审计（见 `P6E_engineering_risk_audit.md`）
- 零 fallback 已修复，strict protocol 已确认

### 4. 主模型 config 创建 ✅
- `configs/models/textft_xlstm_awaf_mosi_mainline.yaml`

---

## 二、CMU 计算序列状态

| 服务器 | 状态 |
|--------|------|
| `immortal.multicomp.cs.cmu.edu` | ❌ HTTP 404 — .csd 文件不可下载 |
| MOSEI Labels (.csd) | ❌ 不可用 |
| MOSEI High-level features (.csd) | ❌ 不可用 |

**当前可用**：MOSEI CSV (labels + raw text) + WAV 音频块。Vision 特征不可用。

---

## 三、当前最优结果

| 数据集 | 模型 | 模态 | ACC2 | 状态 |
|--------|------|------|------|------|
| MOSI | P5E V2 (2-seed) | T+A+V | 82.17% | 最优多模态 |
| MOSI | P6C RoBERTa text-only | T | 85.37% | 文本上限 |
| MOSEI | P6D RoBERTa text-only 3ep | T | 88.13% | 文本基线 |
| MOSEI | — | T+A+V | — | 未开始 |

---

## 四、未完成

| 任务 | 状态 | 原因 |
|------|------|------|
| MOSI TextFT 多模态训练 | ⏳ | 358M参数训练慢，待启动 |
| MOSEI 多模态训练 | ⏳ | CMU .csd 不可用，AV特征缺失 |
| Baseline runner | ❌ | 未开始 |
| 图表/xlsx输出 | ⏳ | 待训练完成后执行 |

---

## 五、需要用户协助

1. **MOSEI Audio/Vision 特征** — CMU 服务器 404，CSV仅有文本+标签。需提供 COVAREP/OpenFace 特征文件或替代下载地址
2. **Baseline 优先级确认** — 是否立即开始 Self-MM / MulT / MMIM 本地复现？
3. **GPU 策略确认** — TextFT 358M 训练慢，是否接受 LoRA 或使用 P5E V2 frozen 路线？

---

## 六、GitHub

| 项目 | 值 |
|------|-----|
| **当前分支** | `p6e-textft-mainline-dual-baseline` |
| **最新 commit** | `bbe49a0` |
| **URL** | https://github.com/ailiwood/main_leo0614/tree/p6e-textft-mainline-dual-baseline |
