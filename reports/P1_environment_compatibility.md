# P1 环境兼容性分析

> 生成时间：2026-06-16  
> 用途：P2 前环境建设规划，不执行安装

---

## 一、当前环境状态

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16GB VRAM |
| Conda | E:\Anaconda3 |
| 当前可用环境 | mme (Python 3.9.21, PyTorch 2.3.0+cu118, Transformers 4.34.1) |
| CUDA (driver) | 13.3 (UMD) |
| CUDA (PyTorch) | 11.8 |

---

## 二、Python 版本推荐

**推荐 Python 3.10**，理由：

| 维度 | Python 3.10 | Python 3.11 |
|------|:--:|:--:|
| PyTorch 2.3.0 官方支持 | ✅ 完整支持 | ✅ 支持 |
| CUDA 11.8 兼容性 | ✅ 成熟 | ✅ 支持 |
| Windows conda 预编译包 | ✅ 丰富 | ⚠ 部分包需源码编译 |
| MLCL/CASP 兼容性 | ✅ (≥3.8) | ✅ (≥3.8) |
| 已知 Windows CUDA bug | ✅ 无 | ⚠ 部分用户报告 |
| **综合推荐** | ✅ **推荐** | ⚠ 保守用户可选 3.10 |

---

## 三、推荐环境规划

### 3.1 环境分层

```text
E:\Anaconda3\envs\
├── mme (Python 3.9.21)            ← 保留，只读对照，不新增包
├── mme_xlstm (Python 3.10)        ← ★ 主模型环境，P2 前创建
├── mme_mlcl (Python 3.10)         ← MLCL baseline，P6 按需创建
└── mme_casp (Python 3.10)         ← CASP baseline，P6 按需创建
```

### 3.2 mme_xlstm 建议安装命令（草案，不执行）

```bash
# 创建环境
conda create -n mme_xlstm python=3.10 -y
conda activate mme_xlstm

# PyTorch (CUDA 11.8 — 与当前 mme 环境保持一致)
conda install pytorch==2.3.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 核心依赖
pip install transformers==4.34.1
pip install numpy pandas scipy scikit-learn tqdm einops
pip install matplotlib seaborn openpyxl python-docx

# 可选
pip install tensorboard soundfile librosa
```

### 3.3 mme_mlcl / mme_casp 建议（P6 时再评估）

```bash
# 仅在 mme_xlstm 依赖冲突时创建
conda create -n mme_mlcl python=3.10 -y --clone mme_xlstm  # 或从零安装
```

---

## 四、环境创建时机

| 阶段 | 动作 |
|------|------|
| P1 (当前) | ✅ 规划完成，不安装 |
| P2 开始前 | 创建 mme_xlstm，安装核心依赖 |
| P6 开始前 | 评估是否创建 mme_mlcl / mme_casp |

---

## 五、当前环境缺失包检查

以下是在 P1 审计中发现的缺失包（仅记录，不安装）：

| 包 | 用途 | 当前缺失 |
|----|------|:--:|
| gensim | Tri_modal_ER w2v.vectors 读取 | ✅ 在 mme 中缺失 |
| soundfile | TMDC 音频提取（读取 wav） | 待确认 |

这些包不影响 P1 决策，P2 前统一安装到 mme_xlstm。
