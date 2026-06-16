# P0 环境审计与环境规划

> 生成时间：2026-06-16  
> 用途：记录当前环境状态并规划后续环境建设

---

## 一、当前环境状态

### 1.1 硬件

| 项目 | 状态 |
|------|------|
| GPU | NVIDIA GeForce RTX 5070 Ti |
| 显存 | 16,303 MiB (~16 GB) |
| 驱动版本 | 610.47 |
| CUDA (UMD) | 13.3 |

### 1.2 Conda 环境

Conda 安装路径：`E:\Anaconda3`

| 环境名 | Python 版本 | PyTorch | 状态 |
|--------|------------|---------|------|
| **mme** | 3.9.21 | 2.3.0+cu118 | ★ 当前项目环境 |
| base | — | — | Anaconda 基础环境 |
| ml_py310 | — | — | — |
| pytorch_py310 | — | — | — |
| dl311 | — | — | — |
| FIGR_torch171 | — | — | — |
| 其他 (~15个) | — | — | 不相关 |

### 1.3 mme 环境关键依赖

| 包 | 版本 |
|----|------|
| Python | 3.9.21 |
| PyTorch | 2.3.0+cu118 |
| CUDA (available) | True |
| Transformers | 4.34.1 |

### 1.4 环境问题

| 问题 | 影响 | 建议 |
|------|------|------|
| conda 不在 Git Bash PATH 中 | 需完整路径调用 | 使用 `E:\Anaconda3\Scripts\conda.exe` |
| 无 mme_xlstm 环境 | 需要新建 | P2 前创建 |
| mme 环境可能多轮依赖混乱 | 后续可能冲突 | 倾向于新建环境 |
| CUDA 13.3 (UMD) vs CUDA 11.8 (PyTorch) | nvidia-smi 报告 UMD 版本更高 | 注意 CUDA toolkit 与 PyTorch CUDA 版本匹配 |

---

## 二、环境建设计划

### 2.1 推荐新环境

```text
名称：mme_xlstm
路径：E:\Anaconda3\envs\mme_xlstm
Python：3.10 或 3.11
PyTorch：2.3.0+cu118 或 2.5.0+cu124
```

### 2.2 环境分层策略

| 环境 | 用途 | 优先级 |
|------|------|--------|
| mme_xlstm | 主模型训练与评估 | P2 前必须创建 |
| mme | 旧实验对照 & 只读 | 保留，不再新增包 |
| mme_mlcl | MLCL baseline（如需专属环境） | P6 时评估 |
| mme_casp | CASP baseline（如需专属环境） | P6 时评估 |

### 2.3 环境文件输出目标

```
env/
├── install_commands.md               # 安装命令记录
├── environment_mme_xlstm.yml         # Conda 环境导出
├── pip_freeze_mme_xlstm.txt          # pip freeze
├── requirements_main.txt             # 主模型依赖
├── requirements_mlcl.txt             # MLCL 依赖
└── requirements_casp.txt             # CASP 依赖
```

### 2.4 核心依赖预估

```
torch >= 2.3.0
transformers >= 4.34.0
numpy, pandas, scipy, scikit-learn
tqdm, einops, matplotlib, seaborn
tensorboard (可选)
openpyxl, python-docx (P8 表图输出)
```

### 2.5 环境创建时机

| 阶段 | 动作 |
|------|------|
| P0 (当前) | 记录旧环境，不安装新包 |
| P1 | 评估是否需要安装新包（如 scikit-learn） |
| P2 前 | 创建 mme_xlstm 环境，安装核心依赖 |
| P6 | 评估是否创建 mme_mlcl / mme_casp |
