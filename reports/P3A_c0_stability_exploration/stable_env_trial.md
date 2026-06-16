# P3A Stable 环境尝试报告

> 日期：2026-06-16

---

## 结论：✅ 成功！切换到 stable PyTorch

---

## 尝试过程

| 尝试 | PyTorch 版本 | CUDA | GPU 可用 | 失败原因 |
|:--:|------|------|:--:|------|
| 1 | 2.3.0+cu118 | 11.8 | ❌ | 无 sm_120 kernel |
| 2 | 2.5.1+cu124 | 12.4 | ❌ | 无 sm_120 kernel |
| 3 | 2.6.0+cu124 | 12.4 | ❌ | 无 sm_120 kernel |
| 4 | 2.12.0.dev nightly | 12.8 | ✅ | 但为 dev 版本 |
| **5** | **2.11.0+cu128 stable** | **12.8** | **✅** | **完全可用** |

## Stable 环境信息

| 项目 | 值 |
|------|-----|
| 环境名 | mme_xlstm_stable |
| 路径 | E:\Anaconda3\envs\mme_xlstm_stable |
| Python | 3.10.20 |
| PyTorch | **2.11.0+cu128 (STABLE)** |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Capability | (12, 0) = sm_120 ✅ native support |
| Transformers | 4.34.1 |
| NumPy | 1.26.4 |

## 全部测试通过

- ✅ Metrics 单元测试
- ✅ sLSTM 单元测试（含 mask 状态冻结）
- ✅ AWAF 单元测试（7 模式 + dropout + context）
- ✅ MainModel forward/backward
- ✅ MOSI GPU smoke test (P2)
- ✅ P3A 14 个 20-epoch run（全部完成，无错误）

## 决策

**P3A 正式使用 mme_xlstm_stable 环境。**  
P4 无需切换。旧 mme_xlstm (nightly) 保留为备份。

## 建议

后续 P3B-P10 全部使用 mme_xlstm_stable。
