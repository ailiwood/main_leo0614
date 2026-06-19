# HANDOFF PHASE P6S-REPAIR-5: MOSEI Collapse Diagnosis + Clean Restart

**Generated**: 2026-06-19  
**Branch**: p6s-repair4-mosei-fast-finish

## 本阶段完成
1. 诊断并修复了系统性 collapse 根因（COVAREP 音频特征含 -inf → NaN 传播）
2. 隔离了 12 个无效 P6S-Repair-4 baseline 输出到 `outputs/_invalid/`
3. 修复 `collate_textft()` 添加 `_clean_features()` 自动清理 -inf/+inf
4. 修复 `utils/metrics.py` 添加 NaN fail-fast (>10% NaN → ValueError)
5. 修复 `scripts/train_baseline_lite.py` 添加 collapse guard (epoch 级检查)
6. 新增 `tests/test_metrics_mosei_non0.py` — 8/8 测试通过
7. MulT-lite 修复后结果: ACC2=82.4%, F1=86.2%, MAE=0.611, Corr=0.705
8. SelfMM-lite 修复后结果: ACC2=82.6%, F1=86.3%, MAE=0.623, Corr=0.682
9. 剩余 6 个 baselines (TFN/LMF/MISA/MMIM/MLCL/DLF) 正在训练中

## 关键发现
- **Root cause**: COVAREP 中 ~2-5% 文件含 `-inf` 值（`log(0)` 操作产物）
- **NaN 传播链**: `-inf → Linear → NaN → ReLU → GRU → 全模型 NaN`
- **38.04% 真相**: MOSEI test set neg_non0/non0 = 1253/3294 = 38.0389%，即模型预测全负
- **修复方式**: `_clean_features()` 在 collate 时自动将 -inf/+inf 替换为 0.0

## 新增/修改文件

### 修改
- `data/textft_multimodal_dataset.py` — 添加 `_clean_features()` 和 inf 清理到 `collate_textft()`
- `utils/metrics.py` — 添加 NaN fail-fast
- `scripts/train_baseline_lite.py` — 添加 collapse guard、train_subset、fix f-string bug

### 新增
- `tests/test_metrics_mosei_non0.py` — 8 个指标单元测试
- `configs/experiments/p6s_repair5_debug/` — 200-sample overfit 配置
- `configs/experiments/p6s_repair5_mosei_baselines_fixed/` — 8 个修复后 baseline 配置
- `configs/experiments/p6s_repair5_mosei_mainline_fixed/` — 主模型配置目录
- `reports/P6S_repair5_collapse_fix/` — 7 个诊断/修复报告
- `outputs/_invalid/P6S_repair4_collapse_20260619_190122/` — 隔离的无效输出
- `HANDOFF_PHASE_P6S_REPAIR5.md` — 本文件

## 实验运行与核心结果

### Baseline-Lite MOSEI 修复后结果
| Model | ACC2_Non0 | F1_Non0 | MAE | Corr | Status |
|-------|-----------|---------|-----|------|--------|
| MulT-lite | 82.39% | 86.17% | 0.611 | 0.705 | ✅ Complete |
| SelfMM-lite | 82.60% | 86.33% | 0.623 | 0.682 | ✅ Complete |
| TFN-lite | — | — | — | — | 🔄 Training |
| LMF-lite | — | — | — | — | 🔄 Training |
| MISA-lite | — | — | — | — | 🔄 Training |
| MMIM-lite | — | — | — | — | 🔄 Training |
| MLCL-lite | — | — | — | — | 🔄 Training |
| DLF-lite | — | — | — | — | 🔄 Training |

## 无法完成或仍需确认
1. 剩余 6 个 baselines 结果（正在训练中，约 3 分钟完成）
2. 主模型尚未用修复后代码重训（使用同一 collate，应已免疫 collapse）
3. CUDA 兼容性：RTX 5070 Ti (sm_120) 不在此 PyTorch 2.3.0+cu118 支持列表，建议升级

## 需用户/网页版 AI 判断
1. 剩余 6 baselines 完成后是否自动继续 P6S 后续阶段？
2. 是否需要升级 PyTorch 以完全支持 RTX 5070 Ti？
3. 主模型是否需要立即重训，还是先等 8 baselines 全部跑完？
4. MOSI 数据集是否也需要同类型 collapse 检查？

## 下一步建议
1. 等待 6 baselines 完成 → 汇总 8 model fast matrix
2. 运行主模型 short train smoke test 确认无 collapse
3. 建议升级 PyTorch: `pip install torch==2.5.0+cu124 --index-url https://download.pytorch.org/whl/cu124`
4. 继续 P6S baseline 全量训练（若有需要）
