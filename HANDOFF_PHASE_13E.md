# HANDOFF_PHASE_13E.md — P5E UGR-AWAF Residual Optimization

## 本阶段完成

### 代码修复 ✅
- Attention pooling return expression fixed (`return pooled, attn if return_weights else pooled` → explicit if/else)
- V2 training script supports encoding='utf-8' yaml loading
- V2 losses use per-sample reduction='none' (correct sample reweight)

### V2 模型接入 ✅
- `models/deeptext_xlstm_awaf_residual_v2.py` (DeepTextXLSTMAWAFResidualV2, 3.67M params)
- `models/modules/uncertainty_residual_gate.py` (UncertaintyGuidedResidualGate)
- `engine/residual_losses_v2.py` (ResidualLossV2)
- All 12 unit tests passed

### 单元测试 ✅
9/9 P5D tests + 12/12 V2 tests

### Diagnostic Ablation ✅
| Ablation | ACC2 (30ep) | Δ vs Full |
|----------|-------------|-----------|
| Full V2 | 82.47% | — |
| no_audio | 82.32% | -0.15% |
| no_awaf_mean | 81.71% | -0.76% |
| no_vision | 81.25% | -1.22% |
| no_residual | 80.49% | -1.98% |

### V2 One-Stage Results ✅

| Seed | ACC2_NZ | MAE | Corr |
|------|---------|-----|------|
| 42 | 82.47% | 0.8111 | 0.7385 |
| 2024 | 81.86% | 0.8486 | 0.7316 |
| **Mean** | **82.17%** | **0.8299** | **0.7351** |

### Checkpoint 清理 ✅
35→4 pth files, 551MB→52MB

### 最佳候选
**P5E V2 one-stage**: 82.17% 2-seed mean — current best

### 是否超过关键阈值

| 阈值 | 当前 | 状态 |
|------|------|------|
| 80.2% (text-only) | 82.17% | ✅ +1.97% |
| 82% | 82.17% | ✅ +0.17% |
| 83% | 82.17% | ❌ -0.83% |
| 85% | 82.17% | ❌ -2.83% |

### 待完成实验
- Two-stage training
- Weak_neg reweight
- ConditionalGate seed2024 (if needed)

### GitHub
- **分支**: `p5e-ugr-awaf-residual-optimization`
- **Commit**: (pending)

### 仍需确认的问题
1. 82.17% 是否足够进入论文主结果？
2. 是否继续 P5E two-stage/weakneg 实验？
3. 是否接受 V2 并转向 P5F 强特征升级？
4. MAE 退化 (0.8056→0.8299) 是否可接受？
