# P6H-R Model Patch Report — AWAF/Delta/Gate 修复开关

**Date**: 2026-06-18  
**Status**: ✅ 已完成

---

## 修改文件

| 文件 | 修改内容 |
|---|---|
| `models/fusion/awaf.py` | 新增 `use_modal_layernorm`, `awaf_uniform_mix`, `return_diagnostics`, `compute_entropy` |
| `models/modules/uncertainty_residual_gate.py` | 无需修改 (init_bias 已是构造函数参数) |
| `models/textft_lora_xlstm_awaf_residual.py` | **新建** — 完整主模型类 |
| `scripts/train_textft_lora_mainline.py` | **新建** — 完整训练脚本 |

---

## 修复开关详情

### 2.1 模态输入 LayerNorm ✅

在 AWAF 模块 `__init__` 中新增 3 个独立 LayerNorm:

```python
self.modal_ln_t = nn.LayerNorm(hidden_dim)  # 文本
self.modal_ln_a = nn.LayerNorm(hidden_dim)  # 音频
self.modal_ln_v = nn.LayerNorm(hidden_dim)  # 视觉
```

在 `forward` 开头应用:
```python
h_t = self.modal_ln_t(h_t)  # → L2 norm 标准化
h_a = self.modal_ln_a(h_a)
h_v = self.modal_ln_v(h_v)
```

- 可通过 `use_modal_layernorm` config 开关
- 输出 norm 前后的 L2 均值到 diagnostics

### 2.2 AWAF 温度 τ ✅

- 已有 `tau_raw → softplus → tau ≥ 0.1` 机制
- 通过 `tau_init` config 控制
- R1-R4 默认: 3.0 / 3.0 / 3.0 / 5.0
- 最终 τ 值输出到 result.json

### 2.3 AWAF entropy regularization ✅

- 训练脚本中实现: `loss = loss - lambda_entropy * entropy`
- 通过 `lambda_awaf_entropy` config 控制
- R1: 0.0, R2: 0.01, R3: 0.005, R4: 0.01

### 2.4 Uniform mix (weight floor) ✅

在 AWAF forward 中:
```python
if awaf_uniform_mix > 0:
    weights = (1-eps)*softmax(logits/τ) + eps/3
```
- 通过 `awaf_uniform_mix` config 控制
- R1: 0.0, R2: 0.0, R3: 0.10, R4: 0.05

### 2.5 Delta scale ✅

```python
self.delta_scale_reg = nn.Parameter(torch.tensor(delta_scale_init))
```
- 可学习参数
- R1-R3: 0.2, R4: 0.1
- 最终值输出到 result.json

### 2.6 Gate bias ✅

- 通过 `gate_init_bias` config 传入 `UncertaintyGuidedResidualGate.__init__`
- R1-R3: 2.0 (sigmoid→0.88), R4: 1.0 (sigmoid→0.73)
- gate_mean 输出到 result.json

---

## 模型架构 (TextFTLoRAXLSTMAWAFResidual)

```text
RoBERTa-large + LoRA(r=16,α=32) → CLS [1024]
  → TextMLP [1024→512→256] → h_t [256] ─→ reg_head_text, cls_head_text
                                         ─→ AWAF
                                         ─→ delta experts

Audio [T×768] → Linear+H → sLSTM → AttnPool → h_a [256] ─→ AWAF
                                                           ─→ delta experts

Vision [T×768] → Linear+H → sLSTM → AttnPool → h_v [256] ─→ AWAF
                                                           ─→ delta experts

AWAF(h_t,h_a,h_v) → Z [256] + weights [3]
  → Gate(ht, Z, reg_text_base, ...) → gate_reg [0-1]
  → Delta experts → bounded_delta [≤max_delta]
  → reg = reg_text_base + gate_reg * delta_scale * bounded_delta
```

参数: total 360M, trainable 4.65M
