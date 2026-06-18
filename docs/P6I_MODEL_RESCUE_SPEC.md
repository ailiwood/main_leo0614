# P6I_MODEL_RESCUE_SPEC.md

## 目标

P6I 不再继续做 AWAF 参数微调，而是验证 audio/vision 是否有可用情感信号，并在信号存在时实现 `Text-Confidence Conditioned Residual`。

当前失败事实：

- P6H original：text_base=86.43%，final=86.43%，gain=0。
- P6H-R：AWAF 权重恢复，但 final 仍等于 text_base。
- 说明：AWAF 权重分布不是唯一瓶颈，当前 residual/gate/delta 结构没有学到有效修正。

## 保留内容

继续保留：

- `models/modules/minimal_lora.py`
- `models/fusion/awaf.py`
- `models/textft_lora_xlstm_awaf_residual.py`
- `scripts/train_textft_lora_mainline.py`
- P6H-R 的 per-sample 输出系统

## 新增核心结构

新增：

```text
models/modules/text_confidence_residual.py
```

核心思想：

```text
text_base 是主预测；
residual 只学习 label - detach(text_base_pred)；
gate 由 text confidence 控制；
高置信文本少修正；
低置信文本允许 audio/vision 修正。
```

不要继续使用自由 UGR gate 作为主方案，因为它在 P6H/P6H-R 中学会关闭 residual。

## 新 forward 逻辑

输入：

```text
h_t: text hidden [B,H]
h_a: audio hidden [B,H]
h_v: vision hidden [B,H]
z_av: AWAF or AV fused hidden [B,H]
reg_text_base: [B,1]
cls_text_base: [B,1]
label: optional [B,1]
```

输出：

```text
reg_final
delta
gate
text_confidence
target_delta
diagnostics
```

公式：

```text
text_prob = sigmoid(cls_text_base)
text_margin = abs(text_prob - 0.5) * 2
pred_strength = clamp(abs(reg_text_base) / 3, 0, 1)
text_confidence = 0.5 * text_margin + 0.5 * pred_strength

target_delta = label - detach(reg_text_base)

gate = gate_floor + (1 - gate_floor) * sigmoid(gate_mlp([...]))
delta = max_delta * tanh(delta_mlp([...]))
reg_final = reg_text_base + gate * delta
```

注意：

1. `target_delta` 必须 detach text_base。
2. residual branch 不应反向破坏 text_base。
3. 必须支持 `freeze_text_base_after_warmup`。
4. 必须支持 `residual_only_stage`。
5. 必须保存 per-sample confidence/gate/delta。
6. 必须记录 final 是否真正超过 text_base。

## 训练阶段

建议三阶段：

### Stage 1：text warmup

```text
训练 text LoRA + text heads
关闭 residual
目标：恢复 text_base >= 86%
```

### Stage 2：residual only

```text
冻结 RoBERTa LoRA / text_mlp / text heads
训练 audio/vision/AWAF/text_conf_residual
target_delta = label - detach(text_base)
```

### Stage 3：joint fine-tune

```text
小学习率联合微调
防止 text_base 退化
```

## P6I 实验顺序

1. `text_only recovery`
2. `audio_only`
3. `vision_only`
4. `av_only`
5. `text_audio_residual`
6. `text_vision_residual`
7. `text_av_residual`
8. `text_confidence_residual`

## 进入 MOSEI 的条件

只有满足：

```text
MOSI final_ACC2_Non0 > text_base_ACC2_Non0
且 residual_gain >= 0.5%
```

才允许启动 MOSEI dataset / training。

如果 MOSI final 仍不超过 text_base：

```text
不启动 MOSEI；
不补 seed；
转入 text-dominant conservative multimodal model。
```
