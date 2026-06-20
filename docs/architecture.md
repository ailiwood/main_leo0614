# Model Architecture: Canonical Text-Audio AWAF+sLSTM

## Overview

The model performs multimodal sentiment regression on MOSEI.
Input: text tokens + COVAREP audio features. Output: sentiment score [-3, +3].

```
Text [B,128]  → RoBERTa+LoRA → TextMLP → h_t [B,256]  ┐
                                                          ├→ AWAF → Head → y_hat
Audio [B,100,74] → Proj → sLSTM → Pool → h_a [B,256]  ┘   + [w_t, w_a]
```

## Module-to-File Mapping

| Module | Code File | Key Class/Function | Line |
|--------|-----------|-------------------|------|
| Full Model | `models/textft_lora_xlstm_awaf_residual.py` | `TextFTLoRAXLSTMAWAFResidual` | L120 |
| Config | `models/textft_lora_xlstm_awaf_residual.py` | `TextFTLoRAConfig` | L30 |
| Text Encoder | `models/textft_lora_xlstm_awaf_residual.py` | `_compute_text()` | L304 |
| Audio Encoder | `models/textft_lora_xlstm_awaf_residual.py` | `_compute_audio()` | L328 |
| Canonical Forward | `models/textft_lora_xlstm_awaf_residual.py` | `_forward_canonical_ta_awaf()` | L407 |
| sLSTM | `models/encoders/slstm.py` | `SLSTMEncoder` | — |
| AWAF Fusion | `models/fusion/awaf.py` | `AdaptiveWeightedAttentionFusion` | L31 |
| LoRA | `models/modules/minimal_lora.py` | `apply_lora_to_roberta()` | — |
| Pooling | `models/pooling/attention_pooling.py` | `MaskedAttentionPooling` | — |

## Module Design

### 1. Text Encoder
```
input_ids [B,128] + attention_mask [B,128]
  → RoBERTa-large (LoRA: r=16, alpha=32, targets=query/value)
  → CLS token [B,1024]
  → TextMLP: Linear(1024→512, LN, GELU, Drop) → Linear(512→256, LN, GELU, Drop)
  → h_t [B,256]
```
Trainable: 1.6M (LoRA) + 658K (TextMLP)

### 2. Audio Encoder
```
audio_features [B,100,74] + audio_mask [B,100]
  → Linear(74→256) + LayerNorm + GELU + Dropout(0.2)
  → 1-layer sLSTM (exponential gate + stabilizer, mask-aware)
  → MaskedAttentionPooling(256)
  → h_a [B,256]
```
Trainable: 20K (proj) + 525K (sLSTM) + 34K (pool)

### 3. AWAF Fusion
```
h_t, h_a → AdaptiveWeightedAttentionFusion
  Stage 1: Cross-modal context
    c_t = CrossAttn(query=h_t, key/value=h_a)
    c_a = CrossAttn(query=h_a, key/value=h_t)
    h_t = LayerNorm(h_t + c_t)
    h_a = LayerNorm(h_a + c_a)

  Stage 2: Interaction scoring
    g_ta = h_t * h_a  (Hadamard product)
    e = MLP(concat(h_t, h_a, g_ta))  → [B, 3]
    [w_t, w_a, w_v] = softmax(e / tau), tau=3.0

  Stage 3: Weighted fusion
    z = w_t * h_t + w_a * h_a + w_v * 0  → [B, 256]
```
Trainable: 627K

### 4. Prediction Head
```
z [B,256] → Linear(256→128) + ReLU → Linear(128→1) → y_hat [B,1]
```
Trainable: 33K

## Parameter Summary

| Component | Params | Trainable |
|-----------|--------|-----------|
| RoBERTa-large | 355M | 1.6M (LoRA only) |
| TextMLP | 658K | 658K |
| Audio Projection | 20K | 20K |
| sLSTM | 525K | 525K |
| Audio Pooling | 34K | 34K |
| AWAF | 627K | 627K |
| Head | 33K | 33K |
| **Total** | **357M** | **3.5M** |

## Training

- Loss: L1Loss(y_hat, y_label)
- Optimizer: AdamW (lr=3e-5 text, 3e-6 LoRA, wd=0.03)
- Batch: 8 × grad_accum 8 = effective 64
- Epochs: 12 (MOSEI), 20 (MOSI)
- Early stop: patience=4, min_epochs=3
- Seed: 42

## Key Design Decisions

1. NO gate/delta text bypass — AWAF output directly feeds prediction head
2. Independent `canonical_text_audio_awaf_slstm` mode — does NOT modify legacy paths
3. 2-modality AWAF (text+audio) with dummy vision ignored (w_v→0)
4. LoRA-efficient RoBERTa fine-tuning (only query/value matrices)
5. COVAREP inf cleaning in data loader (replaces -inf with 0.0)
