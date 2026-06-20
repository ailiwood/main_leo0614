# Canonical Text-Audio AWAF+sLSTM — Architecture Specification

**Architecture Frozen** | Commit: `8a42a46` | Mode: `canonical_text_audio_awaf_slstm`

## 1. System Overview

```
Input               Encoder              Fusion            Head          Output
------               -------              ------            ----          ------
text tokens [B,128] → RoBERTa+LoRA → h_t ╮
                                          ├→ AWAF → z [B,256] → MLP → y_hat [B,1]
audio [B,100,74]    → Proj+sLSTM   → h_a ╯                    + [w_t,w_a]
```

## 2. Component Specification

### 2.1 Text Encoder
| Component | Specification | Code |
|-----------|--------------|------|
| Backbone | RoBERTa-large | `L134-139` |
| LoRA | r=16, alpha=32, targets=['query','value'] | `L136-139` |
| CLS extraction | `last_hidden_state[:, 0, :]` → [B, 1024] | `L306` |
| Text MLP | 1024→512(LN+GELU+Drop)→256(LN+GELU+Drop) | `L142-147` |
| Output | h_t [B, 256] | `L308` |

### 2.2 Audio Encoder
| Component | Specification | Code |
|-----------|--------------|------|
| Projection | Linear(74→256) + LN + GELU + Drop(0.2) | `L158-161` |
| Temporal | 1-layer sLSTM (exponential gate + stabilizer) | `L162-165` |
| Pooling | MaskedAttentionPooling(256) | `L166` |
| Output | h_a [B, 256] | `L338` |

### 2.3 AWAF Fusion
| Component | Specification | Code |
|-----------|--------------|------|
| Module | AdaptiveWeightedAttentionFusion | `models/fusion/awaf.py` |
| Fusion mode | 'awaf' (full) | `L234` |
| Context | Cross-modal attention (t↔a) | `awaf.py` L85-110 |
| Interaction | Hadamard product g_ta = h_t ⊙ h_a | `awaf.py` L115-120 |
| Scoring | MLP(3H→H→3) → softmax(/tau), tau=3.0 | `awaf.py` L125-140 |
| Fusion | z = Σ w_m · h_m | `awaf.py` L145 |
| Output | z [B, 256] + weights [B, 3] | `L407-423` |

### 2.4 Prediction Head
| Component | Specification | Code |
|-----------|--------------|------|
| Canonical head | Linear(256→128) + ReLU + Linear(128→1) | `L220-223` |
| Output | y_hat [B, 1] | `L409` |

## 3. Parameter Count (Verified from Live Code)

| Module | Total | Trainable |
|--------|-------|-----------|
| roberta (LoRA only) | 356,932,608 | 1,572,864 |
| text_mlp | 657,664 | 657,664 |
| audio_proj | 19,712 | 19,712 |
| audio_temporal (sLSTM) | 525,312 | 525,312 |
| audio_pool | 33,537 | 33,537 |
| canonical_fusion (AWAF) | 627,204 | 627,204 |
| canonical_head | 33,025 | 33,025 |
| **Total trainable** | | **~3,469,318** |

## 4. Training Protocol

| Parameter | Value |
|-----------|-------|
| Loss | L1Loss(y_hat, y_label) |
| Optimizer | AdamW (lr=3e-5 text, lr=3e-6 LoRA) |
| Weight decay | 0.03 |
| Batch size | 8 × grad_accum 8 = effective 64 |
| Epochs (MOSEI) | 12 |
| Early stop | patience=4, min_epochs=3 |
| Seed | 42 |

## 5. Key Design Decisions

1. **No gate/delta bypass**: Unlike P6K conservative path, the canonical model has no `reg = rtb + 0` shortcut
2. **AWAF explicitly created**: `needs_awaf=True` for canonical mode
3. **Independent mode**: Does NOT modify old `text_audio_residual` path
4. **2-modality AWAF**: Text + Audio fused; dummy vision ignored (w_v → 0)

## 6. What This Model IS

- A text-audio multimodal sentiment regressor
- Using AWAF for learnable sample-level modality weighting
- Using sLSTM for audio temporal encoding
- With LoRA-efficient RoBERTa text encoding
- Producing interpretable per-sample modality weights

## 7. What This Model IS NOT

- NOT a 3-modality (text+audio+vision) model
- NOT using gate/delta residual mechanism
- NOT validated on MOSI (training blocked)
- NOT claiming audio significantly improves MOSEI accuracy
- NOT a text-only model with audio as auxiliary

Generated: 2026-06-20 | Verified from live code
