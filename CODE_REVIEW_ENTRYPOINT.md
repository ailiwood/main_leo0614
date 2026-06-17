# CODE_REVIEW_ENTRYPOINT.md

## For Web AI: Priority Reading Order

1. **CODE_REVIEW_ENTRYPOINT.md** (this file) — quick overview + latest architecture
2. **docs/主模型大修决策06171310.md** — latest architecture overhaul decision (D031)
3. **docs/FINAL_MODEL_SPEC_DRAFT.md** — new DeepText-xLSTM-AWAF Residual spec (P5C)
4. **memory.md** — project memory & phase history (P0→P5C)
5. **docs/DECISIONS.md** — key decisions D001-D031
6. **HANDOFF_PHASE_10B.md** — P5B breakthrough (sLSTM counterproductive on text)
7. **models/deeptext_xlstm_awaf_residual.py** — new main model (P5C)
8. **models/encoders/slstm.py** — sLSTM implementation (unchanged, used in audio/vision only)
9. **models/fusion/awaf.py** — AWAF fusion module (unchanged, now used as residual weight generator)
10. **engine/strict_trainer.py** — strict protocol trainer
11. **engine/losses.py** — new loss functions (P5C)
12. **utils/metrics.py** — unified metrics
13. **configs/models/deeptext_xlstm_awaf_residual_mosi.yaml** — MOSI config

## P5C Architecture: DeepText-xLSTM-AWAF Residual

### Breakthrough Discovery (P5B)
- **sLSTM on DeBERTa text tokens is COUNTERPRODUCTIVE**: 76.2% → 80.2% by removing it
- **DeepMLP text-only (592K) BEATS Full AWAF-Seq multimodal (4.15M)**: 80.2% > 78.8%
- **Multimodal fusion currently HURTS**: need residual, not replacement approach

### New Architecture
```
Text [B,Tt,1024] → AttentionPool → DeepMLP → reg_text_base, cls_text_base
Audio [B,Ta,768] → sLSTM → AttentionPool → h_a
Vision [B,Tv,768] → sLSTM → AttentionPool → h_v
Text_residual [B,Tt,1024] → projection → AttentionPool → h_t_residual (NO sLSTM)

AWAF(h_t_residual, h_a, h_v) → z_residual, awaf_weights
  → delta_reg, delta_cls

reg = reg_text_base + δ_reg * delta_reg
cls = cls_text_base + δ_cls * delta_cls
```

### Key Design Decisions
1. **Text branch = DeepMLP (NO sLSTM)** — main discriminant
2. **xLSTM only on audio/vision** — temporal residual enhancement
3. **AWAF = residual correction weight generator** — not main fusion
4. **Final = text_base + λ * delta** — residual correction, not replacement
5. **δ_scale_init = 0.1** — learnable, small correction by default

### Ablation Plan
- `no_residual` → text-only baseline
- `no_audio` / `no_vision` → single-modal residual
- `text_slstm_on` → prove sLSTM hurts on text (消融验证)
- `no_awaf_mean_residual` → AWAF → mean pooling in residual branch

## Data: MOSI Strong Sequence v3_T40
- text: DeBERTa-large token-level, 1024d, T~8-32
- audio: wav2vec2-base frame-level, 768d, T~100
- vision: CLIP frame-level, 768d, T=40 (real frames!)

## Key Config (MOSI)
- hidden_dim: 256, text_mlp_hidden: 512, text_mlp_layers: 3
- batch_size: 32, epochs: 60, lr: 1e-4
- delta_scale_init: 0.1

## Questions for Web AI
1. Is residual correction approach scientifically sound?
2. How should AWAF weights be interpreted in residual context?
3. Should δ_reg and δ_cls share the same scale or be independent?
4. Best way to frame this in the paper (Chapter 3)?
