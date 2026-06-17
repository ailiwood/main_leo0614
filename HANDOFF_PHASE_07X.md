# HANDOFF_PHASE_07X.md — P4X MOSI Optimization + MOSEI Prep

## MOSI Results Summary

| Stage | ACC2_NZ_reg | MAE | Corr | Key Change |
|-------|:--:|:--:|:--:|------|
| C0 baseline | 72.7% | 1.096 | 0.572 | sLSTM+AWAF only |
| P4V.1 AWAF-Seq | 73.9% | 1.070 | 0.629 | +Cross-modal Transformer |
| **P4W Enhanced** | **78.4%** | **0.980** | **0.666** | +Vision T>1 + sign_cons + aux + 60ep |

## AWAF-Seq Best Config (P4W)

```yaml
hidden_dim: 256, cross_layers: 2, batch_size: 32
vision: CLIP-ViT-B/32, T=20 (from MP4), dim=768
text: DeBERTa-large, T~18, dim=1024
audio: wav2vec2-base, T~93, dim=768
sign_consistency: 0.1, aux_loss: 0.1
epochs: 60, scheduler: ReduceLROnPlateau, AMP: true
strict protocol: val MAE → best, test final once
```

## MOSEI Status

- Text: ✅ (CSV with transcript)
- Audio: ✅ (wav files)
- **Vision: ❌ BLOCKED** (no MP4 video files in local data)
- Need: MOSEI videos OR public pre-extracted features

## In Progress (P4X)

- Vision T=32/T=40 extraction (background)
- 5-config component ablation (background)

## GitHub

- Branch: p4x-mosi-vision-mosei-prep (from p4w-mosi-awafseq-performance)
- Latest push: 3a8246c (p4w-mosi-awafseq-performance)
- URL: https://github.com/ailiwood/main_leo0614

## For Web AI Decision

1. 78.4% is solid — should this be the MOSI main result?
2. Path to 80%+: upgrade wav2vec2 → large? Vision T=32/40?
3. MOSEI vision blocked — accept text+audio or find video source?
