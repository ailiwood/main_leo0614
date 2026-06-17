# P5F P5E State Confirmation

## P5E V2 Results

| Seed | ACC2_NZ | MAE | Corr | ACC7 |
|------|---------|-----|------|------|
| 42 | 82.47% | 0.8111 | 0.7385 | 41.11% |
| 2024 | 81.86% | 0.8486 | 0.7316 | 40.23% |
| **Mean** | **82.17%** | **0.8299** | **0.7351** | **40.67%** |

## Diagnostic Ablation

| Ablation | ACC2 (30ep) |
|----------|-------------|
| Full V2 | 82.47% |
| no_audio | 82.32% |
| no_vision | 81.25% |
| no_awaf_mean | 81.71% |
| no_residual | 80.49% |

## Current Best Config

`configs/models/deeptext_xlstm_awaf_residual_v2_mosi.yaml`
- hidden_dim=256, text_mlp_hidden=512, text_mlp_layers=3
- slstm_num_layers=1, delta_scale_init=0.1, max_delta=1.5
- use_uncertainty_gate=true, use_delta_experts=true, use_bounded_delta=true
- delta_target_loss_weight=0.2, margin_sign_loss_weight=0.05

## Current Features

- text: DeBERTa-large token-level, 1024d, T~8-32
- audio: wav2vec2-base frame-level, 768d, T~100
- vision: CLIP ViT-B/32 frame-level, 768d, T=40

## Current Checkpoints

4 pth files, 52MB total (cleaned in P5D)

## Open Issues

- 82.17% < 83% target
- MAE degraded vs P5D
- Vision contributes more than audio (1.22% vs 0.15%)
- Need stronger features to reach 85-87%
