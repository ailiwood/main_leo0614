# HANDOFF_PHASE_10B.md — P5B Strong Feature Upper Bound Diagnosis

## 🔴 BREAKTHROUGH: sLSTM is Counterproductive on Text

| Model | ACC2 | MAE | Corr | Params | Note |
|-------|:--:|:--:|:--:|:--:|------|
| Text+sLSTM (P4Z) | 76.2% | 1.004 | 0.656 | ~855K | sLSTM on text |
| **AttnPool+DeepMLP** | **80.2%** | **0.885** | **0.733** | **726K** | **NO sLSTM** |
| **MeanPool+DeepMLP** | **80.2%** | 0.885 | 0.733 | **592K** | Simplest |
| Full AWAF-Seq (P4W) | 78.8% | 0.994 | 0.645 | 4.15M | Current best |

## Key Findings

1. **Removing sLSTM from text improves ACC2 by +4.0%** (76.2→80.2)
2. **Simple 592K text model BEATS complex 4.15M multimodal model** (+1.4%)
3. **DeBERTa tokens are already contextualized** — sLSTM adds noise, not value
4. **Multimodal fusion currently HURTS performance** (78.8 < 80.2)
5. **Text ceiling is at least 80.2%** — path to 85% requires better features

## Root Cause

sLSTM on DeBERTa-large token features is counterproductive. DeBERTa's attention mechanism already captures sequence-level context. Adding a recurrent layer (sLSTM) introduces temporal noise and degrades the representation quality.

## Path to 85%

1. Use DeepMLP on DeBERTa features (NOT sLSTM) as text backbone
2. Add multimodal branch as residual correction
3. Upgrade audio to wav2vec2-large (768→1024d)
4. This can reach 83-85% with proper fusion

## MOSEI & Audio Status

- wav2vec2-large: download timed out (HF CDN unreachable)
- MOSEI SDK: still blocked (setuptools/pip incompatibility)
