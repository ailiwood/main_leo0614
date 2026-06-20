# MME: Multimodal Sentiment Analysis with AWAF+sLSTM

**Status**: Architecture frozen, MOSEI Canonical ablation in progress  
**Branch**: `p6w-canonical-ta-awaf-slstm`  
**Last Commit**: `7d08ab4`

## Overview

Research code for multimodal sentiment analysis on CMU-MOSI and CMU-MOSEI datasets.  
Core innovation: **Adaptive Weighted Attention Fusion (AWAF)** + **sLSTM temporal encoder** for text-audio fusion.

## Model

### Canonical Text-Audio AWAF+sLSTM (`canonical_text_audio_awaf_slstm`)

```
Text [B,128]  → RoBERTa+LoRA → MLP → h_t [B,256]  ┐
                                                      ├→ AWAF → Head → y_hat
Audio [B,100,74] → Proj → sLSTM → Pool → h_a [B,256] ┘   + [w_t, w_a]
```

- **Text**: RoBERTa-large + LoRA (r=16, 1.6M trainable)
- **Audio**: COVAREP 74d → projection → 1-layer sLSTM → masked attention pooling
- **Fusion**: AWAF with cross-modal context + Hadamard interaction scoring
- **Head**: Linear(256→128)→ReLU→Linear(128→1)
- **Output**: Sentiment score [-3, +3] + per-sample modality weights [w_t, w_a]
- **Params**: 357M total, 3.5M trainable

### Architecture Documents
- `docs/MODEL_ARCHITECTURE_SPEC.md` — Full component specification with code references
- `docs/MODEL_ARCHITECTURE_CANONICAL.md` — Paper-oriented description with formulas
- `docs/MODEL_ARCHITECTURE_DIAGRAM.mmd` — Mermaid flowchart
- `docs/MODEL_TRUTH_SOURCES.md` — Final model classification

## Baselines

7 baseline-lite reimplementations (not official):
TFN, LMF, MulT, Self-MM, MISA, MLCL, DLF  
MMIM-lite excluded (byte-identical to MISA-lite).

## Results

### MOSEI (text_audio, seed=42)
| Model | ACC2_Non0 | F1_Non0 | MAE | Corr |
|-------|-----------|---------|-----|------|
| **Canonical AWAF+sLSTM** | **87.83%** | 90.28% | 0.526 | 0.794 |
| MISA-lite (best baseline) | 82.67% | 86.62% | 0.621 | 0.687 |

*Fair ablation (fusion/encoder variants) in progress.*

### MOSI (text_audio, seed=42)
| Model | ACC2_Non0 | F1_Non0 |
|-------|-----------|---------|
| P6K Conservative Reference | 88.72% | 86.64% |

*Note: P6K 88.72% uses text-audio co-training without AWAF. Canonical AWAF+sLSTM blocked on MOSI (insufficient training samples).*

## Quick Start

```bash
# Environment
conda env create -f env/environment_mme_canonical.yml
conda activate mme

# Verify imports
python -c "from models.textft_lora_xlstm_awaf_residual import TextFTLoRAXLSTMAWAFResidual; print('OK')"

# Smoke test (1 epoch, 5 batches)
python scripts/train_textft_lora_mainline.py \
  --config configs/experiments/p6w_canonical/mosei/control_awaf_slstm_s42.yaml \
  --device cuda --smoke

# Unit tests
python tests/test_metrics_mosei_non0.py
```

## Data

CMU-MOSI and CMU-MOSEI datasets are NOT included.  
Feature paths in configs assume local preprocessed data.  
See `docs/DATA_AND_FEATURE_SPEC.md` for expected structure.

## Environment

- Python 3.9, PyTorch 2.3.0+cu118, CUDA 11.8
- `env/environment_mme_canonical.yml` — Conda environment
- `env/requirements_mme_canonical.txt` — pip freeze

## Disclaimer

- Baseline-lite are lightweight reimplementations, NOT official
- No data, weights, or checkpoints are uploaded
- MOSEI audio (COVAREP 74d) has limited discriminative value
- MOSI Canonical training is blocked (1284 samples insufficient)

## License

Research code. License TBD.
