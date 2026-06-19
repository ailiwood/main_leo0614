# P6U: Active Code Manifest

**Date**: 2026-06-20 | **Status**: ✅ Verified

## KEEP — Active Production Code

### Entry Scripts (3)
- `scripts/train_textft_lora_mainline.py` — Main model training (P6T/P6U)
- `scripts/train_baseline_lite.py` — Baseline training (P6S_repair5/P6U)
- `scripts/eval_checkpoint.py` — Model evaluation

### Main Model (7)
- `models/textft_lora_xlstm_awaf_residual.py` — **Active main model**
- `models/encoders/slstm.py` — sLSTM encoder
- `models/fusion/awaf.py` — AWAF fusion (core innovation)
- `models/pooling/attention_pooling.py` — Masked attention pooling
- `models/modules/minimal_lora.py` — LoRA adapter for RoBERTa
- `models/modules/uncertainty_residual_gate.py` — Gate mechanism
- `models/modules/text_confidence_residual.py` — Text confidence residual

### Baselines (8)
- `models/baselines/base_baseline.py` — Shared baseline base class
- `models/baselines/tfn_lite.py` — Tensor Fusion Network lite
- `models/baselines/lmf_lite.py` — Low-rank Multimodal Fusion lite
- `models/baselines/mult_lite.py` — Multimodal Transformer lite
- `models/baselines/self_mm_lite.py` — Self-MM lite
- `models/baselines/misa_lite.py` — MISA lite
- `models/baselines/mlcl_lite.py` — MLCL lite
- `models/baselines/dlf_lite.py` — DLF lite

### Shared (3)
- `data/textft_multimodal_dataset.py` — Dataset + collate (with -inf fix)
- `utils/metrics.py` — Unified metrics (with NaN fail-fast)
- `utils/seed.py` — Seed control (if exists)

### Tests (1)
- `tests/test_metrics_mosei_non0.py` — Metrics unit tests (8/8 pass)

## EXCLUDE — Inactive/Old Code (archive candidates)

### Old Main Models (3)
- `models/deeptext_xlstm_awaf_residual.py` — Superseded by textft_lora version
- `models/deeptext_xlstm_awaf_residual_v2.py` — Superseded
- `models/textft_xlstm_awaf_residual.py` — Superseded

### Old Baseline (1)
- `models/baselines/mmim_lite.py` — Byte-identical to MISA-lite (excluded)

### Old Entry Scripts (5)
- `scripts/p6g_train_lora_mosi.py`
- `scripts/p6h_train_lora_main.py`
- `scripts/p6c_text_roberta_finetune.py`
- `scripts/p6d_mosei_text_roberta.py`
- `scripts/eval_deeptext_xlstm_awaf_residual.py`

### Old Data Loaders (2)
- `data/sequence_dataset.py`
- `data/strong_sequence_dataset.py`

Generated: 2026-06-20
