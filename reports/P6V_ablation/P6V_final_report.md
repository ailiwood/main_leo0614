# P6V: Core Ablation — Final Report

**Date**: 2026-06-20  
**Branch**: `p6v-core-ablation-fast`  
**Status**: 🔄 Training launched, awaiting completion

## 1. Branch & Commit
- Branch: `p6v-core-ablation-fast` (based on `p6u-strict-freeze-cleanup`)
- Last P6U commit: `327ad4e`

## 2. Infrastructure Changes

### Model modifications (`models/textft_lora_xlstm_awaf_residual.py`)
- Added `fusion_type` config field: awaf | mean | concat | gated | fixed
- Added `temporal_encoder` config field: slstm | gru | lstm | none
- Added `awaf_context` and `awaf_interaction` config fields
- Modified encoder creation to support GRU/LSTM/none
- Modified `_forward_text_x_residual` to use AWAF for text_audio mode
- Updated `needs_awaf` property to always create AWAF for ablation
- Fixed `collect_trainable_params` references (`audio_slstm` → `audio_temporal`)

### Data fix (`data/textft_multimodal_dataset.py`)
- Fixed `split_map` to handle both 'valid' (MOSEI) and 'val' (MOSI) feature directories

### Configs (16 total)
- `configs/experiments/p6v_ablation/mosi/` — 10 MOSI ablation configs
- `configs/experiments/p6v_ablation/mosei/` — 6 MOSEI ablation configs

## 3. Ablation Matrix Status

### MOSI (10 experiments, 8 epochs each)
| # | Ablation | Smoke | Training |
|---|----------|:---:|:---:|
| 1 | main_awaf_slstm | Inherited (P6K) | ✅ |
| 2 | fusion_mean | ✅ | 🔄 |
| 3 | fusion_concat | Pending | 🔄 |
| 4 | fusion_gated | Pending | 🔄 |
| 5 | fusion_fixed | Pending | 🔄 |
| 6 | awaf_no_interaction | Pending | 🔄 |
| 7 | awaf_no_context | Pending | 🔄 |
| 8 | encoder_gru | Pending | 🔄 |
| 9 | encoder_lstm | Pending | 🔄 |
| 10 | encoder_no_temporal | Pending | 🔄 |

### MOSEI (6 experiments, 4 epochs each)
| # | Ablation | Smoke | Training |
|---|----------|:---:|:---:|
| 1 | main_awaf_slstm | Inherited (P6T) | ✅ |
| 2 | fusion_mean | ✅ | 🔄 |
| 3 | fusion_gated | Pending | 🔄 |
| 4 | awaf_no_interaction | Pending | 🔄 |
| 5 | encoder_gru | Pending | 🔄 |
| 6 | encoder_no_temporal | Pending | 🔄 |

## 4. Recovery Commands

```bash
# Check training progress
cd E:\00project_code\main_leo\new_code
conda activate mme
python scripts/p6u_status.py

# Launch remaining MOSI ablations (if needed)
for config in fusion_mean fusion_concat fusion_gated fusion_fixed awaf_no_interaction awaf_no_context encoder_gru encoder_lstm encoder_no_temporal; do
  python scripts/train_textft_lora_mainline.py \
    --config "configs/experiments/p6v_ablation/mosi/${config}_s42.yaml" \
    --device cuda
done

# Launch remaining MOSEI ablations (if needed)
for config in fusion_mean fusion_gated awaf_no_interaction encoder_gru encoder_no_temporal; do
  python scripts/train_textft_lora_mainline.py \
    --config "configs/experiments/p6v_ablation/mosei/${config}_s42.yaml" \
    --device cuda
done

# Collect results
python scripts/p6u_compile_results.py  # (needs adaptation for P6V paths)
```

## 5. Pending
- Complete all ablation training (~3-5 hours total)
- Generate ablation figures (1000 DPI)
- Generate ablation tables (CSV/MD/XLSX)
- Write ablation interpretation for thesis
- Update experiment registry

## 6. Files to Commit
- `models/textft_lora_xlstm_awaf_residual.py` (ablation switches)
- `data/textft_multimodal_dataset.py` (val/valid fix)
- `configs/experiments/p6v_ablation/` (16 configs)
- `scripts/p6v_generate_ablation_configs.py`
- `reports/P6V_ablation/`

Generated: 2026-06-20 01:00 UTC+8
