# HANDOFF P6V: Core Ablation Fast

**Date**: 2026-06-20  
**Branch**: `p6v-core-ablation-fast`  
**Status**: 🔄 Ablation training launched, awaiting completion

## What Was Done
1. Added ablation switches to main model: fusion_type, temporal_encoder, awaf_context, awaf_interaction
2. Fixed data loading bug (MOSI 'val' vs 'valid' directory naming)
3. Created 16 ablation configs (10 MOSI + 6 MOSEI)
4. Smoke-tested and launched ablation matrix training (background)
5. P6U freeze results preserved unmodified

## Ablation Changes
- `models/textft_lora_xlstm_awaf_residual.py`: +4 config fields, encoder switching, fusion switching
- `data/textft_multimodal_dataset.py`: fixed split_map for MOSI/MOSEI compatibility

## Recovery After Training
```bash
cd E:\00project_code\main_leo\new_code
conda activate mme

# Check outputs
ls outputs/P6V_ablation/mosi/
ls outputs/P6V_ablation/mosei/

# Compile results
python -c "
import json, os, glob
for d in sorted(glob.glob('outputs/P6V_ablation/*/*_s42_*/result.json')):
    r = json.load(open(d))
    print(f'{r.get(\"ablation_name\",\"?\"):>25s}: ACC2={r.get(\"final_ACC2\",\"?\"):.2f}%')
"

# Generate figures and tables
# (scripts TBD — use P6U figure generation as template)
```
