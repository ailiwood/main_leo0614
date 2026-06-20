#!/usr/bin/env python
"""G1: Static validation of canonical model architecture."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.textft_lora_xlstm_awaf_residual import TextFTLoRAConfig, TextFTLoRAXLSTMAWAFResidual

def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition

print("G1: Canonical Static Validation")
print("=" * 50)

config = TextFTLoRAConfig(
    mode='canonical_text_audio_awaf_slstm',
    audio_input_dim=74, hidden_dim=256, device='cpu',
    text_model_name='roberta-large',
)
model = TextFTLoRAXLSTMAWAFResidual(config)

results = []
results.append(check("AWAF created (canonical_fusion exists)", hasattr(model, 'canonical_fusion')))
results.append(check("Canonical head exists", hasattr(model, 'canonical_head')))
results.append(check("Audio projection exists", hasattr(model, 'audio_proj')))
results.append(check("Audio temporal exists (sLSTM)", hasattr(model, 'audio_temporal')))
results.append(check("Audio pool exists", hasattr(model, 'audio_pool')))
results.append(check("Mode is canonical", model.config.mode == 'canonical_text_audio_awaf_slstm'))
results.append(check("needs_awaf=True", model.config.needs_awaf))
results.append(check("needs_audio_branch=True", model.config.needs_audio_branch))
results.append(check("needs_text=True", model.config.needs_text))
results.append(check("uses_gate=False (default)", not getattr(model.config, 'use_uncertainty_gate', False)))
results.append(check("uses_delta=False (default)", not getattr(model.config, 'use_delta_experts', False)))
results.append(check("Canonical head params > 0", sum(p.numel() for p in model.canonical_head.parameters()) > 0))
results.append(check("Canonical fusion params > 0", sum(p.numel() for p in model.canonical_fusion.parameters()) > 0))

# Verify forward path exists
results.append(check("_forward_canonical_ta_awaf method exists", hasattr(model, '_forward_canonical_ta_awaf')))

passed = sum(results)
total = len(results)
print(f"\nG1 RESULT: {passed}/{total} PASS")
if passed == total:
    print("G1 GATE: PASS")
else:
    print("G1 GATE: FAIL — STOP, do not proceed to G2")
    sys.exit(1)
