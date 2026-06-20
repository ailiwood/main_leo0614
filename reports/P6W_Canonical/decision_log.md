# P6W-C Decision Log

## Attempt 1 (Failed)
- **Date**: 2026-06-20
- **Config**: control_awaf_slstm_s42.yaml (full training, 20 epochs)
- **Result**: ACC2=42.38% — all-positive collapse
- **Evidence**: val stuck at 57.41% (MOSI val pos/non0), 11 epochs, best=epoch 5
- **Root cause**: RoBERTa+AWAF co-training diverges on 1284 MOSI samples
- **Verdict**: FAILED — canonical model cannot converge on MOSI with end-to-end training

## Attempt 2 (Running)
- **Date**: 2026-06-20
- **Config**: control_awaf_slstm_staged_s42.yaml
- **Strategy**: freeze_text_base=true — frozen RoBERTa+LoRA text features
- **Trainable**: audio_proj, audio_temporal(sLSTM), canonical_fusion(AWAF), canonical_head
- **Protocol**: Stage A: 4 epochs frozen text → Stage B: 16 epochs unfrozen LoRA at lr=5e-7
- **Pre-registered**: LAST attempt. If fails, MOSI Canonical permanently blocked
- **Gate conditions**: ACC2 must exceed 60%, predictions non-constant, AWAF weights active

## MOSEI Control (Passed)
- **Date**: 2026-06-20
- **Result**: ACC2=87.83%, delta vs P6T=0.03pp
- **Verdict**: PASS — canonical model reproduces P6T frozen result
- **Ready for**: Fair ablation matrix (4 variants) if MOSI Attempt 2 fails

Generated: 2026-06-20
