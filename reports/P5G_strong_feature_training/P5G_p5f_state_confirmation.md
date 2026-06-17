# P5G P5F State Confirmation

| Item | Value |
|------|-------|
| P5E baseline | V2 2-seed 82.17% |
| Audio large model | wav2vec2-large-960h-lv60-self (1024d) ✅ cached |
| Vision large model | openai/clip-vit-large-patch14 (1024d) ✅ cached |
| Current feature root | data/features_strong_sequence_mosi_v3_T40/ |
| Current dims | text=1024, audio=768, vision=768 |
| Dataset | strong_sequence_dataset.py (supports dynamic dims via npz fields) |
| Extraction scripts | New scripts needed for audio/vision large |
| V2 code | models/deeptext_xlstm_awaf_residual_v2.py |
| V2 config | configs/models/deeptext_xlstm_awaf_residual_v2_mosi.yaml |
