# P6X: Model Truth Sources

**Commit**: `8a42a46` | **Branch**: `p6w-canonical-ta-awaf-slstm`

## A. MOSEI Canonical — `canonical_text_audio_awaf_slstm`

| Property | Value |
|----------|-------|
| Status | `architecture_frozen_evaluation_pending` |
| Code | `models/textft_lora_xlstm_awaf_residual.py` L407-423 |
| Config | `configs/experiments/p6w_canonical/mosei/control_awaf_slstm_s42.yaml` |
| Control result | ACC2=87.83% (MOSEI, 12ep, s42) |
| Gates | G1(14/14), G2(14/14), G3(6/6) all PASS |
| AWAF created | Yes (canonical_fusion) |
| Text enters AWAF | Yes |
| Audio enters AWAF | Yes (projection→sLSTM→pooling→AWAF) |
| Gate/delta bypass | No |
| Ablation | 4 variants running |

## B. MOSI P6K — `P6K_text_audio_conservative_reference`

| Property | Value |
|----------|-------|
| Status | `historical_engineering_reference_not_awaf_evidence` |
| Commit | `59b216f` |
| Result | ACC2=88.72% (MOSI, 20ep, s42) |
| AWAF created | **No** — needs_awaf returns False for text_audio_residual |
| Gate/delta | Both off (gr=1.0, bdr=0) |
| Final output | reg = rtb (pure text base) |
| Can be called AWAF+sLSTM | **No** |
| Canonical MOSI | Permanently blocked (2 attempts, all collapsed) |

## C. Baselines

All `baseline_lite_reimplementation` — NOT official reimplementation.

| Model | Code | Active |
|-------|------|:---:|
| TFN-lite | `models/baselines/tfn_lite.py` | ✅ |
| LMF-lite | `models/baselines/lmf_lite.py` | ✅ |
| MulT-lite | `models/baselines/mult_lite.py` | ✅ |
| SelfMM-lite | `models/baselines/self_mm_lite.py` | ✅ |
| MISA-lite | `models/baselines/misa_lite.py` | ✅ |
| MLCL-lite | `models/baselines/mlcl_lite.py` | ✅ |
| DLF-lite | `models/baselines/dlf_lite.py` | ✅ |
| MMIM-lite | **excluded_duplicate_of_misa_lite** | ❌ |

Generated: 2026-06-20
