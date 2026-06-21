# P6AJ Handoff — MOSI 87+ Freeze

**Date**: 2026-06-21
**Branch**: `p6aj-mosi-87-reproducible-recovery`
**Status**: **FREEZE GATE PASSED** — 3-seed test ACC2_Non0 mean > 87.00%

---

## Final Multi-Seed Results

| Seed | ACC2_Non0 | F1_Non0 | MAE | Corr | ACC7 | Best Epoch | Best Val ACC2 |
|------|-----------|---------|-----|------|------|------------|---------------|
| 42 | 87.65% | 85.03 | 0.668 | 0.837 | 48.10 | 2 | 87.50% |
| 2024 | 87.50% | 85.51 | 0.699 | 0.816 | 45.48 | 1 | 87.96% |
| 3407 | 88.26% | 86.03 | 0.644 | 0.852 | 48.54 | 3 | 88.43% |
| **Mean** | **87.80%** | **85.52** | **0.670** | **0.835** | **47.37** | — | — |
| **Std** | **0.33** | **0.41** | **0.022** | **0.015** | **1.34** | — | — |

## Freeze Gate Verification

| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| 3-seed mean ACC2_Non0 | > 87.00% | 87.80% | ✅ PASS |
| Seeds >= 86.50% | >= 2/3 | 3/3 (87.65, 87.50, 88.26) | ✅ PASS |
| F1_Non0 no degradation | — | 85.52 ± 0.41 | ✅ Stable |
| Corr always positive | > 0 | 0.835 ± 0.015 | ✅ PASS |
| Test per seed once | — | Each seed: 1 test eval | ✅ PASS |
| Best ckpt by validation | — | Seeds: epoch 2, 1, 3 by val | ✅ PASS |
| TAV genuine paths | — | r_a=0.14, r_v=0.15, non-zero gates | ✅ PASS |

---

## Final Model Specification

### Architecture
- **Name**: `canonical_mosi_text_anchored_tav`
- **Text**: RoBERTa-large + LoRA (r=16) + TextMLP → h_t
- **Audio**: data2vec 768d → projection → sLSTM → masked attention pooling → h_a
- **Vision**: CLIP-L14 1024d → projection → sLSTM → masked attention pooling → h_v
- **Fusion**: Text-Anchored Reliable Fusion (`models/fusion/text_anchored_reliable_fusion.py`)
  - y_text = TextHead(h_t)
  - g_ta = h_t ⊙ h_a, g_tv = h_t ⊙ h_v
  - delta_a = MLP_a([h_t, h_a, g_ta]), r_a = sigmoid(GateMLP_a([h_t, h_a, g_ta]) + b_a)
  - delta_v = MLP_v([h_t, h_v, g_tv]), r_v = sigmoid(GateMLP_v([h_t, h_v, g_tv]) + b_v)
  - y_hat = clamp(y_text + α_a·r_a·delta_a + α_v·r_v·delta_v, -3, 3)
- **Params**: 4.16M trainable (total ~359.5M with frozen RoBERTa)

### Initialization
- **P6K checkpoint**: Text + Audio branches initialized from P6K T+A conservative (seed 42, ACC2=88.72%)
- **Vision**: Random initialization (CLIP-L14 1024d projection)
- **Fusion**: Random initialization with gate_bias=-2.0 (gates start near 0)

### Training Protocol
```yaml
batch_size: 8
grad_accum_steps: 8
max_epochs: 35
early_stopping_patience: 8
min_epochs: 10
lr: 5e-5 (new modules)
lr_text_lora: 1e-5
weight_decay: 0.01
optimizer: AdamW
loss: MSE
test_final_once: true
```

### Feature Contract
```yaml
dataset: mosi
data_version: mosi_tav_v1
cohort_id: mosi_official_tav_complete_case_v1
train: 1284, val: 229, test: 686
vision: CLIP-L14 1024d, 32 frames, Route-S
audio: data2vec 768d, 100 frames, Route-S
text: raw → RoBERTa-large
```

---

## Evidence Chain

### All MOSI Attempts

| Phase | Architecture | ACC2 | Corr | Status |
|-------|-------------|------|------|--------|
| P6K | T+A residual (v3_T40) | 88.72% | 0.851 | Historical reference, diff features |
| P6AD | Canonical AWAF | 42.23% | -0.03 | Collapse (softmax blending) |
| P6AF | Canonical AWAF (35ep) | 44.21% | -0.11 | Collapse (same root cause) |
| P6AG | Text-Anchored V1 | 82.01% | 0.749 | Working, below target |
| P6AI B | Text-Anchored V2 | 84.60% | 0.795 | Best from-scratch TAV |
| P6AI A | T+A residual (retry) | 46.04% | 0.145 | Gate collapsed |
| **P6AJ** | **P6K-init Text-Anchored** | **87.80%** | **0.835** | **TARGET ACHIEVED** |

### Why Canonical AWAF Fails on MOSI

The canonical AWAF architecture uses softmax-weighted blending:
```
[w_t, w_a, w_v] = softmax(scorer(h_t, h_a, h_v) / τ)
z = w_t·h_t + w_a·h_a + w_v·h_v
```
On MOSI's small dataset (1,284 train), audio and vision provide no standalone signal (probes: P2=56.94%, P3=57.41%, both below majority 59.8%). Softmax forces the model to distribute weight across three modalities, allowing noise from weak modalities to degrade the fused prediction below text-only (86.11%).

### Why Text-Anchored Fusion Works

The text-anchored architecture preserves the text prediction as an anchor:
```
y_hat = y_text + α_a·r_a·delta_a + α_v·r_v·delta_v
```
- y_text always contributes (anchor never removed)
- r_a, r_v initialized near 0 → model starts text-only
- Gates learn per-sample which corrections to trust
- If audio/vision are unreliable, gates stay small → model stays near text-only
- Architecture is naturally safe for small datasets

---

## Relationship to P6K

| Aspect | P6K | P6AJ |
|--------|-----|------|
| Architecture | T+A residual (gate + delta) | T+A+V text-anchored (dual gate + dual delta) |
| Features | v3_T40 (768d vision) | mosi_tav_v1 (1024d CLIP-L14 vision) |
| Modalities | Text + Audio | Text + Audio + Vision |
| Initialization | From scratch | P6K text+audio weights |
| ACC2 | 88.72% | 87.80% (3-seed mean) |
| Paper role | Historical reference | Final TAV candidate |

P6K weights were used only as initialization for text and audio branches. The vision branch and fusion module were trained from scratch. P6K's text+audio backbone provides a strong starting point, but the final model is a distinct three-modal architecture.

---

## Paper Claims

### Allowed

✅ "The final MOSI TAV model achieves 3-seed test ACC2_Non0 mean of 87.80% (87.50–88.26%) with text-anchored reliable fusion, demonstrating that a text-preserving architecture enables stable multimodal learning on small datasets where canonical softmax-weighted fusion collapses."

✅ "The text-anchored reliable fusion architecture uses per-sample reliability gates initialized near zero, ensuring the model starts from a strong text-only baseline and only incorporates audio/vision corrections when the gates learn they are reliable."

✅ "On MOSEI (13,239 train), canonical AWAF with Hadamard interaction achieves strong performance. On MOSI (1,284 train), the same architecture collapses due to forced softmax blending of weak auxiliary modalities. A text-anchored architecture resolves this, suggesting dataset size is a critical factor in multimodal fusion design."

✅ "The P6K text-audio residual model (88.72%) was used as a historical reference and weight initialization source, but is not the final TAV model and uses a different feature version."

### NOT Allowed

❌ "Canonical AWAF achieves state-of-the-art on both MOSI and MOSEI"
❌ Direct numerical comparison of MOSI and MOSEI results without noting feature version differences
❌ "P6K 88.72% is the TAV result" — P6K is T+A only, different features
❌ "Dual-dataset three-modal evidence complete with identical models" — architectures differ (AWAF vs text-anchored)
❌ "Vision significantly improves over text+audio on MOSI" — gates show small contribution, text anchor dominates

---

## Output Locations

### Code
- `models/fusion/text_anchored_reliable_fusion.py` — Fusion module
- `models/textft_lora_xlstm_awaf_residual.py` — Main model (added `canonical_mosi_text_anchored_tav` mode)
- `scripts/train_textft_lora_mainline.py` — Training script (added `--init_checkpoint`)

### Configs
- `configs/experiments/p6aj_mosi_87_recovery/P1_p6k_init_tav_s42.yaml`
- `configs/experiments/p6aj_mosi_87_recovery/P1_p6k_init_tav_s2024.yaml`
- `configs/experiments/p6aj_mosi_87_recovery/P1_p6k_init_tav_s3407.yaml`

### Training Outputs
- `outputs/P6AJ_mosi_87_recovery/P1_p6k_init_tav_s42_s42_20260621_143839/`
- `outputs/P6AJ_mosi_87_recovery/P1_p6k_init_tav_s2024_s2024_20260621_144824/`
- `outputs/P6AJ_mosi_87_recovery/P1_p6k_init_tav_s3407_s3407_20260621_145600/`

### Reports
- `reports/P6AJ_mosi_87_recovery/G0_p6k_feature_bridge_audit.md`
- `reports/P6AJ_mosi_87_recovery/P6AJ_final_model_freeze_report.md`
- `HANDOFF_PHASE_P6AJ_MOSI_87_FREEZE.md` (this file)

### P6K Reference
- `outputs/P6K/text_audio_conservative_s42_s42_20260619_031645/` — Historical T+A reference
- `configs/references/mosi_conservative/text_audio_conservative_s42.yaml`

---

## Remaining Work (NOT in P6AJ scope)

- [ ] MOSI TAV baseline-lite (TFN, LMF, MulT, SelfMM, MISA, MLCL, DLF)
- [ ] MOSI TAV key ablation (w/o vision, w/o audio, w/o gate, w/o interaction)
- [ ] MOSEI strict complete-case cohort re-run
- [ ] Cross-dataset analysis and paper writing
- [ ] Git cleanup, final release
- [ ] Do NOT delete any P6AD/P6AF/P6AG/P6AI intermediate evidence

---

## Environment

- **Python**: 3.10.20
- **PyTorch**: 2.11.0+cu128
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16GB)
- **Conda env**: `mme_xlstm_stable`
- **Base commit**: `604b773` (P6AA)
- **Branch**: `p6aj-mosi-87-reproducible-recovery`
