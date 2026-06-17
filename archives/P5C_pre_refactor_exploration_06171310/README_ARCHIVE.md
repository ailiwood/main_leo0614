# P5C Pre-Refactor Exploration Archive

> Created: 2026-06-17 13:10
> Phase: P5C — DeepText-xLSTM-AWAF Residual Refactor
> Decision: D031 — Architecture Overhaul

## Purpose

This archive preserves exploration code, scripts, configs, reports, and handoffs
from phases P2 through P5B that were superseded by the new DeepText-xLSTM-AWAF Residual architecture.

## Contents

```
P5C_pre_refactor_exploration_06171310/
├── README_ARCHIVE.md                    ← This file
├── file_manifest_before_move.csv        ← Full file listing
├── P4V_architecture_upgrade_files.zip   ← P4V upgrade archive
├── models_exploration/                  ← Old model implementations
│   ├── ours_xlstm_fusion.py             ← P2-P4: three-way sLSTM fusion
│   ├── ours_awaf_seq_xlstm.py           ← P4V-P4W: AWAF-Seq + CrossModalTransformer
│   ├── ours_text_guided_awaf_xlstm.py   ← P5A: Text-Guided AWAF-xLSTM (FAILED)
│   ├── enhancements/                    ← Candidate modules (not adopted)
│   │   ├── cme.py                       ← Cross-Modal Encoder
│   │   ├── cme_residual.py              ← CME residual variant
│   │   └── deconv.py                    ← DEConv
│   └── interaction/
│       └── modality_reliability.py      ← P5A modality reliability scoring
├── scripts_exploration/                 ← Old training/test scripts
│   ├── p3a_hparam_search.py
│   ├── p3b_candidate_runs.py
│   ├── train_main.py                    ← Old training entry
│   ├── train_awaf_seq.py                ← P4V training
│   ├── train_text_guided_awaf_xlstm.py  ← P5A training
│   ├── smoke_forward.py                 ← P2 smoke test
│   ├── test_awaf_seq_modules.py         ← P4V unit test
│   ├── test_text_guided_awaf_xlstm.py   ← P5A unit test
│   ├── extract_mosi_data2vec_audio.py   ← Data2Vec extraction
│   └── extract_mosi_strong_sequence.py  ← Strong sequence extraction
├── configs_exploration/                 ← Old model configs
│   ├── ours_awaf_seq_xlstm.yaml
│   ├── ours_backbone.yaml
│   ├── ours_c0_best.yaml
│   ├── ours_c0_p4a_final.yaml
│   ├── ours_c0_strong_sequence.yaml
│   ├── ours_c1_deconv.yaml
│   ├── ours_c3_cme.yaml
│   ├── ours_text_guided_awaf_xlstm_mosi.yaml
│   ├── ours_text_guided_awaf_xlstm_mosei_sdk.yaml
│   └── ...
├── reports_exploration/                 ← Historical reports (copies)
│   ├── P0_*.md through P4Z_*/
│   ├── P5A_*/ P5B_*/
│   └── README_PROJECT_STATUS.md
└── handoffs_exploration/                ← Historical handoffs (copies)
    ├── HANDOFF_PHASE_00.md through HANDOFF_PHASE_10B.md
```

## Key Modules NOT Archived (active in P5C)

These remain in `models/`, `engine/`, `utils/`, `data/`:

- `models/encoders/slstm.py` — Core sLSTM (used in audio/vision branches)
- `models/fusion/awaf.py` — Core AWAF (used as residual weight generator)
- `models/heads.py` — Prediction heads
- `models/interaction/cross_modal_transformer.py` — Cross-modal transformer (optional)
- `models/pooling/attention_pooling.py` — Masked attention pooling
- `engine/strict_trainer.py` — Strict protocol trainer
- `utils/metrics.py` — Unified metrics
- `data/strong_sequence_dataset.py` — Strong sequence data loading
- `data/dataset.py` — T=1 dataset
- `configs/default.yaml` — Default config
- `configs/data/mosi_strong_sequence.yaml` — Data config

## Restoration

To restore any archived file to its original location:
```bash
cp archives/P5C_pre_refactor_exploration_06171310/<subdir>/<file> <original_path>
```

Note: Original reports and handoffs were COPIED (not moved) — originals remain in place
for evidence chain integrity.
