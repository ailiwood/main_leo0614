# Code Structure

## Directory Layout

```
new_code/
├── README.md                     # Project overview + quick start
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
│
├── configs/                      # Experiment configurations
│   ├── canonical/mosei/          # Canonical AWAF+sLSTM configs (5)
│   ├── baselines/                # Baseline-lite configs
│   └── references/               # Historical reference configs
│
├── models/                       # Model implementations
│   ├── textft_lora_xlstm_awaf_residual.py  # Canonical main model
│   ├── encoders/slstm.py         # sLSTM temporal encoder
│   ├── fusion/awaf.py            # AWAF fusion module
│   ├── pooling/attention_pooling.py  # Masked attention pooling
│   ├── modules/                  # LoRA, gate, text confidence
│   └── baselines/                # 7 baseline-lite models
│
├── scripts/                      # Executable scripts (5 total)
│   ├── train_textft_lora_mainline.py   # Main model training
│   ├── train_baseline_lite.py          # Baseline training
│   ├── eval_checkpoint.py              # Model evaluation
│   ├── run_main_experiments.py         # One-click main experiments
│   └── run_ablation_experiments.py     # One-click ablation matrix
│
├── data/                         # Dataset loading
│   └── textft_multimodal_dataset.py    # Data loader + collate
│
├── utils/                        # Utilities
│   └── metrics.py                # Unified metrics (7 metrics)
│
├── tests/                        # Unit tests
│   └── test_metrics_mosei_non0.py    # Metrics tests (8/8 pass)
│
├── docs/                         # Documentation
│   ├── architecture.md           # Model architecture
│   ├── code-structure.md         # This file
│   ├── model-status.md           # Model classification
│   ├── data-and-features.md      # Data specification
│   ├── baselines.md              # Baseline descriptions
│   ├── reproducibility.md        # Reproduction guide
│   ├── environment.md            # Environment setup
│   └── limitations.md            # Known limitations
│
├── env/                          # Environment exports
├── results/                      # Final results (tables + figures)
└── outputs/                      # Training outputs (git-ignored)
```

## Active Code Files (17)

### Entry Points (5)
| File | Purpose |
|------|---------|
| `scripts/train_textft_lora_mainline.py` | Main model training (Canonical + ablation) |
| `scripts/train_baseline_lite.py` | Baseline-lite training |
| `scripts/eval_checkpoint.py` | Model evaluation from checkpoint |
| `scripts/run_main_experiments.py` | One-click: all main experiments |
| `scripts/run_ablation_experiments.py` | One-click: all ablation variants |

### Model (8)
| File | Purpose |
|------|---------|
| `models/textft_lora_xlstm_awaf_residual.py` | Canonical AWAF+sLSTM model |
| `models/encoders/slstm.py` | sLSTM temporal encoder |
| `models/fusion/awaf.py` | Adaptive Weighted Attention Fusion |
| `models/pooling/attention_pooling.py` | Masked attention pooling |
| `models/modules/minimal_lora.py` | LoRA adapter for RoBERTa |
| `models/modules/uncertainty_residual_gate.py` | Uncertainty gate (legacy) |
| `models/modules/text_confidence_residual.py` | Text confidence residual (legacy) |
| `models/baselines/base_baseline.py` | Baseline base class |

### Data + Metrics (2)
| File | Purpose |
|------|---------|
| `data/textft_multimodal_dataset.py` | Dataset + collate (with COVAREP inf fix) |
| `utils/metrics.py` | Unified 7-metric computation (with NaN fail-fast) |

### Test (1)
| File | Purpose |
|------|---------|
| `tests/test_metrics_mosei_non0.py` | Metrics unit tests (8/8) |

### Baselines (7)
| File | Reference |
|------|-----------|
| `models/baselines/tfn_lite.py` | TFN (Zadeh et al., EMNLP 2017) |
| `models/baselines/lmf_lite.py` | LMF (Liu et al., ACL 2018) |
| `models/baselines/mult_lite.py` | MulT (Tsai et al., ACL 2019) |
| `models/baselines/misa_lite.py` | MISA (Hazarika et al., ACM MM 2020) |
| `models/baselines/self_mm_lite.py` | Self-MM (Yu et al., AAAI 2021) |
| `models/baselines/mlcl_lite.py` | MLCL (Zhuang et al., IEEE TMM 2025) |
| `models/baselines/dlf_lite.py` | DLF |

Note: All baselines are **lite reimplementations**, not official code.
MMIM-lite excluded (byte-identical to MISA-lite).
