# Reproduction Guide

## Environment Setup

```bash
# Create conda environment
conda env create -f env/environment_mme_canonical.yml
conda activate mme

# Or install from requirements
pip install -r env/requirements_mme_canonical.txt

# Verify
python -c "import torch; print(torch.__version__)"
python -c "from models.textft_lora_xlstm_awaf_residual import TextFTLoRAXLSTMAWAFResidual; print('OK')"
```

Requirements: Python 3.9, PyTorch 2.3.0+, CUDA 11.8+, transformers, numpy, pandas, scikit-learn.

## Data Preparation

CMU-MOSI and CMU-MOSEI datasets must be downloaded separately and placed in:
```
data/
├── mosi/label.csv
├── features_strong_sequence_mosi_v3_T40/
│   ├── train/*.npz
│   ├── val/*.npz
│   └── test/*.npz
└── processed/mosei_full/
    ├── label.csv
    ├── train/*.npz
    ├── valid/*.npz
    ├── test/*.npz
    └── roberta_cache/
        ├── train_roberta_cls.npy
        ├── valid_roberta_cls.npy
        ├── test_roberta_cls.npy
        └── *_ids.json
```

Each `.npz` file must contain: `audio_seq`, `audio_mask`, `vision_seq`, `vision_mask`.

## Quick Smoke Test

```bash
# Verify everything works (1 epoch, 5 batches per model)
python scripts/train_textft_lora_mainline.py \
  --config configs/canonical/mosei/control_awaf_slstm_s42.yaml \
  --device cuda --smoke

python scripts/train_baseline_lite.py \
  --config configs/baselines/mosei/misa_lite_s42.yaml \
  --device cuda --max_epochs 1 --limit_batches 5
```

## Run Full Experiments

### Main Model (MOSEI)
```bash
python scripts/train_textft_lora_mainline.py \
  --config configs/canonical/mosei/control_awaf_slstm_s42.yaml \
  --device cuda
```

### All Main + Baselines (one command)
```bash
python scripts/run_main_experiments.py --dataset mosei --device cuda
```

### Ablation Matrix (MOSEI, 5 variants)
```bash
python scripts/run_ablation_experiments.py --dataset mosei --device cuda
```

### Baseline Training
```bash
python scripts/train_baseline_lite.py \
  --config configs/baselines/mosei/mult_lite_s42.yaml \
  --device cuda
```

## Output Structure

Each run saves to `outputs/<phase>/<dataset>/<model>_s42_<timestamp>/`:

| File | Content |
|------|---------|
| `config.yaml` | Experiment configuration |
| `command.txt` | Exact command executed |
| `train.log` | Training output |
| `metrics_epoch.csv` | Per-epoch metrics |
| `metrics_best.json` | Best validation metrics |
| `result.json` | Final test metrics |
| `best_model.pth` | Best checkpoint weights |
| `last_model.pth` | Last epoch weights |
| `predictions_test.csv` | Per-sample predictions |
| `awaf_weights_test.csv` | Per-sample AWAF weights |
| `loss_curve.png` | Training loss plot |

## Metrics

All results computed by `utils/metrics.py`:
- ACC2_Non0 (primary) — binary accuracy excluding zero labels
- F1_Non0 — F1 score excluding zero labels
- ACC2_Has0, F1_Has0 — including zero labels
- MAE — Mean Absolute Error
- Corr — Pearson Correlation
- ACC7 — 7-class accuracy

## Unit Tests

```bash
python tests/test_metrics_mosei_non0.py
# Expected: 8/8 tests passed
```

## Reproducibility Notes

- All experiments use seed=42
- Official train/valid/test splits
- Baseline-lite are lightweight reimplementations, NOT official code
- MOSEI COVAREP audio has limited discriminative value (text-dominant dataset)
- MOSI Canonical AWAF+sLSTM training is blocked (1284 samples insufficient)
- Weights and data NOT included in repository
