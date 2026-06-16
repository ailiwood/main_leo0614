# P4S Strict Protocol Implementation

## Fixes Applied

### 1. New StrictTrainer (`engine/strict_trainer.py`)

- train() ONLY on train split ✅
- check_val() on val split every epoch ✅
- best checkpoint selected by VAL MAE ✅
- test() called ONCE on best checkpoint ✅
- val_metrics_epoch.csv: only val metrics per epoch ✅
- test_metrics_final.json: only at end ✅

### 2. Key Differences from P4A

| Aspect | P4A (Flawed) | P4S (Fixed) |
|--------|:--:|:--:|
| Best epoch selection | test ACC2_Non0 ❌ | val MAE ✅ |
| Test evaluation | every epoch ❌ | once at end ✅ |
| Metrics saved | metrics_epoch (test!) | val_metrics_epoch (val) ✅ |
