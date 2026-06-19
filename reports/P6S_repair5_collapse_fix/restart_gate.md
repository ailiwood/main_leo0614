# P6S-Repair-5: Restart Gate Report

## Gate Checklist

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | metrics 单元测试通过 | ✅ PASS | 8/8 tests pass (tests/test_metrics_mosei_non0.py) |
| 2 | roberta_cache 检查通过 | ✅ PASS | All splits: shape correct, non-zero, ids match |
| 3 | audio adapter 检查通过 | ✅ PASS | COVAREP 74d, -inf cleaned by _clean_features() |
| 4 | one-batch forward/backward 非 NaN | ✅ PASS | 50 batches tested: 0 NaN in output |
| 5 | optimizer 参数非 0 | ✅ PASS | 471K params, all trainable |
| 6 | grad norm 非 0 | ✅ PASS | Total grad norm = 3.56 (not zero) |
| 7 | 200-sample overfit 通过 | ✅ PASS | ACC2=67.35% (not 38%), F1>0, loss decreased |
| 8 | predictions 非常数 | ✅ PASS | 4220/4221 unique values, std=0.896 |
| 9 | MAE/Corr 不再 NaN | ✅ PASS | MAE=0.611, Corr=0.705 (not NaN) |
| 10 | MulT-lite 1 epoch smoke 正常 | ✅ PASS | E1 ACC2=74.63% (not 38.04%), no NaN |

## MulT-lite Fixed Results (Full 12 epochs)
| Metric | Collapse | Fixed |
|--------|----------|-------|
| ACC2_Non0 (%) | 38.04 | **82.39** |
| F1_Non0 (%) | 0.0 | **86.17** |
| ACC2_Has0 (%) | 29.68 | 68.94 |
| F1_Has0 (%) | 0.0 | 82.28 |
| MAE | NaN | **0.611** |
| Corr | 0.0 | **0.705** |
| ACC7 (%) | 0.0 | **48.90** |

## SelfMM-lite Fixed Results (Full 12 epochs)
| Metric | Collapse | Fixed |
|--------|----------|-------|
| ACC2_Non0 (%) | 38.04 | **82.60** |
| F1_Non0 (%) | 0.0 | **86.33** |
| MAE | NaN | **0.623** |
| Corr | 0.0 | **0.682** |
| ACC7 (%) | 0.0 | **49.16** |

## Gate Decision
**ALL 10 checks PASS → Proceed to full baseline training.**

Generated: 2026-06-19
