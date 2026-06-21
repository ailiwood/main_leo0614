# P6AA: Old T+A vs New TAV Comparability Assessment

**Date**: 2026-06-21  
**Conclusion**: `not_directly_comparable` — no valid gain claim possible without same-cohort baselines

---

## 1. What Cannot Be Claims

❌ "Vision contributes +0.15pp ACC2 improvement" (TAV 87.98% vs old T+A 87.83%)

**Why this claim is invalid:**

| Factor | TAV (P6AA) | Old T+A (P6T) | Match? |
|--------|-----------|---------------|--------|
| Data version | `mosei_tav_openface2_v1` | `mosei_full` (inferred) | ❌ Different feature source |
| Vision features | OpenFace2 713-dim | N/A | ❌ Different data |
| Cohort | `mosei_official_tav_intersection_v1` | Not filtered | ❌ Possibly different samples |
| Architecture | Canonical AWAF (no residual) | Residual with gate | ❌ Different fusion |
| Epoch selection | 12 epochs, best epoch 11 | 12 epochs, best epoch 12 | ⚠️ Different selection |
| Modalities | Text + Audio + Vision | Text + Audio | ❌ |
| Trainable params | 4.84M | 2.81M | ❌ |

---

## 2. What P6AA Has Actually Proven

✅ **TAV data pipeline is functional**:
- OpenFace2 713-dim vision features can be loaded, encoded, temporally modeled via sLSTM
- AWAF fusion can weight three modalities per-sample
- Model produces distinct, non-collapsed predictions

✅ **TAV model trains stably on MOSEI**:
- No NaN, no gradient collapse, no constant output
- AWAF weights converge (w_t≈0.48, w_a≈0.26, w_v≈0.25)

❌ **NOT yet proven**:
- That vision features improve over text+audio
- That dynamic AWAF weights improve over fixed mean
- That sLSTM improves over plain LSTM for vision/audio
- That Hadamard interaction terms contribute

These require **same-cohort, same-budget controlled ablation** (P6AB-1).

---

## 3. What P6AB Must Deliver for Valid Claims

### 3.1 Required Same-Cohort Baselines

| Variant | What it measures |
|---------|-----------------|
| F1: No vision (T+A only) | Vision contribution |
| F2: No audio (T+V only) | Audio contribution |
| F3: Global static weights | Sample-level vs global fusion |
| F4: Fixed mean (1/3 each) | Learned vs fixed fusion |
| F5: No Hadamard interaction | Interaction term contribution |
| E1: LSTM instead of sLSTM | Temporal encoder contribution |

### 3.2 Required Controls

- Same data version: `mosei_tav_openface2_v1`
- Same cohort: `mosei_official_tav_intersection_v1`
- Same seed: 42
- Same epochs: 4 (ablation budget)
- Same batch, optimizer, scheduler, loss
- Same text encoder (frozen RoBERTa + LoRA)

### 3.3 Required Statistical Tests

- Paired bootstrap 95% CI for ACC2 difference
- McNemar test for binary classification agreement
- AWAF weight distribution comparison (F0 vs F3/F4)
- Counterfactual: zero-vision vs full-vision prediction delta

---

## 4. P6AA 12-epoch Result Classification

| Classification | Status |
|---------------|--------|
| Canonical main candidate | ✅ Valid for this role |
| Evidence for vision benefit | ❌ Not without same-cohort ablation |
| Comparable to old T+A | ❌ Different data/cohort/fusion |
| Paper-ready result | ⚠️ Only if accompanied by P6AB ablation |
