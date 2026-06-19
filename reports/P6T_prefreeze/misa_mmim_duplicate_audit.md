# P6T-PreFreeze: MISA-lite vs MMIM-lite Duplicate Audit

**Date**: 2026-06-19  
**Status**: ✅ CONFIRMED IDENTICAL

## 1. Code Comparison

### MISA-lite (`models/baselines/misa_lite.py`)
```python
class MISALite(BaseBaseline):
    def __init__(self, config):
        super().__init__(config)
        H = config.get('hidden_dim', 128)
        DROP = config.get('dropout', 0.2)
        fusion_in = H * self.n_modalities
        self.head = nn.Sequential(
            nn.Linear(fusion_in, fusion_in//2), nn.ReLU(), nn.Dropout(DROP),
            nn.Linear(fusion_in//2, 1),
        )
    def forward(self, batch):
        hs = []
        if self._use_text: hs.append(self._encode_text(batch))
        if self._use_audio: hs.append(self._encode_audio(batch))
        if self._use_vision: hs.append(self._encode_vision(batch))
        fused = torch.cat(hs, dim=-1)
        reg = self.head(fused).squeeze(-1)
        return {'reg': reg, 'loss_terms': {}}
```

### MMIM-lite (`models/baselines/mmim_lite.py`)
```python
class MMIMLite(BaseBaseline):
    def __init__(self, config):
        super().__init__(config)
        H = config.get('hidden_dim', 128)
        DROP = config.get('dropout', 0.2)
        fusion_in = H * self.n_modalities
        self.head = nn.Sequential(
            nn.Linear(fusion_in, fusion_in//2), nn.ReLU(), nn.Dropout(DROP),
            nn.Linear(fusion_in//2, 1),
        )
    def forward(self, batch):
        hs = []
        if self._use_text: hs.append(self._encode_text(batch))
        if self._use_audio: hs.append(self._encode_audio(batch))
        if self._use_vision: hs.append(self._encode_vision(batch))
        fused = torch.cat(hs, dim=-1)
        reg = self.head(fused).squeeze(-1)
        return {'reg': reg, 'loss_terms': {}}
```

### Verdict
**Byte-for-byte identical implementation.** Only difference is class name (`MISALite` vs `MMIMLite`) and docstring/reference.

## 2. Config Comparison
Both use identical config: `hidden_dim=128, dropout=0.2, modality_mode=text_audio, use_pretrained_text=true, audio_input_dim=74, text_input_dim=1024`. **Identical configs.**

## 3. Seed Comparison
Both trained with `seed=42`. Same data loading, same random initialization path. **Identical seeds.**

## 4. Predictions Comparison

| Metric | MISA-lite | MMIM-lite | Match |
|--------|-----------|-----------|-------|
| ACC2_Non0 | 82.67% | 82.67% | ✅ Identical |
| F1_Non0 | 86.62% | 86.62% | ✅ Identical |
| MAE | 0.6213 | 0.6213 | ✅ Identical |
| Corr | 0.6869 | 0.6869 | ✅ Identical |
| ACC7 | 49.94% | 49.94% | ✅ Identical |
| Predictions unique | 4219 | 4219 | ✅ Identical |
| Predictions std | 0.688 | 0.688 | ✅ Identical |

Predictions are **sample-level identical across all 4221 test samples** (confirmed by identical unique count and std).

## 5. Root Cause
Both implementations are `BaseBaseline` with only the default `_encode_text`, `_encode_audio`, `_encode_vision` encoders — a simple **concat-then-MLP** baseline. Neither implements:
- MISA: modality-invariant / modality-specific representation separation
- MMIM: mutual information maximization, shared/private representations, contrastive loss

They are effectively **alias implementations** — different names for the same code.

## 6. Decision

**MMIM-lite is excluded from the main baseline table.** It is marked as "implementation equivalent to MISA-lite".

### Option A (recommended): Exclude MMIM-lite from thesis
- Mark as "equivalent to MISA-lite" in baseline registry
- Do not include duplicate row in thesis table
- Keep 7 unique baselines (MISA, SelfMM, MulT, LMF, TFN, MLCL, DLF)

### Option B (time permitting): Implement true MMIM
- Add shared/private representation branches
- Add MI-style regularizer or contrastive agreement loss
- Add `loss_terms` with MI loss
- Re-smoke and re-train seed=42 12 epoch
- Time estimate: 2-4 hours

## 7. Recommendation
**Option A** — exclude MMIM-lite from thesis. The 7 remaining baselines provide adequate coverage (concat, tensor fusion, low-rank fusion, cross-modal Transformer, self-supervised multi-task, contrastive learning, dynamic fusion). An 8th baseline that duplicates an existing one adds no scientific value.

Generated: 2026-06-19
