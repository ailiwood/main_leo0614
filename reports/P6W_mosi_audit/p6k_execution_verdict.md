# P6K Execution Trace Verdict

**Date**: 2026-06-20 | **Commit**: `59b216f`

## Verdict

**P6K text_audio (88.72%) does NOT use AWAF, gate, or delta.**

### Code Evidence

```python
# P6K needs_awaf:
def needs_awaf(self) -> bool:
    return self.mode in ('text_av_residual', 'text_confidence_residual', 'av_only')
# text_audio_residual NOT included → AWAF NOT created

# P6K _forward_text_x_residual:
if which == 'audio':
    z = hap  # just audio features, NOT AWAF output
# gate defaults to off → gr = ones_like(rtb)
# delta defaults to off → bdr = zeros, dsr = 0
reg = rtb + 0  # pure text base prediction
```

### What P6K 88.72% Actually Is

- RoBERTa-large + LoRA text encoder
- Co-trained with audio features (audio branch exists but output ignored)
- No fusion, no gate, no delta, no AWAF
- The 88.72% comes from multimodal co-training improving text representations

### What P6K text_av (86.13%) Is

- Uses AWAF (text_av_residual mode)
- AWAF fuses text + audio + vision
- Actually calls `self.awaf(ht, hap, hvp)`

### Implications for Thesis

1. P6K text_audio 88.72% CANNOT be called "AWAF+sLSTM main model"
2. It should be called "text_audio conservative mainline" (as P6K named it)
3. No ablation of AWAF/sLSTM is possible on this path — AWAF doesn't participate
4. AWAF ablation requires text_av mode (which includes vision)

### Recommendation

The P6K text_audio path is NOT suitable for AWAF/fusion ablation. 
For MOSI ablation with AWAF, use text_av mode (86.13% baseline) where AWAF actually runs.
Or accept that the MOSI main result (88.72%) is a text-co-training result, not a fusion result.

Generated: 2026-06-20
