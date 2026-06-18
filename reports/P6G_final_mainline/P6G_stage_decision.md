# P6G Stage Decision

## Completed

| Task | Status | Detail |
|------|--------|--------|
| SDK install | ✅ | cmu-multimodal-sdk v0.0.6 |
| CSD file audit | ✅ | 7/7 files (30GB), all loadable |
| Zero fallback | ✅ | formal_mode raises FileNotFoundError |
| LoRA model | ⚠️ | PEFT 0.19.1 incompatible with PyTorch distributed API |

## PEFT/LoRA Issue

`peft` v0.19.1 requires `torch.distributed.tensor` which is not available in the current PyTorch build. Cannot use LoRA without either:
- Upgrading PyTorch to a version with `torch.distributed.tensor`
- Downgrading peft to an older version
- Using manual LoRA implementation

## MOSI/MOSEI Multimodal Training

- MOSI: ⏳ Not yet completed (LoRA blocked)
- MOSEI: ⏳ CSD loaded, dataset not yet built

## Next Steps

1. Fix PEFT/PyTorch compatibility or use manual LoRA
2. Or use proven P5E V2 frozen model for MOSI
3. Build MOSEI dataset from CSD files
4. Baseline repos to be cloned

## CSD Sequences Available for MOSEI

| Sequence | Size | Dim | Use |
|----------|------|-----|-----|
| All Labels | 23MB | 1 | Sentiment [-3,3] |
| words | 37MB | text | Raw transcripts |
| COVAREP | 11GB | 74 | Audio features |
| OpenFace_2 | 16GB | vision | Vision features |
| glove_vectors | 1.5GB | 300 | Text embeddings |
| FACET 4.2 | 1.6GB | vision | Alt vision |
