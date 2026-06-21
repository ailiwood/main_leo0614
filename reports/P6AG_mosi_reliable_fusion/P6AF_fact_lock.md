# P6AF Fact Lock — Basis for P6AG

**Date**: 2026-06-21
**Purpose**: Freeze diagnostic facts from P6AF before P6AG implementation

## Established Facts

1. **MOSI train samples**: 1,284 (val: 229, test: 686, total: 2,199)
2. **Text-only probe (P1)**: ACC2 = 86.11% — healthy, text pipeline works
3. **Audio-only probe (P2)**: ACC2 = 56.94% — below majority (59.8%), no standalone signal
4. **Vision-only probe (P3)**: ACC2 = 57.41% — below majority, no standalone signal
5. **Feature temporality (G0)**: Both audio (100 frames) and vision (32 frames) are Route-S — valid temporal sequences supporting sLSTM
6. **Canonical AWAF failure**:
   - P6AD: 4-epoch frozen → ACC2=42.23% (collapse)
   - P6AF Route A: 35-epoch all-trainable → ACC2=44.21% (collapse)
7. **Failure scope**: Limited to "forced three-modal softmax weighted-sum Canonical AWAF"
8. **Not disproven**: "MOSI multimodal information is completely useless"
9. **P6AG hypothesis**: Text-anchored architecture with reliability-gated auxiliary corrections can recover multimodal benefit on MOSI's small dataset

## P6K Reference

- P6K text-audio residual: ACC2=88.72% (seed 42)
- Architecture: text_base + gate * delta(audio) — preserves text baseline
- This demonstrates that with text preservation, AUXILIARY modalities CAN help on MOSI
