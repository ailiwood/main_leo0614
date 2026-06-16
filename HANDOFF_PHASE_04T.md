# HANDOFF_PHASE_04T.md

## P4T Status: Partially Complete

### Done
1. **Cleanup classification**: A/B/C/D/E categories defined. No files deleted.
2. **GitHub prep**: .gitignore updated, README_PROJECT_STATUS.md written.
3. **Strong sequence extraction script**: `scripts/extract_mosi_strong_sequence.py` written.
4. **P4R/P4S audit reports**: Complete.
5. **Key finding from P4S**: TMDC-v1 T=1 (75.25%) >> MLCL T~12 (44.9%).

### In Progress
- Strong sequence feature extraction: DeBERTa-large (1.63GB) downloaded; wav2vec2-large (1.26GB) downloading; then 2200 sample processing.

### Blocked
- C0 strict protocol retraining on strong sequence features (need extraction first)

## Path Forward

**Option 1**: Wait for full extraction (~2-4 hours) then retrain C0.
**Option 2**: Accept TMDC-v1 T=1 as primary features (with T=1 limitation noted in paper).
**Option 3**: Use TMDC features for paper + add discussion of why sequence features were explored.

## Project State Summary
- Core model: C0 (sLSTM+AWAF), 3.06M params, MOSI ACC2_NZ=75.25%
- Strict protocol: ✅ Fixed (val-tuned, test-once)
- Metrics: ACC2 reg_sign (MMSA convention)
- Main limitation: T=1 features (honest in paper)
