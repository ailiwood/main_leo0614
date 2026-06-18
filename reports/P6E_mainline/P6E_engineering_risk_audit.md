# P6E Engineering Risk Audit

| Risk | Severity | Status |
|------|----------|--------|
| Zero fallback AV features | 🔴 CRITICAL | ✅ Fixed — formal_mode raises error |
| Text-only disguised as multimodal | 🔴 CRITICAL | ⚠️ Must verify in training |
| Test set used for best epoch | 🔴 CRITICAL | ✅ strict protocol enforced |
| Hardcoded config | 🟡 MEDIUM | ⏳ Moving to YAML config |
| No baseline runner | 🟡 MEDIUM | ❌ Not started |
| MOSEI no AV features | 🔴 CRITICAL | ⏳ SDK installed, need .csd data |
| 358M model too slow | 🟡 MEDIUM | ⚠️ May need LoRA or frozen approach |
