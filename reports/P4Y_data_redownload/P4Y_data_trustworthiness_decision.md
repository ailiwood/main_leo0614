# P4Y Data Trustworthiness Decision

## MOSI: Grade B+ (Trustworthy with notes)

| Criteria | Status | Note |
|----------|:--:|------|
| MP4 video files | ✅ 2199 files | Complete, extracted from YouTube by previous user |
| WAV audio | ✅ 2199 files | Complete |
| Text/transcript | ✅ label.csv | 2199 rows with text field |
| Labels [-3,+3] | ✅ | Standard sentiment labels |
| Split (train/val/test) | ✅ | 1284/229/686 |
| Source traceable | ⚠ | Not from CMU SDK - likely user assembled |
| **Decision** | **KEEP** | Too valuable to delete. MP4 files enable Vision T>1. |

## MOSEI: Grade C (Incomplete - needs SDK features)

| Criteria | Status | Note |
|----------|:--:|------|
| MP4 video | ❌ 0 files | **Not available from CMU** (YouTube privacy restriction) |
| WAV audio | ✅ 8636 files | Complete |
| Text/transcript | ✅ CSV labels | With ASR and text |
| Labels [-3,+3] | ✅ | Standard sentiment |
| Old .features | ❌ Unreliable | 0-1 labels, inconsistent dimensions |
| SDK .csd features | ❌ Missing | Need to download via CMU-MultimodalSDK |
| **Decision** | **DELETE old .features, DOWNLOAD SDK** | SDK provides standard COVAREP+FACET+GloVe |

## Key Finding from Official Source Audit

**MOSEI raw MP4 videos are NOT publicly available.** CMU MultimodalSDK explicitly states: "we are not allowed to share the raw videos due to privacy of YouTube content creators." The official dataset provides computational sequences (.csd files) with pre-extracted features, not raw videos. This is the standard research practice - most papers use these features.

For MOSI, our local MP4 collection is actually a rare asset that enables Vision T>1 extraction.
