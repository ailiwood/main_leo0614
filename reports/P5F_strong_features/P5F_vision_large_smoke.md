# P5F Vision Large Smoke Test

## Model: openai/clip-vit-large-patch14 (via transformers)

| Item | Result |
|------|--------|
| Downloaded | ✅ 1.71GB |
| Hidden size | **1024** (vs base B/32 768, +33%) |
| Loading | ✅ Successful via transformers CLIPVisionModel |
| Video path | data/mosi/Frames/<id>/*.jpg (frame images) |

## Recommendation

✅ Proceed with full extraction if time allows (~30-45 min for 2199 samples)
⚠️ Need to adapt extraction for frame-level images
⚠️ Feature format: 1024d frames (vs current 768d)
