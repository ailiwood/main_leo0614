#!/usr/bin/env python
"""
scripts/p5g_extract_mosi_vision_large_full.py
P5G: Extract MOSI vision features using CLIP ViT-L/14 via transformers (1024d).

Output: data/features_strong_sequence_mosi_v3_vision_large/{train,val,test}/*.npz
"""
import sys, os, time, glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Config ---
MODEL_NAME = 'openai/clip-vit-large-patch14'
SOURCE_ROOT = 'data/features_strong_sequence_mosi_v3_T40'
FRAMES_ROOT = 'data/mosi/Frames'  # <video_id>/*.jpg
OUTPUT_ROOT = 'data/features_strong_sequence_mosi_v3_vision_large'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FRAMES = 20  # Uniformly sample N frames
BATCH_SIZE = 8  # Batch frames for faster encoding

# --- Load model ---
print(f'Loading {MODEL_NAME}...')
from transformers import CLIPProcessor, CLIPVisionModel
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPVisionModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
dim = model.config.hidden_size
print(f'Model loaded. Hidden size: {dim}, Device: {DEVICE}')

# --- Collect all samples ---
all_samples = []
for split in ['train', 'val', 'test']:
    src_dir = os.path.join(SOURCE_ROOT, split)
    if not os.path.exists(src_dir):
        continue
    for fname in sorted(os.listdir(src_dir)):
        if fname.endswith('.npz'):
            all_samples.append((split, fname, os.path.join(src_dir, fname)))

print(f'Total: {len(all_samples)} (Train:{sum(1 for s in all_samples if s[0]=="train")} '
      f'Val:{sum(1 for s in all_samples if s[0]=="val")} Test:{sum(1 for s in all_samples if s[0]=="test")})')

# --- Process ---
failed = []
t0 = time.time()
success = 0

for split, fname, src_path in tqdm(all_samples, desc='Vision extract'):
    out_dir = os.path.join(OUTPUT_ROOT, split)
    out_path = os.path.join(out_dir, fname)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        success += 1
        continue

    try:
        src = np.load(src_path, allow_pickle=True)
        sample_id = str(src['sample_id'])
        label = float(src['label'][0]) if 'label' in src else 0.0

        # sample_id is like "03bSnISJMiM_1", frame dir is "03bSnISJMiM"
        video_id = '_'.join(sample_id.split('_')[:-1]) if '_' in sample_id else sample_id
        frame_dir = os.path.join(FRAMES_ROOT, video_id)
        if not os.path.exists(frame_dir):
            frame_dir = os.path.join(FRAMES_ROOT, sample_id)  # fallback
        if not os.path.exists(frame_dir):
            failed.append({'sample_id': sample_id, 'split': split, 'reason': 'no frames dir'})
            continue

        frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
        if not frame_files:
            failed.append({'sample_id': sample_id, 'split': split, 'reason': 'no jpg files'})
            continue

        # Uniform sampling
        n_total = len(frame_files)
        if n_total <= N_FRAMES:
            sampled = frame_files
        else:
            indices = np.linspace(0, n_total - 1, N_FRAMES, dtype=int)
            sampled = [frame_files[i] for i in indices]

        # Batch process frames
        images = [Image.open(f).convert('RGB') for f in sampled]
        inputs = processor(images=images, return_tensors='pt')
        pixel_values = inputs.pixel_values.to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values)
            # Use pooler_output for per-image representation
            vision_seq = outputs.pooler_output.cpu().numpy()  # [N_FRAMES, dim]
            # Or use last_hidden_state[:,0,:] for CLS token
            # vision_seq = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        T = vision_seq.shape[0]
        vision_mask = np.ones(T, dtype=np.int64)

        # Copy text and audio from source
        text_seq = src['text_seq']
        text_mask = src['text_mask']
        audio_seq = src['audio_seq']
        audio_mask = src['audio_mask']

        np.savez_compressed(
            out_path,
            text_seq=text_seq, text_mask=text_mask,
            audio_seq=audio_seq, audio_mask=audio_mask,
            vision_seq=vision_seq, vision_mask=vision_mask,
            label=np.array([label], dtype=np.float32),
            sample_id=sample_id,
        )
        success += 1

    except Exception as e:
        failed.append({'sample_id': fname, 'split': split, 'reason': str(e)[:100]})

elapsed = time.time() - t0
print(f'\n=== Done ===')
print(f'Success: {success}/{len(all_samples)}, Failed: {len(failed)}, Time: {elapsed/60:.1f}min')

if failed:
    print(f'First 10 failures:')
    for f in failed[:10]:
        print(f'  {f["sample_id"]} ({f["split"]}): {f["reason"]}')

# Audit
for split in ['train', 'val', 'test']:
    d = os.path.join(OUTPUT_ROOT, split)
    if os.path.exists(d):
        files = [f for f in os.listdir(d) if f.endswith('.npz')]
        print(f'{split}: {len(files)} files')
