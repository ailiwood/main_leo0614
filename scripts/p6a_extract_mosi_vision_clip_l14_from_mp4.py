#!/usr/bin/env python
"""
P6A: Extract MOSI vision features from MP4 using CLIP ViT-L/14 (1024d).
Input: data/mosi/Raw/<video_id>/<segment>.mp4
Output: data/features_strong_sequence_mosi_v6_vision_l14_T32/{train,val,test}/*.npz
"""
import sys, os, time, glob
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config
MODEL_NAME = 'openai/clip-vit-large-patch14'
SOURCE_ROOT = 'data/features_strong_sequence_mosi_v3_T40'
MP4_ROOT = 'data/mosi/Raw'
OUTPUT_ROOT = 'data/features_strong_sequence_mosi_v6_vision_l14_T32'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FRAMES = 32

print(f'Loading {MODEL_NAME}...')
from transformers import CLIPProcessor, CLIPVisionModel
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPVisionModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
dim = model.config.hidden_size
print(f'✅ dim={dim}, Device={DEVICE}')

# Collect samples
all_samples = []
for split in ['train', 'val', 'test']:
    src_dir = os.path.join(SOURCE_ROOT, split)
    if not os.path.exists(src_dir): continue
    for fname in sorted(os.listdir(src_dir)):
        if fname.endswith('.npz'):
            all_samples.append((split, fname, os.path.join(src_dir, fname)))

print(f'Total: {len(all_samples)}')

failed, success = [], 0
t0 = time.time()

for split, fname, src_path in tqdm(all_samples, desc='Vision-L/14'):
    out_dir = os.path.join(OUTPUT_ROOT, split)
    out_path = os.path.join(out_dir, fname)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_path): success += 1; continue

    try:
        src = np.load(src_path, allow_pickle=True)
        sample_id = str(src['sample_id'])
        video_id = '_'.join(sample_id.split('_')[:-1])
        segment_num = sample_id.split('_')[-1]
        mp4_path = os.path.join(MP4_ROOT, video_id, f'{segment_num}.mp4')
        if not os.path.exists(mp4_path):
            mp4_path = os.path.join(MP4_ROOT, sample_id, f'{segment_num}.mp4')

        cap = cv2.VideoCapture(mp4_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release(); failed.append({'id': sample_id, 'reason': 'no frames'}); continue

        indices = np.linspace(0, frame_count-1, N_FRAMES, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()

        if len(frames) < N_FRAMES // 2:
            failed.append({'id': sample_id, 'reason': f'only {len(frames)} frames'}); continue

        # Pad/repeat if needed
        while len(frames) < N_FRAMES:
            frames.append(frames[-1])

        inputs = processor(images=frames, return_tensors='pt')
        with torch.no_grad():
            outputs = model(inputs.pixel_values.to(DEVICE))
            vision_seq = outputs.pooler_output.cpu().numpy()

        vision_mask = np.ones(N_FRAMES, dtype=np.int64)
        label = float(src['label'][0])
        text_seq, text_mask = src['text_seq'], src['text_mask']
        audio_seq, audio_mask = src['audio_seq'], src['audio_mask']

        np.savez_compressed(out_path,
            text_seq=text_seq, text_mask=text_mask,
            audio_seq=audio_seq, audio_mask=audio_mask,
            vision_seq=vision_seq, vision_mask=vision_mask,
            label=np.array([label], dtype=np.float32), sample_id=sample_id)
        success += 1
    except Exception as e:
        failed.append({'id': fname, 'reason': str(e)[:100]})

elapsed = time.time() - t0
print(f'Success: {success}/{len(all_samples)}, Failed: {len(failed)}, Time: {elapsed/60:.1f}min')
if failed:
    print(f'First 5 failures: {[f["id"] for f in failed[:5]]}')
for split in ['train','val','test']:
    d = os.path.join(OUTPUT_ROOT, split)
    if os.path.exists(d):
        files = [f for f in os.listdir(d) if f.endswith('.npz')]
        print(f'{split}: {len(files)} files')
