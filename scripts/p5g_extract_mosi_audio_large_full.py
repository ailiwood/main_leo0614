#!/usr/bin/env python
"""
scripts/p5g_extract_mosi_audio_large_full.py
P5G: Extract MOSI audio features using wav2vec2-large-960h-lv60-self (1024d).

Output: data/features_strong_sequence_mosi_v3_audio_large/{train,val,test}/*.npz
Each npz contains: audio_seq (1024d), audio_mask — text/vision/label inherited from v3_T40.
"""
import sys, os, time, csv, json
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Config ---
MODEL_NAME = 'facebook/wav2vec2-large-960h-lv60-self'
SOURCE_ROOT = 'data/features_strong_sequence_mosi_v3_T40'  # For text, vision, labels, splits
AUDIO_ROOT = 'data/mosi/wav'  # Raw segmented WAVs per utterance
OUTPUT_ROOT = 'data/features_strong_sequence_mosi_v3_audio_large'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_SR = 16000
MAX_AUDIO_LEN = 200000  # ~12.5s at 16kHz
BATCH_SIZE = 1  # Process one sample at a time

# --- Load model ---
print(f'Loading {MODEL_NAME}...')
from transformers import Wav2Vec2Processor, Wav2Vec2Model
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print(f'Model loaded. Hidden size: {model.config.hidden_size}, Device: {DEVICE}')

# --- Collect all samples from source ---
all_samples = []
for split in ['train', 'val', 'test']:
    src_dir = os.path.join(SOURCE_ROOT, split)
    if not os.path.exists(src_dir):
        print(f'WARNING: {src_dir} not found, skipping')
        continue
    for fname in sorted(os.listdir(src_dir)):
        if fname.endswith('.npz'):
            all_samples.append((split, fname, os.path.join(src_dir, fname)))

print(f'Total samples to process: {len(all_samples)}')
print(f'  Train: {sum(1 for s in all_samples if s[0]=="train")}')
print(f'  Val:   {sum(1 for s in all_samples if s[0]=="val")}')
print(f'  Test:  {sum(1 for s in all_samples if s[0]=="test")}')

# --- Process ---
failed = []
t0_total = time.time()
success_count = 0

for split, fname, src_path in tqdm(all_samples, desc='Extracting'):
    out_dir = os.path.join(OUTPUT_ROOT, split)
    out_path = os.path.join(out_dir, fname)
    os.makedirs(out_dir, exist_ok=True)

    # Skip if already done
    if os.path.exists(out_path):
        success_count += 1
        continue

    try:
        # Load source npz for metadata
        src = np.load(src_path, allow_pickle=True)
        sample_id = str(src['sample_id'])
        label = float(src['label'][0]) if 'label' in src else 0.0

        # Find audio segments: sample_id is like "03bSnISJMiM_1", audio dir is "03bSnISJMiM"
        # Strip trailing _N segment suffix to get video_id
        video_id = '_'.join(sample_id.split('_')[:-1]) if '_' in sample_id else sample_id
        # Also try direct match for samples without segment suffix
        audio_dir = os.path.join(AUDIO_ROOT, video_id)
        if not os.path.exists(audio_dir):
            audio_dir = os.path.join(AUDIO_ROOT, sample_id)  # fallback
        if not os.path.exists(audio_dir):
            failed.append({'sample_id': sample_id, 'split': split, 'reason': 'no audio directory'})
            continue

        # Find the specific segment WAV: sample_id "X_Y" → wav file "Y.wav" in dir "X"
        segment_num = sample_id.split('_')[-1] if '_' in sample_id else '1'
        wav_path = os.path.join(audio_dir, f'{segment_num}.wav')
        if not os.path.exists(wav_path):
            # Try loading all wavs and concatenating as fallback
            wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')],
                              key=lambda x: int(x.replace('.wav', '')))
            if not wav_files:
                failed.append({'sample_id': sample_id, 'split': split, 'reason': 'no wav files'})
                continue
            all_audio = []
            for wf in wav_files:
                audio, sr = sf.read(os.path.join(audio_dir, wf))
                if len(audio.shape) > 1: audio = audio.mean(axis=1)
                if sr != TARGET_SR:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
                all_audio.append(audio)
            audio_concat = np.concatenate(all_audio)
        else:
            audio_concat, sr = sf.read(wav_path)
            if len(audio_concat.shape) > 1: audio_concat = audio_concat.mean(axis=1)
            if sr != TARGET_SR:
                import librosa
                audio_concat = librosa.resample(audio_concat, orig_sr=sr, target_sr=TARGET_SR)

        audio_concat = audio_concat[:MAX_AUDIO_LEN]

        # Process through wav2vec2-large
        inputs = processor(audio_concat, sampling_rate=TARGET_SR, return_tensors='pt')
        input_values = inputs.input_values.to(DEVICE)

        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=False)

        audio_seq = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # [T, 1024]
        T = audio_seq.shape[0]
        audio_mask = np.ones(T, dtype=np.int64)

        # Copy text and vision from source
        text_seq = src['text_seq']
        text_mask = src['text_mask']
        vision_seq = src['vision_seq']
        vision_mask = src['vision_mask']

        # Save
        np.savez_compressed(
            out_path,
            text_seq=text_seq, text_mask=text_mask,
            audio_seq=audio_seq, audio_mask=audio_mask,
            vision_seq=vision_seq, vision_mask=vision_mask,
            label=np.array([label], dtype=np.float32),
            sample_id=sample_id,
        )
        success_count += 1

    except Exception as e:
        failed.append({'sample_id': fname, 'split': split, 'reason': str(e)})

elapsed = time.time() - t0_total
print(f'\n=== Extraction Complete ===')
print(f'Success: {success_count}/{len(all_samples)}')
print(f'Failed:  {len(failed)}')
print(f'Time:    {elapsed/60:.1f} min')

if failed:
    print('\nFailed samples:')
    for f in failed[:10]:
        print(f'  {f["sample_id"]} ({f["split"]}): {f["reason"]}')

# Audit
print('\n=== Feature Audit ===')
for split in ['train', 'val', 'test']:
    d = os.path.join(OUTPUT_ROOT, split)
    if os.path.exists(d):
        files = [f for f in os.listdir(d) if f.endswith('.npz')]
        print(f'{split}: {len(files)} files')
        if files:
            s = np.load(os.path.join(d, files[0]))
            print(f'  audio_dim={s["audio_seq"].shape[-1]}, text_dim={s["text_seq"].shape[-1]}, vision_dim={s["vision_seq"].shape[-1]}')
