#!/usr/bin/env python
"""A1: Extract OpenFace2 vision features aligned to MOSEI utterance time windows."""
import os, sys, numpy as np, h5py, pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSD_PATH = 'data/cmu_mosei_comp_seq/CMU_MOSEI_OpenFace2.csd'
LABEL_PATH = 'data/processed/mosei_full/label.csv'
SRC_NPZ_DIR = 'data/processed/mosei_full'
OUT_DIR = 'data/processed/mosei_tav_openface2_v1'
MAX_VISION_LEN = 50  # Fixed vision sequence length

def main():
    label = pd.read_csv(LABEL_PATH)
    print(f'Segments: {len(label)}')

    with h5py.File(CSD_PATH, 'r') as of_f:
        of_data = of_f['OpenFace_2']['data']

        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)
            subset = label[label['mode'] == split]

            for _, row in tqdm(subset.iterrows(), total=len(subset), desc=split):
                video_id = row['video_id']
                clip_id = row['clip_id']
                sample_id = f'{video_id}_{clip_id}'

                # Read source npz (has audio + text features)
                src_npz = os.path.join(SRC_NPZ_DIR, split, f'{sample_id}.npz')
                if not os.path.exists(src_npz):
                    continue

                src = np.load(src_npz, allow_pickle=True)

                # Extract vision from OpenFace2 using time alignment
                vision_seq = np.zeros((MAX_VISION_LEN, 713), dtype=np.float32)
                vision_mask = np.zeros(MAX_VISION_LEN, dtype=np.int64)

                if video_id in of_data:
                    # Parse clip_id to get frame range (COVAREP frames)
                    parts = clip_id.split('_')
                    if len(parts) == 2:
                        start_frame = int(parts[0])
                        end_frame = int(parts[1])

                        # Get COVAREP time mapping (from source npz or estimate)
                        # COVAREP is at ~100Hz, so time = frame / 100
                        start_time = start_frame / 100.0
                        end_time = end_frame / 100.0

                        of_intervals = of_data[video_id]['intervals'][:]
                        of_features = of_data[video_id]['features'][:]

                        # Find OpenFace frames in time range
                        mask = (of_intervals[:, 0] >= start_time) & (of_intervals[:, 0] <= end_time)
                        vision_frames = of_features[mask]

                        if len(vision_frames) > 0:
                            # Downsample to MAX_VISION_LEN
                            n_frames = len(vision_frames)
                            if n_frames <= MAX_VISION_LEN:
                                vision_seq[:n_frames] = vision_frames
                                vision_mask[:n_frames] = 1
                            else:
                                indices = np.linspace(0, n_frames-1, MAX_VISION_LEN, dtype=int)
                                vision_seq[:MAX_VISION_LEN] = vision_frames[indices]
                                vision_mask[:MAX_VISION_LEN] = 1

                # Clean inf/nan
                vision_seq = np.nan_to_num(vision_seq, nan=0.0, posinf=0.0, neginf=0.0)

                # Save TAV npz
                out_path = os.path.join(OUT_DIR, split, f'{sample_id}.npz')
                np.savez_compressed(
                    out_path,
                    audio_seq=src['audio_seq'],
                    audio_mask=src['audio_mask'],
                    vision_seq=vision_seq,
                    vision_mask=vision_mask,
                )

    print(f'Done. Output: {OUT_DIR}')
    # Print stats
    for split in ['train', 'valid', 'test']:
        d = os.path.join(OUT_DIR, split)
        files = os.listdir(d) if os.path.exists(d) else []
        nz = 0
        for f in files[:100]:
            feat = np.load(os.path.join(d, f), allow_pickle=True)
            if feat['vision_mask'].sum() > 0:
                nz += 1
        print(f'  {split}: {len(files)} files, ~{nz}/min(100,{len(files)}) have non-zero vision')

if __name__ == '__main__':
    main()
