#!/usr/bin/env python
"""P6S: Extract MOSEI features from CSD files to TextFTMultimodalDataset format."""
import h5py, numpy as np, os, csv, json, argparse, random
from collections import defaultdict

random.seed(42)
CSD_DIR = 'data/cmu_mosei_comp_seq'
OUT_DIR = 'data/processed/mosei_text_audio'


def get_segments(csd_path):
    f = h5py.File(csd_path, 'r')
    key = list(f.keys())[0]
    segs = list(f[key]['data'].keys())
    f.close()
    return segs


def get_features(csd_path, seg_id):
    f = h5py.File(csd_path, 'r')
    key = list(f.keys())[0]
    feats = f[key]['data'][seg_id]['features'][:]
    f.close()
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    # Find aligned segments
    labels_segs = set(get_segments(os.path.join(CSD_DIR, 'CMU_MOSEI_Labels.csd')))
    audio_segs = set(get_segments(os.path.join(CSD_DIR, 'CMU_MOSEI_COVAREP.csd')))
    # Use WordVectors for text features (GloVe 300d)
    text_segs = set(get_segments(os.path.join(CSD_DIR, 'CMU_MOSEI_TimestampedWordVectors.csd')))

    aligned = sorted(labels_segs & audio_segs & text_segs)
    if args.limit:
        aligned = aligned[:args.limit]
    print(f'Aligned segments: {len(aligned)}')

    if args.dry_run:
        seg = aligned[0]
        lab = get_features(os.path.join(CSD_DIR, 'CMU_MOSEI_Labels.csd'), seg)
        aud = get_features(os.path.join(CSD_DIR, 'CMU_MOSEI_COVAREP.csd'), seg)
        txt = get_features(os.path.join(CSD_DIR, 'CMU_MOSEI_TimestampedWordVectors.csd'), seg)
        print(f'Label shape: {lab.shape} (sentiment={lab[0,0]:.3f})')
        print(f'Audio shape: {aud.shape} (COVAREP 74d)')
        print(f'Text shape: {txt.shape} (GloVe 300d)')
        return

    # Split: 70/15/15
    random.shuffle(aligned)
    n = len(aligned)
    train = aligned[:int(n*0.7)]
    val = aligned[int(n*0.7):int(n*0.85)]
    test = aligned[int(n*0.85):]
    print(f'Split: train={len(train)} val={len(val)} test={len(test)}')

    # Create output
    os.makedirs(OUT_DIR, exist_ok=True)
    for split_name, segs in [('train', train), ('val', val), ('test', test)]:
        split_dir = os.path.join(OUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        audio_dir = os.path.join(split_dir, 'audio')
        text_dir = os.path.join(split_dir, 'text')
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)

        for seg in segs:
            # Audio
            aud = get_features(os.path.join(CSD_DIR, 'CMU_MOSEI_COVAREP.csd'), seg)
            np.save(os.path.join(audio_dir, f'{seg}.npy'), aud.astype(np.float32))
            # Text (GloVe)
            txt = get_features(os.path.join(CSD_DIR, 'CMU_MOSEI_TimestampedWordVectors.csd'), seg)
            np.save(os.path.join(text_dir, f'{seg}.npy'), txt.astype(np.float32))
        print(f'  {split_name}: {len(segs)} segments saved')

    # label.csv
    with open(os.path.join(OUT_DIR, 'label.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video_id', 'clip_id', 'text', 'label', 'mode'])
        for split_name, segs in [('train', train), ('val', val), ('test', test)]:
            for seg in segs:
                lab = get_features(os.path.join(CSD_DIR, 'CMU_MOSEI_Labels.csd'), seg)
                sentiment = float(lab[0, 0])
                # segment ID format: video_id_clip_id
                parts = seg.rsplit('_', 1) if '_' in seg else [seg, '0']
                w.writerow([parts[0], parts[1] if len(parts) > 1 else '0', '', sentiment, split_name])

    # Stats
    stats = {
        'total_segments': n,
        'train': len(train), 'val': len(val), 'test': len(test),
        'audio_dim': 74, 'text_dim': 300,
        'split_source': 'random_70_15_15', 'official_split': False,
        'note': 'CMU-MOSEI CSD subset, NOT official full MOSEI (22,856). Split is random, not official.'
    }
    json.dump(stats, open(os.path.join(OUT_DIR, 'feature_stats.json'), 'w'), indent=2)
    print(f'Saved: {OUT_DIR}')
    print(f'WARNING: Split is random 70/15/15, NOT official MOSEI split.')


if __name__ == '__main__':
    main()
