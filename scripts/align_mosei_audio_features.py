#!/usr/bin/env python
"""P6S-Repair-2: Align MOSEI CSD audio features with CSV labels via time intervals."""
import h5py, numpy as np, os, csv, json

CSD_DIR = 'data/cmu_mosei_comp_seq'
CSV_PATH = 'data/mosei/label.csv'
OUT_DIR = 'data/processed/mosei_full'

os.makedirs(OUT_DIR, exist_ok=True)

# Load audio CSD
print('Loading COVAREP CSD...')
f_aud = h5py.File(os.path.join(CSD_DIR, 'CMU_MOSEI_COVAREP.csd'), 'r')
audio_data = f_aud['COVAREP']['data']

# Load labels CSD for interval matching
print('Loading Labels CSD...')
f_lab = h5py.File(os.path.join(CSD_DIR, 'CMU_MOSEI_Labels.csd'), 'r')
label_data = f_lab['All Labels']['data']

# Load CSV segments
print('Loading CSV...')
csv_segments = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        csv_segments.append(row)

# Build label CSD interval index: video_id -> [(start, end, segment_idx), ...]
label_intervals = {}
for vid in label_data.keys():
    seg = label_data[vid]
    intervals = seg['intervals'][:]
    label_intervals[vid] = intervals

# Match: for each CSV row, find the closest label CSD interval
matched = 0
unmatched = 0
match_report = []

for split_name in ['train', 'valid', 'test']:
    split_dir = os.path.join(OUT_DIR, split_name, 'audio')
    os.makedirs(split_dir, exist_ok=True)

for row in csv_segments:
    vid = row['video_id']
    clip_id = row['clip_id']  # format: start_end (e.g., 82_7645)
    mode = row['mode']

    # Parse clip_id to get start time
    # clip_id format: start_end → convert to start.end (e.g., 82_7645 → 82.7645)
    start_str = clip_id.replace('_', '.')
    try:
        start_time = float(start_str)
    except ValueError:
        unmatched += 1
        continue

    # Find matching interval in label CSD
    if vid not in label_intervals:
        unmatched += 1
        continue

    intervals = label_intervals[vid]
    # Find closest start time (within tolerance)
    tol = 0.5  # 500ms tolerance (handle annotation timing jitter)
    diffs = np.abs(intervals[:, 0] - start_time)
    best_idx = np.argmin(diffs)
    best_diff = diffs[best_idx]

    if best_diff > tol:
        unmatched += 1
        if unmatched <= 5:
            match_report.append(f'UNMATCHED: {vid}_{clip_id} start={start_time} best_diff={best_diff:.3f}')
        continue

    # Extract audio for this segment from COVAREP
    matched_start = intervals[best_idx, 0]
    matched_end = intervals[best_idx, 1]

    if vid in audio_data:
        aud_seg = audio_data[vid]
        aud_intervals = aud_seg['intervals'][:]
        aud_feats = aud_seg['features'][:]  # [T, 74]

        # Time-slice audio features
        time_mask = (aud_intervals[:, 0] >= matched_start) & (aud_intervals[:, 1] <= matched_end)
        if time_mask.sum() == 0:
            # Fallback: use all audio for this video
            audio_slice = aud_feats
        else:
            audio_slice = aud_feats[time_mask]

        # Save audio feature
        sample_id = f'{vid}_{clip_id}'
        out_path = os.path.join(OUT_DIR, row['mode'], 'audio', f'{sample_id}.npy')
        np.save(out_path, audio_slice.astype(np.float32))
        matched += 1

print(f'\nMatched: {matched}/{len(csv_segments)} ({matched/len(csv_segments)*100:.1f}%)')
print(f'Unmatched: {unmatched}')
print(f'Sample unmatched: {match_report[:10]}')

# Save stats
stats = {'csv_total': len(csv_segments), 'matched': matched, 'unmatched': unmatched,
         'match_rate': matched/len(csv_segments), 'audio_dim': 74}
json.dump(stats, open(os.path.join(OUT_DIR, 'feature_stats.json'), 'w'), indent=2)

f_aud.close(); f_lab.close()
print(f'Saved to {OUT_DIR}')
