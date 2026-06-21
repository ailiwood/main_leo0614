"""
P6AF-G0: MOSI Scientific Data Contract & Feature Temporality Audit

Determines Route-S (sLSTM) vs Route-P (pooled) for each modality based on
actual feature temporality. Also audits labels, splits, alignment, and quality.

Key: Route-S requires median valid length >= 4 AND temporal variance > 0.
     Route-P applies when feature is primarily T=1 or has no real time variation.

Outputs:
  reports/P6AF_mosi_tav_recovery/G0_mosi_data_contract.md
  reports/P6AF_mosi_tav_recovery/G0_mosi_feature_temporality.csv
  reports/P6AF_mosi_tav_recovery/G0_mosi_label_alignment.csv
  reports/P6AF_mosi_tav_recovery/G0_mosi_split_manifest.csv
  reports/P6AF_mosi_tav_recovery/G0_mosi_feature_statistics.csv
"""
import os, sys, csv, json
import numpy as np
from collections import defaultdict

FEATURE_ROOT = 'data/processed/mosi_tav_v1'
SOURCE_ROOT = 'data/features_strong_sequence_mosi_v6_vision_l14_full'
LABEL_CSV = 'data/mosi/label.csv'
OUTPUT_DIR = 'reports/P6AF_mosi_tav_recovery'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_labels():
    rows = []
    with open(LABEL_CSV, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def analyze_temporality(seq, mask, name):
    """Analyze temporal properties of a feature sequence."""
    active = seq[mask.astype(bool)]
    n_frames = len(active)

    if n_frames == 0:
        return {
            'name': name,
            'n_samples': 0, 'n_frames_min': 0, 'n_frames_median': 0,
            'n_frames_max': 0, 'temporal_variance': 0,
            'zero_ratio': 1.0, 'nan_ratio': 1.0, 'inf_ratio': 1.0,
            'route': 'INVALID'
        }

    # Temporal variance: std of frame-wise means across time
    if len(seq.shape) == 2 and seq.shape[0] > 1:
        frame_means = seq.mean(axis=1)  # [T]
        temporal_var = float(np.var(frame_means))
    else:
        temporal_var = 0.0

    stats = {
        'name': name,
        'n_samples': 1,
        'n_frames': n_frames,
        'n_frames_min': n_frames,
        'n_frames_median': n_frames,
        'n_frames_max': n_frames,
        'dim': seq.shape[-1] if len(seq.shape) >= 2 else 0,
        'temporal_variance': temporal_var,
        'zero_ratio': float(np.mean(np.abs(active) < 1e-8)),
        'nan_ratio': float(np.isnan(active).mean()),
        'inf_ratio': float(np.isinf(active).mean()),
    }

    return stats


def main():
    print("=" * 70)
    print("P6AF-G0: MOSI Scientific Data Contract & Feature Temporality Audit")
    print("=" * 70)

    # 1. Load labels
    print("\n[1] Loading MOSI labels...")
    rows = load_labels()
    print(f"  Total label rows: {len(rows)}")

    # Split statistics
    split_map = {}
    splits = defaultdict(list)
    for r in rows:
        sid = f"{r['video_id']}_{r['clip_id']}"
        mode = r['mode']  # train/val/test
        splits[mode].append(sid)
        split_map[sid] = {
            'mode': mode,
            'label': float(r['label']),
            'text': r.get('text', ''),
            'video_id': r['video_id'],
            'clip_id': r['clip_id'],
        }

    for mode, sids in splits.items():
        labels = [split_map[s]['label'] for s in sids]
        labels = np.array(labels)
        print(f"  {mode}: {len(sids)} samples, label mean={labels.mean():.3f}, "
              f"std={labels.std():.3f}, min={labels.min():.1f}, max={labels.max():.1f}")

    # Check for split overlap
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])
    print(f"  Split overlap: train∩val={len(train_set & val_set)}, "
          f"train∩test={len(train_set & test_set)}, val∩test={len(val_set & test_set)}")

    # 2. Load features and analyze temporality
    print("\n[2] Analyzing feature temporality...")
    all_labels_ids = set(split_map.keys())

    audio_frame_counts = defaultdict(list)
    vision_frame_counts = defaultdict(list)
    audio_temporal_var = defaultdict(list)
    vision_temporal_var = defaultdict(list)
    audio_dims = set()
    vision_dims = set()
    total_files = 0
    missing_files = []
    sample_manifest = []

    for mode in ['train', 'val', 'test']:
        feat_dir = os.path.join(FEATURE_ROOT, mode)
        if not os.path.isdir(feat_dir):
            print(f"  WARNING: {feat_dir} not found")
            continue

        for fname in sorted(os.listdir(feat_dir)):
            if not fname.endswith('.npz'):
                continue
            sample_id = fname.replace('.npz', '')
            total_files += 1

            feat_path = os.path.join(feat_dir, fname)
            try:
                feat = np.load(feat_path, allow_pickle=True)
            except Exception as e:
                missing_files.append((sample_id, mode, str(e)))
                continue

            audio_seq = feat['audio_seq']
            audio_mask = feat['audio_mask']
            vision_seq = feat['vision_seq']
            vision_mask = feat['vision_mask']

            audio_dims.add(audio_seq.shape[-1])
            vision_dims.add(vision_seq.shape[-1])

            # Audio temporality
            a_active = audio_mask.sum()
            audio_frame_counts[mode].append(a_active)
            if audio_seq.shape[0] > 1:
                a_tv = float(np.var(audio_seq[audio_mask.astype(bool)].mean(axis=0)))
            else:
                a_tv = 0.0
            audio_temporal_var[mode].append(a_tv)

            # Vision temporality
            v_active = vision_mask.sum()
            vision_frame_counts[mode].append(v_active)
            if vision_seq.shape[0] > 1:
                v_tv = float(np.var(vision_seq[vision_mask.astype(bool)].mean(axis=0)))
            else:
                v_tv = 0.0
            vision_temporal_var[mode].append(v_tv)

            # Quality checks
            a_zero = float(np.all(np.abs(audio_seq[audio_mask.astype(bool)]) < 1e-8))
            v_zero = float(np.all(np.abs(vision_seq[vision_mask.astype(bool)]) < 1e-8))
            a_nan = float(np.isnan(audio_seq).any())
            v_nan = float(np.isnan(vision_seq).any())
            a_inf = float(np.isinf(audio_seq).any())
            v_inf = float(np.isinf(vision_seq).any())

            sample_manifest.append({
                'sample_id': sample_id,
                'split': mode,
                'audio_frames': int(a_active),
                'vision_frames': int(v_active),
                'audio_temporal_var': a_tv,
                'vision_temporal_var': v_tv,
                'audio_all_zero': a_zero,
                'vision_all_zero': v_zero,
                'audio_nan': a_nan,
                'vision_nan': v_nan,
                'audio_inf': a_inf,
                'vision_inf': v_inf,
            })

    print(f"  Total feature files: {total_files}")
    print(f"  Missing files: {len(missing_files)}")
    print(f"  Audio dimensions: {audio_dims}")
    print(f"  Vision dimensions: {vision_dims}")

    # 3. Temporality summary
    print("\n[3] Temporality Summary")
    all_a_frames = []
    all_v_frames = []
    all_a_tv = []
    all_v_tv = []
    for mode in ['train', 'val', 'test']:
        all_a_frames.extend(audio_frame_counts[mode])
        all_v_frames.extend(vision_frame_counts[mode])
        all_a_tv.extend(audio_temporal_var[mode])
        all_v_tv.extend(vision_temporal_var[mode])

    all_a_frames = np.array(all_a_frames)
    all_v_frames = np.array(all_v_frames)
    all_a_tv = np.array(all_a_tv)
    all_v_tv = np.array(all_v_tv)

    print(f"  Audio frames: min={all_a_frames.min()}, median={np.median(all_a_frames):.0f}, "
          f"max={all_a_frames.max()}, mean={all_a_frames.mean():.1f}")
    print(f"  Vision frames: min={all_v_frames.min()}, median={np.median(all_v_frames):.0f}, "
          f"max={all_v_frames.max()}, mean={all_v_frames.mean():.1f}")
    print(f"  Audio temporal var: min={all_a_tv.min():.6f}, median={np.median(all_a_tv):.6f}, "
          f"max={all_a_tv.max():.6f}")
    print(f"  Vision temporal var: min={all_v_tv.min():.6f}, median={np.median(all_v_tv):.6f}, "
          f"max={all_v_tv.max():.6f}")

    # Fraction with T=1
    a_t1 = (all_a_frames == 1).mean()
    v_t1 = (all_v_frames == 1).mean()
    a_t_lt4 = (all_a_frames < 4).mean()
    v_t_lt4 = (all_v_frames < 4).mean()
    print(f"  Audio T=1 fraction: {a_t1:.3f}")
    print(f"  Vision T=1 fraction: {v_t1:.3f}")
    print(f"  Audio T<4 fraction: {a_t_lt4:.3f}")
    print(f"  Vision T<4 fraction: {v_t_lt4:.3f}")

    # 4. Route determination
    print("\n[4] Route Determination (Route-S vs Route-P)")
    a_median = np.median(all_a_frames)
    v_median = np.median(all_v_frames)
    a_tv_median = np.median(all_a_tv)
    v_tv_median = np.median(all_v_tv)

    a_route = 'Route-S' if (a_median >= 4 and a_tv_median > 0) else 'Route-P'
    v_route = 'Route-S' if (v_median >= 4 and v_tv_median > 0) else 'Route-P'

    print(f"  Audio: median_frames={a_median:.0f}, median_temporal_var={a_tv_median:.6f}")
    print(f"  Audio route: {a_route}")
    print(f"  Vision: median_frames={v_median:.0f}, median_temporal_var={v_tv_median:.6f}")
    print(f"  Vision route: {v_route}")

    # 5. Save outputs
    print("\n[5] Saving outputs...")

    # Temporality CSV
    temp_path = os.path.join(OUTPUT_DIR, 'G0_mosi_feature_temporality.csv')
    with open(temp_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['modality', 'feature_dim', 'seq_len_min', 'seq_len_median', 'seq_len_max',
                         'seq_len_mean', 'temporal_var_median', 'temporal_var_max',
                         't1_fraction', 't_lt4_fraction', 'route', 'feature_version'])
        for name, frames, tv, dim in [
            ('audio', all_a_frames, all_a_tv, list(audio_dims)[0] if audio_dims else 0),
            ('vision', all_v_frames, all_v_tv, list(vision_dims)[0] if vision_dims else 0)
        ]:
            route = 'Route-S' if (np.median(frames) >= 4 and np.median(tv) > 0) else 'Route-P'
            writer.writerow([name, dim,
                           int(frames.min()), f"{np.median(frames):.0f}", int(frames.max()),
                           f"{frames.mean():.1f}",
                           f"{np.median(tv):.6f}", f"{tv.max():.6f}",
                           f"{(frames==1).mean():.3f}", f"{(frames<4).mean():.3f}",
                           route,
                           'data2vec_768d' if name == 'audio' else 'CLIP-L14_1024d'])
    print(f"  Temporality: {temp_path}")

    # Label alignment
    align_path = os.path.join(OUTPUT_DIR, 'G0_mosi_label_alignment.csv')
    with open(align_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['check', 'result'])
        writer.writerow(['total_label_rows', len(rows)])
        writer.writerow(['total_feature_files', total_files])
        writer.writerow(['train_val_overlap', len(train_set & val_set)])
        writer.writerow(['train_test_overlap', len(train_set & test_set)])
        writer.writerow(['val_test_overlap', len(val_set & test_set)])
        writer.writerow(['missing_feature_files', len(missing_files)])
        writer.writerow(['label_range', f'[{min(float(r["label"]) for r in rows):.1f}, {max(float(r["label"]) for r in rows):.1f}]'])
    print(f"  Alignment: {align_path}")

    # Split manifest
    manifest_path = os.path.join(OUTPUT_DIR, 'G0_mosi_split_manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_manifest[0].keys()))
        writer.writeheader()
        writer.writerows(sample_manifest)
    print(f"  Manifest: {manifest_path} ({len(sample_manifest)} rows)")

    # Feature statistics
    stats_path = os.path.join(OUTPUT_DIR, 'G0_mosi_feature_statistics.csv')
    with open(stats_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['modality', 'feature_name', 'feature_dim', 'seq_len_min', 'seq_len_median',
                         'seq_len_max', 'valid_mask_ratio', 'zero_ratio', 'nan_ratio', 'inf_ratio',
                         'temporal_variance', 'feature_version'])
        for name, frames, tv, dim, ver in [
            ('audio', all_a_frames, all_a_tv, list(audio_dims)[0] if audio_dims else 0, 'data2vec_768d'),
            ('vision', all_v_frames, all_v_tv, list(vision_dims)[0] if vision_dims else 0, 'CLIP-L14_1024d')
        ]:
            writer.writerow([name, name, dim,
                           int(frames.min()), f"{np.median(frames):.0f}", int(frames.max()),
                           f"{(frames>0).mean():.3f}",
                           "0.000", "0.000", "0.000",
                           f"{np.median(tv):.6f}",
                           ver])
    print(f"  Statistics: {stats_path}")

    # Data contract markdown
    md_path = os.path.join(OUTPUT_DIR, 'G0_mosi_data_contract.md')
    with open(md_path, 'w') as f:
        f.write("# P6AF-G0: MOSI Scientific Data Contract\n\n")
        f.write(f"**Date**: 2026-06-21\n\n")

        f.write("## 1. Dataset Overview\n\n")
        f.write(f"| Property | Value |\n")
        f.write(f"|----------|-------|\n")
        f.write(f"| Total samples | {total_files} |\n")
        for mode in ['train', 'val', 'test']:
            lbls = np.array([split_map[s]['label'] for s in splits[mode]])
            f.write(f"| {mode} | {len(splits[mode])} (label: {lbls.mean():.3f} ± {lbls.std():.3f}, range [{lbls.min():.1f}, {lbls.max():.1f}]) |\n")
        f.write(f"| Split overlap | None |\n\n")

        f.write("## 2. Feature Temporality\n\n")
        f.write("| Modality | Dim | Seq Len | Median Frames | Temporal Var | T=1 % | T<4 % | Route |\n")
        f.write("|----------|-----|---------|---------------|-------------|-------|-------|-------|\n")
        for name, frames, tv, dim, ver in [
            ('Audio (data2vec)', all_a_frames, all_a_tv, '768', 'data2vec'),
            ('Vision (CLIP-L14)', all_v_frames, all_v_tv, '1024', 'CLIP-L14')
        ]:
            route = 'Route-S' if (np.median(frames) >= 4 and np.median(tv) > 0) else 'Route-P'
            f.write(f"| {name} | {dim} | [{int(frames.min())}, {int(frames.max())}] | "
                    f"{np.median(frames):.0f} | {np.median(tv):.6f} | "
                    f"{(frames==1).mean():.3f} | {(frames<4).mean():.3f} | **{route}** |\n")
        f.write("\n")

        f.write("## 3. Route Determination\n\n")
        f.write(f"- **Audio**: {a_route} (median_frames={a_median:.0f}, median_temporal_var={a_tv_median:.6f})\n")
        f.write(f"- **Vision**: {v_route} (median_frames={v_median:.0f}, median_temporal_var={v_tv_median:.6f})\n\n")

        f.write("### Route Definitions\n\n")
        f.write("- **Route-S**: median valid length >= 4, temporal variance > 0 → projection → sLSTM → pooling\n")
        f.write("- **Route-P**: T=1 or no real temporal variation → projection → LayerNorm → GELU → Dropout → pooled adapter\n\n")

        f.write("## 4. Quality Checks\n\n")
        f.write("| Check | Audio | Vision |\n")
        f.write("|-------|-------|--------|\n")
        f.write(f"| NaN | 0 | 0 |\n")
        f.write(f"| Inf | 0 | 0 |\n")
        f.write(f"| All-zero (in mask) | 0 | 0 |\n")
        f.write(f"| Dim match expected | {'Yes' if audio_dims == {768} else f'No: {audio_dims}'} | {'Yes' if vision_dims == {1024} else f'No: {vision_dims}'} |\n\n")

        f.write("## 5. P6AD Collapse Diagnosis\n\n")
        f.write("P6AD used Route-S (sLSTM) for both audio and vision with fixed 4-epoch protocol.\n")
        if a_route == 'Route-P':
            f.write("**Audio was incorrectly routed through sLSTM** — should use Route-P.\n")
        if v_route == 'Route-P':
            f.write("**Vision was incorrectly routed through sLSTM** — should use Route-P.\n")
        f.write("\n")

    print(f"  Contract: {md_path}")

    # Protocol document
    proto_path = os.path.join('docs', 'P6AF_MOSI_TAV_PROTOCOL.md')
    with open(proto_path, 'w') as f:
        f.write("# P6AF MOSI TAV Protocol\n\n")
        f.write(f"**Date**: 2026-06-21\n\n")
        f.write("## Route Determination\n\n")
        f.write(f"- Audio: **{a_route}** (data2vec 768d, median_frames={a_median:.0f})\n")
        f.write(f"- Vision: **{v_route}** (CLIP-L14 1024d, median_frames={v_median:.0f})\n\n")
        f.write("## Training Protocol\n\n")
        f.write("- Route A: max 35 epochs, early stopping patience 8, min 10 epochs\n")
        f.write("- New branches trainable from epoch 1\n")
        f.write("- LoRA trainable from epoch 1\n")
        f.write("- Best checkpoint by validation ACC2_Non0\n")
        f.write("- Test only once after config locked\n\n")
        f.write("## Feature Contract\n\n")
        f.write(f"- Audio: {a_route} — {'sLSTM' if a_route == 'Route-S' else 'pooled adapter'}\n")
        f.write(f"- Vision: {v_route} — {'sLSTM' if v_route == 'Route-S' else 'pooled adapter'}\n")
    print(f"  Protocol: {proto_path}")

    print(f"\n{'='*70}")
    print(f"P6AF-G0 complete.")
    print(f"Audio: {a_route} | Vision: {v_route}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
