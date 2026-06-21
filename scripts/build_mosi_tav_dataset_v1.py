"""
P6AD-G1: Build MOSI TAV Dataset v1

Converts v6_vision_l14_full features into standard TextFTMultimodalDataset format.
Source: data/features_strong_sequence_mosi_v6_vision_l14_full/
Target: data/processed/mosi_tav_v1/

Standard format per .npz: {audio_seq, audio_mask, vision_seq, vision_mask}
Uses data/mosi/label.csv for labels and split info.
"""
import os, sys, csv, shutil
import numpy as np

SOURCE_ROOT = 'data/features_strong_sequence_mosi_v6_vision_l14_full'
TARGET_ROOT = 'data/processed/mosi_tav_v1'
LABEL_CSV = 'data/mosi/label.csv'
OUTPUT_DIR = 'reports/P6AD_strict_tav_mosi'

os.makedirs(TARGET_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("P6AD-G1: Build MOSI TAV Dataset v1")
    print("=" * 70)

    # Load labels for split assignment
    split_map = {}
    with open(LABEL_CSV, 'r') as f:
        for r in csv.DictReader(f):
            sid = f"{r['video_id']}_{r['clip_id']}"
            split_map[sid] = r['mode']  # train/val/test

    # Statistics
    stats = {'train': {'n': 0, 'v_dim': None, 'a_dim': None, 'v_len': None, 'a_len': None},
             'val': {'n': 0, 'v_dim': None, 'a_dim': None, 'v_len': None, 'a_len': None},
             'test': {'n': 0, 'v_dim': None, 'a_dim': None, 'v_len': None, 'a_len': None}}

    for split in ['train', 'val', 'test']:
        src_dir = os.path.join(SOURCE_ROOT, split)
        dst_dir = os.path.join(TARGET_ROOT, split)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"  WARNING: {src_dir} not found, skipping")
            continue

        files = os.listdir(src_dir)
        print(f"\n  {split}: {len(files)} source files")

        for fname in files:
            if not fname.endswith('.npz'):
                continue

            src_path = os.path.join(src_dir, fname)
            sample_id = fname.replace('.npz', '')

            feat = np.load(src_path, allow_pickle=True)

            # Extract standard keys
            audio_seq = feat['audio_seq']
            audio_mask = feat['audio_mask'].astype(np.int64)
            vision_seq = feat['vision_seq']
            vision_mask = feat['vision_mask'].astype(np.int64)

            # Record dimensions
            if stats[split]['v_dim'] is None:
                stats[split]['v_dim'] = vision_seq.shape[1]
                stats[split]['a_dim'] = audio_seq.shape[1]
                stats[split]['v_len'] = vision_seq.shape[0]
                stats[split]['a_len'] = audio_seq.shape[0]

            # Save in standard format
            dst_path = os.path.join(dst_dir, f'{sample_id}.npz')
            np.savez_compressed(
                dst_path,
                audio_seq=audio_seq,
                audio_mask=audio_mask,
                vision_seq=vision_seq,
                vision_mask=vision_mask,
            )
            stats[split]['n'] += 1

    print(f"\n{'='*70}")
    print("Build Summary")
    print(f"{'='*70}")
    total = 0
    for split in ['train', 'val', 'test']:
        n = stats[split]['n']
        total += n
        vd = stats[split].get('v_dim', '?')
        ad = stats[split].get('a_dim', '?')
        vl = stats[split].get('v_len', '?')
        al = stats[split].get('a_len', '?')
        print(f"  {split}: {n} samples | vision={vl}×{vd} | audio={al}×{ad}")
    print(f"  TOTAL: {total}")
    print(f"  Target: {TARGET_ROOT}")

    # Write README
    with open(os.path.join(TARGET_ROOT, 'README.md'), 'w') as f:
        f.write("# MOSI TAV Dataset v1\n\n")
        f.write(f"Source: {SOURCE_ROOT}\n")
        f.write(f"Build date: 2026-06-21\n\n")
        f.write("## Splits\n\n")
        for split in ['train', 'val', 'test']:
            f.write(f"- {split}: {stats[split]['n']} samples\n")
        f.write(f"\nTotal: {total}\n\n")
        f.write("## Feature Dimensions\n\n")
        f.write(f"- Vision: {stats['train']['v_len']}×{stats['train']['v_dim']} (CLIP-L14)\n")
        f.write(f"- Audio: {stats['train']['a_len']}×{stats['train']['a_dim']} (data2vec)\n\n")
        f.write("## Important Notes\n\n")
        f.write("- Vision features are CLIP-L14 (1024-dim), NOT OpenFace2 (713-dim)\n")
        f.write("- Audio features are data2vec (768-dim), NOT COVAREP (74-dim)\n")
        f.write("- Text is pre-encoded (1024-dim tokens) — not raw text for roberta-large\n")
        f.write("- These are DIFFERENT feature sources from MOSEI TAV\n")
        f.write("- Must be documented as: mosi_tav_v1 (CLIP-L14 + data2vec)\n")

    # Create manifest
    os.makedirs(os.path.join(TARGET_ROOT, 'manifests'), exist_ok=True)
    for split in ['train', 'val', 'test']:
        dst_dir = os.path.join(TARGET_ROOT, split)
        if not os.path.isdir(dst_dir):
            continue
        ids = sorted([f.replace('.npz', '') for f in os.listdir(dst_dir)])
        with open(os.path.join(TARGET_ROOT, 'manifests', f'{split}_ids.txt'), 'w') as f:
            for sid in ids:
                f.write(sid + '\n')

    # Save split statistics
    csv_path = os.path.join(OUTPUT_DIR, 'G1_mosi_tav_alignment_manifest.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'split'])
        for split in ['train', 'val', 'test']:
            dst_dir = os.path.join(TARGET_ROOT, split)
            if not os.path.isdir(dst_dir):
                continue
            for fname in sorted(os.listdir(dst_dir)):
                writer.writerow([fname.replace('.npz', ''), split])
    print(f"\n  Manifest: {csv_path}")

    print(f"\n{'='*70}")
    print("MOSI TAV Dataset v1 build complete.")
    print(f"Status: ready")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
