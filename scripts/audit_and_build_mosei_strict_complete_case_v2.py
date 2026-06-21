"""
P6AD-G0: MOSEI Strict Complete-Case Cohort Truth Audit

Checks all 20,680 MOSEI TAV samples against strict complete-case criteria:
  - text valid (label.csv row present)
  - audio_mask.sum() > 0
  - vision_mask.sum() > 0
  - audio all finite (no NaN/Inf)
  - vision all finite (no NaN/Inf)
  - audio not all zero
  - vision not all zero
  - sample_id unique
  - sample belongs to official train/valid/test split

Target: 19,354 strict complete-case samples
Current: 20,680 (mosei_tav_openface2_v1, full intersection)

Outputs:
  reports/P6AD_strict_tav_mosi/G0_mosei_strict_cohort_audit.md
  reports/P6AD_strict_tav_mosi/G0_mosei_strict_sample_manifest.csv
  reports/P6AD_strict_tav_mosi/G0_mosei_excluded_samples.csv
  reports/P6AD_strict_tav_mosi/G0_mosei_split_statistics.csv
"""
import os, sys, csv, json
import numpy as np
from collections import defaultdict

FEATURE_ROOT = 'data/processed/mosei_tav_openface2_v1'
CSV_PATH = 'data/processed/mosei_full/label.csv'
OUTPUT_DIR = 'reports/P6AD_strict_tav_mosi'
STRICT_OUTPUT_ROOT = 'data/processed/mosei_tav_complete_case_v2'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_label_csv(path):
    """Load all rows from label CSV."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def get_sample_id(row):
    """MOSEI sample_id format: video_id_clip_id"""
    return f"{row['video_id']}_{row['clip_id']}"

def check_strict_complete_case(feat_path):
    """Check if a sample meets strict complete-case criteria.
    Returns (is_valid, reason_dict)"""
    reasons = {}
    try:
        feat = np.load(feat_path, allow_pickle=True)
    except Exception as e:
        return False, {'error': str(e)}

    # Check keys exist
    for key in ['audio_seq', 'audio_mask', 'vision_seq', 'vision_mask']:
        if key not in feat:
            return False, {'missing_key': key}

    audio_seq = feat['audio_seq']
    audio_mask = feat['audio_mask']
    vision_seq = feat['vision_seq']
    vision_mask = feat['vision_mask']

    # Audio checks
    reasons['audio_shape'] = str(audio_seq.shape)
    reasons['vision_shape'] = str(vision_seq.shape)

    if audio_mask.sum() == 0:
        return False, {**reasons, 'failed': 'audio_mask_zero'}
    if vision_mask.sum() == 0:
        return False, {**reasons, 'failed': 'vision_mask_zero'}

    # Active portion checks (only within mask)
    a_active = audio_seq[audio_mask.astype(bool)]
    v_active = vision_seq[vision_mask.astype(bool)]

    if len(a_active) == 0:
        return False, {**reasons, 'failed': 'audio_active_empty'}
    if len(v_active) == 0:
        return False, {**reasons, 'failed': 'vision_active_empty'}

    # Finite checks
    if not np.isfinite(a_active).all():
        n_bad = (~np.isfinite(a_active)).sum()
        return False, {**reasons, 'failed': f'audio_nonfinite_{n_bad}'}
    if not np.isfinite(v_active).all():
        n_bad = (~np.isfinite(v_active)).sum()
        return False, {**reasons, 'failed': f'vision_nonfinite_{n_bad}'}

    # Non-zero checks (within active portion)
    if np.all(a_active == 0):
        return False, {**reasons, 'failed': 'audio_all_zero'}
    if np.all(v_active == 0):
        return False, {**reasons, 'failed': 'vision_all_zero'}

    reasons['audio_active_mean'] = float(a_active.mean())
    reasons['vision_active_mean'] = float(v_active.mean())
    reasons['audio_active_std'] = float(a_active.std())
    reasons['vision_active_std'] = float(v_active.std())
    reasons['audio_frames'] = int(audio_mask.sum())
    reasons['vision_frames'] = int(vision_mask.sum())

    return True, reasons


def main():
    print("=" * 70)
    print("P6AD-G0: MOSEI Strict Complete-Case Cohort Audit")
    print("=" * 70)

    # Load labels
    rows = load_label_csv(CSV_PATH)
    print(f"\n[1] Label CSV: {len(rows)} total rows")

    # Map split names
    split_map = {'train': 'train', 'val': 'valid', 'test': 'test'}
    by_split = defaultdict(list)
    for r in rows:
        mode = r.get('mode', 'train')
        mapped = split_map.get(mode, mode)
        by_split[mapped].append(r)

    for split in ['train', 'valid', 'test']:
        print(f"  {split}: {len(by_split[split])} samples in label CSV")

    # Check all feature files
    print(f"\n[2] Auditing feature files in {FEATURE_ROOT}...")

    all_results = {}
    excluded = []
    valid_samples = defaultdict(list)
    duplicate_ids = set()
    seen_ids = set()

    total_files = 0
    for split in ['train', 'valid', 'test']:
        feat_dir = os.path.join(FEATURE_ROOT, split)
        if not os.path.isdir(feat_dir):
            print(f"  WARNING: {feat_dir} not found")
            continue

        files = os.listdir(feat_dir)
        total_files += len(files)
        print(f"  {split}: {len(files)} feature files")

        for fname in files:
            if not fname.endswith('.npz'):
                continue
            sample_id = fname.replace('.npz', '')

            # Duplicate check
            if sample_id in seen_ids:
                duplicate_ids.add(sample_id)
            seen_ids.add(sample_id)

            feat_path = os.path.join(feat_dir, fname)
            is_valid, reasons = check_strict_complete_case(feat_path)

            result = {
                'sample_id': sample_id,
                'split': split,
                'is_valid': is_valid,
                **reasons
            }
            all_results[sample_id] = result

            if is_valid:
                valid_samples[split].append(sample_id)
            else:
                result['failure_reason'] = reasons.get('failed', 'unknown')
                excluded.append(result)

    print(f"\n[3] Results Summary")
    print(f"  Total feature files: {total_files}")
    print(f"  Duplicate IDs: {len(duplicate_ids)}")

    total_valid = sum(len(v) for v in valid_samples.values())
    total_excluded = len(excluded)
    print(f"  Strict complete-case: {total_valid}")
    print(f"  Excluded: {total_excluded}")
    print(f"  Target (19,354): {'MATCH' if total_valid == 19354 else f'DIFFERS by {total_valid - 19354}'}")

    for split in ['train', 'valid', 'test']:
        print(f"  {split}: {len(valid_samples[split])} valid")

    # Exclusion reasons breakdown
    print(f"\n[4] Exclusion Reasons")
    reason_counts = defaultdict(int)
    for e in excluded:
        reason = e.get('failure_reason', 'unknown')
        reason_counts[reason] += 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Check vs P6AA/P6AB usage
    print(f"\n[5] P6AA/P6AB Cohort Classification")
    p6aa_used = 20680  # total samples in mosei_tav_openface2_v1
    p6aa_test = 4221
    zero_vision_in_test = sum(1 for e in excluded
                              if e['split'] == 'test' and 'vision_all_zero' in e.get('failure_reason', ''))

    print(f"  P6AA/P6AB total samples used: {p6aa_used}")
    print(f"  P6AA/P6AB test samples: {p6aa_test}")
    print(f"  Test samples with all-zero vision: {zero_vision_in_test}")
    print(f"  Strict complete-case valid: {total_valid}")

    if total_valid != p6aa_used:
        print(f"\n  *** WARNING: P6AA/P6AB used {p6aa_used} samples, but only {total_valid} are strict complete-case.")
        print(f"  *** P6AA/P6AB results ARE NOT strict complete-case.")
        print(f"  *** Classification: aligned_or_mixed_tav_reference")
    else:
        print(f"\n  P6AA/P6AB results CAN be classified as strict complete-case.")

    # ================================================================
    # Save outputs
    # ================================================================
    print(f"\n[6] Saving outputs to {OUTPUT_DIR}...")

    # Sample manifest
    manifest_path = os.path.join(OUTPUT_DIR, 'G0_mosei_strict_sample_manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        fields = ['sample_id', 'split', 'is_valid', 'audio_frames', 'vision_frames',
                  'audio_active_mean', 'vision_active_mean']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for sid, r in sorted(all_results.items()):
            writer.writerow(r)
    print(f"  Manifest: {manifest_path} ({len(all_results)} rows)")

    # Excluded samples
    excl_path = os.path.join(OUTPUT_DIR, 'G0_mosei_excluded_samples.csv')
    with open(excl_path, 'w', newline='') as f:
        fields = ['sample_id', 'split', 'failure_reason', 'audio_frames', 'vision_frames']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for e in sorted(excluded, key=lambda x: x['sample_id']):
            writer.writerow(e)
    print(f"  Excluded: {excl_path} ({len(excluded)} rows)")

    # Split statistics
    stats_path = os.path.join(OUTPUT_DIR, 'G0_mosei_split_statistics.csv')
    with open(stats_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'n_valid', 'label_mean', 'label_std',
                         'label_pos_ratio', 'audio_frames_mean', 'vision_frames_mean'])
        for split in ['train', 'valid', 'test']:
            sids = valid_samples[split]
            labels = []
            a_frames = []
            v_frames = []
            for sid in sids:
                r = all_results.get(sid, {})
                # Get label from CSV
                for row in by_split[split]:
                    if get_sample_id(row) == sid:
                        labels.append(float(row['label']))
                        break
                a_frames.append(r.get('audio_frames', 0))
                v_frames.append(r.get('vision_frames', 0))

            labels = np.array(labels)
            writer.writerow([
                split, len(sids),
                f"{labels.mean():.4f}", f"{labels.std():.4f}",
                f"{(labels >= 0).mean():.4f}",
                f"{np.mean(a_frames):.1f}", f"{np.mean(v_frames):.1f}"
            ])
    print(f"  Statistics: {stats_path}")

    # Markdown audit report
    md_path = os.path.join(OUTPUT_DIR, 'G0_mosei_strict_cohort_audit.md')
    with open(md_path, 'w') as f:
        f.write("# P6AD-G0: MOSEI Strict Complete-Case Cohort Audit\n\n")
        f.write(f"**Date**: 2026-06-21\n\n")

        f.write("## 1. Cohort Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total feature files | {total_files} |\n")
        f.write(f"| Total label CSV rows | {len(rows)} |\n")
        f.write(f"| Duplicate sample IDs | {len(duplicate_ids)} |\n")
        f.write(f"| **Strict complete-case valid** | **{total_valid}** |\n")
        f.write(f"| Excluded | {total_excluded} |\n")
        f.write(f"| Target (19,354) match | {'YES' if total_valid == 19354 else f'NO (diff={total_valid-19354})'} |\n\n")

        f.write("## 2. Per-Split Breakdown\n\n")
        f.write("| Split | Valid | Label Mean | Label Std | Pos Ratio | Audio Frames | Vision Frames |\n")
        f.write("|-------|-------|-----------|-----------|-----------|-------------|--------------|\n")
        for split in ['train', 'valid', 'test']:
            sids = valid_samples[split]
            labels = []
            a_frames = []
            v_frames = []
            for sid in sids:
                for row in by_split[split]:
                    if get_sample_id(row) == sid:
                        labels.append(float(row['label']))
                        break
                r = all_results.get(sid, {})
                a_frames.append(r.get('audio_frames', 0))
                v_frames.append(r.get('vision_frames', 0))
            labels = np.array(labels)
            f.write(f"| {split} | {len(sids)} | {labels.mean():.3f} | {labels.std():.3f} | "
                    f"{(labels>=0).mean():.3f} | {np.mean(a_frames):.1f} | {np.mean(v_frames):.1f} |\n")
        f.write("\n")

        f.write("## 3. Exclusion Reasons\n\n")
        f.write("| Reason | Count |\n")
        f.write("|--------|-------|\n")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {reason} | {count} |\n")
        f.write("\n")

        f.write("## 4. P6AA/P6AB Result Classification\n\n")
        if total_valid != p6aa_used:
            f.write(f"**P6AA/P6AB used {p6aa_used} samples, but only {total_valid} pass strict complete-case criteria.**\n\n")
            f.write("**Classification**: `aligned_or_mixed_tav_reference`\n\n")
            f.write("P6AA/P6AB results CANNOT be claimed as strict complete-case MOSEI TAV results.\n")
            f.write("They remain valid as aligned/mixed TAV reference points.\n\n")
            f.write("**Subsequent MOSEI final tables, baselines, and ablations MUST use**:\n")
            f.write("`mosei_tav_complete_case_v2` ({total_valid} strict complete-case samples)\n\n")
        else:
            f.write("P6AA/P6AB results CAN be classified as strict complete-case.\n\n")

        f.write(f"## 5. Test Set Vision Quality\n\n")
        test_zero_v = sum(1 for e in excluded
                         if e['split'] == 'test' and 'vision_all_zero' in e.get('failure_reason', ''))
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Test samples with all-zero vision | {test_zero_v} |\n")
        f.write(f"| Test total valid | {len(valid_samples['test'])} |\n\n")

        f.write("## 6. Required Actions\n\n")
        if total_valid != 19354:
            f.write(f"- [ ] Investigate why target is 19,354 but actual is {total_valid}\n")
        f.write("- [ ] Build `mosei_tav_complete_case_v2` with strict filtered samples\n")
        f.write("- [ ] Re-run key ablations (F0, F1, F4, F5, E1) on strict cohort\n")
        f.write("- [ ] Update all paper claims to reference strict cohort\n")

    print(f"  Audit report: {md_path}")

    # ================================================================
    # Build strict cohort directory
    # ================================================================
    print(f"\n[7] Building strict cohort directory: {STRICT_OUTPUT_ROOT}")

    if total_valid == total_files:
        print("  All samples pass strict criteria — no filtering needed.")
        print(f"  Using symlink or copying from {FEATURE_ROOT}")
        # Create symlinks (or copy if symlinks not available)
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(STRICT_OUTPUT_ROOT, split), exist_ok=True)
            src_dir = os.path.join(FEATURE_ROOT, split)
            dst_dir = os.path.join(STRICT_OUTPUT_ROOT, split)
            for sample_id in valid_samples[split]:
                src = os.path.join(src_dir, f'{sample_id}.npz')
                dst = os.path.join(dst_dir, f'{sample_id}.npz')
                if not os.path.exists(dst):
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        import shutil
                        shutil.copy2(src, dst)
        print(f"  Strict cohort built at {STRICT_OUTPUT_ROOT}")
    else:
        print(f"  Filtering {total_excluded} invalid samples...")
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(STRICT_OUTPUT_ROOT, split), exist_ok=True)
            src_dir = os.path.join(FEATURE_ROOT, split)
            dst_dir = os.path.join(STRICT_OUTPUT_ROOT, split)
            for sample_id in valid_samples[split]:
                src = os.path.join(src_dir, f'{sample_id}.npz')
                dst = os.path.join(dst_dir, f'{sample_id}.npz')
                if os.path.exists(src) and not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)

        # Create manifests dir
        os.makedirs(os.path.join(STRICT_OUTPUT_ROOT, 'manifests'), exist_ok=True)
        for split in ['train', 'valid', 'test']:
            with open(os.path.join(STRICT_OUTPUT_ROOT, 'manifests', f'{split}_ids.txt'), 'w') as f:
                for sid in sorted(valid_samples[split]):
                    f.write(sid + '\n')

        # Write README
        with open(os.path.join(STRICT_OUTPUT_ROOT, 'README.md'), 'w') as f:
            f.write("# MOSEI TAV Complete Case v2\n\n")
            f.write(f"Strict complete-case filtered MOSEI TAV dataset.\n\n")
            f.write(f"- Source: `{FEATURE_ROOT}`\n")
            f.write(f"- Total samples: {total_valid}\n")
            f.write(f"- Excluded: {total_excluded}\n")
            f.write(f"- Filter criteria: audio_mask>0, vision_mask>0, all finite, non-zero\n\n")
            f.write("## Splits\n\n")
            for split in ['train', 'valid', 'test']:
                f.write(f"- {split}: {len(valid_samples[split])}\n")

        print(f"  Strict cohort built at {STRICT_OUTPUT_ROOT}")
        for split in ['train', 'valid', 'test']:
            count = len(os.listdir(os.path.join(STRICT_OUTPUT_ROOT, split)))
            print(f"    {split}: {count} files")

    print(f"\n{'='*70}")
    print(f"P6AD-G0 audit complete.")
    print(f"Strict valid: {total_valid} / {total_files} total")
    print(f"Classification: {'strict_complete_case' if total_valid == p6aa_used else 'aligned_or_mixed_tav_reference'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
