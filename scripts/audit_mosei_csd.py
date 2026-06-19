#!/usr/bin/env python
"""P6Q: Audit MOSEI CSD files."""
import os, sys

CSD_DIR = 'data/cmu_mosei_comp_seq'
print("MOSEI CSD Audit")
print("=" * 60)

if not os.path.isdir(CSD_DIR):
    print(f"[BLOCKED] CSD directory not found: {CSD_DIR}")
    sys.exit(1)

files = sorted(os.listdir(CSD_DIR))
total_size = 0
for f in files:
    path = os.path.join(CSD_DIR, f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    total_size += size_mb
    print(f"  {f:50s} {size_mb:8.0f} MB")
print(f"  Total: {total_size / 1024:.1f} GB")

key_files = ['CMU_MOSEI_Labels.csd', 'CMU_MOSEI_COVAREP.csd', 'CMU_MOSEI_OpenFace2.csd',
             'CMU_MOSEI_TimestampedWords.csd', 'CMU_MOSEI_TimestampedWordVectors.csd',
             'CMU_MOSEI_TimestampedPhones.csd', 'CMU_MOSEI_VisualFacet42.csd']
print("\nKey files check:")
all_present = True
for kf in key_files:
    found = any(kf in f for f in files)
    if not found:
        all_present = False
    print(f"  {kf}: {'FOUND' if found else 'MISSING'}")
print(f"  All present: {all_present}")

# Check mmsdk
print("\nmmsdk check:")
try:
    import mmsdk
    from mmsdk import mmdatasdk
    print("  mmsdk imported OK")
except ImportError as e:
    print(f"  mmsdk not available: {e}")

# Try loading labels
print("\nAttempting to load labels CSD...")
try:
    from mmsdk import mmdatasdk
    csd_path = os.path.join(CSD_DIR, 'CMU_MOSEI_Labels.csd')
    if os.path.exists(csd_path):
        labels = mmdatasdk.mmdataset(csd_path)
        keys = list(labels.keys())
        print(f"  Labels loaded: {len(keys)} segments")
        print(f"  Sample keys: {keys[:5]}")
        print("  MOSEI Labels CSD loadable.")
    else:
        print(f"  Not found: {csd_path}")
except Exception as e:
    print(f"  Load failed: {e}")

print("\nNext steps:")
print("  1. Extract label.csv from Labels CSD")
print("  2. Extract audio features from COVAREP CSD (or TimestampedWordVectors)")
print("  3. Extract vision features from OpenFace2 CSD (or VisualFacet42)")
print("  4. Create data/mosei/ directory structure")
