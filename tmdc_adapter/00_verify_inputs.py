"""MOSI/MOSEI 输入数据一致性自检 — 一次性只读脚本"""
import os, csv, json, pickle
import torch

DATA = r'D:\business\pycharm\project\Tri_modal_ER\data'

def check_mosi():
    print('='*60); print('MOSI 检查'); print('='*60)
    with open(os.path.join(DATA, 'mosi/label.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    csv_clips = [(r['video_id'], r['clip_id']) for r in rows]
    print(f'  label.csv clips: {len(csv_clips)}')
    modes = {}
    for r in rows:
        modes[r['mode']] = modes.get(r['mode'], 0) + 1
    print(f'  切分: {modes}')

    with open(os.path.join(DATA, 'mosi/audio_features_raw.pkl'), 'rb') as f:
        audio = pickle.load(f)
    csv_keys = set(csv_clips)
    print(f'  audio_features_raw.pkl: {len(audio)} ∩ csv: {len(set(audio.keys()) & csv_keys)}')

    frames_root = os.path.join(DATA, 'mosi/Frames')
    disk_keys = set()
    for vid in os.listdir(frames_root):
        sub = os.path.join(frames_root, vid)
        if os.path.isdir(sub):
            for c in os.listdir(sub):
                if c.endswith('.pt'):
                    disk_keys.add((vid, c[:-3]))
    print(f'  Frames .pt: {len(disk_keys)} ∩ csv: {len(disk_keys & csv_keys)}')

    wav_root = os.path.join(DATA, 'mosi/wav')
    wav_keys = set()
    for vid in os.listdir(wav_root):
        sub = os.path.join(wav_root, vid)
        if os.path.isdir(sub):
            for c in os.listdir(sub):
                if c.endswith('.wav'):
                    wav_keys.add((vid, c[:-3]))
    print(f'  wav: {len(wav_keys)} ∩ csv: {len(wav_keys & csv_keys)}')

    mp4_root = os.path.join(DATA, 'mosi/Raw')
    mp4_keys = set()
    for vid in os.listdir(mp4_root):
        sub = os.path.join(mp4_root, vid)
        if os.path.isdir(sub):
            for c in os.listdir(sub):
                if c.endswith('.mp4'):
                    mp4_keys.add((vid, c[:-3]))
    print(f'  mp4: {len(mp4_keys)} ∩ csv: {len(mp4_keys & csv_keys)}')

    sample = csv_clips[0]
    print(f'  sample ({sample[0]}/{sample[1]}):')
    print(f'    audio: {audio[sample].shape}')
    pt_path = os.path.join(frames_root, sample[0], sample[1] + '.pt')
    t = torch.load(pt_path, map_location='cpu', weights_only=False)
    print(f'    frames: {t.shape}, dtype={t.dtype}')
    return rows

def check_mosei():
    print('='*60); print('MOSEI 检查'); print('='*60)
    with open(os.path.join(DATA, 'mosei_sampled_indices.json'), encoding='utf-8') as f:
        idx = json.load(f)
    print(f'  切分: train={len(idx["train"])}, val={len(idx["val"])}, test={len(idx["test"])}')

    audio_chunk_root = os.path.join(DATA, 'CMU-MOSEI/CMU-MOSEI-20230514T151450Z-001/CMU-MOSEI/Audio_chunk')
    print(f'  Audio_chunk: {os.listdir(audio_chunk_root)}')
    for sub in os.listdir(audio_chunk_root):
        n = len(os.listdir(os.path.join(audio_chunk_root, sub)))
        print(f'    {sub}: {n} wav')

    for split in ['train', 'val', 'test']:
        cap = split[0].upper() + split[1:]
        csv_path = os.path.join(DATA, f'CMU-MOSEI/CMU-MOSEI-20230514T151450Z-001/CMU-MOSEI/Labels/Data_{cap}_modified.csv')
        with open(csv_path, encoding='utf-8') as f:
            n = sum(1 for _ in f) - 1
        print(f'  Data_{cap}_modified.csv: {n} 行')

    with open(os.path.join(DATA, 'CMU-MOSEI/train.features'), 'rb') as f:
        feats = pickle.load(f)
    print(f'  train.features: {len(feats)}, sample: {feats[0]["feature"].shape}')

if __name__ == '__main__':
    check_mosi()
    check_mosei()
    print('\n自检完成。')