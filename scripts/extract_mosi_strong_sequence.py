"""
scripts/extract_mosi_strong_sequence.py — MOSI Strong Pretrained Sequence Features (v2)

P4T.1 enhancements:
  - Audio resampling via librosa or fallback
  - Missing modality tracking
  - Sample-level metadata
  - Resume/skip-existing
  - Failed samples CSV
"""
import os, csv, time, json, argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model, Wav2Vec2Processor, CLIPVisionModel, CLIPImageProcessor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

DATA = 'data/mosi'
OUT = 'data/features_strong_sequence_mosi'
TEXT_MODEL = 'microsoft/deberta-large'
AUDIO_MODEL = 'facebook/wav2vec2-base-960h'  # base=768d, cached; large=1024d but 1.26GB download
VISION_MODEL = 'openai/clip-vit-base-patch32'
MAX_TEXT_LEN = 50
MAX_AUDIO_LEN = 100
MAX_VISION_FRAMES = 20
TARGET_SR = 16000

def load_models(device, modalities):
    models = {}
    if 'text' in modalities:
        print(f'Loading {TEXT_MODEL}...')
        models['text_tk'] = AutoTokenizer.from_pretrained(TEXT_MODEL)
        models['text_md'] = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
        models['text_dim'] = models['text_md'].config.hidden_size
    if 'audio' in modalities:
        print(f'Loading {AUDIO_MODEL}...')
        models['audio_proc'] = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL)
        models['audio_md'] = Wav2Vec2Model.from_pretrained(AUDIO_MODEL).to(device).eval()
        models['audio_dim'] = models['audio_md'].config.hidden_size
    if 'vision' in modalities:
        print(f'Loading {VISION_MODEL}...')
        models['vis_proc'] = CLIPImageProcessor.from_pretrained(VISION_MODEL)
        models['vis_md'] = CLIPVisionModel.from_pretrained(VISION_MODEL).to(device).eval()
        models['vis_dim'] = models['vis_md'].config.hidden_size
    return models

def extract_text(text_str, models, device):
    tk, md = models['text_tk'], models['text_md']
    inp = tk(text_str, return_tensors='pt', truncation=True, max_length=MAX_TEXT_LEN)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad(): out = md(**inp)
    seq = out.last_hidden_state[0].cpu().numpy().astype(np.float32)
    mask = inp['attention_mask'][0].cpu().numpy().astype(np.int64)
    T = mask.sum()
    truncated = (T >= MAX_TEXT_LEN)
    return seq, mask, {'text_len': int(T), 'text_truncated': truncated}

def extract_audio(wav_path, models, device):
    import soundfile as sf
    proc, md = models['audio_proc'], models['audio_md']
    audio, sr = sf.read(wav_path)
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    # Resample if needed
    if sr != TARGET_SR:
        try:
            import librosa
            audio = librosa.resample(y=audio.astype(np.float64), orig_sr=sr, target_sr=TARGET_SR)
            audio = audio.astype(np.float32)
        except ImportError:
            pass  # fallback: pass as-is, Wav2Vec2Processor may handle
    inp = proc(audio, sampling_rate=TARGET_SR, return_tensors='pt')
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad(): out = md(**inp)
    seq = out.last_hidden_state[0].cpu().numpy().astype(np.float32)
    T = seq.shape[0]
    truncated = (T > MAX_AUDIO_LEN)
    if truncated:
        factor = max(T // MAX_AUDIO_LEN, 1)
        seq = seq[::factor][:MAX_AUDIO_LEN]
        T = seq.shape[0]
    mask = np.ones(T, dtype=np.int64)
    return seq, mask, {'audio_len': int(mask.sum()), 'audio_truncated': truncated, 'audio_original_sr': int(sr), 'audio_original_len': len(audio)}

def extract_vision(frames_dir, models, device):
    """Extract CLIP features from pre-extracted .pt files or raw images.

    P4T.1: MOSI Frames/ directory contains .pt files (pre-extracted CLIP features).
    Each .pt is (1, 1024) — a single pooled feature vector per frame.
    We stack multiple frames to create a T>1 sequence where possible.
    If only raw images exist, we use CLIP-ViT to extract features.
    """
    if not os.path.isdir(frames_dir): return None, None, None

    # First try .pt files (pre-extracted CLIP features)
    pt_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.pt')])
    if pt_files:
        n_original = len(pt_files)
        truncated = (n_original > MAX_VISION_FRAMES)
        if truncated:
            indices = np.linspace(0, n_original-1, MAX_VISION_FRAMES, dtype=int)
            pt_files = [pt_files[i] for i in indices]
        feats = []
        for fname in pt_files:
            t = torch.load(os.path.join(frames_dir, fname), map_location='cpu', weights_only=False)
            if isinstance(t, torch.Tensor):
                feat = t.squeeze(0).numpy().astype(np.float32)  # (1024,) or (1, 1024) -> (1024,)
                if feat.ndim == 1: feat = feat[np.newaxis, :]  # (1, 1024)
                feats.append(feat)
        if feats:
            seq = np.concatenate(feats, axis=0)  # (T, 1024)
            T = seq.shape[0]
            mask = np.ones(T, dtype=np.int64)
            return seq, mask, {'vision_len': int(T), 'vision_truncated': truncated,
                              'vision_original_frames': n_original, 'vision_source': 'pre-extracted .pt CLIP features'}

    # Fallback: raw images
    proc, md = models['vis_proc'], models['vis_md']
    frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    if not frames: return None, None, None
    n_original = len(frames)
    truncated = (n_original > MAX_VISION_FRAMES)
    if truncated:
        indices = np.linspace(0, n_original-1, MAX_VISION_FRAMES, dtype=int)
        frames = [frames[i] for i in indices]
    images = []
    for fname in frames:
        try:
            img = Image.open(os.path.join(frames_dir, fname)).convert('RGB')
            images.append(img)
        except: pass
    if not images: return None, None, None
    inp = proc(images=images, return_tensors='pt')
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad(): out = md(**inp)
    seq = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    T = seq.shape[0]
    mask = np.ones(T, dtype=np.int64)
    return seq, mask, {'vision_len': int(T), 'vision_truncated': truncated,
                       'vision_original_frames': n_original, 'vision_source': 'clip-vit raw image extraction'}

# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modalities', nargs='+', default=['text','audio','vision'])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    models = load_models(device, args.modalities)

    # Load labels
    with open(os.path.join(DATA, 'label.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0: rows = rows[:args.limit]

    split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
    splits = {'train': [], 'val': [], 'test': []}
    for r in rows:
        splits[split_map.get(r['mode'], 'train')].append(r)

    stats = {'total': 0, 'complete_trimodal': 0, 'errors': 0,
             'missing_text': 0, 'missing_audio': 0, 'missing_vision': 0}
    failed_samples = []
    t0 = time.time()

    for split_name, split_rows in splits.items():
        out_dir = os.path.join(OUT, split_name)
        os.makedirs(out_dir, exist_ok=True)
        done = 0
        for r in split_rows:
            vid, cid = r['video_id'], r['clip_id']
            uid = f'{vid}_{cid}'
            out_path = os.path.join(out_dir, uid + '.npz')

            if args.resume and os.path.exists(out_path):
                # Quick integrity check
                try:
                    d = np.load(out_path, allow_pickle=True)
                    has_all = all(k in d for k in ['text_seq','audio_seq','vision_seq','label'])
                    if has_all:
                        done += 1; stats['total'] += 1; stats['complete_trimodal'] += 1
                        continue
                except:
                    pass  # corrupted, re-extract

            result = {'label': np.array([float(r['label'])], dtype=np.float32), 'sample_id': uid}
            meta = {'text_model': TEXT_MODEL, 'audio_model': AUDIO_MODEL, 'vision_model': VISION_MODEL}
            has_text = has_audio = has_vision = False

            try:
                # Text
                if 'text' in args.modalities and r.get('text'):
                    tseq, tmask, tmeta = extract_text(r['text'], models, device)
                    result['text_seq'] = tseq; result['text_mask'] = tmask
                    meta.update(tmeta); has_text = True
                else: stats['missing_text'] += 1

                # Audio
                if 'audio' in args.modalities:
                    wav_path = os.path.join(DATA, 'wav', vid, cid + '.wav')
                    if os.path.exists(wav_path):
                        aseq, amask, ameta = extract_audio(wav_path, models, device)
                        result['audio_seq'] = aseq; result['audio_mask'] = amask
                        meta.update(ameta); has_audio = True
                    else: stats['missing_audio'] += 1
                else: stats['missing_audio'] += 1

                # Vision: check both .pt file (MOSI) and directory (MOSEI)
                if 'vision' in args.modalities:
                    # MOSI format: Frames/<vid>/<cid>.pt (single file per clip)
                    pt_single = os.path.join(DATA, 'Frames', vid, cid + '.pt')
                    frames_dir = os.path.join(DATA, 'Frames', vid, cid)

                    vseq, vmask, vmeta = None, None, None
                    if os.path.exists(pt_single):
                        # Single .pt file: load as T=1 sequence
                        t = torch.load(pt_single, map_location='cpu', weights_only=False)
                        if isinstance(t, (torch.Tensor, np.ndarray)):
                            feat = np.array(t).reshape(-1).astype(np.float32)
                            if feat.shape[0] > 1024:  # flattened (C*H*W)
                                feat = feat[:1024]
                            elif feat.shape[0] < 1024:
                                feat = np.pad(feat, (0, 1024 - feat.shape[0]))
                            vseq = feat[np.newaxis, :]  # (1, D)
                            vmask = np.ones(1, dtype=np.int64)
                            vmeta = {'vision_len': 1, 'vision_source': 'single .pt CLIP feature'}

                    if vseq is None and os.path.isdir(frames_dir):
                        vseq, vmask, vmeta = extract_vision(frames_dir, models, device)

                    if vseq is not None:
                        result['vision_seq'] = vseq; result['vision_mask'] = vmask
                        meta.update(vmeta or {}); has_vision = True
                    else:
                        stats['missing_vision'] += 1
                else:
                    stats['missing_vision'] += 1

                result['metadata_json'] = json.dumps(meta, default=lambda x: bool(x) if isinstance(x, (np.bool_,)) else str(x))
                stats['total'] += 1
                if has_text and has_audio and has_vision:
                    stats['complete_trimodal'] += 1

                np.savez_compressed(out_path, **result)
                done += 1
            except Exception as e:
                stats['errors'] += 1
                failed_samples.append({'uid': uid, 'split': split_name, 'error': str(e)[:200]})
                if stats['errors'] <= 5: print(f'  Error [{uid}]: {e}')

            if done % 100 == 0 and done > 0:
                print(f'  [{split_name}] {done}/{len(split_rows)} ({time.time()-t0:.0f}s)')

        print(f'  [{split_name}] Done: {done}/{len(split_rows)}')

    elapsed = time.time() - t0
    print(f'\nTotal: {stats["total"]} extracted, {stats["complete_trimodal"]} trimodal '
          f'({100*stats["complete_trimodal"]/max(stats["total"],1):.1f}%), {stats["errors"]} errors')
    print(f'Missing: text={stats["missing_text"]}, audio={stats["missing_audio"]}, vision={stats["missing_vision"]}')
    print(f'Time: {elapsed:.0f}s ({elapsed/60:.1f}min)')

    # Save failed samples
    if failed_samples:
        failed_path = 'reports/P4T_strong_sequence_and_cleanup/P4T_strong_sequence_failed_samples.csv'
        os.makedirs(os.path.dirname(failed_path), exist_ok=True)
        with open(failed_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['uid','split','error']); w.writeheader(); w.writerows(failed_samples)

    # Per-split summary
    print('\nFeature Audit Summary:')
    for split_name in ['train','val','test']:
        d = os.path.join(OUT, split_name)
        if not os.path.isdir(d): continue
        files = [f for f in os.listdir(d) if f.endswith('.npz')]
        if files:
            s = np.load(os.path.join(d, files[0]), allow_pickle=True)
            tdim = s['text_seq'].shape[-1] if 'text_seq' in s else '?'
            adim = s['audio_seq'].shape[-1] if 'audio_seq' in s else '?'
            vdim = s['vision_seq'].shape[-1] if 'vision_seq' in s else '?'
            tlens = []; alens = []; vlens = []
            for f in files[:50]:
                ss = np.load(os.path.join(d, f), allow_pickle=True)
                if 'text_seq' in ss: tlens.append(ss['text_seq'].shape[0])
                if 'audio_seq' in ss: alens.append(ss['audio_seq'].shape[0])
                if 'vision_seq' in ss: vlens.append(ss['vision_seq'].shape[0])
            print(f'  [{split_name}] N={len(files)}, text_dim={tdim}, audio_dim={adim}, vision_dim={vdim}')
            if tlens: print(f'    text: T={np.mean(tlens):.0f}±{np.std(tlens):.0f} [{min(tlens)},{max(tlens)}]')
            if alens: print(f'    audio: T={np.mean(alens):.0f}±{np.std(alens):.0f} [{min(alens)},{max(alens)}]')
            if vlens: print(f'    vision: T={np.mean(vlens):.0f}±{np.std(vlens):.0f} [{min(vlens)},{max(vlens)}]')

if __name__ == '__main__':
    main()
