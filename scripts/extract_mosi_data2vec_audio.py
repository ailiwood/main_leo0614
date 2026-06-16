"""
Extract Data2Vec-Audio features for CMU-MOSI.
Model: facebook/data2vec-audio-base-960h (Apache 2.0, hidden=768)
Output: data/features_mosi_data2vec/<uid>.npy
"""
import os, csv, time
import numpy as np
import torch
import soundfile as sf
from transformers import Data2VecAudioModel, Wav2Vec2Processor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DATA = 'data/mosi'
OUT = 'data/features_mosi_data2vec'
os.makedirs(OUT, exist_ok=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model_name = 'facebook/data2vec-audio-base-960h'
    print(f'Loading {model_name}...')
    proc = Wav2Vec2Processor.from_pretrained(model_name)
    model = Data2VecAudioModel.from_pretrained(model_name).to(device).eval()
    print(f'  hidden_size={model.config.hidden_size}')

    with open(os.path.join(DATA, 'label.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    wav_root = os.path.join(DATA, 'wav')
    print(f'Total clips: {len(rows)}')

    done = 0; skipped = 0; errors = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        vid, cid = r['video_id'], r['clip_id']
        uid = f'{vid}_{cid}'
        out_path = os.path.join(OUT, uid + '.npy')
        if os.path.exists(out_path):
            skipped += 1; continue
        wav_path = os.path.join(wav_root, vid, cid + '.wav')
        if not os.path.exists(wav_path):
            errors += 1; continue
        try:
            audio, sr = sf.read(wav_path)
            if sr != 16000:
                continue  # would need resampling
            inp = proc(audio, sampling_rate=16000, return_tensors='pt')
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            feat = out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32).squeeze(0)
            np.save(out_path, feat)
            done += 1
            if done % 200 == 0:
                print(f'  {done}/{len(rows)} ({time.time()-t0:.0f}s)')
        except Exception as e:
            errors += 1
    elapsed = time.time() - t0
    print(f'Done: {done} extracted, {skipped} skipped, {errors} errors, {elapsed:.0f}s')

    # Verify
    import pickle
    with open('tmdc_adapter/features/mosi_pkls/CMUMOSI_features_raw_2way.pkl', 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    vids, labels, spks, sents, tr, va, te = obj
    for split_name, split_vids in [('train', tr), ('val', va), ('test', te)]:
        count = 0
        for vid in split_vids:
            for idx in range(len(vids[vid])):
                uid = vids[vid][idx]
                if os.path.exists(os.path.join(OUT, uid + '.npy')):
                    count += 1
        print(f'  {split_name}: {count} features found')

if __name__ == '__main__':
    main()
