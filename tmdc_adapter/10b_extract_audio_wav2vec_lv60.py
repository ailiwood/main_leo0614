"""MOSI wav2vec2-large-960h-lv60-self 特征抽取 — 论文等价"""
import os, csv, time
import numpy as np
import torch
import soundfile as sf
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HOME'] = r'D:\business\pycharm\project\Tri_modal_ER\tmdc_adapter\hf_cache'

from transformers import Wav2Vec2Model, Wav2Vec2Processor

DATA = r'D:\business\pycharm\project\Tri_modal_ER\data'
OUT = r'D:\business\pycharm\project\Tri_modal_ER\tmdc_adapter\features\mosi\wav2vec-large-c-UTT'
os.makedirs(OUT, exist_ok=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = 'facebook/wav2vec2-large-960h-lv60-self'
    print(f'loading {name}...')
    proc = Wav2Vec2Processor.from_pretrained(name)
    model = Wav2Vec2Model.from_pretrained(name).to(device).eval().half()
    print(f'hidden_size: {model.config.hidden_size}')

    with open(os.path.join(DATA, r'mosi\label.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    wav_root = os.path.join(DATA, r'mosi\wav')

    done, skipped, errors = 0, 0, 0
    t0 = time.time()
    for i, r in enumerate(rows):
        vid, cid = r['video_id'], r['clip_id']
        uid = f'{vid}_{cid}'
        out_path = os.path.join(OUT, uid + '.npy')
        if os.path.exists(out_path):
            skipped += 1
            continue
        wav_path = os.path.join(wav_root, vid, cid + '.wav')
        if not os.path.exists(wav_path):
            errors += 1
            continue
        try:
            audio, sr = sf.read(wav_path)
            if audio.ndim > 1: audio = audio.mean(axis=1)
            if sr != 16000:
                tgt = int(len(audio) * 16000 / sr)
                audio = np.interp(np.linspace(0, len(audio), tgt), np.arange(len(audio)), audio).astype(np.float32)
            inputs = proc(audio, sampling_rate=16000, return_tensors='pt', padding=True)
            inputs = {k: (v.to(device).half() if v.dtype.is_floating_point else v.to(device)) for k,v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            feat = out.last_hidden_state.squeeze(0).mean(dim=0).float().cpu().numpy()
            np.save(out_path, feat)
            done += 1
        except Exception as e:
            errors += 1
        if (i+1) % 100 == 0:
            print(f'  [{i+1}/{len(rows)}] done={done} skip={skipped} err={errors} | {(time.time()-t0)/60:.1f}min')
    print(f'\n完成: done={done} skipped={skipped} errors={errors} 用时 {(time.time()-t0)/60:.1f}min')

if __name__ == '__main__':
    main()
