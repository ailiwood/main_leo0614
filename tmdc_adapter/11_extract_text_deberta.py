"""MOSI 文本 DeBERTa-v3-large 特征抽取 — 一次性, fp16, 跳过已抽取

论文用 DeBERTa-large (v1, He et al. 2021). 我们用 deberta-v3-large,
hidden_size 都是 1024, 与 TMDC 的 Conv1d 输入兼容.
"""
import os, csv, time
import numpy as np
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HOME'] = r'D:\business\pycharm\project\Tri_modal_ER/tmdc_adapter/hf_cache'

from transformers import AutoTokenizer, AutoModel

DATA = r'D:\business\pycharm\project\Tri_modal_ER\data'
OUT = r'D:\business\pycharm\project\Tri_modal_ER/tmdc_adapter/features/mosi/deberta-large-4-UTT'
os.makedirs(OUT, exist_ok=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    name = 'microsoft/deberta-large'
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval().half()

    with open(os.path.join(DATA, 'mosi/label.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    done, skipped, errors = 0, 0, 0
    t0 = time.time()
    for i, r in enumerate(rows):
        vid, cid = r['video_id'], r['clip_id']
        uid = f'{vid}_{cid}'
        out_path = os.path.join(OUT, uid + '.npy')
        if os.path.exists(out_path):
            skipped += 1
            continue
        text = r['text']
        if not text or text.strip() == '':
            np.save(out_path, np.zeros((1024,), dtype=np.float32))
            skipped += 1
            continue
        try:
            inputs = tok(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
            inputs = {k: (v.to(device).half() if v.dtype.is_floating_point else v.to(device)) for k,v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            feat = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            feat = feat.squeeze(0).cpu().numpy()
            np.save(out_path, feat)
            done += 1
        except Exception as e:
            errors += 1
            print(f'  err {uid}: {e}')
        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1)
            eta = (len(rows) - i - 1) / max(rate, 0.01)
            print(f'  [{i+1}/{len(rows)}] done={done} skip={skipped} err={errors} | {rate:.1f}/s ETA {eta/60:.1f}min')
    print(f'\n完成: done={done} skipped={skipped} errors={errors} 用时 {(time.time()-t0)/60:.1f}min')

if __name__ == '__main__':
    main()