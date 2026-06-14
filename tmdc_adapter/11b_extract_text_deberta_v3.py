"""MOSI DeBERTa-v3-large 特征抽取 — sentencepiece + 硬编码 CLS=1, SEP=2, PAD=0"""
import os, csv, time
import numpy as np
import torch
import sentencepiece as spm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HOME'] = r'D:\business\pycharm\project\Tri_modal_ER\tmdc_adapter\hf_cache'

from transformers import AutoModel

DATA = r'D:\business\pycharm\project\Tri_modal_ER\data'
OUT = r'D:\business\pycharm\project\Tri_modal_ER\tmdc_adapter\features\mosi\deberta-large-4-UTT'
os.makedirs(OUT, exist_ok=True)

SPM_PATH = r'D:\business\pycharm\project\Tri_modal_ER\tmdc_adapter\hf_cache\hub\models--microsoft--deberta-v3-large\snapshots\64a8c8eab3e352a784c658aef62be1662607476f\spm.model'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = 'microsoft/deberta-v3-large'
    model = AutoModel.from_pretrained(name).to(device).eval().half()
    print(f'hidden_size: {model.config.hidden_size}')

    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_PATH)
    print('spm loaded, vocab size:', sp.GetPieceSize())

    CLS, SEP, PAD = 1, 2, 0  # DeBERTa-v3 standard

    with open(os.path.join(DATA, r'mosi\label.csv'), encoding='utf-8') as f:
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
            text_lower = ' '.join(text.lower().split())
            ids = sp.EncodeAsIds(text_lower)
            ids = ids[:126]  # 留出 CLS/SEP 位置
            ids = [CLS] + ids + [SEP]
            ids = ids[:128]
            n = len(ids)
            input_ids = torch.tensor([ids + [PAD] * (128 - n)], dtype=torch.long, device=device)
            attn_mask = torch.tensor([[1]*n + [0]*(128-n)], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            mask = attn_mask.unsqueeze(-1).float()
            feat = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            feat = feat.squeeze(0).cpu().numpy()
            np.save(out_path, feat)
            done += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f'  err {uid}: {e}')
        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1)
            eta = (len(rows) - i - 1) / max(rate, 0.01)
            print(f'  [{i+1}/{len(rows)}] done={done} skip={skipped} err={errors} | {rate:.1f}/s ETA {eta/60:.1f}min')
    print(f'\n完成: done={done} skipped={skipped} errors={errors} 用时 {(time.time()-t0)/60:.1f}min')

if __name__ == '__main__':
    main()
