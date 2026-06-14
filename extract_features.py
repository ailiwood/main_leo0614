"""
extract_roberta_mosei.py
========================
用 RoBERTa-large 替代 BERT-base 重提 MOSEI text features (B, 50, 1024)
- 不用动 BERT 版本 (train/val/test_bert.pkl 已备份到 back/v7_features_bert/)
- 输出 train/val/test_roberta.pkl (新文件名, 不覆盖)
- 预计大小: 1024/768 = 1.33x, 单个文件 train ~ 4.5G
- 耗时: ~30-60 分钟 (RoBERTa-large 比 BERT-base 慢 2-3x)

下游:
- 写一个 mosei_roberta_dataset.py (与 mosei_bert_dataset.py 并行)
- 写一个 model_main_v6.py (ModalEncoder 的 TEXT_DIM 从 768 改 1024)

注意: HF 缓存中 roberta-large 已存在, 所以离线运行 (HF_HUB_OFFLINE=1)
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from transformers import AutoTokenizer, AutoModel

LABEL_DIR = r'D:\business\pycharm\project\Tri_modal_ER\data\CMU-MOSEI\CMU-MOSEI-20230514T151450Z-001\CMU-MOSEI\Labels'
CASP_PKL = r'D:\business\pycharm\project\jqxxtest\CASP-main\data\casp_mosei.pkl'
OUT_DIR = r'D:\business\pycharm\project\Tri_modal_ER\data\CMU-MOSEI'

T_TEXT = 50
DEVICE = 'cuda'
BATCH = 32  # RoBERTa-large 比 BERT-base 显存大, 32 比较稳
MODEL_NAME = 'roberta-large'


def load_model():
    tk = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    mdl = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True).to(DEVICE).eval()
    print(f'  Loaded {MODEL_NAME}, hidden_size={mdl.config.hidden_size}')
    return tk, mdl


def extract_text(texts, tk, mdl):
    all_emb, all_mask = [], []
    N = len(texts)
    for i in range(0, N, BATCH):
        batch = [str(t) if pd.notna(t) else '' for t in texts[i:i + BATCH]]
        inp = tk(batch, return_tensors='pt', padding='max_length',
                 truncation=True, max_length=T_TEXT)
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        with torch.no_grad():
            out = mdl(**inp)
        all_emb.append(out.last_hidden_state.cpu().numpy().astype(np.float32))
        all_mask.append(inp['attention_mask'].cpu().numpy().astype(np.int64))
        if (i + BATCH) % (BATCH * 20) == 0 or i + BATCH >= N:
            print(f'  text {i+len(batch)}/{N}')
    return np.concatenate(all_emb, 0), np.concatenate(all_mask, 0)


def main():
    print('Loading CASP for audio/vision...')
    with open(CASP_PKL, 'rb') as f:
        casp = pickle.load(f)

    print(f'Loading {MODEL_NAME}...')
    tk, mdl = load_model()

    splits = [('train', 'Train', 'train'), ('val', 'Val', 'valid'), ('test', 'Test', 'test')]

    for sp_short, sp_cap, sp_casp in splits:
        print(f'\n=== {sp_short} ===')
        csv = os.path.join(LABEL_DIR, f'Data_{sp_cap}_modified.csv')
        df = pd.read_csv(csv)
        casp_split = casp[sp_casp]
        N_csv = len(df); N_casp = casp_split['regression_labels'].shape[0]
        print(f'CSV rows: {N_csv}, CASP rows: {N_casp}')
        if N_csv != N_casp:
            print(f'  WARNING: size mismatch. Will use min(N_csv, N_casp)')
            N = min(N_csv, N_casp)
            df = df.iloc[:N]
        texts = df['text'].astype(str).tolist()
        print(f'  Sample text 0: {texts[0][:80]}')
        labels_csv = df['sentiment'].values.astype(np.float32)
        labels_casp = casp_split['regression_labels'][:len(df)]
        match_rate = np.isclose(labels_csv, labels_casp, atol=0.05).mean() * 100
        print(f'  Label match CSV vs CASP: {match_rate:.1f}%')

        text_emb, text_mask = extract_text(texts, tk, mdl)
        audio = casp_split['audio'][:len(df)].astype(np.float32)
        vision = casp_split['vision'][:len(df)].astype(np.float32)

        out = {
            'text': text_emb,
            'text_mask': text_mask,
            'audio': audio,
            'vision': vision,
            'regression_labels': labels_csv,
        }
        out_path = os.path.join(OUT_DIR, f'{sp_short}_roberta.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(out, f, protocol=4)
        print(f'  saved -> {out_path}')
        print(f'  shapes: text={text_emb.shape}, audio={audio.shape}, vision={vision.shape}')

    print('\nDONE.')


if __name__ == '__main__':
    main()
