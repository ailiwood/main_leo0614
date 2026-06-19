#!/usr/bin/env python
"""P6S-Repair-3: Cache RoBERTa CLS features for all MOSEI segments."""
import torch, os, json, csv, numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = 'cuda'; BATCH = 32; MAX_LEN = 64
CSV_PATH = 'data/processed/mosei_full/label.csv'
CACHE_DIR = 'data/processed/mosei_full/roberta_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Load CSV
rows = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        rows.append(row)
print(f'Total samples: {len(rows)}')

# Load model
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
model = AutoModel.from_pretrained('roberta-large').to(DEVICE)
model.eval()

# Cache per split
for split in ['train', 'valid', 'test']:
    split_rows = [r for r in rows if r['mode'] == split]
    if not split_rows:
        continue
    texts = [r['text'] for r in split_rows]
    ids = [f"{r['video_id']}_{r['clip_id']}" for r in split_rows]

    all_cls = []
    for i in tqdm(range(0, len(texts), BATCH), desc=f'Caching {split}'):
        batch_texts = texts[i:i+BATCH]
        tokens = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()  # [B, 1024]
        all_cls.append(cls)

    feats = np.concatenate(all_cls, axis=0).astype(np.float32)
    np.save(os.path.join(CACHE_DIR, f'{split}_roberta_cls.npy'), feats)
    with open(os.path.join(CACHE_DIR, f'{split}_ids.json'), 'w') as f:
        json.dump(ids, f)
    print(f'{split}: {feats.shape[0]} samples, shape={feats.shape}')

json.dump({'model': 'roberta-large', 'max_length': MAX_LEN, 'dim': 1024, 'batch_size': BATCH},
          open(os.path.join(CACHE_DIR, 'cache_config.json'), 'w'))
print(f'Cache complete: {CACHE_DIR}')
