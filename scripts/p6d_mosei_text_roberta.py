#!/usr/bin/env python
"""P6D: RoBERTa-large text-only fine-tune on CMU-MOSEI raw text."""
import sys, os, csv, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils.metrics import compute_all_metrics

MODEL_NAME = 'roberta-large'; MAX_LEN = 128
BATCH_SIZE = 4; GRAD_ACCUM = 4; EPOCHS = 10; LR = 1e-5
DEVICE = 'cuda'

class MOSEITextDataset(Dataset):
    def __init__(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            self.data = list(csv.DictReader(f))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        r = self.data[idx]
        return {'text': r['text'], 'label': float(r['sentiment']),
                'id': f"{r['video']}_{r['start_time']}_{r['end_time']}"}

def main():
    torch.manual_seed(42)
    BASE = 'data/CMU-MOSEI/CMU-MOSEI-20230514T151450Z-001/CMU-MOSEI/Labels'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MOSEITextDataset(f'{BASE}/Data_Train_modified.csv')
    val_ds = MOSEITextDataset(f'{BASE}/Data_Val_modified.csv')
    test_ds = MOSEITextDataset(f'{BASE}/Data_Test_modified.csv')
    print(f'MOSEI: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}')

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, problem_type='regression').to(DEVICE)
    print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params')

    def collate(batch):
        tokens = tokenizer([b['text'] for b in batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        return {'input_ids': tokens.input_ids, 'attention_mask': tokens.attention_mask,
                'labels': torch.tensor([b['label'] for b in batch], dtype=torch.float32)}

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) // GRAD_ACCUM * EPOCHS
    scheduler = get_linear_schedule_with_warmup(opt, int(total_steps*0.1), total_steps)
    scaler = torch.amp.GradScaler('cuda')

    best_mae = float('inf'); best_state = None
    for epoch in range(1, EPOCHS+1):
        model.train(); total_loss = 0.0; opt.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f'E{epoch}', leave=False)):
            with torch.amp.autocast('cuda'):
                out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
                loss = torch.nn.L1Loss()(out.logits.squeeze(-1), batch['labels'].to(DEVICE)) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (i+1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); scheduler.step()
            total_loss += loss.item() * GRAD_ACCUM

        model.eval(); vp, vl = [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast('cuda'):
                    out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
                vp.append(out.logits.squeeze(-1).cpu()); vl.append(batch['labels'])
        rp = torch.cat(vp); tg = torch.cat(vl); rs = torch.where(rp >= 0, 1.0, -1.0)
        m = compute_all_metrics(rp, rs, tg)
        print(f'E{epoch}: loss={total_loss/len(train_loader):.4f} val_ACC2={m["ACC2_Non0"]:.2f}% val_MAE={m["MAE"]:.4f}')
        if m['MAE'] < best_mae: best_mae = m['MAE']; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    model.eval(); tp, tl = [], []
    with torch.no_grad():
        for batch in test_loader:
            with torch.amp.autocast('cuda'):
                out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            tp.append(out.logits.squeeze(-1).cpu()); tl.append(batch['labels'])
    rp = torch.cat(tp); tg = torch.cat(tl); rs = torch.where(rp >= 0, 1.0, -1.0)
    m = compute_all_metrics(rp, rs, tg)
    print(f'\n=== MOSEI RoBERTa-large text-only ===')
    print(f'ACC2_Non0: {m["ACC2_Non0"]:.2f}%  F1: {m["F1_Non0"]:.2f}%  MAE: {m["MAE"]:.4f}  Corr: {m["Corr"]:.4f}')
    os.makedirs('outputs/P6D/mosei_text', exist_ok=True)
    with open('outputs/P6D/mosei_text/test_metrics_s42.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}, f)

if __name__ == '__main__':
    main()
