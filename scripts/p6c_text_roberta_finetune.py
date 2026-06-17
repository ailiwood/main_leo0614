#!/usr/bin/env python
"""P6C: RoBERTa-large text-only fine-tune on MOSI raw text."""
import sys, os, csv, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

# Config
MODEL_NAME = 'roberta-large'
MAX_LEN = 128
BATCH_SIZE = 8
GRAD_ACCUM = 2  # effective batch = 16
EPOCHS = 30
LR = 1e-5
WARMUP = 0.1
WEIGHT_DECAY = 0.01
DEVICE = 'cuda'
USE_AMP = True
SEED = 42

class MOSITextDataset(Dataset):
    def __init__(self, csv_path, split):
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        split_map = {'train': 'train', 'val': 'valid', 'test': 'test'}
        self.data = [r for r in rows if r.get('mode','train') == split_map.get(split, split)]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        return {'text': r['text'], 'label': float(r['label']),
                'id': f"{r['video_id']}_{r['clip_id']}"}

def main():
    set_seed()
    print(f'RoBERTa-large fine-tune on MOSI')
    print(f'Device: {DEVICE}, BS={BATCH_SIZE}, Accum={GRAD_ACCUM}, LR={LR}')

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MOSITextDataset('data/mosi/label.csv', 'train')
    val_ds = MOSITextDataset('data/mosi/label.csv', 'val')
    test_ds = MOSITextDataset('data/mosi/label.csv', 'test')
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type='regression'
    ).to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M')

    # Collate
    def collate(batch):
        texts = [b['text'] for b in batch]
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32)
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        return {'input_ids': tokens.input_ids, 'attention_mask': tokens.attention_mask,
                'labels': labels, 'ids': [b['id'] for b in batch]}

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) // GRAD_ACCUM * EPOCHS
    scheduler = get_linear_schedule_with_warmup(opt, int(total_steps*WARMUP), total_steps)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    best_val_mae = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        opt.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f'Train E{epoch}', leave=False)):
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
                loss = torch.nn.L1Loss()(out.logits.squeeze(-1), batch['labels'].to(DEVICE))
                loss = loss / GRAD_ACCUM

            if scaler: scaler.scale(loss).backward()
            else: loss.backward()

            if (i+1) % GRAD_ACCUM == 0:
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad()
                scheduler.step()
            total_loss += loss.item() * GRAD_ACCUM

        # Val
        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
                v_preds.append(out.logits.squeeze(-1).cpu())
                v_labels.append(batch['labels'])

        rp = torch.cat(v_preds); tg = torch.cat(v_labels)
        rs = torch.where(rp >= 0, 1.0, -1.0)
        from utils.metrics import compute_all_metrics
        m = compute_all_metrics(rp, rs, tg)
        val_mae, val_acc = m['MAE'], m['ACC2_Non0']

        if val_mae < best_val_mae:
            best_val_mae = val_mae; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'E{epoch:3d} | loss={total_loss/len(train_loader):.4f} | val_MAE={val_mae:.4f} | val_ACC2={val_acc:.2f}%')

    # Test
    if best_state: model.load_state_dict(best_state)
    model.eval()
    t_preds, t_labels, t_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            t_preds.append(out.logits.squeeze(-1).cpu())
            t_labels.append(batch['labels'])
            t_ids.extend(batch['ids'])

    rp = torch.cat(t_preds); tg = torch.cat(t_labels)
    rs = torch.where(rp >= 0, 1.0, -1.0)
    from utils.metrics import compute_all_metrics
    m = compute_all_metrics(rp, rs, tg)

    print(f'\n=== RoBERTa-large text-only results ===')
    print(f'Best epoch: {best_epoch}')
    print(f'ACC2_Non0: {m["ACC2_Non0"]:.2f}%  F1_Non0: {m["F1_Non0"]:.2f}%')
    print(f'MAE: {m["MAE"]:.4f}  Corr: {m["Corr"]:.4f}  ACC7: {m["ACC7"]:.2f}%')

    os.makedirs('outputs/P6C/text_roberta_large', exist_ok=True)
    with open(f'outputs/P6C/text_roberta_large/test_metrics.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}, f)

    return m['ACC2_Non0']

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

if __name__ == '__main__':
    main()
