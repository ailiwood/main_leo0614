#!/usr/bin/env python
"""P6P: Baseline-Lite diagnosis script."""
import sys, os, csv, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft

# Load MOSI data
train_ds = TextFTMultimodalDataset(split='train', formal_mode=True)
val_ds = TextFTMultimodalDataset(split='val', formal_mode=True)
test_ds = TextFTMultimodalDataset(split='test', formal_mode=True)

print("=" * 60)
print("P6P Baseline-Lite Diagnosis")
print("=" * 60)

# === 1. Label distribution ===
for name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
    labels = np.array([r['label'] for r in ds.data], dtype=np.float32)
    pos = (labels > 0).sum()
    neg = (labels < 0).sum()
    zero = (labels == 0).sum()
    pos_nonzero = (labels > 0).sum()
    neg_nonzero = (labels < 0).sum()
    total = len(labels)
    print(f"\n[{name}] N={total}")
    print(f"  positive: {pos} ({pos/total*100:.1f}%)")
    print(f"  negative: {neg} ({neg/total*100:.1f}%)")
    print(f"  zero:     {zero} ({zero/total*100:.1f}%)")

# === 2. Majority baseline (always predict majority non-zero class) ===
for name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
    labels = np.array([float(r['label']) for r in ds.data])
    non_zero = labels[labels != 0]
    maj_class = 1 if (non_zero > 0).sum() > (non_zero < 0).sum() else -1
    preds = np.full_like(labels, maj_class)
    # Compute ACC2 manually for non-zero samples
    nz_mask = labels != 0
    acc2 = (preds[nz_mask] * labels[nz_mask] > 0).mean() * 100
    acc2_has0 = (preds * labels > 0).mean() * 100
    print(f"\n[MAJORITY {name}] maj_class={maj_class}  ACC2_Non0={acc2:.1f}%  ACC2_Has0={acc2_has0:.1f}%")

# === 3. Feature stats ===
tl = DataLoader(train_ds, 16, shuffle=False, collate_fn=collate_textft)
batch = next(iter(tl))
for k in ['audio', 'vision', 'input_ids', 'label']:
    if k in batch:
        v = batch[k]
        if isinstance(v, torch.Tensor):
            print(f"\n[FEAT] {k}: shape={list(v.shape)}  mean={v.float().mean():.4f}  std={v.float().std():.4f}  "
                  f"zeros={(v==0).float().mean()*100:.1f}%  NaN={(torch.isnan(v.float())).float().sum()}")

# === 4. Check text input quality ===
print(f"\n[TEXT] input_ids range: [{batch['input_ids'].min()}, {batch['input_ids'].max()}]")
print(f"[TEXT] attention_mask mean: {batch['attention_mask'].float().mean():.4f}")
print(f"[TEXT] Sample 0 tokens (first 20): {batch['input_ids'][0, :20].tolist()}")

# === 5. Read P6O predictions to check ===
print("\n=== P6O Predictions Check ===")
for model in ['tfn_lite', 'lmf_lite', 'mult_lite', 'self_mm_lite']:
    path = f'outputs/P6O/baseline_lite/mosi/{model}_s42_20260619_131407/predictions_test.csv'
    if model == 'lmf_lite':
        path = f'outputs/P6O/baseline_lite/mosi/{model}_s42_20260619_131607/predictions_test.csv'
    elif model == 'mult_lite':
        path = f'outputs/P6O/baseline_lite/mosi/{model}_s42_20260619_131717/predictions_test.csv'
    elif model == 'self_mm_lite':
        path = f'outputs/P6O/baseline_lite/mosi/{model}_s42_20260619_131805/predictions_test.csv'
    if not os.path.exists(path):
        # Try to find it
        import glob
        matches = glob.glob(f'outputs/P6O/baseline_lite/mosi/{model}_*/predictions_test.csv')
        if matches:
            path = matches[0]
        else:
            print(f"  {model}: predictions_test.csv NOT FOUND")
            continue
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    preds = np.array([float(r['prediction']) for r in rows])
    labels = np.array([float(r['label']) for r in rows])
    print(f"  {model}: pred_mean={preds.mean():.4f} pred_std={preds.std():.4f} "
          f"pred_pos={(preds>0).mean()*100:.1f}%  "
          f"label_pos={(labels>0).mean()*100:.1f}%  "
          f"corr={np.corrcoef(preds, labels)[0,1]:.4f}")

    # Sign flip test
    acc2_orig = ((preds * labels > 0) & (labels != 0)).mean() * 100
    acc2_flip = ((-preds * labels > 0) & (labels != 0)).mean() * 100
    print(f"    ACC2_Non0 orig={acc2_orig:.1f}%  flipped={acc2_flip:.1f}%  "
          f"{'FLIP_BETTER!' if acc2_flip > acc2_orig else 'orig_better'}")

# === 6. Text-only probe: simple Linear from raw input_ids ===
print("\n=== Text-Only Probe ===")
# Use mean-pooled token embeddings (learned) as simple probe
class TextProbe(nn.Module):
    def __init__(self, vocab_size=50265, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, ids, mask):
        x = self.embed(ids)  # [B, T, H]
        x = x * mask.unsqueeze(-1).float()
        x = x.sum(dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-8)
        return self.head(x).squeeze(-1)

probe = TextProbe().cuda()
opt = torch.optim.Adam(probe.parameters(), lr=3e-4)
for ep in range(20):
    probe.train()
    for batch in tl:
        ids = batch['input_ids'].cuda(); am = batch['attention_mask'].cuda()
        lbl = batch['label'].squeeze(-1).cuda()
        nz = lbl != 0
        if nz.sum() == 0:
            continue
        pred = probe(ids, am)
        loss = nn.L1Loss()(pred[nz], lbl[nz])
        opt.zero_grad(); loss.backward(); opt.step()
    # Val
    probe.eval()
    vp, vl_ = [], []
    with torch.no_grad():
        for batch in DataLoader(val_ds, 16, shuffle=False, collate_fn=collate_textft):
            ids = batch['input_ids'].cuda(); am = batch['attention_mask'].cuda()
            pred = probe(ids, am)
            vp.append(pred.cpu()); vl_.append(batch['label'].cpu())
    rp = torch.cat(vp); tg = torch.cat(vl_).squeeze(-1)
    nz = tg != 0
    acc2 = ((rp[nz] * tg[nz]) > 0).float().mean() * 100
    if ep == 0 or ep == 19 or ep == 9:
        print(f"  E{ep+1:2d}: val_ACC2_Non0={acc2:.1f}%")

# Test
probe.eval()
tp, tl_ = [], []
with torch.no_grad():
    for batch in DataLoader(test_ds, 16, shuffle=False, collate_fn=collate_textft):
        ids = batch['input_ids'].cuda(); am = batch['attention_mask'].cuda()
        pred = probe(ids, am)
        tp.append(pred.cpu()); tl_.append(batch['label'].cpu())
rp = torch.cat(tp); tg = torch.cat(tl_).squeeze(-1)
nz = tg != 0
acc2 = ((rp[nz] * tg[nz]) > 0).float().mean() * 100
print(f"  Test ACC2_Non0={acc2:.1f}%")

# === 7. 200-sample overfit test on SelfMM-lite ===
print("\n=== 200-Sample Overfit ===")
from models.baselines.self_mm_lite import SelfMMLite
subset = torch.utils.data.Subset(train_ds, range(200))
sub_loader = DataLoader(subset, 16, shuffle=True, collate_fn=collate_textft)
model = SelfMMLite({'modality_mode': 'text_audio_vision', 'hidden_dim': 128, 'dropout': 0.0}).cuda()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
for ep in range(100):
    model.train()
    for batch in sub_loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        lbl = batch['label'].squeeze(-1).cuda()
        loss = out.get('loss_terms', {}).get('loss_total', nn.L1Loss()(out['reg'], lbl))
        opt.zero_grad(); loss.backward(); opt.step()
    if ep == 0 or ep == 99 or ep == 49:
        model.eval()
        with torch.no_grad():
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in next(iter(sub_loader)).items()}
            out = model(batch)
            lbl = batch['label'].squeeze(-1).cuda()
            nz = lbl != 0
            acc2 = ((out['reg'][nz] * lbl[nz]) > 0).float().mean() * 100
        print(f"  E{ep+1:3d}: train_ACC2_Non0={acc2:.1f}%  loss={loss.item():.4f}")
print("Diagnosis complete.")
