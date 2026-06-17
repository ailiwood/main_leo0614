#!/usr/bin/env python
"""P6D: Two-stage training for TextFT-xLSTM-AWAF Residual on MOSI."""
import sys, os, csv, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from models.textft_xlstm_awaf_residual import TextFTXLSTMAWAFResidual
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from utils.metrics import compute_all_metrics
from utils.seed import set_seed

# Config
DEVICE = 'cuda'
BATCH_SIZE = 2
GRAD_ACCUM = 8
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 12
STAGE3_EPOCHS = 3
LR_STAGE1 = 1e-5
LR_STAGE2 = 1e-4
LR_STAGE3 = 2e-6
SEED = 42

def freeze_module(m, freeze=True):
    for p in m.parameters(): p.requires_grad = not freeze

def compute_metrics(preds, labels):
    rp = torch.cat(preds); tg = torch.cat(labels)
    rs = torch.where(rp >= 0, 1.0, -1.0)
    return compute_all_metrics(rp, rs, tg)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    regs, clss, lbls, reg_bases, aws, gates_reg = [], [], [], [], [], []
    for batch in loader:
        out = model(
            input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE),
            audio=batch['audio'].to(DEVICE), audio_mask=batch['audio_mask'].to(DEVICE),
            vision=batch['vision'].to(DEVICE), vision_mask=batch['vision_mask'].to(DEVICE))
        regs.append(out['reg'].cpu()); clss.append(out['cls'].cpu())
        lbls.append(batch['label']); reg_bases.append(out['reg_text_base'].cpu())
        aws.append(out['awaf_weights'].cpu()); gates_reg.append(out['residual_gate_reg'].cpu())
    return compute_metrics(regs, lbls), compute_all_metrics(torch.cat(regs), torch.cat(clss), torch.cat(lbls)), \
           torch.cat(aws), torch.cat(gates_reg), torch.cat(reg_bases)

def stage1_text_pretrain(model, train_loader, val_loader):
    """Stage 1: Fine-tune RoBERTa text branch only."""
    print(f'\n=== Stage 1: Text pretrain ({STAGE1_EPOCHS} epochs) ===')
    model.ablation = 'no_residual'
    freeze_module(model.roberta, False)  # unfreeze RoBERTa
    for n, p in model.named_parameters():
        if not n.startswith('roberta') and not n.startswith('text_mlp') \
           and not n.startswith('reg_text_head') and not n.startswith('cls_text_head'):
            p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE1)
    total_steps = len(train_loader) // GRAD_ACCUM * STAGE1_EPOCHS
    scheduler = get_linear_schedule_with_warmup(opt, int(total_steps*0.1), total_steps)
    scaler = torch.amp.GradScaler('cuda')

    best_mae = float('inf')
    for epoch in range(1, STAGE1_EPOCHS+1):
        model.train(); total_loss = 0.0; opt.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f'S1 E{epoch}', leave=False)):
            with torch.amp.autocast('cuda'):
                out = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE),
                           audio=batch['audio'].to(DEVICE), audio_mask=batch['audio_mask'].to(DEVICE),
                           vision=batch['vision'].to(DEVICE), vision_mask=batch['vision_mask'].to(DEVICE))
                loss = torch.nn.L1Loss()(out['reg'], batch['label'].to(DEVICE)) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (i+1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); scheduler.step()
            total_loss += loss.item() * GRAD_ACCUM
        m_reg, _, _, _, _ = evaluate(model, val_loader)
        print(f'S1 E{epoch}: loss={total_loss/len(train_loader):.4f} val_ACC2={m_reg["ACC2_Non0"]:.2f}% val_MAE={m_reg["MAE"]:.4f}')
        if m_reg['MAE'] < best_mae: best_mae = m_reg['MAE']
    model.ablation = 'none'

def stage2_residual(model, train_loader, val_loader):
    """Stage 2: Train sLSTM + AWAF residual, freeze text."""
    print(f'\n=== Stage 2: Residual training ({STAGE2_EPOCHS} epochs) ===')
    # Unfreeze all first, then selectively freeze
    for p in model.parameters(): p.requires_grad = True
    freeze_module(model.roberta, True)
    for p in model.text_mlp.parameters(): p.requires_grad = False
    for p in model.reg_text_head.parameters(): p.requires_grad = False
    for p in model.cls_text_head.parameters(): p.requires_grad = False
    for p in model.text_dropout_layer.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE2)
    scaler = torch.amp.GradScaler('cuda')
    best_mae = float('inf'); best_state = None

    for epoch in range(1, STAGE2_EPOCHS+1):
        model.train(); total_loss = 0.0; opt.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f'S2 E{epoch}', leave=False)):
            with torch.amp.autocast('cuda'):
                out = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE),
                           audio=batch['audio'].to(DEVICE), audio_mask=batch['audio_mask'].to(DEVICE),
                           vision=batch['vision'].to(DEVICE), vision_mask=batch['vision_mask'].to(DEVICE))
                lbl = batch['label'].to(DEVICE)
                # Final loss + delta target loss
                loss_reg = torch.nn.L1Loss()(out['reg'], lbl)
                # Delta target: encourage delta to match text error
                target_delta = lbl - out['reg_text_base'].detach()
                loss_delta = torch.nn.SmoothL1Loss()(out['effective_delta_reg'], target_delta)
                loss = (loss_reg + 0.2 * loss_delta) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (i+1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            total_loss += loss.item() * GRAD_ACCUM
        m_reg, _, aw, gate, _ = evaluate(model, val_loader)
        print(f'S2 E{epoch}: loss={total_loss/len(train_loader):.4f} val_ACC2={m_reg["ACC2_Non0"]:.2f}% val_MAE={m_reg["MAE"]:.4f} gate={gate.mean().item():.3f}')
        if m_reg['MAE'] < best_mae:
            best_mae = m_reg['MAE']; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)

def stage3_joint_finetune(model, train_loader, val_loader):
    """Stage 3: Joint fine-tune with small LR."""
    print(f'\n=== Stage 3: Joint fine-tune ({STAGE3_EPOCHS} epochs) ===')
    freeze_module(model.roberta, False)  # unfreeze all
    for p in model.parameters(): p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=LR_STAGE3)
    scaler = torch.amp.GradScaler('cuda')
    best_mae = float('inf'); best_state = None

    for epoch in range(1, STAGE3_EPOCHS+1):
        model.train(); total_loss = 0.0; opt.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f'S3 E{epoch}', leave=False)):
            with torch.amp.autocast('cuda'):
                out = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE),
                           audio=batch['audio'].to(DEVICE), audio_mask=batch['audio_mask'].to(DEVICE),
                           vision=batch['vision'].to(DEVICE), vision_mask=batch['vision_mask'].to(DEVICE))
                lbl = batch['label'].to(DEVICE)
                loss_reg = torch.nn.L1Loss()(out['reg'], lbl)
                target_delta = lbl - out['reg_text_base'].detach()
                loss_delta = torch.nn.SmoothL1Loss()(out['effective_delta_reg'], target_delta)
                loss = (loss_reg + 0.2 * loss_delta) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (i+1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            total_loss += loss.item() * GRAD_ACCUM
        m_reg, _, _, gate, _ = evaluate(model, val_loader)
        print(f'S3 E{epoch}: loss={total_loss/len(train_loader):.4f} val_ACC2={m_reg["ACC2_Non0"]:.2f}% val_MAE={m_reg["MAE"]:.4f}')
        if m_reg['MAE'] < best_mae:
            best_mae = m_reg['MAE']; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)

def main():
    set_seed(SEED)
    print(f'P6D TextFT-xLSTM-AWAF Residual Training')
    print(f'Device: {DEVICE}, BS={BATCH_SIZE}, Accum={GRAD_ACCUM}')

    # Data
    train_ds = TextFTMultimodalDataset(split='train')
    val_ds = TextFTMultimodalDataset(split='val')
    test_ds = TextFTMultimodalDataset(split='test')
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_textft)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_textft)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_textft)

    # Model
    print('Loading TextFT-xLSTM-AWAF Residual...')
    model = TextFTXLSTMAWAFResidual(freeze_bottom_k_layers=12).to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M total')

    # Stage 1: Text pretrain
    stage1_text_pretrain(model, train_loader, val_loader)

    # Stage 2: Residual training
    stage2_residual(model, train_loader, val_loader)

    # Stage 3: Joint fine-tune
    stage3_joint_finetune(model, train_loader, val_loader)

    # Final test
    m_reg, m_cls, aw, gate, reg_base = evaluate(model, test_loader)
    reg_final = None  # will be computed from evaluate

    print(f'\n=== FINAL RESULTS (seed={SEED}) ===')
    print(f'ACC2_Non0 (reg_sign): {m_reg["ACC2_Non0"]:.2f}%')
    print(f'F1_Non0: {m_reg["F1_Non0"]:.2f}%')
    print(f'MAE: {m_reg["MAE"]:.4f}  Corr: {m_reg["Corr"]:.4f}  ACC7: {m_reg["ACC7"]:.2f}%')
    print(f'ACC2_Non0 (cls): {m_cls["ACC2_Non0"]:.2f}%')
    print(f'AWAF: w_t={aw[:,0].mean():.4f} w_a={aw[:,1].mean():.4f} w_v={aw[:,2].mean():.4f}')
    print(f'Gate reg: {gate.mean().item():.4f}')

    # Save
    os.makedirs('outputs/P6D/textft_mosi', exist_ok=True)
    with open(f'outputs/P6D/textft_mosi/test_metrics_s{SEED}.json', 'w') as f:
        json.dump({k: float(v) for k, v in m_reg.items()}, f)
    print(f'Saved to outputs/P6D/textft_mosi/')

    return m_reg['ACC2_Non0']

if __name__ == '__main__':
    main()
