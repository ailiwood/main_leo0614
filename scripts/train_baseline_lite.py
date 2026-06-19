#!/usr/bin/env python
"""P6N: Baseline-Lite training script."""
import sys, os, json, time, csv, argparse, yaml
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from utils.metrics import compute_all_metrics

MODEL_MAP = {
    'tfn_lite': 'models.baselines.tfn_lite.TFNLite',
    'lmf_lite': 'models.baselines.lmf_lite.LMFLite',
    'mult_lite': 'models.baselines.mult_lite.MulTLite',
    'self_mm_lite': 'models.baselines.self_mm_lite.SelfMMLite',
}


def import_model(name):
    mod_path, cls_name = MODEL_MAP[name].rsplit('.', 1)
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_epochs', type=int, default=None)
    p.add_argument('--limit_batches', type=int, default=0)
    p.add_argument('--debug_shapes', action='store_true')
    p.add_argument('--val_only', action='store_true')
    p.add_argument('--quick_test', action='store_true')
    args = p.parse_args()

    with open(args.config) as f:
        yc = yaml.safe_load(f)
    m = yc.get('model', {})
    t = yc.get('training', {})
    d = yc.get('data', {})
    o = yc.get('output', {})

    DEVICE = args.device
    SEED = t.get('seed', 42)
    EPOCHS = args.max_epochs or t.get('epochs', 1)
    BATCH = t.get('batch_size', 16)
    LR = t.get('lr', 1e-4)
    WD = t.get('weight_decay', 0.01)
    MODE = m.get('modality_mode', 'text_audio_vision')
    MODEL_NAME = m.get('model_name', 'tfn_lite')
    torch.manual_seed(SEED); np.random.seed(SEED)

    # Data
    ds_name = d.get('dataset', 'mosi')
    csv_path = d.get('csv_path', f'data/{ds_name}/label.csv')
    feat_root = d.get('feature_root', f'data/features_strong_sequence_{ds_name}_v3_T40')
    print(f'[DATA] {ds_name.upper()} from {csv_path}')
    train_ds = TextFTMultimodalDataset(csv_path=csv_path, feature_root=feat_root, split='train')
    val_ds = TextFTMultimodalDataset(csv_path=csv_path, feature_root=feat_root, split='val')
    test_ds = TextFTMultimodalDataset(csv_path=csv_path, feature_root=feat_root, split='test')
    print(f'  Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}')
    tl = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_textft)
    vl = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate_textft)
    tlt = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_textft)

    # Model
    model_cls = import_model(MODEL_NAME)
    model = model_cls({**m, 'modality_mode': MODE}).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[MODEL] {MODEL_NAME} mode={MODE} params={n_params/1e6:.2f}M')

    if args.debug_shapes:
        batch = next(iter(tl))
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        print(f'[SHAPES] reg={out["reg"].shape}')
        for k, v in out.get('loss_terms', {}).items():
            if isinstance(v, torch.Tensor):
                print(f'[SHAPES] {k}={v.shape}')
        if args.quick_test:
            return

    if args.val_only:
        print('[VAL_ONLY] Skipping training')
        return

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # Output
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(o.get('root', 'outputs/P6N'), f'{MODEL_NAME}_s{SEED}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    # Save config + command
    yaml.dump(yc, open(os.path.join(out_dir, 'config.yaml'), 'w'))
    with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    metrics_epoch = {'epoch': [], 'train_loss': [], 'val_ACC2': [], 'val_MAE': [], 'val_Corr': []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(tqdm(tl, desc=f'E{epoch}', leave=False)):
            if args.limit_batches and i >= args.limit_batches:
                break
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            lbl = batch['label'].squeeze(-1)
            loss = out.get('loss_terms', {}).get('loss_total', nn.L1Loss()(out['reg'], lbl))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / min(len(tl), args.limit_batches or len(tl))

        # Val
        model.eval()
        vp, vl_ = [], []
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch)
                vp.append(out['reg'].cpu()); vl_.append(batch['label'].cpu())
        rp = torch.cat(vp); tg = torch.cat(vl_).squeeze(-1)
        rs = torch.where(rp >= 0, 1.0, -1.0)
        m = compute_all_metrics(rp, rs, tg)
        print(f'E{epoch:2d}: loss={avg_loss:.4f} val_ACC2={m["ACC2_Non0"]:.2f}% val_MAE={m["MAE"]:.4f}')

        metrics_epoch['epoch'].append(epoch)
        metrics_epoch['train_loss'].append(avg_loss)
        metrics_epoch['val_ACC2'].append(m['ACC2_Non0'])
        metrics_epoch['val_MAE'].append(m['MAE'])
        metrics_epoch['val_Corr'].append(m['Corr'])

        if m['ACC2_Non0'] > best_val_acc:
            best_val_acc = m['ACC2_Non0']
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))

    # Test
    model.eval()
    tp, tl_ = [], []
    with torch.no_grad():
        for batch in tqdm(tlt, desc='Test'):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            tp.append(out['reg'].cpu()); tl_.append(batch['label'].cpu())
    rp = torch.cat(tp); tg = torch.cat(tl_).squeeze(-1)
    rs = torch.where(rp >= 0, 1.0, -1.0)
    m = compute_all_metrics(rp, rs, tg)

    result = {
        'model': MODEL_NAME, 'mode': MODE, 'seed': SEED, 'epochs': EPOCHS,
        'ACC2_Non0': m['ACC2_Non0'], 'F1_Non0': m['F1_Non0'],
        'MAE': m['MAE'], 'Corr': m['Corr'], 'ACC7': m['ACC7'],
        'best_val_ACC2': best_val_acc, 'params_M': n_params / 1e6,
    }
    json.dump(result, open(os.path.join(out_dir, 'result.json'), 'w'), indent=2)
    print(f'Test ACC2={m["ACC2_Non0"]:.2f}% MAE={m["MAE"]:.4f} Corr={m["Corr"]:.4f}')
    print(f'Saved: {out_dir}')


if __name__ == '__main__':
    main()
