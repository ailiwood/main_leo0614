#!/usr/bin/env python
"""P6O: Baseline-Lite formal training with full metrics and checkpointing."""
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
    return getattr(importlib.import_module(mod_path), cls_name)


def evaluate(model, loader, device):
    """Return predictions, labels, and compute_all_metrics dict."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            preds.append(out['reg'].cpu())
            labels.append(batch['label'].cpu())
    rp = torch.cat(preds)
    tg = torch.cat(labels).squeeze(-1)
    rs = torch.where(rp >= 0, 1.0, -1.0)
    m = compute_all_metrics(rp, rs, tg)
    return rp.numpy(), tg.numpy(), m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_epochs', type=int, default=None)
    p.add_argument('--limit_batches', type=int, default=0)
    p.add_argument('--debug_shapes', action='store_true')
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
    EPOCHS = args.max_epochs or t.get('epochs', 50)
    MIN_EPOCHS = t.get('min_epochs', 30)
    PATIENCE = t.get('early_stopping_patience', 10)
    BATCH = t.get('batch_size', 16)
    LR = t.get('lr', 1e-4)
    WD = t.get('weight_decay', 0.01)
    MODE = m.get('modality_mode', 'text_audio_vision')
    MODEL_NAME = m.get('model_name', 'tfn_lite')
    TEST_ONCE = t.get('test_final_once', True)
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
        if args.quick_test:
            return

    # Output dir
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(o.get('root', 'outputs/P6O'), f'{MODEL_NAME}_s{SEED}_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    yaml.dump(yc, open(os.path.join(out_dir, 'config.yaml'), 'w'))
    with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # Training state
    metrics_epoch = {
        'epoch': [], 'train_loss': [],
        'ACC2_Non0': [], 'F1_Non0': [], 'ACC2_Has0': [], 'F1_Has0': [],
        'MAE': [], 'Corr': [], 'ACC7': [],
    }
    best_val = {'ACC2_Non0': 0.0, 'MAE': 999, 'Corr': -999}
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
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
        _, _, val_m = evaluate(model, vl, DEVICE)

        # Record
        metrics_epoch['epoch'].append(epoch)
        metrics_epoch['train_loss'].append(avg_loss)
        for k in ['ACC2_Non0', 'F1_Non0', 'ACC2_Has0', 'F1_Has0', 'MAE', 'Corr', 'ACC7']:
            metrics_epoch[k].append(val_m[k])

        # Best checkpoint: ACC2_Non0 primary, MAE tiebreaker
        is_better = False
        if val_m['ACC2_Non0'] > best_val['ACC2_Non0']:
            is_better = True
        elif val_m['ACC2_Non0'] == best_val['ACC2_Non0'] and val_m['MAE'] < best_val['MAE']:
            is_better = True

        if is_better:
            best_val = {k: val_m[k] for k in best_val}
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f'E{epoch:2d}: loss={avg_loss:.4f}  '
              f'ACC2_Non0={val_m["ACC2_Non0"]:.2f}%  MAE={val_m["MAE"]:.4f}  '
              f'Corr={val_m["Corr"]:.4f}  best={best_epoch}({best_val["ACC2_Non0"]:.2f}%)')

        # Save per-epoch CSV
        with open(os.path.join(out_dir, 'metrics_epoch.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            keys = list(metrics_epoch.keys())
            w.writerow(keys)
            for i in range(len(metrics_epoch['epoch'])):
                w.writerow([metrics_epoch[k][i] for k in keys])

        # Early stopping
        if PATIENCE > 0 and no_improve >= PATIENCE and epoch >= MIN_EPOCHS:
            print(f'[EARLY STOP] epoch={epoch} >= {MIN_EPOCHS}, patience={PATIENCE}')
            break

    # Save best and last
    if best_state:
        torch.save(best_state, os.path.join(out_dir, 'best_model.pth'))
    torch.save(model.state_dict(), os.path.join(out_dir, 'last_model.pth'))

    # Test final once with best model
    if TEST_ONCE and best_state:
        model.load_state_dict(best_state)
    test_preds, test_labels, test_m = evaluate(model, tlt, DEVICE)

    # Save predictions
    with open(os.path.join(out_dir, 'predictions_test.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'label', 'prediction'])
        for i in range(len(test_labels)):
            w.writerow([f'sample_{i}', float(test_labels[i]), float(test_preds[i])])

    # Save metrics_best.json
    metrics_best = {
        'best_epoch': best_epoch,
        'monitor_metric': 'ACC2_Non0',
        'best_val': {k: best_val[k] for k in best_val},
        'test_at_best': {k: test_m[k] for k in ['ACC2_Non0', 'F1_Non0', 'ACC2_Has0', 'F1_Has0', 'MAE', 'Corr', 'ACC7']},
    }
    json.dump(metrics_best, open(os.path.join(out_dir, 'metrics_best.json'), 'w'), indent=2)

    # Save result.json
    result = {
        'phase': 'P6O', 'model': MODEL_NAME, 'dataset': ds_name, 'modalities': MODE,
        'seed': SEED, 'epochs_run': epoch, 'best_epoch': best_epoch,
        'params_M': n_params / 1e6,
        **{k: test_m[k] for k in ['ACC2_Non0', 'F1_Non0', 'ACC2_Has0', 'F1_Has0', 'MAE', 'Corr', 'ACC7']},
        'best_val_ACC2_Non0': best_val['ACC2_Non0'],
    }
    json.dump(result, open(os.path.join(out_dir, 'result.json'), 'w'), indent=2)

    # Save loss curve plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(metrics_epoch['epoch'], metrics_epoch['train_loss'], 'r-', label='Train Loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title(f'{MODEL_NAME} s{SEED} Training Loss')

        ax2 = ax.twinx()
        ax2.plot(metrics_epoch['epoch'], metrics_epoch['ACC2_Non0'], 'b-', label='Val ACC2_Non0')
        ax2.set_ylabel('ACC2_Non0 (%)'); ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[PLOT] loss_curve.png saved')
    except ImportError:
        print('[WARN] matplotlib not available, skipping plot')

    print(f'\n=== P6O {MODEL_NAME} s{SEED} ===')
    print(f'  Best epoch: {best_epoch}  val_ACC2_Non0={best_val["ACC2_Non0"]:.2f}%')
    print(f'  Test ACC2_Non0={test_m["ACC2_Non0"]:.2f}%  F1={test_m["F1_Non0"]:.2f}%')
    print(f'  MAE={test_m["MAE"]:.4f}  Corr={test_m["Corr"]:.4f}  ACC7={test_m["ACC7"]:.2f}%')
    print(f'  Saved: {out_dir}')


if __name__ == '__main__':
    main()
