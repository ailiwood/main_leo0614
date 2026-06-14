"""
train_main_v6.py — v9 (RoBERTa-large) 3-seed 集成 + 阈值校准 (冲 87%)
======================================================================

策略:
  1) 3 seeds x 4 epochs (同 v4/v5 配置)
  2) MainModelV9 = v6 架构 + RoBERTa-large 1024d text
  3) 集成 + val 阈值校准
  4) 重点看 weak_neg ACC (v7 = 67.52%)

依赖:
  - model_main_v6.py (新建): MainModelV9
  - mosei_roberta_dataset.py (新建): MOSEIRobertaDataset
  - data/CMU-MOSEI/{train,val,test}_roberta.pkl (extract_roberta_mosei.py 产出)

用法:
  python train_main_v6.py --epochs 4 --seeds 42 100 2024
"""
import os, sys, argparse, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, r'D:\business\pycharm\project\Tri_modal_ER')
from mosei_roberta_dataset import MOSEIRobertaDataset
from model_main_v6 import MainModelV9
from balanced_sampler import get_balanced_sampler


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def class_weighted_l1(pred, target, neg_weight=1.5):
    w = torch.where(target >= 0, torch.ones_like(target), torch.full_like(target, neg_weight))
    return (w * (pred - target).abs()).mean()


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)
        return backup

    def restore(self, model, backup):
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)


def compute_metrics(reg_pred, pol_logit, target, threshold=0.5):
    rp = reg_pred.cpu().numpy().flatten()
    pl = pol_logit.cpu().numpy().flatten()
    tg = target.cpu().numpy().flatten()
    valid = ~(np.isnan(rp) | np.isnan(tg))
    rp = rp[valid]; pl = pl[valid]; tg = tg[valid]
    if len(rp) == 0:
        return {'MAE': 999, 'Corr': 0, 'ACC2': 0, 'ACC2_Non0': 0, 'F1_Non0': 0, 'threshold': threshold}
    mae = float(np.mean(np.abs(rp - tg)))
    corr = float(np.corrcoef(rp, tg)[0, 1]) if np.std(rp) > 0 and np.std(tg) > 0 else 0.0
    if np.isnan(corr):
        corr = 0.0
    pol_prob = 1.0 / (1.0 + np.exp(-pl))
    pred_bin = (pol_prob >= threshold).astype(int)
    tgt_bin = (tg >= 0).astype(int)
    acc2 = float((pred_bin == tgt_bin).mean()) * 100
    nz = tg != 0
    if nz.sum() > 0:
        pb = pred_bin[nz]; tb = tgt_bin[nz]
        acc2_nz = float((pb == tb).mean()) * 100
        tp = int(((pb == 1) & (tb == 1)).sum())
        fp = int(((pb == 1) & (tb == 0)).sum())
        fn = int(((pb == 0) & (tb == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    else:
        acc2_nz = acc2; f1 = 0.0
    return {'MAE': mae, 'Corr': corr, 'ACC2': acc2, 'ACC2_Non0': acc2_nz, 'F1_Non0': f1 * 100, 'threshold': threshold}


def best_threshold_for_acc2_nz(pol_logit, target, n_steps=101):
    pl = pol_logit.cpu().numpy().flatten()
    tg = target.cpu().numpy().flatten()
    pol_prob = 1.0 / (1.0 + np.exp(-pl))
    nz = tg != 0
    if nz.sum() == 0:
        return 0.5, 0.0
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.30, 0.70, n_steps):
        pb = (pol_prob[nz] >= t).astype(int)
        tb = (tg[nz] >= 0).astype(int)
        acc = float((pb == tb).mean()) * 100
        if acc > best_acc:
            best_acc = acc; best_t = float(t)
    return best_t, best_acc


def train_one_epoch(model, loader, optimizer, scheduler, ema, neg_weight=1.5, polarity_weight=0.5):
    model.train()
    total_loss = 0; cnt = 0
    bce = nn.BCEWithLogitsLoss()
    for batch in tqdm(loader, desc='Train', leave=False):
        text = batch['text'].cuda(non_blocking=True)
        mask = batch['text_mask'].cuda(non_blocking=True)
        audio = batch['audio'].cuda(non_blocking=True)
        vision = batch['vision'].cuda(non_blocking=True)
        labels = batch['label'].cuda(non_blocking=True)
        polar_tgt = (labels >= 0).float()

        optimizer.zero_grad()
        out = model(text, audio, vision, mask)
        if torch.isnan(out['M']).any():
            continue
        reg_loss = (0.6 * class_weighted_l1(out['M'], labels, neg_weight)
                    + 0.15 * class_weighted_l1(out['T'], labels, neg_weight)
                    + 0.10 * class_weighted_l1(out['A'], labels, neg_weight)
                    + 0.05 * class_weighted_l1(out['V'], labels, neg_weight))
        pol_loss = bce(out['polarity'], polar_tgt)
        loss = reg_loss + polarity_weight * pol_loss
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        if ema:
            ema.update(model)
        total_loss += loss.item() * text.size(0)
        cnt += text.size(0)
    return total_loss / cnt if cnt > 0 else 999


@torch.no_grad()
def collect_outputs(model, loader):
    model.eval()
    all_M, all_P, all_T = [], [], []
    for batch in tqdm(loader, desc='Eval', leave=False):
        out = model(batch['text'].cuda(), batch['audio'].cuda(),
                    batch['vision'].cuda(), batch['text_mask'].cuda())
        all_M.append(out['M'].cpu())
        all_P.append(out['polarity'].cpu())
        all_T.append(batch['label'])
    return torch.cat(all_M), torch.cat(all_P), torch.cat(all_T)


def train_one_seed(seed, epochs, batch_size, lr, hidden_dim, dropout, neg_weight, polarity_weight, ema_decay, save_dir):
    set_seed(seed)
    tr = MOSEIRobertaDataset('train')
    val = MOSEIRobertaDataset('val')
    te = MOSEIRobertaDataset('test')
    sampler = get_balanced_sampler(tr, batch_size=batch_size, seed=seed)
    tr_loader = DataLoader(tr, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MainModelV9(hidden_dim=hidden_dim, head_dropout=dropout).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_steps = len(tr_loader) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_steps,
                                                 pct_start=0.1, anneal_strategy='cos',
                                                 div_factor=25.0, final_div_factor=1000.0)
    ema = ModelEMA(model, decay=ema_decay)

    best_val = -1.0
    best_state = None
    best_test_raw = -1.0
    history = []

    for ep in range(epochs):
        train_loss = train_one_epoch(model, tr_loader, opt, sched, ema,
                                     neg_weight=neg_weight, polarity_weight=polarity_weight)
        backup = ema.apply_to(model)
        M, P, T = collect_outputs(model, val_loader)
        v_m = compute_metrics(M, P, T, threshold=0.5)
        M_t, P_t, T_t = collect_outputs(model, te_loader)
        t_m = compute_metrics(M_t, P_t, T_t, threshold=0.5)
        ema.restore(model, backup)

        if v_m['ACC2_Non0'] > best_val:
            best_val = v_m['ACC2_Non0']
            best_state = {k: v.detach().clone().cpu() for k, v in ema.shadow.items()}
            best_test_raw = t_m['ACC2_Non0']
            torch.save({'state_dict': best_state, 'epoch': ep + 1, 'val_nz': best_val,
                        'test_nz_raw': best_test_raw, 'args': {'seed': seed, 'epochs': epochs, 'lr': lr}},
                       os.path.join(save_dir, f'best_seed{seed}.pth'))
        history.append({'epoch': ep + 1, 'train_loss': train_loss,
                        'val_nz': v_m['ACC2_Non0'], 'test_nz_raw': t_m['ACC2_Non0']})
        print(f'  [seed {seed}] ep {ep+1}/{epochs}: train_loss={train_loss:.4f} '
              f'val_nz={v_m["ACC2_Non0"]:.2f}% test_nz={t_m["ACC2_Non0"]:.2f}% '
              f'(best_val={best_val:.2f}%, best_test_raw={best_test_raw:.2f}%)')

    return {'seed': seed, 'best_val_nz': best_val, 'best_test_nz_raw': best_test_raw,
            'state_dict': best_state, 'history': history}


@torch.no_grad()
def ensemble_predict(states, val_loader, te_loader, hidden_dim, dropout):
    models = []
    for st in states:
        m = MainModelV9(hidden_dim=hidden_dim, head_dropout=dropout).cuda()
        m.load_state_dict(st, strict=False)
        m.eval()
        models.append(m)

    def run(loader):
        all_M, all_P, all_T = [], [], []
        for batch in tqdm(loader, desc='Ensemble', leave=False):
            t = batch['text'].cuda(); a = batch['audio'].cuda(); v = batch['vision'].cuda(); m = batch['text_mask'].cuda()
            Ms, Ps = [], []
            for mm in models:
                o = mm(t, a, v, m)
                Ms.append(o['M'])
                Ps.append(o['polarity'])
            M = torch.stack(Ms, dim=0).mean(dim=0)
            P = torch.stack(Ps, dim=0).mean(dim=0)
            all_M.append(M.cpu()); all_P.append(P.cpu()); all_T.append(batch['label'])
        return torch.cat(all_M), torch.cat(all_P), torch.cat(all_T)

    return run(val_loader), run(te_loader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=24)  # RoBERTa-large 显存大, 降到 24
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--neg_weight', type=float, default=1.5)
    p.add_argument('--polarity_weight', type=float, default=0.5)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024])
    p.add_argument('--out_dir', type=str, default='experiments/v9_roberta')
    args = p.parse_args()
    print(f'Args: {vars(args)}')

    os.makedirs(args.out_dir, exist_ok=True)
    states = []
    t_start = time.time()

    for seed in args.seeds:
        print(f'\n========== Training seed={seed} ==========')
        r = train_one_seed(seed=seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                           hidden_dim=args.hidden_dim, dropout=args.dropout,
                           neg_weight=args.neg_weight, polarity_weight=args.polarity_weight,
                           ema_decay=args.ema_decay, save_dir=args.out_dir)
        states.append(r['state_dict'])
        print(f'  >> seed {seed} best val_nz={r["best_val_nz"]:.2f}% '
              f'best test_nz (raw 0.5)={r["best_test_nz_raw"]:.2f}%')

    print(f'\n========== Ensemble {len(states)} seeds ==========')
    val = MOSEIRobertaDataset('val')
    te = MOSEIRobertaDataset('test')
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=0)
    (Mv, Pv, Tv), (Mt, Pt, Tt) = ensemble_predict(states, val_loader, te_loader,
                                                  args.hidden_dim, args.dropout)

    best_t, best_acc_v = best_threshold_for_acc2_nz(Pv, Tv)
    print(f'Best threshold (val): {best_t:.3f} -> ACC2_Non0={best_acc_v:.2f}%')

    t_m_raw = compute_metrics(Mt, Pt, Tt, threshold=0.5)
    t_m_cal = compute_metrics(Mt, Pt, Tt, threshold=best_t)
    print(f'\n[Ensemble Test] raw (t=0.5): ACC2_Non0={t_m_raw["ACC2_Non0"]:.2f}%, F1={t_m_raw["F1_Non0"]:.2f}%, MAE={t_m_raw["MAE"]:.4f}, Corr={t_m_raw["Corr"]:.4f}')
    print(f'[Ensemble Test] cal (t={best_t:.3f}): ACC2_Non0={t_m_cal["ACC2_Non0"]:.2f}%, F1={t_m_cal["F1_Non0"]:.2f}%, MAE={t_m_cal["MAE"]:.4f}, Corr={t_m_cal["Corr"]:.4f}')

    # 关键对比: 弱负子群
    pol_prob = 1.0 / (1.0 + np.exp(-Pt.numpy().flatten()))
    pred_bin = (pol_prob >= best_t).astype(int)
    tgt_bin = (Tt.numpy().flatten() >= 0).astype(int)
    wn = (Tt.numpy().flatten() >= -1) & (Tt.numpy().flatten() < 0)
    wn_acc = (pred_bin[wn] == tgt_bin[wn]).mean() * 100
    print(f'\n[KEY] weak_neg [-1, 0) ACC: {wn_acc:.2f}% (v7 was 67.52%)')

    summary = {
        'args': vars(args),
        'seeds': args.seeds,
        'best_val_threshold': best_t,
        'val_acc2_nz_at_best_t': best_acc_v,
        'test_raw_t05': t_m_raw,
        'test_calibrated': t_m_cal,
        'weak_neg_acc': float(wn_acc),
        'time_min': (time.time() - t_start) / 60,
    }
    with open(os.path.join(args.out_dir, 'ensemble_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nDone in {(time.time()-t_start)/60:.1f} min. Summary -> {args.out_dir}/ensemble_summary.json')


if __name__ == '__main__':
    main()
