"""
ensemble_5seed.py — 合并 v9 (3 seeds) + v9_roberta_5seed (2 seeds) = 5 seeds 集成
==================================================================================

v9 (3 seeds: 42/100/2024):           test ACC2_NZ raw t=0.5 = 86.54%
v9_roberta_5seed (2 seeds: 7/1234): test ACC2_NZ raw t=0.5 = 86.34%
合并: 5 seeds 集成期望更稳

注意: 这次是 "post-hoc", 纯集成报告, 不训练
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, r'D:\business\pycharm\project\Tri_modal_ER')
from mosei_roberta_dataset import MOSEIRobertaDataset
from model_main_v6 import MainModelV9


@torch.no_grad()
def ensemble_collect(model_paths, hidden_dim, dropout, batch_size):
    models = []
    for p in model_paths:
        m = MainModelV9(hidden_dim=hidden_dim, head_dropout=dropout).cuda()
        ckpt = torch.load(p, map_location='cuda', weights_only=False)
        m.load_state_dict(ckpt['state_dict'], strict=False)
        m.eval()
        models.append(m)

    def run(split):
        ds = MOSEIRobertaDataset(split)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        all_M, all_P, all_T = [], [], []
        for batch in tqdm(loader, desc=f'{split}', leave=False):
            t = batch['text'].cuda(); a = batch['audio'].cuda(); v = batch['vision'].cuda(); m = batch['text_mask'].cuda()
            Ms, Ps = [], []
            for mm in models:
                o = mm(t, a, v, m)
                Ms.append(o['M'])
                Ps.append(o['polarity'])
            M = torch.stack(Ms, dim=0).mean(dim=0)
            P = torch.stack(Ps, dim=0).mean(dim=0)
            all_M.append(M.cpu()); all_P.append(P.cpu()); all_T.append(batch['label'])
        return torch.cat(all_M).numpy().flatten(), torch.cat(all_P).numpy().flatten(), torch.cat(all_T).numpy().flatten()

    return run('val'), run('test')


def compute_full_metrics(pol_logit, target, threshold):
    pl = pol_logit.flatten()
    tg = target.flatten()
    pol_prob = 1.0 / (1.0 + np.exp(-pl))
    pred_bin = (pol_prob >= threshold).astype(int)
    tgt_bin = (tg >= 0).astype(int)
    nz = tg != 0
    pb_nz = pred_bin[nz]; tb_nz = tgt_bin[nz]
    acc2_nz = float((pb_nz == tb_nz).mean()) * 100
    tp = int(((pb_nz == 1) & (tb_nz == 1)).sum())
    fp = int(((pb_nz == 1) & (tb_nz == 0)).sum())
    fn = int(((pb_nz == 0) & (tb_nz == 1)).sum())
    tn = int(((pb_nz == 0) & (tb_nz == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (tpr + specificity) / 2
    youden_j = tpr - fpr
    acc2 = float((pred_bin == tgt_bin).mean()) * 100
    return {
        'threshold': float(threshold),
        'acc2': acc2, 'acc2_nz': acc2_nz, 'f1': f1 * 100,
        'balanced_acc': balanced_acc * 100, 'youden_j': youden_j,
    }


def find_best_threshold(pol_logit, target, metric='acc2_nz', n_steps=401):
    pl = pol_logit.flatten()
    tg = target.flatten()
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.20, 0.80, n_steps):
        m = compute_full_metrics(pl, tg, t)
        v = m[metric]
        if v > best_v:
            best_v = v; best_t = t
    return float(best_t), best_v


def main():
    seeds = [42, 100, 2024, 7, 1234]
    model_paths = (
        [f'experiments/v9_roberta/best_seed{s}.pth' for s in [42, 100, 2024]] +
        [f'experiments/v9_roberta_5seed/best_seed{s}.pth' for s in [7, 1234]]
    )
    print(f'Loading {len(model_paths)} models ({len(seeds)} seeds)...')
    (Mv, Pv, Tv), (Mt, Pt, Tt) = ensemble_collect(model_paths, 256, 0.3, 24)
    print(f'val N={len(Tv)}, test N={len(Tt)}')

    print('\n=== 5-seed Ensemble on TEST ===')
    print(f'{"strategy":<35} {"threshold":>10} {"ACC2_NZ":>10} {"F1":>8} {"Balanced":>10} {"Youden":>8}')
    for name, fixed_t, metric in [
        ('raw t=0.5 (no calib)', 0.5, None),
        ('val acc2_nz optimal', None, 'acc2_nz'),
        ('val F1 optimal', None, 'f1'),
        ('val Youden J optimal', None, 'youden_j'),
        ('val Balanced Acc optimal', None, 'balanced_acc'),
    ]:
        if fixed_t is not None:
            t = fixed_t
        else:
            t, _ = find_best_threshold(Pv, Tv, metric=metric)
        m = compute_full_metrics(Pt, Tt, t)
        print(f'{name:<35} {t:>10.4f} {m["acc2_nz"]:>10.2f} {m["f1"]:>8.2f} {m["balanced_acc"]:>10.2f} {m["youden_j"]:>8.4f}')

    print('\n=== Sanity: test-optimal (上限) ===')
    for metric in ['acc2_nz', 'f1', 'balanced_acc']:
        t, v = find_best_threshold(Pt, Tt, metric=metric)
        print(f'  test {metric} optimal: t={t:.4f}, val={v:.2f}')

    print('\n=== weak_neg [-1, 0) ACC under different thresholds (test) ===')
    wn = (Tt >= -1) & (Tt < 0)
    print(f'  weak_neg N = {wn.sum()}')
    for name, fixed_t, metric in [
        ('raw t=0.5', 0.5, None),
        ('val acc2_nz', None, 'acc2_nz'),
        ('val Youden J', None, 'youden_j'),
    ]:
        if fixed_t is not None:
            t = fixed_t
        else:
            t, _ = find_best_threshold(Pv, Tv, metric=metric)
        pol_prob = 1.0 / (1.0 + np.exp(-Pt))
        pred_bin = (pol_prob >= t).astype(int)
        tgt_bin = (Tt >= 0).astype(int)
        wn_acc = (pred_bin[wn] == tgt_bin[wn]).mean() * 100
        print(f'  {name:<20} t={t:.4f}  weak_neg ACC = {wn_acc:.2f}%')

    summary = {
        'n_seeds': len(seeds),
        'seeds': seeds,
        'note': '5-seed 集成: 3 from v9_roberta + 2 from v9_roberta_5seed',
        'test_raw_t05': compute_full_metrics(Pt, Tt, 0.5),
    }
    with open('experiments/v9_roberta_5seed/ensemble_5seed_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved 5-seed summary')


if __name__ == '__main__':
    main()
