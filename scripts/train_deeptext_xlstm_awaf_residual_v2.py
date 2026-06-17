#!/usr/bin/env python
"""
scripts/train_deeptext_xlstm_awaf_residual_v2.py
P5E V2 one-stage training with UGR gate, delta experts, delta target loss.

Usage:
    python scripts/train_deeptext_xlstm_awaf_residual_v2.py [--seed SEED] [--epochs EPOCHS] [--ablation ABLATION]
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, yaml, numpy as np
from torch.utils.data import DataLoader

from models.deeptext_xlstm_awaf_residual_v2 import DeepTextXLSTMAWAFResidualV2
from engine.residual_losses_v2 import ResidualLossV2
from data.strong_sequence_dataset import StrongSequenceMOSIDataset, collate_strong_sequence
from utils.metrics import compute_all_metrics
from utils.seed import set_seed
from models.fusion.awaf import save_awaf_weights_csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv


class V2Trainer:
    def __init__(self, model, loss_computer, device='cuda', lr=1e-4, weight_decay=0.01,
                 use_amp=True, grad_clip=1.0):
        self.model = model.to(device)
        self.device = device
        self.loss_computer = loss_computer
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=8)
        self.scaler = torch.amp.GradScaler('cuda') if use_amp and device == 'cuda' else None
        self.use_amp = use_amp and device == 'cuda'
        self.grad_clip = grad_clip
        self.best_val_mae = float('inf')
        self.best_epoch = 0
        self.best_state = None
        self.val_history = []
        self.current_lr = lr

    def train_epoch(self, loader, epoch):
        self.model.train()
        total, cnt = 0.0, 0
        for batch in tqdm(loader, desc=f'Train E{epoch}', leave=False):
            txt = batch['text'].to(self.device)
            aud = batch['audio'].to(self.device)
            vis = batch['vision'].to(self.device)
            lbl = batch['label'].to(self.device)
            tm = batch.get('text_mask'); am = batch.get('audio_mask'); vm = batch.get('vision_mask')
            if isinstance(tm, torch.Tensor): tm = tm.to(self.device)
            if isinstance(am, torch.Tensor): am = am.to(self.device)
            if isinstance(vm, torch.Tensor): vm = vm.to(self.device)
            self.opt.zero_grad()
            out = self.model(txt, aud, vis, text_mask=tm, audio_mask=am, vision_mask=vm)
            if torch.isnan(out['reg']).any(): continue
            losses = self.loss_computer.compute(out, lbl)
            loss = losses['total']
            if torch.isnan(loss): continue
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
            bs = aud.size(0)
            total += loss.item() * bs; cnt += bs
        return total / cnt if cnt > 0 else float('nan')

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_reg, all_cls, all_lbl, all_w = [], [], [], []
        all_reg_base, all_delta, all_eff_delta, all_gate = [], [], [], []
        all_ids = []
        total_loss, cnt = 0.0, 0
        for batch in tqdm(loader, desc='Eval', leave=False):
            txt = batch['text'].to(self.device)
            aud = batch['audio'].to(self.device)
            vis = batch['vision'].to(self.device)
            lbl = batch['label'].to(self.device)
            tm = batch.get('text_mask'); am = batch.get('audio_mask'); vm = batch.get('vision_mask')
            if isinstance(tm, torch.Tensor): tm = tm.to(self.device)
            if isinstance(am, torch.Tensor): am = am.to(self.device)
            if isinstance(vm, torch.Tensor): vm = vm.to(self.device)
            out = self.model(txt, aud, vis, text_mask=tm, audio_mask=am, vision_mask=vm)
            losses = self.loss_computer.compute(out, lbl)
            bs = aud.size(0)
            total_loss += losses['total'].item() * bs; cnt += bs
            all_reg.append(out['reg'].cpu()); all_cls.append(out['cls'].cpu())
            all_lbl.append(lbl.cpu()); all_w.append(out['awaf_weights'].cpu())
            ids = batch.get('id', [str(i) for i in range(len(lbl))]); all_ids.extend(ids)
            if out.get('reg_text_base') is not None:
                all_reg_base.append(out['reg_text_base'].cpu())
                all_delta.append(out['delta_reg'].cpu())
                all_eff_delta.append(out['effective_delta_reg'].cpu())
            if out.get('residual_gate_reg') is not None:
                all_gate.append(out['residual_gate_reg'].cpu())

        rp = torch.cat(all_reg); cp = torch.cat(all_cls); tg = torch.cat(all_lbl)
        aw = torch.cat(all_w)
        rs = torch.where(rp >= 0, 1.0, -1.0)
        m_reg = compute_all_metrics(rp, rs, tg)
        m_cls = compute_all_metrics(rp, cp, tg)
        wc = aw.clamp(1e-8); ent = -(wc * torch.log(wc)).sum(-1)
        aw_stats = {
            'w_t_mean': float(aw[:, 0].mean()), 'w_a_mean': float(aw[:, 1].mean()), 'w_v_mean': float(aw[:, 2].mean()),
            'sum_w_max_dev': float((aw.sum(-1) - 1).abs().max()),
        }
        result = {
            'reg_preds': rp, 'cls_preds': cp, 'targets': tg, 'awaf_weights': aw,
            'metrics_regsign': m_reg, 'metrics_cls': m_cls, 'awaf_stats': aw_stats,
            'ids': all_ids, 'loss': total_loss / cnt if cnt > 0 else float('nan'),
        }
        if all_reg_base:
            result['reg_text_base'] = torch.cat(all_reg_base)
            result['delta_reg'] = torch.cat(all_delta)
            result['effective_delta_reg'] = torch.cat(all_eff_delta)
        if all_gate:
            result['residual_gate_reg'] = torch.cat(all_gate)
        return result

    def check_val(self, epoch, val_loader):
        val_r = self.evaluate(val_loader)
        val_mae = val_r['metrics_regsign']['MAE']
        record = {'epoch': epoch, 'val_loss': val_r['loss'],
                  'val_MAE': val_mae, 'val_ACC2_Non0_regsign': val_r['metrics_regsign']['ACC2_Non0']}
        self.val_history.append(record)
        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
            self.best_epoch = epoch
            self.best_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        self.scheduler.step(val_mae)
        self.current_lr = self.opt.param_groups[0]['lr']
        return val_r, record

    def final_test(self, test_loader):
        if self.best_state is not None:
            current = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_state)
        test_r = self.evaluate(test_loader)
        if self.best_state is not None:
            self.model.load_state_dict(current)
        return test_r

    def save_run(self, out_dir, config, cmd, epoch, test_result):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'config.json'), 'w') as f: json.dump(config, f, indent=2)
        with open(os.path.join(out_dir, 'command.txt'), 'w') as f: f.write(cmd)
        # Val history
        if self.val_history:
            with open(os.path.join(out_dir, 'val_metrics_epoch.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(self.val_history[0].keys()))
                w.writeheader(); w.writerows(self.val_history)
        m = test_result; mr = m['metrics_regsign']; mc = m['metrics_cls']; a = m['awaf_stats']
        test_metrics = {
            'best_epoch_by_val': self.best_epoch, 'best_val_MAE': self.best_val_mae,
            'ACC2_Non0_regsign': mr['ACC2_Non0'], 'F1_Non0_regsign': mr['F1_Non0'],
            'ACC2_Has0_regsign': mr['ACC2_Has0'], 'MAE': mr['MAE'], 'Corr': mr['Corr'], 'ACC7': mr['ACC7'],
            'ACC2_Non0_cls': mc['ACC2_Non0'], **a
        }
        with open(os.path.join(out_dir, 'test_metrics_final.json'), 'w') as f: json.dump(test_metrics, f, indent=2)
        # Predictions
        with open(os.path.join(out_dir, 'predictions_test.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['sample_id', 'reg_pred', 'cls_logit', 'target'])
            rp = m['reg_preds'].numpy().flatten(); cp = m['cls_preds'].numpy().flatten()
            tg = m['targets'].numpy().flatten(); ids = m.get('ids', range(len(rp)))
            for i in range(len(rp)): w.writerow([ids[i] if i < len(ids) else i, rp[i], cp[i], tg[i]])
        save_awaf_weights_csv(m['awaf_weights'], m.get('ids', []), os.path.join(out_dir, 'awaf_weights_test.csv'))
        # text_base_delta_gate_test.csv
        with open(os.path.join(out_dir, 'text_base_delta_gate_test.csv'), 'w', newline='') as f:
            fieldnames = ['sample_id', 'label', 'reg_text_base', 'delta_reg', 'effective_delta_reg',
                          'reg_final', 'gate_reg', 'awaf_w_t', 'awaf_w_a', 'awaf_w_v']
            w = csv.writer(f); w.writerow(fieldnames)
            for i in range(len(rp)):
                w.writerow([
                    ids[i] if i < len(ids) else i, tg[i],
                    m.get('reg_text_base', rp)[i].item() if m.get('reg_text_base') is not None else '',
                    m.get('delta_reg', rp)[i].item() if m.get('delta_reg') is not None else '',
                    m.get('effective_delta_reg', rp)[i].item() if m.get('effective_delta_reg') is not None else '',
                    rp[i],
                    m.get('residual_gate_reg', rp)[i].item() if m.get('residual_gate_reg') is not None else '',
                    m['awaf_weights'][i, 0].item(), m['awaf_weights'][i, 1].item(), m['awaf_weights'][i, 2].item(),
                ])
        # Only save best_model.pth
        torch.save({'epoch': epoch, 'model_state_dict': self.best_state or self.model.state_dict(),
                    'best_val_mae': self.best_val_mae, 'best_epoch': self.best_epoch,
                    'test_metrics': test_metrics}, os.path.join(out_dir, 'best_model.pth'))
        # Plot
        if self.val_history:
            fig, ax = plt.subplots(figsize=(10, 5))
            eps = [r['epoch'] for r in self.val_history]
            ax.plot(eps, [r['val_MAE'] for r in self.val_history], 'r-', label='Val MAE')
            ax.plot(eps, [r['val_ACC2_Non0_regsign'] for r in self.val_history], 'g-', label='Val ACC2')
            ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(out_dir, 'val_curves.png'), dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/models/deeptext_xlstm_awaf_residual_v2_mosi.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--ablation', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f: cfg = yaml.safe_load(f)
    m_cfg, t_cfg, d_cfg = cfg['model'], cfg['training'], cfg['data']
    seed = args.seed
    epochs = args.epochs or t_cfg['epochs']
    ablation = args.ablation or m_cfg.get('ablation', 'none')
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'

    print(f"P5E V2 Training: seed={seed}, epochs={epochs}, ablation={ablation}")
    print(f"Device: {DEVICE}")
    set_seed(seed)

    # Data
    train_ds = StrongSequenceMOSIDataset('train', feature_root=d_cfg['feature_root'])
    val_ds = StrongSequenceMOSIDataset('val', feature_root=d_cfg['feature_root'])
    test_ds = StrongSequenceMOSIDataset('test', feature_root=d_cfg['feature_root'])
    print(f"Data: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    cfn = lambda b: collate_strong_sequence(b, max_text_len=d_cfg.get('max_text_len', 50),
                                             max_audio_len=d_cfg.get('max_audio_len', 100),
                                             max_vision_len=d_cfg.get('max_vision_len', 50))
    train_loader = DataLoader(train_ds, batch_size=t_cfg['batch_size'], shuffle=True, collate_fn=cfn)
    val_loader = DataLoader(val_ds, batch_size=t_cfg['batch_size'], shuffle=False, collate_fn=cfn)
    test_loader = DataLoader(test_ds, batch_size=t_cfg['batch_size'], shuffle=False, collate_fn=cfn)

    # Model
    model = DeepTextXLSTMAWAFResidualV2(
        text_dim=m_cfg['text_dim'], audio_dim=m_cfg['audio_dim'], vision_dim=m_cfg['vision_dim'],
        hidden_dim=m_cfg['hidden_dim'],
        text_mlp_hidden=m_cfg.get('text_mlp_hidden', 512), text_mlp_layers=m_cfg.get('text_mlp_layers', 3),
        text_mlp_dropout=m_cfg.get('text_mlp_dropout', 0.3),
        slstm_num_layers=m_cfg.get('slstm_num_layers', 1), slstm_dropout=m_cfg.get('slstm_dropout', 0.3),
        awaf_fusion_mode=m_cfg.get('awaf_fusion_mode', 'awaf'),
        awaf_modality_dropout=m_cfg.get('awaf_modality_dropout', True),
        awaf_modality_dropout_prob=m_cfg.get('awaf_modality_dropout_prob', 0.1),
        delta_scale_init=m_cfg.get('delta_scale_init', 0.1),
        max_delta=m_cfg.get('max_delta', 1.5),
        use_bounded_delta=m_cfg.get('use_bounded_delta', True),
        use_uncertainty_gate=m_cfg.get('use_uncertainty_gate', True),
        use_delta_experts=m_cfg.get('use_delta_experts', True),
        head_dropout=m_cfg.get('head_dropout', 0.3),
        ablation=ablation,
    )
    print(f"Model: {model._init_info['total_params']:,} params")

    # Loss
    loss_computer = ResidualLossV2(
        reg_loss_weight=t_cfg.get('reg_loss_weight', 1.0),
        cls_loss_weight=t_cfg.get('cls_loss_weight', 0.5),
        sign_consistency_weight=t_cfg.get('sign_consistency_weight', 0.1),
        delta_reg_weight=t_cfg.get('delta_reg_weight', 0.02),
        delta_target_loss_weight=t_cfg.get('delta_target_loss_weight', 0.2),
        margin_sign_loss_weight=t_cfg.get('margin_sign_loss_weight', 0.05),
        sample_reweight_enabled=t_cfg.get('sample_reweight_enabled', False),
        weak_neg_weight=t_cfg.get('weak_neg_weight', 1.5),
        weak_pos_weight=t_cfg.get('weak_pos_weight', 1.1),
        near_zero_weight=t_cfg.get('near_zero_weight', 1.0),
        sign_focal_enabled=t_cfg.get('sign_focal_enabled', False),
    )

    # Trainer
    trainer = V2Trainer(model, loss_computer, device=DEVICE, lr=t_cfg['lr'],
                        weight_decay=t_cfg.get('weight_decay', 0.01),
                        use_amp=t_cfg.get('amp', True))

    # Training
    print(f"{'='*60}")
    print(f"Training: {epochs} epochs")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_result, val_record = trainer.check_val(epoch, val_loader)
        elapsed = time.time() - t0
        print(f"E{epoch:3d} | loss={train_loss:.4f} | val_MAE={val_record['val_MAE']:.4f} | "
              f"val_ACC2_NZ={val_record['val_ACC2_Non0_regsign']:.2f}% | lr={trainer.current_lr:.2e} | {elapsed:.1f}s")

    # Test
    print(f"\n{'='*60}")
    print(f"Final Test (best epoch: {trainer.best_epoch})")
    test_result = trainer.final_test(test_loader)
    mr = test_result['metrics_regsign']
    mc = test_result['metrics_cls']
    a = test_result['awaf_stats']
    print(f"\n=== V2 RESULTS (seed={seed}, epochs={epochs}, ablation={ablation}) ===")
    print(f"ACC2_Non0: {mr['ACC2_Non0']:.2f}%  F1_Non0: {mr['F1_Non0']:.2f}%")
    print(f"MAE: {mr['MAE']:.4f}  Corr: {mr['Corr']:.4f}  ACC7: {mr['ACC7']:.2f}%")
    print(f"ACC2_Non0_cls: {mc['ACC2_Non0']:.2f}%")
    print(f"AWAF: w_t={a['w_t_mean']:.4f}, w_a={a['w_a_mean']:.4f}, w_v={a['w_v_mean']:.4f}")

    # Save
    ts = time.strftime('%Y%m%d_%H%M%S')
    abl_tag = f"_{ablation}" if ablation != 'none' else ""
    out_dir = args.output_dir or f"outputs/P5E/MOSI/deeptext_xlstm_awaf_residual_v2/{ts}_s{seed}{abl_tag}"
    trainer.save_run(out_dir, {**cfg, 'seed': seed, 'epochs': epochs, 'ablation': ablation},
                     ' '.join(sys.argv), epochs, test_result)
    print(f"Saved to: {out_dir}")

    # Judgment
    acc2 = mr['ACC2_Non0']
    print(f"\nFINAL JUDGMENT: ACC2_NZ_reg = {acc2:.2f}%")
    if acc2 >= 83.0:
        print("✅ >=83% — Strong candidate!")
    elif acc2 >= 82.0:
        print("✅ >=82% — Consider seed=2024")
    elif acc2 >= 81.25:
        print("⚠️ >=P5D mean — Marginal improvement")
    else:
        print("❌ <P5D mean — Degradation")


if __name__ == '__main__':
    main()
