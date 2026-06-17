#!/usr/bin/env python
"""
scripts/train_deeptext_xlstm_awaf_residual.py
P5C Full training — DeepText-xLSTM-AWAF Residual on MOSI

Usage:
    python scripts/train_deeptext_xlstm_awaf_residual.py [--config CONFIG_PATH] [--seed SEED] [--epochs EPOCHS]

Default: configs/models/deeptext_xlstm_awaf_residual_mosi.yaml
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np

from models.deeptext_xlstm_awaf_residual import DeepTextXLSTMAWAFResidual
from engine.strict_trainer import StrictTrainer
from data.strong_sequence_dataset import StrongSequenceMOSIDataset, collate_strong_sequence
from utils.seed import set_seed


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train DeepText-xLSTM-AWAF Residual')
    parser.add_argument('--config', type=str, default='configs/models/deeptext_xlstm_awaf_residual_mosi.yaml')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--ablation', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    cfg = load_config(args.config)
    m_cfg = cfg['model']
    t_cfg = cfg['training']
    d_cfg = cfg['data']

    # Overrides
    seed = args.seed if args.seed is not None else t_cfg['seed']
    epochs = args.epochs if args.epochs is not None else t_cfg['epochs']
    lr = args.lr if args.lr is not None else t_cfg['lr']
    batch_size = args.batch_size if args.batch_size is not None else t_cfg['batch_size']
    ablation = args.ablation if args.ablation is not None else m_cfg['ablation']

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")
    print(f"Config: {args.config}")
    print(f"Seed: {seed}, Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"Ablation: {ablation}")

    # --- Seed ---
    set_seed(seed)

    # --- Data ---
    print("\n=== Loading Data ===")
    train_ds = StrongSequenceMOSIDataset('train', feature_root=d_cfg['feature_root'])
    val_ds = StrongSequenceMOSIDataset('val', feature_root=d_cfg['feature_root'])
    test_ds = StrongSequenceMOSIDataset('test', feature_root=d_cfg['feature_root'])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    max_tl = d_cfg.get('max_text_len', 50)
    max_al = d_cfg.get('max_audio_len', 100)
    max_vl = d_cfg.get('max_vision_len', 50)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_strong_sequence(b, max_text_len=max_tl, max_audio_len=max_al, max_vision_len=max_vl),
        num_workers=d_cfg.get('num_workers', 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_strong_sequence(b, max_text_len=max_tl, max_audio_len=max_al, max_vision_len=max_vl),
        num_workers=d_cfg.get('num_workers', 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_strong_sequence(b, max_text_len=max_tl, max_audio_len=max_al, max_vision_len=max_vl),
        num_workers=d_cfg.get('num_workers', 0))

    # --- Model ---
    print("\n=== Building Model ===")
    model = DeepTextXLSTMAWAFResidual(
        text_dim=m_cfg['text_dim'], audio_dim=m_cfg['audio_dim'], vision_dim=m_cfg['vision_dim'],
        hidden_dim=m_cfg['hidden_dim'],
        text_mlp_hidden=m_cfg.get('text_mlp_hidden', 512),
        text_mlp_layers=m_cfg.get('text_mlp_layers', 3),
        text_mlp_dropout=m_cfg.get('text_mlp_dropout', 0.3),
        slstm_num_layers=m_cfg.get('slstm_num_layers', 1),
        slstm_dropout=m_cfg.get('slstm_dropout', 0.3),
        slstm_pooling=m_cfg.get('slstm_pooling', 'masked_mean'),
        awaf_fusion_mode=m_cfg.get('awaf_fusion_mode', 'awaf'),
        awaf_tau_init=m_cfg.get('awaf_tau_init', 1.0),
        awaf_dropout=m_cfg.get('awaf_dropout', 0.1),
        awaf_modality_dropout=m_cfg.get('awaf_modality_dropout', True),
        awaf_modality_dropout_prob=m_cfg.get('awaf_modality_dropout_prob', 0.1),
        delta_scale_init=m_cfg.get('delta_scale_init', 0.1),
        head_dropout=m_cfg.get('head_dropout', 0.3),
        ablation=ablation,
        use_aux_heads=m_cfg.get('use_aux_heads', False),
    )
    print(f"Params: {model._init_info['total_params']:,}")
    print(f"Ablation: {model.ablation}")
    print(f"Text sLSTM: {model._text_slstm_on}")

    # --- Trainer ---
    print("\n=== Creating Trainer ===")
    use_amp = t_cfg.get('amp', True) and (DEVICE == 'cuda')
    trainer = StrictTrainer(
        model, device=DEVICE,
        lr=lr, weight_decay=t_cfg.get('weight_decay', 0.01),
        reg_loss_weight=t_cfg.get('reg_loss_weight', 1.0),
        cls_loss_weight=t_cfg.get('cls_loss_weight', 0.5),
        aux_loss_weight=t_cfg.get('aux_loss_weight', 0.0),
        awaf_entropy_reg_weight=t_cfg.get('awaf_entropy_reg_weight', 0.0),
        sign_consistency_weight=t_cfg.get('sign_consistency_weight', 0.1),
        delta_reg_weight=t_cfg.get('delta_reg_weight', 0.05),
        use_amp=use_amp,
    )

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training: {epochs} epochs, seed={seed}, ablation={ablation}")
    print(f"{'='*60}")

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_result, val_record = trainer.check_val(epoch, val_loader)
        elapsed = time.time() - t0

        val_acc = val_record['val_ACC2_Non0_regsign']
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"E{epoch:3d} | loss={train_loss:.4f} | "
              f"val_MAE={val_record['val_MAE']:.4f} | "
              f"val_ACC2_NZ={val_acc:.2f}% | "
              f"best_val_ACC2_NZ={best_val_acc:.2f}% | "
              f"lr={trainer.current_lr:.2e} | {elapsed:.1f}s")

    # --- Final Test ---
    print(f"\n{'='*60}")
    print(f"Final Test (best epoch: {trainer.best_epoch})")
    print(f"{'='*60}")
    test_result = trainer.final_test(test_loader)
    mr = test_result['metrics_regsign']
    mc = test_result['metrics_cls']
    a = test_result['awaf_stats']

    print(f"\n=== RESULTS (seed={seed}, epochs={epochs}, ablation={ablation}) ===")
    print(f"Best epoch (by val MAE): {trainer.best_epoch}")
    print(f"--- reg_sign ---")
    print(f"ACC2_Non0: {mr['ACC2_Non0']:.2f}%  F1_Non0: {mr['F1_Non0']:.2f}%")
    print(f"ACC2_Has0: {mr['ACC2_Has0']:.2f}%  F1_Has0: {mr['F1_Has0']:.2f}%")
    print(f"MAE: {mr['MAE']:.4f}  Corr: {mr['Corr']:.4f}  ACC7: {mr['ACC7']:.2f}%")
    print(f"--- cls ---")
    print(f"ACC2_Non0: {mc['ACC2_Non0']:.2f}%  F1_Non0: {mc['F1_Non0']:.2f}%")
    print(f"--- AWAF ---")
    print(f"w_t: {a['w_t_mean']:.4f}±{a['w_t_std']:.4f}")
    print(f"w_a: {a['w_a_mean']:.4f}±{a['w_a_std']:.4f}")
    print(f"w_v: {a['w_v_mean']:.4f}±{a['w_v_std']:.4f}")
    print(f"sum(w) max_dev: {a['sum_w_max_dev']:.2e}")

    # --- Save ---
    ts = time.strftime('%Y%m%d_%H%M%S')
    abl_tag = f"_{ablation}" if ablation != 'none' else ""
    out_dir = args.output_dir or f"outputs/P5C/MOSI/deeptext_xlstm_awaf_residual/{ts}_s{seed}{abl_tag}"
    os.makedirs(out_dir, exist_ok=True)
    trainer.save_run(out_dir, {**cfg, 'seed': seed, 'epochs': epochs, 'ablation': ablation},
                     ' '.join(sys.argv), epochs, test_result)
    print(f"\nOutputs saved to: {out_dir}")

    # --- Judgment ---
    acc2 = mr['ACC2_Non0']
    print(f"\n{'='*60}")
    print(f"FINAL JUDGMENT: ACC2_NZ_reg = {acc2:.2f}%")
    if acc2 > 83:
        print("✅ EXCEEDS 83% — recommend multi-seed formal training!")
    elif acc2 > 80.2:
        print(f"✅ EXCEEDS DeepMLP text-only (80.2%) by {acc2-80.2:.1f}%")
        print("   Residual fusion is HELPING! Consider multi-seed.")
    elif acc2 > 78.8:
        print(f"⚠️  EXCEEDS P4W AWAF-Seq (78.8%) by {acc2-78.8:.1f}%")
        print("   But below DeepMLP text-only. Tune hyperparams.")
    else:
        print(f"❌ BELOW P4W AWAF-Seq (78.8%)")
        print("   Residual fusion is HURTING. Check model/loss.")
    print(f"{'='*60}")

    return acc2


if __name__ == '__main__':
    main()
