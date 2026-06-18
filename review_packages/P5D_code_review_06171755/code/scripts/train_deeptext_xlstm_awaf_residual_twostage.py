#!/usr/bin/env python
"""
scripts/train_deeptext_xlstm_awaf_residual_twostage.py
P5D Two-Stage Training for DeepText-xLSTM-AWAF Residual

Stage 1: Text branch pretrain (freeze residual, train text only)
Stage 2: Residual training (freeze text, train audio/vision xLSTM + AWAF)
Stage 3: Optional joint fine-tune (small LR, unfreeze all)

Usage:
    python scripts/train_deeptext_xlstm_awaf_residual_twostage.py [--config CONFIG_PATH] [--seed SEED]
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


def get_data_loaders(d_cfg, batch_size):
    train_ds = StrongSequenceMOSIDataset('train', feature_root=d_cfg['feature_root'])
    val_ds = StrongSequenceMOSIDataset('val', feature_root=d_cfg['feature_root'])
    test_ds = StrongSequenceMOSIDataset('test', feature_root=d_cfg['feature_root'])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    max_tl = d_cfg.get('max_text_len', 50)
    max_al = d_cfg.get('max_audio_len', 100)
    max_vl = d_cfg.get('max_vision_len', 50)

    cfn = lambda b: collate_strong_sequence(b, max_text_len=max_tl, max_audio_len=max_al, max_vision_len=max_vl)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=cfn, num_workers=d_cfg.get('num_workers', 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=cfn, num_workers=d_cfg.get('num_workers', 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=cfn, num_workers=d_cfg.get('num_workers', 0))
    return train_loader, val_loader, test_loader


def freeze_module(module, freeze=True):
    """Freeze or unfreeze a module's parameters."""
    for p in module.parameters():
        p.requires_grad = not freeze


def freeze_text_branch(model, freeze=True):
    """Freeze all text branch parameters."""
    for name, param in model.named_parameters():
        if any(prefix in name for prefix in ['text_proj', 'text_pool', 'text_mlp', 'reg_text_head', 'cls_text_head']):
            param.requires_grad = not freeze


def freeze_residual_branch(model, freeze=True):
    """Freeze all residual branch parameters (audio, vision, AWAF, delta heads, gate)."""
    text_prefixes = ['text_proj', 'text_pool', 'text_mlp', 'reg_text_head', 'cls_text_head']
    for name, param in model.named_parameters():
        if not any(prefix in name for prefix in text_prefixes):
            param.requires_grad = not freeze


def stage1_pretrain_text(model, train_loader, val_loader, test_loader, cfg, DEVICE, seed, out_dir):
    """Stage 1: Pretrain text branch only, no residual."""
    print(f"\n{'='*60}")
    print(f"Stage 1: Text Branch Pretrain (no residual)")
    print(f"{'='*60}")

    # Set ablation to no_residual for text-only training
    model.ablation = 'no_residual'
    freeze_residual_branch(model, freeze=True)
    freeze_text_branch(model, freeze=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    t_cfg = cfg['training']
    stage1_epochs = t_cfg.get('stage1_epochs', 25)

    trainer = StrictTrainer(
        model, device=DEVICE, lr=t_cfg['lr'], weight_decay=t_cfg.get('weight_decay', 0.01),
        reg_loss_weight=t_cfg.get('reg_loss_weight', 1.0),
        cls_loss_weight=t_cfg.get('cls_loss_weight', 0.5),
        sign_consistency_weight=t_cfg.get('sign_consistency_weight', 0.1),
        delta_reg_weight=0.0,  # No delta in stage 1
        use_amp=t_cfg.get('amp', True) and (DEVICE == 'cuda'),
    )

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_result, val_record = trainer.check_val(epoch, val_loader)
        elapsed = time.time() - t0
        print(f"S1 E{epoch:3d} | loss={train_loss:.4f} | "
              f"val_MAE={val_record['val_MAE']:.4f} | "
              f"val_ACC2_NZ={val_record['val_ACC2_Non0_regsign']:.2f}% | "
              f"lr={trainer.current_lr:.2e} | {elapsed:.1f}s")

    # Save text branch best state
    text_best_state = {k: v.clone().cpu() for k, v in trainer.best_state.items()} if trainer.best_state else None
    text_best_mae = trainer.best_val_mae

    # Stage 1 test
    test_result = trainer.final_test(test_loader)
    mr = test_result['metrics_regsign']
    print(f"\nStage 1 Test: ACC2_NZ={mr['ACC2_Non0']:.2f}%, MAE={mr['MAE']:.4f}, Corr={mr['Corr']:.4f}")

    # Restore ablation
    model.ablation = 'none'

    return text_best_state, text_best_mae


def stage2_train_residual(model, train_loader, val_loader, test_loader, cfg, DEVICE, seed, out_dir):
    """Stage 2: Train residual branch with frozen text."""
    print(f"\n{'='*60}")
    print(f"Stage 2: Residual Branch Training")
    print(f"{'='*60}")

    # Freeze text, unfreeze residual
    freeze_text_branch(model, freeze=True)
    freeze_residual_branch(model, freeze=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    t_cfg = cfg['training']
    stage2_epochs = t_cfg.get('stage2_epochs', 35)

    trainer = StrictTrainer(
        model, device=DEVICE, lr=t_cfg['lr'], weight_decay=t_cfg.get('weight_decay', 0.01),
        reg_loss_weight=t_cfg.get('reg_loss_weight', 1.0),
        cls_loss_weight=t_cfg.get('cls_loss_weight', 0.5),
        sign_consistency_weight=t_cfg.get('sign_consistency_weight', 0.1),
        delta_reg_weight=t_cfg.get('delta_reg_weight', 0.05),
        use_amp=t_cfg.get('amp', True) and (DEVICE == 'cuda'),
    )

    for epoch in range(1, stage2_epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_result, val_record = trainer.check_val(epoch, val_loader)
        elapsed = time.time() - t0
        print(f"S2 E{epoch:3d} | loss={train_loss:.4f} | "
              f"val_MAE={val_record['val_MAE']:.4f} | "
              f"val_ACC2_NZ={val_record['val_ACC2_Non0_regsign']:.2f}% | "
              f"lr={trainer.current_lr:.2e} | {elapsed:.1f}s")

    # Restore all params trainable
    freeze_text_branch(model, freeze=False)

    return trainer


def stage3_joint_finetune(model, train_loader, val_loader, test_loader, cfg, DEVICE, seed, out_dir, stage2_trainer):
    """Stage 3: Optional joint fine-tune with small LR."""
    print(f"\n{'='*60}")
    print(f"Stage 3: Joint Fine-Tune (optional)")
    print(f"{'='*60}")

    t_cfg = cfg['training']
    stage3_epochs = t_cfg.get('stage3_epochs', 10)
    stage3_lr = t_cfg.get('stage3_lr', 2e-5)

    freeze_text_branch(model, freeze=False)
    freeze_residual_branch(model, freeze=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    trainer = StrictTrainer(
        model, device=DEVICE, lr=stage3_lr, weight_decay=t_cfg.get('weight_decay', 0.01),
        reg_loss_weight=t_cfg.get('reg_loss_weight', 1.0),
        cls_loss_weight=t_cfg.get('cls_loss_weight', 0.5),
        sign_consistency_weight=t_cfg.get('sign_consistency_weight', 0.1),
        delta_reg_weight=t_cfg.get('delta_reg_weight', 0.05),
        use_amp=t_cfg.get('amp', True) and (DEVICE == 'cuda'),
    )

    # Keep stage2 best for comparison
    s2_best_mae = stage2_trainer.best_val_mae
    best_improved = False

    for epoch in range(1, stage3_epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_result, val_record = trainer.check_val(epoch, val_loader)
        elapsed = time.time() - t0
        if val_record['val_MAE'] < s2_best_mae * 0.99:  # 1% improvement
            best_improved = True
        print(f"S3 E{epoch:3d} | loss={train_loss:.4f} | "
              f"val_MAE={val_record['val_MAE']:.4f} | "
              f"val_ACC2_NZ={val_record['val_ACC2_Non0_regsign']:.2f}% | "
              f"lr={trainer.current_lr:.2e} | {elapsed:.1f}s")

    if not best_improved:
        print("Stage 3 did not improve over Stage 2 — skipping joint fine-tune")
        model.load_state_dict(stage2_trainer.best_state)
        return stage2_trainer

    return trainer


def main():
    parser = argparse.ArgumentParser(description='Two-Stage Training')
    parser.add_argument('--config', type=str, default='configs/models/deeptext_xlstm_awaf_residual_mosi.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--skip_stage3', action='store_true', help='Skip joint fine-tune')
    parser.add_argument('--use_gate', action='store_true', help='Enable ConditionalResidualGate')
    args = parser.parse_args()

    cfg = load_config(args.config)
    m_cfg = cfg['model']
    t_cfg = cfg['training']
    d_cfg = cfg['data']

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    seed = args.seed
    print(f"Device: {DEVICE}, Seed: {seed}")
    print(f"Use Gate: {args.use_gate}")

    set_seed(seed)

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(d_cfg, t_cfg['batch_size'])

    # Model
    model = DeepTextXLSTMAWAFResidual(
        text_dim=m_cfg['text_dim'], audio_dim=m_cfg['audio_dim'], vision_dim=m_cfg['vision_dim'],
        hidden_dim=m_cfg['hidden_dim'],
        text_mlp_hidden=m_cfg.get('text_mlp_hidden', 512),
        text_mlp_layers=m_cfg.get('text_mlp_layers', 3),
        text_mlp_dropout=m_cfg.get('text_mlp_dropout', 0.3),
        slstm_num_layers=m_cfg.get('slstm_num_layers', 1),
        slstm_dropout=m_cfg.get('slstm_dropout', 0.3),
        awaf_fusion_mode=m_cfg.get('awaf_fusion_mode', 'awaf'),
        awaf_modality_dropout=m_cfg.get('awaf_modality_dropout', True),
        delta_scale_init=m_cfg.get('delta_scale_init', 0.1),
        head_dropout=m_cfg.get('head_dropout', 0.3),
        ablation='none',
        use_residual_gate=args.use_gate,
    ).to(DEVICE)
    print(f"Params: {model._init_info['total_params']:,}")

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = args.output_dir or f"outputs/P5D/MOSI/twostage/{ts}_s{seed}{'_gate' if args.use_gate else ''}"

    # Stage 1: Text pretrain
    text_best, text_mae = stage1_pretrain_text(model, train_loader, val_loader, test_loader, cfg, DEVICE, seed, out_dir)

    # Stage 2: Residual training
    s2_trainer = stage2_train_residual(model, train_loader, val_loader, test_loader, cfg, DEVICE, seed, out_dir)

    # Stage 3: Optional joint fine-tune
    if not args.skip_stage3:
        final_trainer = stage3_joint_finetune(model, train_loader, val_loader, test_loader, cfg, DEVICE, seed, out_dir, s2_trainer)
    else:
        final_trainer = s2_trainer
        print("Stage 3 skipped")

    # Final test
    print(f"\n{'='*60}")
    print(f"Final Test (best epoch: {final_trainer.best_epoch})")
    print(f"{'='*60}")
    test_result = final_trainer.final_test(test_loader)
    mr = test_result['metrics_regsign']
    mc = test_result['metrics_cls']
    a = test_result['awaf_stats']

    print(f"\n=== TWO-STAGE RESULTS (seed={seed}) ===")
    print(f"Stage 1 Text Best MAE: {text_mae:.4f}")
    print(f"Best epoch: {final_trainer.best_epoch}")
    print(f"--- reg_sign ---")
    print(f"ACC2_Non0: {mr['ACC2_Non0']:.2f}%  F1_Non0: {mr['F1_Non0']:.2f}%")
    print(f"MAE: {mr['MAE']:.4f}  Corr: {mr['Corr']:.4f}  ACC7: {mr['ACC7']:.2f}%")
    print(f"--- cls ---")
    print(f"ACC2_Non0: {mc['ACC2_Non0']:.2f}%")
    print(f"--- AWAF ---")
    print(f"w_t: {a['w_t_mean']:.4f}±{a['w_t_std']:.4f}")
    print(f"w_a: {a['w_a_mean']:.4f}±{a['w_a_std']:.4f}")
    print(f"w_v: {a['w_v_mean']:.4f}±{a['w_v_std']:.4f}")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    final_trainer.save_run(out_dir, {**cfg, 'seed': seed, 'use_gate': args.use_gate, 'training_strategy': 'twostage'},
                           ' '.join(sys.argv), t_cfg.get('epochs', 60), test_result, save_last_pth=False)

    print(f"\nOutputs saved to: {out_dir}")


if __name__ == '__main__':
    main()
