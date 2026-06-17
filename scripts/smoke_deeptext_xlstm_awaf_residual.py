#!/usr/bin/env python
"""
scripts/smoke_deeptext_xlstm_awaf_residual.py
P5C 3-epoch MOSI smoke test — DeepText-xLSTM-AWAF Residual

Usage:
    python scripts/smoke_deeptext_xlstm_awaf_residual.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from models.deeptext_xlstm_awaf_residual import DeepTextXLSTMAWAFResidual
from engine.strict_trainer import StrictTrainer
from data.strong_sequence_dataset import StrongSequenceMOSIDataset, collate_strong_sequence
from utils.metrics import compute_all_metrics
from utils.seed import set_seed

# --- Config ---
CONFIG = {
    'text_dim': 1024, 'audio_dim': 768, 'vision_dim': 768,
    'hidden_dim': 256,
    'text_mlp_hidden': 512, 'text_mlp_layers': 3,
    'slstm_num_layers': 1, 'slstm_dropout': 0.3,
    'awaf_fusion_mode': 'awaf', 'awaf_modality_dropout': True,
    'delta_scale_init': 0.1, 'head_dropout': 0.3,
    'ablation': 'none', 'use_aux_heads': False,
    'batch_size': 32, 'epochs': 3, 'lr': 1e-4, 'weight_decay': 0.01,
    'reg_loss_weight': 1.0, 'cls_loss_weight': 0.5,
    'sign_consistency_weight': 0.1, 'delta_reg_weight': 0.05,
    'seed': 42,
    'feature_root': 'data/features_strong_sequence_mosi_v3_T40',
    'max_text_len': 50, 'max_audio_len': 100, 'max_vision_len': 50,
    'use_amp': True,
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Config: {json.dumps(CONFIG, indent=2)}")

# --- Seed ---
set_seed(CONFIG['seed'])

# --- Data ---
print("\nLoading data...")
train_ds = StrongSequenceMOSIDataset('train', feature_root=CONFIG['feature_root'])
val_ds = StrongSequenceMOSIDataset('val', feature_root=CONFIG['feature_root'])
test_ds = StrongSequenceMOSIDataset('test', feature_root=CONFIG['feature_root'])
print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                          collate_fn=lambda b: collate_strong_sequence(b,
                              max_text_len=CONFIG['max_text_len'],
                              max_audio_len=CONFIG['max_audio_len'],
                              max_vision_len=CONFIG['max_vision_len']))
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                        collate_fn=lambda b: collate_strong_sequence(b,
                            max_text_len=CONFIG['max_text_len'],
                            max_audio_len=CONFIG['max_audio_len'],
                            max_vision_len=CONFIG['max_vision_len']))
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                         collate_fn=lambda b: collate_strong_sequence(b,
                             max_text_len=CONFIG['max_text_len'],
                             max_audio_len=CONFIG['max_audio_len'],
                             max_vision_len=CONFIG['max_vision_len']))

# --- Model ---
print("\nBuilding model...")
model = DeepTextXLSTMAWAFResidual(
    text_dim=CONFIG['text_dim'], audio_dim=CONFIG['audio_dim'], vision_dim=CONFIG['vision_dim'],
    hidden_dim=CONFIG['hidden_dim'],
    text_mlp_hidden=CONFIG['text_mlp_hidden'], text_mlp_layers=CONFIG['text_mlp_layers'],
    slstm_num_layers=CONFIG['slstm_num_layers'], slstm_dropout=CONFIG['slstm_dropout'],
    awaf_fusion_mode=CONFIG['awaf_fusion_mode'], awaf_modality_dropout=CONFIG['awaf_modality_dropout'],
    delta_scale_init=CONFIG['delta_scale_init'], head_dropout=CONFIG['head_dropout'],
    ablation=CONFIG['ablation'], use_aux_heads=CONFIG['use_aux_heads'],
)
print(f"  Params: {model._init_info['total_params']:,}")
print(f"  Ablation: {model.ablation}")
print(f"  Text sLSTM: {model._text_slstm_on}")

# --- Trainer ---
print("\nCreating trainer...")
trainer = StrictTrainer(
    model, device=DEVICE, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'],
    reg_loss_weight=CONFIG['reg_loss_weight'], cls_loss_weight=CONFIG['cls_loss_weight'],
    sign_consistency_weight=CONFIG['sign_consistency_weight'],
    delta_reg_weight=CONFIG['delta_reg_weight'],
    use_amp=CONFIG['use_amp'],
)

# --- Training loop ---
print(f"\n{'='*60}")
print(f"Starting {CONFIG['epochs']}-epoch smoke test")
print(f"{'='*60}")

for epoch in range(1, CONFIG['epochs'] + 1):
    t0 = time.time()
    train_loss = trainer.train_epoch(train_loader, epoch)
    val_result, val_record = trainer.check_val(epoch, val_loader)
    elapsed = time.time() - t0

    print(f"  E{epoch:2d} | train_loss={train_loss:.4f} | "
          f"val_MAE={val_record['val_MAE']:.4f} | "
          f"val_ACC2_NZ_regsign={val_record['val_ACC2_Non0_regsign']:.2f}% | "
          f"lr={trainer.current_lr:.2e} | {elapsed:.1f}s")

# --- Final test ---
print(f"\n{'='*60}")
print("Final test on best checkpoint (epoch {})".format(trainer.best_epoch))
print(f"{'='*60}")
test_result = trainer.final_test(test_loader)
mr = test_result['metrics_regsign']
mc = test_result['metrics_cls']
a = test_result['awaf_stats']

print(f"\n  === Test Results (seed={CONFIG['seed']}, epochs={CONFIG['epochs']}) ===")
print(f"  Best epoch (by val MAE): {trainer.best_epoch}")
print(f"  --- reg_sign metrics ---")
print(f"  ACC2_Non0: {mr['ACC2_Non0']:.2f}%")
print(f"  F1_Non0:   {mr['F1_Non0']:.2f}%")
print(f"  ACC2_Has0: {mr['ACC2_Has0']:.2f}%")
print(f"  MAE:       {mr['MAE']:.4f}")
print(f"  Corr:      {mr['Corr']:.4f}")
print(f"  ACC7:      {mr['ACC7']:.2f}%")
print(f"  --- cls metrics ---")
print(f"  ACC2_Non0: {mc['ACC2_Non0']:.2f}%")
print(f"  F1_Non0:   {mc['F1_Non0']:.2f}%")
print(f"  --- AWAF stats ---")
print(f"  w_t: {a['w_t_mean']:.4f}±{a['w_t_std']:.4f}")
print(f"  w_a: {a['w_a_mean']:.4f}±{a['w_a_std']:.4f}")
print(f"  w_v: {a['w_v_mean']:.4f}±{a['w_v_std']:.4f}")
print(f"  sum(w) max_dev: {a['sum_w_max_dev']:.2e}")

# --- Save smoke outputs ---
out_dir = f"outputs/P5C_smoke/MOSI/deeptext_xlstm_awaf_residual/{time.strftime('%Y%m%d_%H%M%S')}_seed{CONFIG['seed']}"
os.makedirs(out_dir, exist_ok=True)
trainer.save_run(out_dir, CONFIG, ' '.join(sys.argv), CONFIG['epochs'], test_result)
print(f"\nOutputs saved to: {out_dir}")

# --- Judgment ---
acc2 = mr['ACC2_Non0']
print(f"\n{'='*60}")
if acc2 > 80.2:
    print(f"✅ SMOKE: ACC2_NZ={acc2:.2f}% > 80.2% (超过 DeepMLP text-only!)")
elif acc2 > 78.8:
    print(f"⚠️  SMOKE: ACC2_NZ={acc2:.2f}% > 78.8% (超过 P4W AWAF-Seq, 但低于 DeepMLP)")
else:
    print(f"❌ SMOKE: ACC2_NZ={acc2:.2f}% < 78.8% (低于 P4W AWAF-Seq)")

print(f"  Note: 3-epoch smoke — not final. Full 60-epoch training needed.")
print(f"{'='*60}")
