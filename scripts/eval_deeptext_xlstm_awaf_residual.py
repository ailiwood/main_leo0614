#!/usr/bin/env python
"""
scripts/eval_deeptext_xlstm_awaf_residual.py
P5C Evaluation — DeepText-xLSTM-AWAF Residual

Load a checkpoint and evaluate on MOSI test set.

Usage:
    python scripts/eval_deeptext_xlstm_awaf_residual.py --checkpoint CHECKPOINT_PATH [--device cuda]
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm

from models.deeptext_xlstm_awaf_residual import DeepTextXLSTMAWAFResidual
from data.strong_sequence_dataset import StrongSequenceMOSIDataset, collate_strong_sequence
from utils.metrics import compute_all_metrics
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepText-xLSTM-AWAF Residual')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--config', type=str, default='configs/models/deeptext_xlstm_awaf_residual_mosi.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    m_cfg = cfg['model']
    d_cfg = cfg['data']

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Best val MAE: {ckpt.get('best_val_mae', 'N/A')}")
    print(f"  Best epoch: {ckpt.get('best_epoch', 'N/A')}")
    if 'test_metrics' in ckpt:
        print(f"  Saved test metrics: {json.dumps(ckpt['test_metrics'], indent=2)}")

    # Build model
    print("\nBuilding model...")
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
        ablation=m_cfg.get('ablation', 'none'),
        use_aux_heads=m_cfg.get('use_aux_heads', False),
    ).to(DEVICE)

    # Load state dict
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Params: {model._init_info['total_params']:,}")

    # Load data
    print("\nLoading test data...")
    test_ds = StrongSequenceMOSIDataset('test', feature_root=d_cfg['feature_root'])
    print(f"  Test: {len(test_ds)} samples")

    max_tl = d_cfg.get('max_text_len', 50)
    max_al = d_cfg.get('max_audio_len', 100)
    max_vl = d_cfg.get('max_vision_len', 50)

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
        collate_fn=lambda b: collate_strong_sequence(b, max_text_len=max_tl, max_audio_len=max_al, max_vision_len=max_vl))

    # Evaluate
    print("\nEvaluating...")
    all_reg, all_cls, all_lbl, all_w, all_ids = [], [], [], [], []
    all_reg_base, all_delta_reg = [], []
    all_delta_scale_reg, all_delta_scale_cls = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Eval'):
            txt = batch['text'].to(DEVICE)
            aud = batch['audio'].to(DEVICE)
            vis = batch['vision'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            tm = batch['text_mask'].to(DEVICE) if batch.get('text_mask') is not None else None
            am = batch['audio_mask'].to(DEVICE) if batch.get('audio_mask') is not None else None
            vm = batch['vision_mask'].to(DEVICE) if batch.get('vision_mask') is not None else None

            out = model(txt, aud, vis, text_mask=tm, audio_mask=am, vision_mask=vm, return_all=True)

            all_reg.append(out['reg'].cpu())
            all_cls.append(out['cls'].cpu())
            all_lbl.append(lbl.cpu())
            all_w.append(out['awaf_weights'].cpu())
            all_ids.extend(batch.get('id', [''] * len(lbl)))

            if out.get('reg_text_base') is not None:
                all_reg_base.append(out['reg_text_base'].cpu())
            if out.get('delta_reg') is not None:
                all_delta_reg.append(out['delta_reg'].cpu())
            all_delta_scale_reg.append(out['delta_scale_reg'].item())
            all_delta_scale_cls.append(out['delta_scale_cls'].item())

    reg_preds = torch.cat(all_reg)
    cls_preds = torch.cat(all_cls)
    targets = torch.cat(all_lbl)
    awaf_w = torch.cat(all_w)

    # Metrics
    rs = torch.where(reg_preds >= 0, 1.0, -1.0)
    m_reg = compute_all_metrics(reg_preds, rs, targets)
    m_cls = compute_all_metrics(reg_preds, cls_preds, targets)

    print(f"\n=== Evaluation Results ===")
    print(f"--- reg_sign ---")
    print(f"ACC2_Non0: {m_reg['ACC2_Non0']:.2f}%  F1_Non0: {m_reg['F1_Non0']:.2f}%")
    print(f"ACC2_Has0: {m_reg['ACC2_Has0']:.2f}%  F1_Has0: {m_reg['F1_Has0']:.2f}%")
    print(f"MAE: {m_reg['MAE']:.4f}  Corr: {m_reg['Corr']:.4f}  ACC7: {m_reg['ACC7']:.2f}%")
    print(f"--- cls ---")
    print(f"ACC2_Non0: {m_cls['ACC2_Non0']:.2f}%  F1_Non0: {m_cls['F1_Non0']:.2f}%")
    print(f"--- AWAF ---")
    print(f"w_t: {awaf_w[:,0].mean():.4f}±{awaf_w[:,0].std():.4f}")
    print(f"w_a: {awaf_w[:,1].mean():.4f}±{awaf_w[:,1].std():.4f}")
    print(f"w_v: {awaf_w[:,2].mean():.4f}±{awaf_w[:,2].std():.4f}")
    print(f"sum(w) max_dev: {(awaf_w.sum(-1)-1).abs().max():.4e}")
    print(f"--- delta_scale ---")
    print(f"δ_reg: {np.mean(all_delta_scale_reg):.6f}, δ_cls: {np.mean(all_delta_scale_cls):.6f}")

    # Delta analysis
    if all_reg_base and all_delta_reg:
        reg_base = torch.cat(all_reg_base)
        delta = torch.cat(all_delta_reg)
        base_std = reg_base.std().item()
        delta_std = delta.std().item()
        print(f"--- delta analysis ---")
        print(f"reg_base std: {base_std:.4f}, delta_reg std: {delta_std:.4f}")
        print(f"delta/base ratio: {delta_std/base_std:.4f}" if base_std > 0 else "delta/base: N/A")

    # Save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        import csv
        from models.fusion.awaf import save_awaf_weights_csv
        # Predictions
        with open(os.path.join(args.output_dir, 'predictions_test.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'reg_pred', 'cls_logit', 'target'])
            for i in range(len(reg_preds)):
                w.writerow([all_ids[i] if i < len(all_ids) else i,
                           reg_preds[i].item(), cls_preds[i].item(), targets[i].item()])
        save_awaf_weights_csv(awaf_w, all_ids, os.path.join(args.output_dir, 'awaf_weights_test.csv'))
        # Metrics
        all_metrics = {
            'reg_sign': m_reg, 'cls': m_cls,
            'awaf_w_t_mean': float(awaf_w[:,0].mean()),
            'awaf_w_a_mean': float(awaf_w[:,1].mean()),
            'awaf_w_v_mean': float(awaf_w[:,2].mean()),
            'delta_scale_reg': float(np.mean(all_delta_scale_reg)),
            'delta_scale_cls': float(np.mean(all_delta_scale_cls)),
        }
        with open(os.path.join(args.output_dir, 'eval_metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
