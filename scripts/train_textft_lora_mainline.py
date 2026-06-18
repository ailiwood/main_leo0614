#!/usr/bin/env python
"""
scripts/train_textft_lora_mainline.py — P6H-R TextFT LoRA AWAF 训练主脚本

支持:
  - YAML 配置文件
  - per-epoch metrics CSV
  - per-sample 预测 / AWAF 权重 / delta / gate 导出
  - 训练曲线、混淆矩阵、权重分布图
  - val-only 模式 (sweep 阶段不碰 test)
  - AWAF entropy regularization

用法:
  python scripts/train_textft_lora_mainline.py --config configs/experiments/p6h_repair/mosi_r1_norm_tau_dsr_gate.yaml
"""
import sys, os, json, time, csv, argparse, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from models.textft_lora_xlstm_awaf_residual import TextFTLoRAConfig, TextFTLoRAXLSTMAWAFResidual
from utils.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True, help='YAML config path')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def load_config_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_config(yc, device='cuda'):
    """从 YAML dict 构建 TextFTLoRAConfig。"""
    m = yc.get('model', yc)
    t = yc.get('training', {})
    return TextFTLoRAConfig(
        text_model_name=m.get('text_model_name', 'roberta-large'),
        hidden_dim=m.get('hidden_dim', 256),
        audio_input_dim=m.get('audio_dim', 768),
        vision_input_dim=m.get('vision_dim', 768),
        slstm_num_layers=m.get('slstm_num_layers', 1),
        slstm_dropout=m.get('slstm_dropout', 0.2),
        text_mlp_hidden=m.get('text_mlp_hidden', 512),
        text_dropout=m.get('text_dropout', 0.1),
        lora_r=m.get('lora_r', 16),
        lora_alpha=m.get('lora_alpha', 32),
        lora_dropout=m.get('lora_dropout', 0.05),
        lora_targets=tuple(m.get('lora_targets', ['query', 'value'])),
        awaf_fusion_mode=m.get('awaf_fusion_mode', 'awaf'),
        tau_init=m.get('tau_init', 3.0),
        awaf_dropout=m.get('awaf_dropout', 0.1),
        use_modality_dropout=m.get('use_modality_dropout', True),
        modality_dropout_prob=m.get('modality_dropout_prob', 0.1),
        use_modal_layernorm=m.get('use_modal_layernorm', True),
        awaf_uniform_mix=m.get('awaf_uniform_mix', 0.0),
        lambda_awaf_entropy=m.get('lambda_awaf_entropy', 0.0),
        use_uncertainty_gate=m.get('use_uncertainty_gate', True),
        gate_init_bias=m.get('gate_init_bias', 2.0),
        max_delta=m.get('max_delta', 0.5),
        delta_scale_init=m.get('delta_scale_init', 0.2),
        device=device,
    )


# ================================================================
# Group label
# ================================================================
def assign_group(label_val: float) -> str:
    """按 label 值分组: strong_neg / weak_neg / near_zero / weak_pos / strong_pos"""
    if label_val <= -1.5:
        return 'strong_neg'
    elif label_val <= -0.3:
        return 'weak_neg'
    elif label_val < 0.3:
        return 'near_zero'
    elif label_val < 1.5:
        return 'weak_pos'
    else:
        return 'strong_pos'


# ================================================================
# Save helpers
# ================================================================
def save_predictions_csv(filepath, sample_ids, labels, rtb, final, delta, gate, weights):
    """保存 per-sample 预测 CSV。"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'label', 'text_base_pred', 'final_pred', 'delta',
                     'gate', 'w_t', 'w_a', 'w_v', 'text_base_correct', 'final_correct', 'group'])
        for i in range(len(sample_ids)):
            tb_correct = int((rtb[i] >= 0) == (labels[i] >= 0))
            fn_correct = int((final[i] >= 0) == (labels[i] >= 0))
            group = assign_group(float(labels[i]))
            w.writerow([
                sample_ids[i], float(labels[i]),
                float(rtb[i]), float(final[i]), float(delta[i]), float(gate[i]),
                float(weights[i][0]), float(weights[i][1]), float(weights[i][2]),
                tb_correct, fn_correct, group,
            ])


def save_group_error_csv(filepath, sample_ids, labels, rtb, final, weights):
    """保存分组错误分析 CSV。"""
    groups = {'strong_neg': [], 'weak_neg': [], 'near_zero': [], 'weak_pos': [], 'strong_pos': []}
    for i in range(len(sample_ids)):
        g = assign_group(float(labels[i]))
        groups[g].append({
            'sample_id': sample_ids[i],
            'label': float(labels[i]),
            'text_base_pred': float(rtb[i]),
            'final_pred': float(final[i]),
            'abs_error_tb': abs(float(rtb[i]) - float(labels[i])),
            'abs_error_final': abs(float(final[i]) - float(labels[i])),
            'tb_correct': int((rtb[i] >= 0) == (labels[i] >= 0)),
            'fn_correct': int((final[i] >= 0) == (labels[i] >= 0)),
            'w_t': float(weights[i][0]), 'w_a': float(weights[i][1]), 'w_v': float(weights[i][2]),
        })
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['group', 'n_samples', 'tb_acc', 'fn_acc', 'mae_tb', 'mae_fn',
                     'mean_w_t', 'mean_w_a', 'mean_w_v'])
        for gname, items in groups.items():
            n = len(items)
            if n == 0:
                w.writerow([gname, 0, '', '', '', '', '', '', ''])
                continue
            tb_acc = sum(it['tb_correct'] for it in items) / n * 100
            fn_acc = sum(it['fn_correct'] for it in items) / n * 100
            mae_tb = sum(it['abs_error_tb'] for it in items) / n
            mae_fn = sum(it['abs_error_final'] for it in items) / n
            mwt = sum(it['w_t'] for it in items) / n
            mwa = sum(it['w_a'] for it in items) / n
            mwv = sum(it['w_v'] for it in items) / n
            w.writerow([gname, n, f'{tb_acc:.2f}', f'{fn_acc:.2f}',
                         f'{mae_tb:.4f}', f'{mae_fn:.4f}',
                         f'{mwt:.4f}', f'{mwa:.4f}', f'{mwv:.4f}'])


def save_plots(out_dir, epochs_list, metrics_dict, rtb, final, labels, weights, gates, deltas):
    """生成训练曲线 + 分布图 (需要 matplotlib)。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[WARN] matplotlib 不可用，跳过绘图')
        return

    # 1. 训练曲线
    if epochs_list and metrics_dict:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax = axes[0, 0]
        ax.plot(epochs_list, metrics_dict.get('val_ACC2', []), 'b-o', label='val_ACC2', markersize=4)
        ax.axhline(y=max(metrics_dict.get('val_ACC2', [0])), color='b', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch'); ax.set_ylabel('ACC2 (%)'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title('Validation ACC2')

        ax = axes[0, 1]
        ax.plot(epochs_list, metrics_dict.get('train_loss', []), 'r-s', label='train_loss', markersize=4)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title('Training Loss')

        ax = axes[1, 0]
        ax.plot(epochs_list, metrics_dict.get('val_MAE', []), 'g-^', label='val_MAE', markersize=4)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title('Validation MAE')

        ax = axes[1, 1]
        ax.plot(epochs_list, metrics_dict.get('val_Corr', []), 'm-d', label='val_Corr', markersize=4)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Corr'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title('Validation Corr')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 2. AWAF 权重分布
    if weights is not None and len(weights) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, (ax, mod, col) in enumerate(zip(axes, ['Text', 'Audio', 'Vision'], ['blue', 'orange', 'green'])):
            ax.hist(weights[:, idx], bins=50, alpha=0.7, color=col, edgecolor='black')
            ax.axvline(x=weights[:, idx].mean(), color='red', linestyle='--', linewidth=2)
            ax.set_xlabel(f'w_{mod.lower()}'); ax.set_ylabel('Count')
            ax.set_title(f'AWAF {mod} Weight (mean={weights[:, idx].mean():.4f})')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_awaf_weight_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Gate 分布
    if gates is not None and len(gates) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(gates, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=gates.mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Gate Value'); ax.set_ylabel('Count')
        ax.set_title(f'Gate Distribution (mean={gates.mean():.4f}, std={gates.std():.4f})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_gate_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Delta 分布
    if deltas is not None and len(deltas) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(deltas, bins=50, alpha=0.7, color='brown', edgecolor='black')
        ax.axvline(x=deltas.mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Effective Delta'); ax.set_ylabel('Count')
        ax.set_title(f'Effective Delta Distribution (mean={deltas.mean():.4f}, std={deltas.std():.4f})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_delta_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Text vs Final scatter
    if rtb is not None and final is not None and labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, preds, title in zip(axes, [rtb, final], ['Text Base', 'Final']):
            ax.scatter(labels, preds, alpha=0.3, s=10)
            ax.plot([-3, 3], [-3, 3], 'r--', linewidth=1)
            ax.set_xlabel('Label'); ax.set_ylabel('Prediction')
            ax.set_title(f'{title} vs Label')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_text_vs_final_scatter.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 6. Confusion matrix (ACC2)
    if labels is not None and final is not None:
        from sklearn.metrics import confusion_matrix
        y_true_bin = (labels >= 0).astype(int)
        y_pred_bin = (final >= 0).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Neg', 'Pos']); ax.set_yticklabels(['Neg', 'Pos'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (ACC2)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mosi_confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f'[PLOTS] 6 张图已保存到 {out_dir}')


# ================================================================
# Main training loop
# ================================================================
def main():
    args = parse_args()
    yc = load_config_yaml(args.config)
    t = yc.get('training', {})
    out_cfg = yc.get('output', {})

    SEED = t.get('seed', 42)
    EPOCHS = t.get('epochs', 30)
    BATCH = t.get('batch_size', 4)
    ACCUM = t.get('grad_accum_steps', 4)
    LR = t.get('lr', 3e-5)
    LR_LORA = t.get('lr_text_lora', 3e-6)
    WEIGHT_DECAY = t.get('weight_decay', 0.03)
    PATIENCE = t.get('early_stopping_patience', 6)
    TEST_EVAL = t.get('test_eval', False)           # 是否在 best epoch 后 eval test
    TEST_FINAL_ONCE = t.get('test_final_once', True) # 是否训练结束后 test once
    VAL_ONLY = t.get('val_only', False)              # sweep 模式: 只看 val
    LAMBDA_ENTROPY = yc.get('model', {}).get('lambda_awaf_entropy', 0.0)

    DEVICE = args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ================================================================
    # Data
    # ================================================================
    print(f'[DATA] Loading MOSI (formal_mode=True)...')
    train_ds = TextFTMultimodalDataset(split='train', formal_mode=True)
    val_ds = TextFTMultimodalDataset(split='val', formal_mode=True)
    test_ds = TextFTMultimodalDataset(split='test', formal_mode=True)
    print(f'  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}')

    tl = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_textft)
    vl = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate_textft)
    tl_test = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_textft)

    # ================================================================
    # Model
    # ================================================================
    config = build_config(yc, DEVICE)
    print(f'[MODEL] Building TextFTLoRAXLSTMAWAFResidual...')
    print(f'  tau_init={config.tau_init}  delta_scale_init={config.delta_scale_init}')
    print(f'  gate_init_bias={config.gate_init_bias}  use_modal_layernorm={config.use_modal_layernorm}')
    print(f'  awaf_uniform_mix={config.awaf_uniform_mix}  lambda_entropy={config.lambda_awaf_entropy}')
    model = TextFTLoRAXLSTMAWAFResidual(config)
    model = model.to(DEVICE)
    print(f'  Params: {model.count_trainable()}')

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)

    # Optimizer: 分组 lr
    lora_params = [p for n, p in model.roberta.named_parameters() if 'lora' in n.lower() and p.requires_grad]
    other_params = [p for p in model.collect_trainable_params() if p not in set(lora_params)]
    param_groups = [
        {'params': lora_params, 'lr': LR_LORA},
        {'params': other_params, 'lr': LR},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device='cuda')

    # ================================================================
    # Output dir
    # ================================================================
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_root = out_cfg.get('root', 'outputs/P6H_repair')
    exp_name = out_cfg.get('exp_name', os.path.splitext(os.path.basename(args.config))[0])
    out_dir = os.path.join(out_root, f'{exp_name}_s{SEED}_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'[OUTPUT] {out_dir}')

    # ================================================================
    # Train
    # ================================================================
    metrics_epoch = {'epoch': [], 'train_loss': [], 'val_ACC2': [], 'val_MAE': [], 'val_Corr': [],
                      'val_F1': [], 'awaf_entropy': [], 'gate_mean': [], 'delta_abs_mean': []}
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        opt.zero_grad()
        pbar = tqdm(tl, desc=f'E{epoch:2d}', leave=False)
        for i, batch in enumerate(pbar):
            out = model(batch)
            lbl = batch['label'].to(DEVICE)

            # Loss: L1 regression + delta alignment
            lr_loss = F.l1_loss(out['reg'], lbl)

            # Delta alignment: effective_delta should match (label - text_base)
            td = lbl - out['reg_text_base'].detach()
            ld = F.smooth_l1_loss(out['effective_delta_reg'], td)

            loss = lr_loss + 0.2 * ld

            # [P6H-R] AWAF entropy regularization (鼓励权重均匀)
            if LAMBDA_ENTROPY > 0.0:
                w = out['awaf_weights']
                awaf_entropy = model.awaf.compute_entropy(w).mean()
                # 最大化 entropy → 最小化 -entropy
                loss = loss - LAMBDA_ENTROPY * awaf_entropy

            loss = loss / ACCUM
            scaler.scale(loss).backward()

            if (i + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.collect_trainable_params(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            total_loss += loss.item() * ACCUM
            pbar.set_postfix({'loss': f'{loss.item()*ACCUM:.4f}'})

        avg_loss = total_loss / len(tl)

        # --- Val ---
        model.eval()
        vp_list, vl_list = [], []
        with torch.no_grad():
            for batch in vl:
                out = model(batch)
                vp_list.append(out['reg'].cpu())
                vl_list.append(batch['label'].cpu())
        rp = torch.cat(vp_list)
        tg = torch.cat(vl_list)
        rs = torch.where(rp >= 0, 1.0, -1.0)
        m = compute_all_metrics(rp, rs, tg)

        # --- Diagnostics ---
        awaf_ent = 0.0
        gate_m = 0.0
        delta_abs_m = 0.0
        with torch.no_grad():
            batch0 = next(iter(vl))
            out0 = model({k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch0.items()})
            awaf_ent = model.awaf.compute_entropy(out0['awaf_weights']).mean().item()
            gate_m = out0['gate_reg'].mean().item()
            delta_abs_m = out0['effective_delta_reg'].abs().mean().item()

        # --- Log ---
        print(f'E{epoch:2d}: loss={avg_loss:.4f}  val_ACC2={m["ACC2_Non0"]:.2f}%  '
              f'val_MAE={m["MAE"]:.4f}  val_Corr={m["Corr"]:.4f}  '
              f'ent={awaf_ent:.4f}  gate={gate_m:.4f}  |δ|={delta_abs_m:.4f}')

        # --- Record ---
        metrics_epoch['epoch'].append(epoch)
        metrics_epoch['train_loss'].append(avg_loss)
        metrics_epoch['val_ACC2'].append(m['ACC2_Non0'])
        metrics_epoch['val_MAE'].append(m['MAE'])
        metrics_epoch['val_Corr'].append(m['Corr'])
        metrics_epoch['val_F1'].append(m['F1_Non0'])
        metrics_epoch['awaf_entropy'].append(awaf_ent)
        metrics_epoch['gate_mean'].append(gate_m)
        metrics_epoch['delta_abs_mean'].append(delta_abs_m)

        # --- Early stopping ---
        if m['ACC2_Non0'] > best_val_acc:
            best_val_acc = m['ACC2_Non0']
            best_epoch = epoch
            no_improve = 0
            # Save best
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
        else:
            no_improve += 1

        # Save per-epoch CSV
        with open(os.path.join(out_dir, 'metrics_epoch.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            keys = list(metrics_epoch.keys())
            w.writerow(keys)
            for i in range(len(metrics_epoch['epoch'])):
                w.writerow([metrics_epoch[k][i] for k in keys])

        if PATIENCE > 0 and no_improve >= PATIENCE:
            print(f'[EARLY STOP] No improvement for {PATIENCE} epochs. Best: epoch={best_epoch} ACC2={best_val_acc:.2f}%')
            break

    print(f'\n=== Training done: best_val_ACC2={best_val_acc:.2f}% at epoch {best_epoch} ===')

    # ================================================================
    # Test evaluation (only if not val_only)
    # ================================================================
    if VAL_ONLY:
        print('[SWEEP MODE] val_only=True, skipping test eval.')
        # Still save sweep summary
        sweep_summary = {
            'config': args.config,
            'seed': SEED,
            'epochs_run': epoch,
            'best_epoch': best_epoch,
            'best_val_ACC2': best_val_acc,
        }
        json.dump(sweep_summary, open(os.path.join(out_dir, 'sweep_summary.json'), 'w'))
        print(f'[DONE] Sweep summary saved to {out_dir}')
        return

    # Load best model for test
    if os.path.exists(os.path.join(out_dir, 'best_model.pth')):
        model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pth'), map_location=DEVICE))
        print(f'[TEST] Loaded best model (epoch {best_epoch})')

    model.eval()
    tp_list, tl_list, aw_list, gv_list, rb_list, delta_list = [], [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(tl_test, desc='Test'):
            out = model(batch)
            tp_list.append(out['reg'].cpu())
            tl_list.append(batch['label'].cpu())
            aw_list.append(out['awaf_weights'].cpu())
            gv_list.append(out['gate_reg'].cpu())
            rb_list.append(out['reg_text_base'].cpu())
            delta_list.append(out['effective_delta_reg'].cpu())

    rp = torch.cat(tp_list)
    tg = torch.cat(tl_list)
    rs = torch.where(rp >= 0, 1.0, -1.0)
    aw = torch.cat(aw_list)
    gv = torch.cat(gv_list)
    rb = torch.cat(rb_list)
    ed = torch.cat(delta_list)

    # Compute metrics
    m_final = compute_all_metrics(rp, rs, tg)
    m_base = compute_all_metrics(rb, rs, tg)

    # ================================================================
    # Print & save results
    # ================================================================
    residual_gain = m_final['ACC2_Non0'] - m_base['ACC2_Non0']
    print(f'\n{"="*60}')
    print(f'P6H-R MOSI s{SEED} ({EPOCHS}ep) — Best epoch: {best_epoch}')
    print(f'  Text-base ACC2: {m_base["ACC2_Non0"]:.2f}%')
    print(f'  Final    ACC2: {m_final["ACC2_Non0"]:.2f}%')
    print(f'  Residual GAIN: {residual_gain:+.2f}%')
    print(f'  MAE: {m_final["MAE"]:.4f}  Corr: {m_final["Corr"]:.4f}  ACC7: {m_final["ACC7"]:.2f}%')
    print(f'  F1_Non0: {m_final["F1_Non0"]:.2f}%')
    print(f'  AWAF: w_t={aw[:,0].mean():.4f} w_a={aw[:,1].mean():.4f} w_v={aw[:,2].mean():.4f}')
    print(f'  Gate mean: {gv.mean().item():.4f}  |δ| mean: {ed.abs().mean().item():.4f}')
    print(f'  δ_scale_reg: {model.delta_scale_reg.item():.4f}')
    print(f'  AWAF τ: {model.awaf.tau.item():.4f}')
    print(f'  AWAF entropy: {model.awaf.compute_entropy(aw).mean().item():.4f}')
    print(f'{"="*60}')

    # Save results JSON
    result = {
        'config': args.config, 'seed': SEED, 'epochs': epoch, 'best_epoch': best_epoch,
        'best_val_ACC2': best_val_acc,
        'text_base_ACC2': m_base['ACC2_Non0'], 'final_ACC2': m_final['ACC2_Non0'],
        'residual_gain': residual_gain,
        'F1_Non0': m_final['F1_Non0'], 'MAE': m_final['MAE'], 'Corr': m_final['Corr'],
        'ACC7': m_final['ACC7'],
        'awaf_w_t': float(aw[:, 0].mean()), 'awaf_w_a': float(aw[:, 1].mean()),
        'awaf_w_v': float(aw[:, 2].mean()),
        'gate_mean': float(gv.mean()), 'gate_std': float(gv.std()),
        'delta_abs_mean': float(ed.abs().mean()), 'delta_std': float(ed.std()),
        'delta_scale_reg': float(model.delta_scale_reg.item()),
        'awaf_tau': float(model.awaf.tau.item()),
        'awaf_entropy': float(model.awaf.compute_entropy(aw).mean().item()),
        'trainable_M': model.count_trainable()['trainable_M'],
    }
    json.dump(result, open(os.path.join(out_dir, 'result.json'), 'w'), indent=2)

    # Save per-sample CSVs
    sample_ids = [f'sample_{i}' for i in range(len(tg))]
    rp_np = rp.numpy().flatten()
    tg_np = tg.numpy().flatten()
    rb_np = rb.numpy().flatten()
    gv_np = gv.numpy().flatten()
    ed_np = ed.numpy().flatten()
    aw_np = aw.numpy()

    save_predictions_csv(
        os.path.join(out_dir, 'predictions_test.csv'),
        sample_ids, tg_np, rb_np, rp_np, ed_np, gv_np, aw_np,
    )
    save_group_error_csv(
        os.path.join(out_dir, 'group_error_analysis.csv'),
        sample_ids, tg_np, rb_np, rp_np, aw_np,
    )

    # Save AWAF weights CSV
    with open(os.path.join(out_dir, 'awaf_weights_test.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'w_t', 'w_a', 'w_v'])
        for i in range(len(sample_ids)):
            w.writerow([sample_ids[i], float(aw_np[i, 0]), float(aw_np[i, 1]), float(aw_np[i, 2])])

    # Save text_base_delta CSV
    with open(os.path.join(out_dir, 'text_base_delta_test.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'label', 'text_base_pred', 'final_pred', 'delta',
                     'gate', 'abs_error_tb', 'abs_error_final'])
        for i in range(len(sample_ids)):
            w.writerow([sample_ids[i], float(tg_np[i]), float(rb_np[i]), float(rp_np[i]),
                         float(ed_np[i]), float(gv_np[i]),
                         abs(float(rb_np[i]) - float(tg_np[i])),
                         abs(float(rp_np[i]) - float(tg_np[i]))])

    # Save gate_delta_stats CSV
    with open(os.path.join(out_dir, 'gate_delta_stats.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stat', 'gate_reg', 'effective_delta_reg'])
        stats = [
            ('mean', gv_np.mean(), ed_np.mean()),
            ('std', gv_np.std(), ed_np.std()),
            ('min', gv_np.min(), ed_np.min()),
            ('max', gv_np.max(), ed_np.max()),
            ('p5', np.percentile(gv_np, 5), np.percentile(ed_np, 5)),
            ('p25', np.percentile(gv_np, 25), np.percentile(ed_np, 25)),
            ('p50', np.percentile(gv_np, 50), np.percentile(ed_np, 50)),
            ('p75', np.percentile(gv_np, 75), np.percentile(ed_np, 75)),
            ('p95', np.percentile(gv_np, 95), np.percentile(ed_np, 95)),
        ]
        for name, g, d in stats:
            w.writerow([name, f'{g:.6f}', f'{d:.6f}'])

    # Save plots
    save_plots(out_dir,
               metrics_epoch['epoch'],
               metrics_epoch,
               rb_np, rp_np, tg_np, aw_np, gv_np, ed_np)

    print(f'\n[DONE] All outputs saved to {out_dir}')


if __name__ == '__main__':
    main()
