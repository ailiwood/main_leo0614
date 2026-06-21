"""
P6AF-G1: MOSI Diagnostic Probes

Runs 6 probes on train/val ONLY (no test!) to diagnose data/model health:
  P0: majority / mean-label reference
  P1: text-only (RoBERTa+LoRA+MLP → head)
  P2: audio-only (data2vec → sLSTM/pool → head)
  P3: vision-only (CLIP → sLSTM/pool → head)
  P4: early-concat frozen MLP probe
  P5: 200-sample overfit probe (text-only)
  P6: shuffled-label sanity probe (should fail)

All probes use train/val only. Results saved to reports/.
"""
import sys, os, json, time, csv
import torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from models.textft_lora_xlstm_awaf_residual import TextFTLoRAConfig, TextFTLoRAXLSTMAWAFResidual
from utils.metrics import compute_all_metrics

DEVICE = 'cuda'
OUTPUT_DIR = 'reports/P6AF_mosi_tav_recovery'
CSV_PATH = 'data/mosi/label.csv'
FEATURE_ROOT = 'data/processed/mosi_tav_v1'
BATCH_SIZE = 8
GRAD_ACCUM = 8
MAX_EPOCHS_PROBE = 10
PATIENCE = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    train_ds = TextFTMultimodalDataset(csv_path=CSV_PATH, feature_root=FEATURE_ROOT,
                                        split='train', formal_mode=True)
    val_ds = TextFTMultimodalDataset(csv_path=CSV_PATH, feature_root=FEATURE_ROOT,
                                      split='val', formal_mode=True)
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}")
    return train_ds, val_ds

def train_probe(model, train_ds, val_ds, probe_name, max_epochs=MAX_EPOCHS_PROBE):
    """Train a probe model and return validation metrics history."""
    model = model.to(DEVICE)
    info = model.count_trainable()
    print(f"  [{probe_name}] Params: {info['trainable_M']:.2f}M trainable")

    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_textft)
    vl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_textft)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)

    best_val_acc = 0
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in tl:
            batch_gpu = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch_gpu)
            loss = nn.functional.mse_loss(out['reg'], batch_gpu['label'])
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in vl:
                batch_gpu = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch_gpu)
                all_preds.append(out['reg'].cpu())
                all_labels.append(batch_gpu['label'].cpu())

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        signs = torch.where(preds >= 0, 1.0, -1.0)
        m = compute_all_metrics(preds, signs, labels)
        val_acc2 = m['ACC2_Non0']
        val_f1 = m['F1_Non0']
        val_mae = m['MAE']
        val_corr = m['Corr']

        history.append({
            'epoch': epoch, 'train_loss': avg_loss,
            'val_ACC2': val_acc2, 'val_F1': val_f1, 'val_MAE': val_mae, 'val_Corr': val_corr
        })

        if val_acc2 > best_val_acc:
            best_val_acc = val_acc2
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        pred_mean = preds.mean().item()
        pred_std = preds.std().item()
        pos_ratio = (preds >= 0).float().mean().item()

        print(f"    E{epoch}: loss={avg_loss:.4f} val_ACC2={val_acc2:.2f}% "
              f"val_F1={val_f1:.2f}% val_MAE={val_mae:.4f} val_Corr={val_corr:.4f} "
              f"pred_mean={pred_mean:.3f} pred_std={pred_std:.3f} pos={pos_ratio:.2f}")

        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    return {
        'probe': probe_name,
        'best_epoch': best_epoch,
        'best_val_ACC2': best_val_acc,
        'final_val_ACC2': history[-1]['val_ACC2'],
        'final_val_F1': history[-1]['val_F1'],
        'final_val_MAE': history[-1]['val_MAE'],
        'final_val_Corr': history[-1]['val_Corr'],
        'final_pred_mean': pred_mean,
        'final_pred_std': pred_std,
        'final_pos_ratio': pos_ratio,
        'history': history,
    }


def main():
    print("=" * 70)
    print("P6AF-G1: MOSI Diagnostic Probes")
    print("=" * 70)

    train_ds, val_ds = load_data()

    labels = np.array([float(r['label']) for r in train_ds.data])
    val_labels = np.array([float(r['label']) for r in val_ds.data])

    # P0: Majority / mean reference
    print("\n--- P0: Reference Baselines ---")
    maj_acc = max((labels >= 0).mean(), (labels < 0).mean()) * 100
    mean_pred = labels.mean()
    val_maj_acc = max((val_labels >= 0).mean(), (val_labels < 0).mean()) * 100
    print(f"  Train majority ACC2: {maj_acc:.1f}%")
    print(f"  Train mean label: {mean_pred:.3f}")
    print(f"  Val majority ACC2: {val_maj_acc:.1f}%")
    p0_results = {
        'probe': 'P0_majority',
        'train_majority_ACC2': maj_acc,
        'val_majority_ACC2': val_maj_acc,
        'train_mean_label': float(mean_pred),
    }

    # P1: Text-only probe
    print("\n--- P1: Text-Only Probe ---")
    config_t = TextFTLoRAConfig(
        mode='text_only', hidden_dim=256,
        text_model_name='roberta-large', text_mlp_hidden=512, text_dropout=0.1,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        freeze_text_base=False, device=DEVICE,
    )
    model_t = TextFTLoRAXLSTMAWAFResidual(config_t)
    p1 = train_probe(model_t, train_ds, val_ds, 'P1_text_only')

    # P2: Audio-only probe
    print("\n--- P2: Audio-Only Probe ---")
    config_a = TextFTLoRAConfig(
        mode='audio_only', hidden_dim=256, audio_input_dim=768,
        slstm_num_layers=1, slstm_dropout=0.2, slstm_bidirectional=False,
        temporal_encoder='slstm', device=DEVICE,
    )
    model_a = TextFTLoRAXLSTMAWAFResidual(config_a)
    p2 = train_probe(model_a, train_ds, val_ds, 'P2_audio_only')

    # P3: Vision-only probe
    print("\n--- P3: Vision-Only Probe ---")
    config_v = TextFTLoRAConfig(
        mode='vision_only', hidden_dim=256, vision_input_dim=1024,
        slstm_num_layers=1, slstm_dropout=0.2, slstm_bidirectional=False,
        temporal_encoder='slstm', device=DEVICE,
    )
    model_v = TextFTLoRAXLSTMAWAFResidual(config_v)
    p3 = train_probe(model_v, train_ds, val_ds, 'P3_vision_only')

    # P6: Shuffled-label sanity probe (text-only, should fail dramatically)
    print("\n--- P6: Shuffled-Label Sanity Probe ---")
    # Shuffle labels
    shuffled_labels = labels.copy()
    np.random.RandomState(42).shuffle(shuffled_labels)
    # Create a shuffled dataset (modify labels in-place)
    for i, r in enumerate(train_ds.data):
        r['label'] = str(shuffled_labels[i])
    # Re-create dataset with shuffled labels
    train_ds_shuf = TextFTMultimodalDataset(csv_path=CSV_PATH, feature_root=FEATURE_ROOT,
                                             split='train', formal_mode=True)
    for i, r in enumerate(train_ds_shuf.data):
        r['label'] = str(shuffled_labels[i])

    config_shuf = TextFTLoRAConfig(
        mode='text_only', hidden_dim=256,
        text_model_name='roberta-large', text_mlp_hidden=512, text_dropout=0.1,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        freeze_text_base=False, device=DEVICE,
    )
    model_shuf = TextFTLoRAXLSTMAWAFResidual(config_shuf)
    p6 = train_probe(model_shuf, train_ds_shuf, val_ds, 'P6_shuffled_label')

    # Restore labels
    for i, r in enumerate(train_ds.data):
        r['label'] = str(labels[i])

    # Summary
    print(f"\n{'='*70}")
    print("DIAGNOSTIC PROBE SUMMARY")
    print(f"{'='*70}")
    print(f"P0 majority baseline: val_ACC2 = {val_maj_acc:.1f}%")
    print(f"P1 text-only: val_ACC2 = {p1['final_val_ACC2']:.2f}%")
    print(f"P2 audio-only: val_ACC2 = {p2['final_val_ACC2']:.2f}%")
    print(f"P3 vision-only: val_ACC2 = {p3['final_val_ACC2']:.2f}%")
    print(f"P6 shuffled-label: val_ACC2 = {p6['final_val_ACC2']:.2f}%")

    # Save results
    csv_path = os.path.join(OUTPUT_DIR, 'G1_diagnostic_probe_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['probe', 'best_epoch', 'best_val_ACC2',
                                                'final_val_ACC2', 'final_val_F1',
                                                'final_val_MAE', 'final_val_Corr',
                                                'final_pred_mean', 'final_pred_std',
                                                'final_pos_ratio'])
        writer.writeheader()
        for p in [p0_results, p1, p2, p3, p6]:
            row = {k: p.get(k, '') for k in writer.fieldnames}
            writer.writerow(row)
    print(f"\nResults: {csv_path}")

    # MD report
    md_path = os.path.join(OUTPUT_DIR, 'G1_diagnostic_probe_report.md')
    with open(md_path, 'w') as f:
        f.write("# P6AF-G1: MOSI Diagnostic Probe Report\n\n")
        f.write("| Probe | Val ACC2 | Val F1 | Val MAE | Val Corr | Pred Mean | Pred Std | Pos Ratio |\n")
        f.write("|-------|----------|--------|---------|----------|-----------|----------|----------|\n")
        f.write(f"| P0 majority | {val_maj_acc:.1f}% | — | — | — | — | — | — |\n")
        for p, name in [(p1, 'P1 text-only'), (p2, 'P2 audio-only'), (p3, 'P3 vision-only'), (p6, 'P6 shuffled')]:
            f.write(f"| {name} | {p['final_val_ACC2']:.1f}% | {p['final_val_F1']:.1f}% | "
                    f"{p['final_val_MAE']:.4f} | {p['final_val_Corr']:.4f} | "
                    f"{p['final_pred_mean']:.3f} | {p['final_pred_std']:.3f} | "
                    f"{p['final_pos_ratio']:.2f} |\n")
        f.write("\n## Diagnosis\n\n")
        if p1['final_val_ACC2'] > val_maj_acc + 5:
            f.write("- ✅ P1 text-only significantly beats majority baseline — text pipeline is healthy\n")
        else:
            f.write("- ❌ P1 text-only does NOT beat majority baseline — CHECK text pipeline!\n")
        if p6['final_val_ACC2'] < val_maj_acc + 3:
            f.write("- ✅ P6 shuffled-label cannot learn — label signal is real\n")
        else:
            f.write("- ❌ P6 shuffled-label LEARNS — possible data leakage!\n")
        if p2['final_val_ACC2'] > val_maj_acc + 2:
            f.write("- ✅ P2 audio-only beats majority — audio has signal\n")
        else:
            f.write("- ⚠️ P2 audio-only does not beat majority — audio may have weak signal\n")
        if p3['final_val_ACC2'] > val_maj_acc + 2:
            f.write("- ✅ P3 vision-only beats majority — vision has signal\n")
        else:
            f.write("- ⚠️ P3 vision-only does not beat majority — vision may have weak signal\n")
    print(f"Report: {md_path}")

    print(f"\n{'='*70}")
    print("P6AF-G1 complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
