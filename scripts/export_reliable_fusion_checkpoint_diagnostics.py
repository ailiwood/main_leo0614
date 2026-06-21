"""
P6AK-G0: Export Reliable Fusion Checkpoint Diagnostics

Re-evaluates frozen P6AJ checkpoints with proper Text-Anchored Reliable Fusion
output fields, counterfactuals, and consistency checks.
No training. No parameter updates. Read-only checkpoint inference.

Outputs per seed:
  predictions_test_reliable_fusion.csv (full diagnostic columns)
  counterfactual_summary.json
"""
import torch, sys, os, csv, json, argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.textft_lora_xlstm_awaf_residual import TextFTLoRAXLSTMAWAFResidual, TextFTLoRAConfig
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from torch.utils.data import DataLoader
from utils.metrics import compute_all_metrics

DEVICE = 'cuda'
SEEDS = [42, 2024, 3407]
CKPT_DIR = 'outputs/P6AJ_mosi_87_recovery'
OUTPUT_DIR = 'outputs/P6AK_mosi_final_ablation'


def load_model_and_checkpoint(seed):
    """Load frozen P6AJ model and its best checkpoint."""
    # Find best_model.pth
    ckpt_dirs = [
        os.path.join(CKPT_DIR, d) for d in os.listdir(CKPT_DIR)
        if f'P1_p6k_init_tav_s{seed}' in d and os.path.isdir(os.path.join(CKPT_DIR, d))
    ]
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoint found for seed {seed}")
    ckpt_dir = sorted(ckpt_dirs)[-1]
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No best_model.pth in {ckpt_dir}")

    config = TextFTLoRAConfig(
        mode='canonical_mosi_text_anchored_tav', hidden_dim=256,
        audio_input_dim=768, vision_input_dim=1024,
        slstm_num_layers=1, slstm_dropout=0.2, slstm_bidirectional=False,
        text_model_name='roberta-large', text_mlp_hidden=512, text_dropout=0.1,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        temporal_encoder='slstm', awaf_dropout=0.1,
        use_modality_dropout=True, modality_dropout_prob=0.1,
        use_modal_layernorm=True, freeze_text_base=False, device=DEVICE,
    )
    model = TextFTLoRAXLSTMAWAFResidual(config).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Get fusion params
    tf = model.text_anchored_fusion
    alpha_a = (tf.max_alpha * torch.sigmoid(tf.raw_alpha_a)).item()
    alpha_v = (tf.max_alpha * torch.sigmoid(tf.raw_alpha_v)).item()
    gate_bias_a = tf.gate_bias_a.item()
    gate_bias_v = tf.gate_bias_v.item()

    return model, ckpt_dir, {
        'alpha_a': alpha_a, 'alpha_v': alpha_v,
        'gate_bias_a': gate_bias_a, 'gate_bias_v': gate_bias_v,
    }


def evaluate_checkpoint(model, seed, out_dir):
    """Run full test evaluation with diagnostics and counterfactuals."""
    os.makedirs(out_dir, exist_ok=True)

    test_ds = TextFTMultimodalDataset(
        csv_path='data/mosi/label.csv',
        feature_root='data/processed/mosi_tav_v1',
        split='test', formal_mode=True,
    )
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_textft)

    # Collect predictions
    rows = []
    all_y_text, all_y_hat, all_r_a, all_r_v = [], [], [], []
    all_da, all_dv, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Seed {seed}'):
            bg = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(bg)

            y_text = out['y_text'].cpu().numpy().flatten()
            y_hat = out['reg'].cpu().numpy().flatten()
            r_a = out['r_a'].cpu().numpy().flatten()
            r_v = out['r_v'].cpu().numpy().flatten()
            delta_a = out['delta_a'].cpu().numpy().flatten()
            delta_v = out['delta_v'].cpu().numpy().flatten()
            labels = bg['label'].cpu().numpy().flatten()
            sample_ids = batch['id']

            # Counterfactual: zero audio
            bg_no_a = {k: v.clone().to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in bg.items()}
            bg_no_a['audio'] = torch.zeros_like(bg['audio'])
            bg_no_a['audio_mask'] = torch.zeros_like(bg['audio_mask'])
            out_no_a = model(bg_no_a)
            y_hat_no_a = out_no_a['reg'].cpu().numpy().flatten()

            # Counterfactual: zero vision
            bg_no_v = {k: v.clone().to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in bg.items()}
            bg_no_v['vision'] = torch.zeros_like(bg['vision'])
            bg_no_v['vision_mask'] = torch.zeros_like(bg['vision_mask'])
            out_no_v = model(bg_no_v)
            y_hat_no_v = out_no_v['reg'].cpu().numpy().flatten()

            # Counterfactual: permute audio
            bg_perm_a = {k: v.clone().to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in bg.items()}
            audio_seq = bg_perm_a['audio']
            audio_mask = bg_perm_a['audio_mask']
            for i in range(audio_seq.shape[0]):
                n_valid = int(audio_mask[i].sum().item())
                if n_valid > 1:
                    perm = torch.randperm(n_valid, device=DEVICE)
                    audio_seq[i, :n_valid] = audio_seq[i, perm]
            out_perm_a = model(bg_perm_a)
            y_hat_perm_a = out_perm_a['reg'].cpu().numpy().flatten()

            # Counterfactual: permute vision
            bg_perm_v = {k: v.clone().to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in bg.items()}
            vision_seq = bg_perm_v['vision']
            vision_mask = bg_perm_v['vision_mask']
            for i in range(vision_seq.shape[0]):
                n_valid = int(vision_mask[i].sum().item())
                if n_valid > 1:
                    perm = torch.randperm(n_valid, device=DEVICE)
                    vision_seq[i, :n_valid] = vision_seq[i, perm]
            out_perm_v = model(bg_perm_v)
            y_hat_perm_v = out_perm_v['reg'].cpu().numpy().flatten()

            # Fusion params
            tf = model.text_anchored_fusion
            alpha_a_val = (tf.max_alpha * torch.sigmoid(tf.raw_alpha_a)).item()
            alpha_v_val = (tf.max_alpha * torch.sigmoid(tf.raw_alpha_v)).item()

            for i in range(len(sample_ids)):
                contrib_a = alpha_a_val * r_a[i] * delta_a[i]
                contrib_v = alpha_v_val * r_v[i] * delta_v[i]
                tb_correct = int((y_text[i] >= 0) == (labels[i] >= 0))
                fn_correct = int((y_hat[i] >= 0) == (labels[i] >= 0))
                rows.append({
                    'sample_id': sample_ids[i],
                    'label': float(labels[i]),
                    'y_text': float(y_text[i]),
                    'delta_a': float(delta_a[i]),
                    'delta_v': float(delta_v[i]),
                    'r_a': float(r_a[i]),
                    'r_v': float(r_v[i]),
                    'alpha_a': float(alpha_a_val),
                    'alpha_v': float(alpha_v_val),
                    'contribution_a': float(contrib_a),
                    'contribution_v': float(contrib_v),
                    'y_hat': float(y_hat[i]),
                    'audio_zero_y_hat': float(y_hat_no_a[i]),
                    'vision_zero_y_hat': float(y_hat_no_v[i]),
                    'audio_permuted_y_hat': float(y_hat_perm_a[i]),
                    'vision_permuted_y_hat': float(y_hat_perm_v[i]),
                    'text_base_correct': int(tb_correct),
                    'final_correct': int(fn_correct),
                })

            all_y_text.append(y_text); all_y_hat.append(y_hat)
            all_r_a.append(r_a); all_r_v.append(r_v)
            all_da.append(delta_a); all_dv.append(delta_v)
            all_labels.append(labels)

    # Concatenate
    y_text = np.concatenate(all_y_text)
    y_hat = np.concatenate(all_y_hat)
    r_a = np.concatenate(all_r_a)
    r_v = np.concatenate(all_r_v)
    da = np.concatenate(all_da)
    dv = np.concatenate(all_dv)
    labels = np.concatenate(all_labels)

    # Compute metrics
    signs = np.where(y_hat >= 0, 1.0, -1.0)
    m = compute_all_metrics(torch.from_numpy(y_hat), torch.from_numpy(signs), torch.from_numpy(labels))
    mask = labels != 0
    yt_acc = (np.where(y_text[mask] >= 0, 1.0, -1.0) == np.where(labels[mask] >= 0, 1.0, -1.0)).mean() * 100

    # Counterfactual summaries
    all_y_hat_no_a = np.array([r['audio_zero_y_hat'] for r in rows])
    all_y_hat_no_v = np.array([r['vision_zero_y_hat'] for r in rows])
    all_y_hat_perm_a = np.array([r['audio_permuted_y_hat'] for r in rows])
    all_y_hat_perm_v = np.array([r['vision_permuted_y_hat'] for r in rows])
    audio_zero_change = float(np.mean(np.abs(y_hat - all_y_hat_no_a)))
    vision_zero_change = float(np.mean(np.abs(y_hat - all_y_hat_no_v)))
    audio_perm_change = float(np.mean(np.abs(y_hat - all_y_hat_perm_a)))
    vision_perm_change = float(np.mean(np.abs(y_hat - all_y_hat_perm_v)))

    alpha_a_vals = np.array([r['alpha_a'] for r in rows])
    alpha_v_vals = np.array([r['alpha_v'] for r in rows])

    # Save predictions CSV
    csv_path = os.path.join(out_dir, 'predictions_test_reliable_fusion.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Save counterfactual summary
    summary = {
        'seed': seed,
        'ACC2_Non0': m['ACC2_Non0'],
        'F1_Non0': m['F1_Non0'],
        'MAE': m['MAE'],
        'Corr': m['Corr'],
        'ACC7': m['ACC7'],
        'y_text_ACC2': yt_acc,
        'acc2_delta_full_vs_text': m['ACC2_Non0'] - yt_acc,
        'r_a_mean': float(np.mean(r_a)), 'r_a_std': float(np.std(r_a)),
        'r_v_mean': float(np.mean(r_v)), 'r_v_std': float(np.std(r_v)),
        'delta_a_abs_mean': float(np.mean(np.abs(da))),
        'delta_v_abs_mean': float(np.mean(np.abs(dv))),
        'contribution_a_abs_mean': float(np.mean(np.abs(alpha_a_vals * r_a * da))),
        'contribution_v_abs_mean': float(np.mean(np.abs(alpha_v_vals * r_v * dv))),
        'y_hat_minus_y_text_abs_mean': float(np.mean(np.abs(y_hat - y_text))),
        'audio_zero_change': float(audio_zero_change),
        'vision_zero_change': float(vision_zero_change),
        'audio_permute_change': float(audio_perm_change),
        'vision_permute_change': float(vision_perm_change),
        'alpha_a': float(alpha_a_vals[0]),
        'alpha_v': float(alpha_v_vals[0]),
        'n_samples': len(rows),
    }

    with open(os.path.join(out_dir, 'counterfactual_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Single seed (0=all)')
    args = parser.parse_args()

    seeds_to_run = SEEDS if args.seed == 0 else [args.seed]

    all_summaries = {}
    for seed in seeds_to_run:
        print(f"\n{'='*60}")
        print(f"Re-evaluating seed {seed}")
        print(f"{'='*60}")
        model, ckpt_dir, fusion_params = load_model_and_checkpoint(seed)
        print(f"  Checkpoint: {ckpt_dir}")
        print(f"  alpha_a={fusion_params['alpha_a']:.4f}, alpha_v={fusion_params['alpha_v']:.4f}")
        print(f"  gate_bias_a={fusion_params['gate_bias_a']:.4f}, gate_bias_v={fusion_params['gate_bias_v']:.4f}")

        out_dir = os.path.join(OUTPUT_DIR, f're_evaluated_s{seed}')
        summary = evaluate_checkpoint(model, seed, out_dir)
        all_summaries[str(seed)] = summary

        print(f"  ACC2={summary['ACC2_Non0']:.2f}%, y_text={summary['y_text_ACC2']:.2f}%")
        print(f"  r_a={summary['r_a_mean']:.4f}±{summary['r_a_std']:.4f}")
        print(f"  r_v={summary['r_v_mean']:.4f}±{summary['r_v_std']:.4f}")
        print(f"  audio_zero_change={summary['audio_zero_change']:.6f}")
        print(f"  vision_zero_change={summary['vision_zero_change']:.6f}")
        print(f"  Predictions saved: {out_dir}")

    # Consistency with result.json
    print(f"\n{'='*60}")
    print("Consistency Check: Re-evaluated vs result.json")
    print(f"{'='*60}")
    all_ok = True
    for seed in seeds_to_run:
        s = str(seed)
        summary = all_summaries[s]
        # Find original result.json
        ckpt_dirs = [d for d in os.listdir(CKPT_DIR) if f'P1_p6k_init_tav_s{seed}' in d]
        ckpt_dir = sorted(ckpt_dirs)[-1]
        result_path = os.path.join(CKPT_DIR, ckpt_dir, 'result.json')
        with open(result_path) as f:
            original = json.load(f)
        orig_acc2 = original['final_ACC2']
        reval_acc2 = summary['ACC2_Non0']
        diff = abs(orig_acc2 - reval_acc2)
        status = 'PASS' if diff < 0.1 else 'FAIL'
        if diff >= 0.1:
            all_ok = False
        print(f"  Seed {seed}: orig={orig_acc2:.4f}% re-eval={reval_acc2:.4f}% diff={diff:.4f}pp [{status}]")

    if all_ok:
        print(f"\n  ARTIFACT STATUS: consistent")
        print(f"  P6AJ checkpoints are verified. Proceed to G1 ablation.")
    else:
        print(f"\n  ARTIFACT STATUS: inconsistent")
        print(f"  STOP: Fix checkpoint loading or metrics before ablation.")

    # Save overall summary
    summary_path = os.path.join(OUTPUT_DIR, 'G0_counterfactual_summary_all.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[list(all_summaries.keys())[0]].keys()))
        writer.writeheader()
        for s in all_summaries:
            writer.writerow(all_summaries[s])
    print(f"\nSummary: {summary_path}")


if __name__ == '__main__':
    main()
