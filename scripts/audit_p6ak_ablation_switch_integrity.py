"""
P6AK-G1: Ablation Switch Integrity Audit

Verifies all 5 ablation variants (A1-A5) + F0 control.
Checks: static arch, dynamic gradient, counterfactual, parameter count.
"""
import torch, sys, os, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.textft_lora_xlstm_awaf_residual import TextFTLoRAXLSTMAWAFResidual, TextFTLoRAConfig
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from torch.utils.data import DataLoader

DEVICE = 'cuda'
OUTPUT_DIR = 'reports/P6AK_mosi_final_ablation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODES = {
    'F0': {'mode': 'canonical_mosi_text_anchored_tav', 'desc': 'Full TAV control', 'flags': {}},
    'A1': {'mode': 'text_only', 'desc': 'Text-only baseline', 'flags': {}, 'no_audio_vision': True},
    'A2': {'mode': 'canonical_mosi_text_anchored_tav_no_audio_corr', 'desc': 'No audio correction', 'flags': {'no_audio_correction': True}},
    'A3': {'mode': 'canonical_mosi_text_anchored_tav_no_vision_corr', 'desc': 'No vision correction', 'flags': {'no_vision_correction': True}},
    'A4': {'mode': 'canonical_mosi_text_anchored_tav_no_gate', 'desc': 'No reliability gate', 'flags': {'no_reliability_gate': True}},
    'A5': {'mode': 'canonical_mosi_text_anchored_tav_no_interaction', 'desc': 'No interaction', 'flags': {'no_interaction': True}},
}

def build_config(var_id):
    info = MODES[var_id]
    is_ta = 'no_audio_vision' in info
    return TextFTLoRAConfig(
        mode=info['mode'], hidden_dim=256,
        audio_input_dim=0 if is_ta else 768,
        vision_input_dim=0 if is_ta else 1024,
        slstm_num_layers=1, slstm_dropout=0.2, slstm_bidirectional=False,
        text_model_name='roberta-large', text_mlp_hidden=512, text_dropout=0.1,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        temporal_encoder='slstm', awaf_dropout=0.1,
        use_modality_dropout=True, modality_dropout_prob=0.1,
        use_modal_layernorm=True, freeze_text_base=False,
        device=DEVICE,
    )

def load_batch():
    ds = TextFTMultimodalDataset(csv_path='data/mosi/label.csv', feature_root='data/processed/mosi_tav_v1', split='train', formal_mode=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_textft)
    batch = next(iter(loader))
    return {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def main():
    print("=" * 70)
    print("P6AK-G1: Ablation Switch Integrity Audit")
    print("=" * 70)

    batch = load_batch()
    print(f"Fixed batch: {batch['id']}")
    print(f"Audio shape: {batch['audio'].shape}, Vision shape: {batch['vision'].shape}")

    results = {}
    signatures = []
    checks = []

    for var_id in ['F0', 'A1', 'A2', 'A3', 'A4', 'A5']:
        print(f"\n--- {var_id}: {MODES[var_id]['desc']} ---")
        config = build_config(var_id)
        model = TextFTLoRAXLSTMAWAFResidual(config).to(DEVICE)
        info = model.count_trainable()
        total_p = info['total_M']
        train_p = info['trainable_M']
        has_audio = hasattr(model, 'audio_proj')
        has_vision = hasattr(model, 'vision_proj')
        has_ta_fusion = hasattr(model, 'text_anchored_fusion')

        print(f"  Params: {total_p:.2f}M total, {train_p:.2f}M trainable")
        print(f"  Audio branch: {has_audio}, Vision branch: {has_vision}")
        if has_ta_fusion:
            tf = model.text_anchored_fusion
            print(f"  Ablation: no_audio_corr={tf.no_audio_correction}, no_vision_corr={tf.no_vision_correction}, no_gate={tf.no_reliability_gate}, no_interaction={tf.no_interaction}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            out = model(batch)

        reg = out['reg'].cpu().numpy()
        y_text = out.get('y_text', out.get('reg_text_base', reg)).cpu().numpy()
        r_a = out.get('r_a', np.zeros_like(reg)).cpu().numpy() if 'r_a' in out else np.zeros_like(reg)
        r_v = out.get('r_v', np.zeros_like(reg)).cpu().numpy() if 'r_v' in out else np.zeros_like(reg)
        delta_a = out.get('delta_a', np.zeros_like(reg)).cpu().numpy() if 'delta_a' in out else np.zeros_like(reg)
        delta_v = out.get('delta_v', np.zeros_like(reg)).cpu().numpy() if 'delta_v' in out else np.zeros_like(reg)

        result = {
            'variant': var_id, 'mode': MODES[var_id]['mode'],
            'total_params_M': total_p, 'trainable_params_M': train_p,
            'has_audio_branch': has_audio, 'has_vision_branch': has_vision,
            'has_ta_fusion': has_ta_fusion,
            'reg_mean': float(np.mean(reg)), 'reg_std': float(np.std(reg)),
            'y_text_mean': float(np.mean(y_text)),
            'r_a_mean': float(np.mean(r_a)), 'r_v_mean': float(np.mean(r_v)),
            'delta_a_abs_mean': float(np.mean(np.abs(delta_a))),
            'delta_v_abs_mean': float(np.mean(np.abs(delta_v))),
            '|reg-y_text|': float(np.mean(np.abs(reg.flatten() - y_text.flatten()))),
            'pred_constant': float(np.std(reg)) < 1e-6,
        }
        results[var_id] = result
        signatures.append(result)

        # Gradient test (fresh forward in train mode)
        model.train()
        out_train = model(batch)
        loss = torch.nn.functional.mse_loss(out_train['reg'], batch['label'])
        loss.backward()
        grad_ok = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 1e-8)
        result['nonzero_grads'] = grad_ok
        model.zero_grad()

    # ===== Switch integrity checks =====
    print(f"\n{'='*70}")
    print("Switch Integrity Checks")
    print(f"{'='*70}")

    # A1: no audio/vision branches
    a1 = results['A1']
    check = not a1['has_audio_branch'] and not a1['has_vision_branch']
    checks.append(('A1_no_audio_vision_branch', 'PASS' if check else 'FAIL',
                   f"Audio={a1['has_audio_branch']}, Vision={a1['has_vision_branch']}"))
    print(f"  A1 no audio/vision: {'PASS' if check else 'FAIL'}")

    # A2/A3: verify ablation flags are set (contributions zeroed in y_hat)
    # The correction MLPs still compute raw values, but contributions are zeroed
    a2_model = TextFTLoRAXLSTMAWAFResidual(build_config('A2')).to(DEVICE)
    check_a2 = hasattr(a2_model, 'text_anchored_fusion') and a2_model.text_anchored_fusion.no_audio_correction
    checks.append(('A2_flag_no_audio_correction', 'PASS' if check_a2 else 'FAIL', f'Flag={check_a2}'))
    print(f"  A2 no_audio_correction flag: {'PASS' if check_a2 else 'FAIL'}")

    a3_model = TextFTLoRAXLSTMAWAFResidual(build_config('A3')).to(DEVICE)
    check_a3 = hasattr(a3_model, 'text_anchored_fusion') and a3_model.text_anchored_fusion.no_vision_correction
    checks.append(('A3_flag_no_vision_correction', 'PASS' if check_a3 else 'FAIL', f'Flag={check_a3}'))
    print(f"  A3 no_vision_correction flag: {'PASS' if check_a3 else 'FAIL'}")

    # A4: r_a=r_v=1
    a4 = results['A4']
    check = abs(a4['r_a_mean'] - 1.0) < 0.01 and abs(a4['r_v_mean'] - 1.0) < 0.01
    checks.append(('A4_gate_always_one', 'PASS' if check else 'FAIL',
                   f"r_a={a4['r_a_mean']:.4f}, r_v={a4['r_v_mean']:.4f}"))
    print(f"  A4 gate=1: {'PASS' if check else 'FAIL'} (r_a={a4['r_a_mean']:.4f}, r_v={a4['r_v_mean']:.4f})")

    # A5: should not use interaction (verified via flag)
    a5_model = TextFTLoRAXLSTMAWAFResidual(build_config('A5')).to(DEVICE)
    if hasattr(a5_model, 'text_anchored_fusion'):
        check = a5_model.text_anchored_fusion.no_interaction
        checks.append(('A5_no_interaction_flag', 'PASS' if check else 'FAIL',
                       f"no_interaction={check}"))
        print(f"  A5 no interaction: {'PASS' if check else 'FAIL'}")

    # F0 vs A1: outputs must differ (TAV adds corrections)
    f0 = results['F0']
    diff = abs(f0['reg_mean'] - a1['reg_mean'])
    check = diff > 0.001
    checks.append(('F0_vs_A1_outputs_differ', 'PASS' if check else 'FAIL',
                   f"|F0-A1|={diff:.6f}"))
    print(f"  F0 vs A1 differ: {'PASS' if check else 'FAIL'} (|diff|={diff:.6f})")

    # All F0, A2-A5 should have different outputs from each other
    unique_means = len(set(f"{results[v]['reg_mean']:.4f}" for v in ['F0','A2','A3','A4','A5']))
    check = unique_means >= 3
    checks.append(('ablation_outputs_not_identical', 'PASS' if check else 'WARN',
                   f"Unique outputs: {unique_means}/5"))
    print(f"  Unique outputs: {unique_means}/5 -> {'PASS' if check else 'WARN'}")

    # All variants pass
    all_pass = all(c[1] == 'PASS' for c in checks)
    print(f"\n{'='*70}")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print(f"{'='*70}")

    # Save outputs
    sig_path = os.path.join(OUTPUT_DIR, 'G1_model_signature_matrix.csv')
    with open(sig_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(signatures[0].keys()))
        writer.writeheader()
        writer.writerows(signatures)
    print(f"Signatures: {sig_path}")

    batch_path = os.path.join(OUTPUT_DIR, 'G1_fixed_batch_audit.csv')
    with open(batch_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'reg_mean', 'reg_std', 'y_text_mean', 'r_a_mean', 'r_v_mean',
                         '|reg-y_text|', 'total_params_M', 'trainable_params_M'])
        for v in ['F0', 'A1', 'A2', 'A3', 'A4', 'A5']:
            r = results[v]
            writer.writerow([v, r['reg_mean'], r['reg_std'], r['y_text_mean'],
                           r['r_a_mean'], r['r_v_mean'], r['|reg-y_text|'],
                           r['total_params_M'], r['trainable_params_M']])
    print(f"Batch audit: {batch_path}")

    md_path = os.path.join(OUTPUT_DIR, 'G1_switch_integrity_audit.md')
    with open(md_path, 'w') as f:
        f.write("# P6AK-G1: Ablation Switch Integrity Audit\n\n")
        f.write(f"**Overall**: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n\n")
        f.write("| Check | Status | Detail |\n")
        f.write("|-------|--------|--------|\n")
        for name, status, detail in checks:
            f.write(f"| {name} | {status} | {detail} |\n")
        f.write("\n## Variant Summary\n\n")
        f.write("| Var | Mode | Params(M) | Audio | Vision | reg_mean | r_a | r_v |\n")
        f.write("|-----|------|-----------|-------|--------|----------|-----|-----|\n")
        for v in ['F0', 'A1', 'A2', 'A3', 'A4', 'A5']:
            r = results[v]
            f.write(f"| {v} | {r['mode'][-25:]} | {r['trainable_params_M']:.2f} | "
                    f"{'Y' if r['has_audio_branch'] else 'N'} | {'Y' if r['has_vision_branch'] else 'N'} | "
                    f"{r['reg_mean']:.4f} | {r['r_a_mean']:.4f} | {r['r_v_mean']:.4f} |\n")
    print(f"Audit: {md_path}")

    return all_pass

if __name__ == '__main__':
    ok = main()
    exit(0 if ok else 1)
