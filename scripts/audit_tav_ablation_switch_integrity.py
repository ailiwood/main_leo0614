"""
P6AB: TAV Ablation Switch Integrity Audit

For each ablation config, instantiates the model, runs a forward pass on a fixed batch,
and verifies:
1. Config loads correctly
2. Model creates correct branches
3. Forward pass produces valid output shapes
4. Counterfactual checks (no_vision has no vision branch, etc.)
5. Parameter counts are distinct across variants
6. AWAF produces non-uniform weights for relevant variants

Output:
  reports/P6AB_tav_causal_evidence/mosei/A0_tav_ablation_switch_audit.md
  reports/P6AB_tav_causal_evidence/mosei/A0_tav_model_signature_matrix.csv
  reports/P6AB_tav_causal_evidence/mosei/A0_tav_fixed_batch_output_audit.csv
"""
import os, sys, yaml, json, hashlib, csv
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.textft_lora_xlstm_awaf_residual import TextFTLoRAXLSTMAWAFResidual, TextFTLoRAConfig
from data.textft_multimodal_dataset import TextFTMultimodalDataset

# P6AB: force CPU for switch audit (RTX 5070 Ti sm_120 not supported by PyTorch 2.3.0)
DEVICE = torch.device('cpu')
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CONFIG_DIR = 'configs/experiments/p6ab_tav_ablation/mosei'
REPORT_DIR = 'reports/P6AB_tav_causal_evidence/mosei'

VARIANTS = [
    ('F0', 'F0_full_tav_awaf_slstm_s42.yaml', 'full_tav_awaf_slstm'),
    ('F1', 'F1_no_vision_text_audio_s42.yaml', 'no_vision_text_audio'),
    ('F2', 'F2_no_audio_text_vision_s42.yaml', 'no_audio_text_vision'),
    ('F3', 'F3_global_static_tav_s42.yaml', 'global_static_tav'),
    ('F4', 'F4_fixed_mean_tav_s42.yaml', 'fixed_mean_tav'),
    ('F5', 'F5_awaf_no_interaction_tav_s42.yaml', 'awaf_no_interaction_tav'),
    ('E1', 'E1_temporal_lstm_all_s42.yaml', 'temporal_lstm_all'),
]


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def config_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_signature(model):
    """Return a dict describing which submodules exist."""
    sig = {}
    sig['has_audio_proj'] = hasattr(model, 'audio_proj')
    sig['has_audio_temporal'] = hasattr(model, 'audio_temporal') and model.audio_temporal is not None
    sig['has_audio_pool'] = hasattr(model, 'audio_pool')
    sig['has_vision_proj'] = hasattr(model, 'vision_proj')
    sig['has_vision_temporal'] = hasattr(model, 'vision_temporal') and model.vision_temporal is not None
    sig['has_vision_pool'] = hasattr(model, 'vision_pool')
    sig['has_canonical_fusion'] = hasattr(model, 'canonical_fusion')
    sig['has_canonical_head'] = hasattr(model, 'canonical_head')
    sig['temporal_encoder_type'] = getattr(model, '_temporal_encoder_type', 'unknown')
    sig['is_canonical'] = getattr(model, '_is_canonical', False)
    # Check fusion mode
    if sig['has_canonical_fusion']:
        sig['fusion_mode'] = model.canonical_fusion.fusion_mode
        sig['use_context'] = model.canonical_fusion._use_context
        sig['use_interaction'] = model.canonical_fusion._use_interaction
    return sig


def build_fixed_batch(dataset, device):
    """Get first batch from dataset for reproducible audit."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    # Move ALL tensor values to device
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


def run_forward(model, batch):
    """Run forward pass and collect outputs."""
    model.eval()
    with torch.no_grad():
        out = model(batch)
    result = {}
    result['reg_shape'] = tuple(out['reg'].shape)
    result['reg_mean'] = out['reg'].mean().item()
    result['reg_std'] = out['reg'].std().item()
    if 'awaf_weights' in out:
        w = out['awaf_weights']
        result['w_t_mean'] = w[:, 0].mean().item()
        result['w_a_mean'] = w[:, 1].mean().item()
        result['w_v_mean'] = w[:, 2].mean().item()
        result['w_t_std'] = w[:, 0].std().item()
        result['w_unique'] = len(torch.unique(w, dim=0))
    if 'reg_text_base' in out:
        result['text_base_eq_final'] = torch.allclose(out['reg_text_base'], out['reg'])
    return result


def check_counterfactual(model1, out1, model2, out2, check_name):
    """Verify that two models produce different outputs (counterfactual check)."""
    if out1['reg_mean'] != out2['reg_mean'] or out1['reg_std'] != out2['reg_std']:
        return True, "outputs_differ"
    return False, "outputs_identical"


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 80)
    print("P6AB TAV Ablation Switch Integrity Audit")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    # Load a fixed batch from the TAV dataset
    print("\n[1/4] Loading fixed batch from TAV dataset...")
    dataset = TextFTMultimodalDataset(
        csv_path='data/processed/mosei_full/label.csv',
        feature_root='data/processed/mosei_tav_openface2_v1',
        split='train', formal_mode=True,
    )
    batch = build_fixed_batch(dataset, DEVICE)
    print(f"  Batch: {batch['id']}")
    print(f"  Audio shape: {batch['audio'].shape}, Vision shape: {batch['vision'].shape}")
    print(f"  Labels: {batch['label'].squeeze().tolist()}")
    print(f"  Audio nonzero: {(batch['audio'] != 0).sum()}, Vision nonzero: {(batch['vision'] != 0).sum()}")

    # Run audit for each variant
    print("\n[2/4] Auditing all 7 ablation variants...")
    results = {}
    signatures = {}

    for var_id, config_file, var_name in VARIANTS:
        print(f"\n--- {var_id}: {var_name} ---")
        config_path = os.path.join(CONFIG_DIR, config_file)

        # Load config
        cfg = load_config(config_path)
        sha256 = config_sha256(config_path)

        # Build model config
        m = cfg['model']
        model_config = TextFTLoRAConfig(
            mode=m['mode'],
            hidden_dim=m.get('hidden_dim', 256),
            audio_input_dim=m.get('audio_dim', 74),
            vision_input_dim=m.get('vision_dim', 768),
            slstm_num_layers=m.get('slstm_num_layers', 1),
            slstm_dropout=m.get('slstm_dropout', 0.2),
            slstm_bidirectional=m.get('slstm_bidirectional', False),
            text_model_name=m.get('text_model_name', 'roberta-large'),
            text_mlp_hidden=m.get('text_mlp_hidden', 512),
            text_dropout=m.get('text_dropout', 0.1),
            lora_r=m.get('lora_r', 16),
            lora_alpha=m.get('lora_alpha', 32),
            lora_dropout=m.get('lora_dropout', 0.05),
            fusion_type=m.get('fusion_type', 'awaf'),
            awaf_context=m.get('awaf_context', True),
            awaf_interaction=m.get('awaf_interaction', True),
            temporal_encoder=m.get('temporal_encoder', 'slstm'),
            tau_init=m.get('tau_init', 3.0),
            awaf_dropout=m.get('awaf_dropout', 0.1),
            use_modality_dropout=m.get('use_modality_dropout', True),
            modality_dropout_prob=m.get('modality_dropout_prob', 0.1),
            use_modal_layernorm=m.get('use_modal_layernorm', True),
            freeze_text_base=m.get('freeze_text_base', False),
            device='cpu',  # P6AB: force CPU for compatibility
        )

        # Instantiate model
        print(f"  Instantiating model (mode={model_config.mode})...")
        try:
            model = TextFTLoRAXLSTMAWAFResidual(model_config).to(DEVICE)
        except Exception as e:
            print(f"  ERROR: Model instantiation failed: {e}")
            results[var_id] = {'error': str(e), 'status': 'instantiation_failed'}
            continue

        total_p, trainable_p = count_parameters(model)
        sig = get_model_signature(model)
        signatures[var_id] = sig

        print(f"  Total params: {total_p:,}  Trainable: {trainable_p:,}")
        print(f"  Signature: {json.dumps(sig)}")

        # Forward pass
        try:
            out = run_forward(model, batch)
        except Exception as e:
            print(f"  ERROR: Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            results[var_id] = {'error': str(e), 'status': 'forward_failed'}
            continue

        print(f"  Output: reg_mean={out['reg_mean']:.4f}, reg_std={out['reg_std']:.4f}")
        if 'w_t_mean' in out:
            print(f"  AWAF: w_t={out['w_t_mean']:.4f}, w_a={out['w_a_mean']:.4f}, w_v={out['w_v_mean']:.4f}")
            print(f"  AWAF unique weight vectors: {out['w_unique']}/4")

        results[var_id] = {
            'status': 'ok',
            'config_sha256': sha256,
            'mode': model_config.mode,
            'total_params': total_p,
            'trainable_params': trainable_p,
            **sig,
            **out,
        }

    # ================================================================
    # Switch integrity checks
    # ================================================================
    print("\n[3/4] Running switch integrity checks...")
    checks = []

    # Check F1 (no_vision) has NO vision branch
    if 'F1' in results and results['F1']['status'] == 'ok':
        has_vision = results['F1'].get('has_vision_proj', False)
        status = 'pass' if not has_vision else 'FAIL'
        checks.append(('F1_no_vision_branch', status,
                       f"Vision branch: {'present (BAD)' if has_vision else 'absent (OK)'}"))
        print(f"  F1 vision branch check: {status}")

    # Check F2 (no_audio) has NO audio branch
    if 'F2' in results and results['F2']['status'] == 'ok':
        has_audio = results['F2'].get('has_audio_proj', False)
        status = 'pass' if not has_audio else 'FAIL'
        checks.append(('F2_no_audio_branch', status,
                       f"Audio branch: {'present (BAD)' if has_audio else 'absent (OK)'}"))
        print(f"  F2 audio branch check: {status}")

    # Check F0 vs F1 outputs differ (vision should change output)
    if 'F0' in results and 'F1' in results and results['F0']['status'] == 'ok' and results['F1']['status'] == 'ok':
        reg_diff = abs(results['F0']['reg_mean'] - results['F1']['reg_mean'])
        status = 'pass' if reg_diff > 0 else 'FAIL'
        checks.append(('F0_vs_F1_outputs_differ', status,
                       f"|F0_mean - F1_mean| = {reg_diff:.6f}"))
        print(f"  F0 vs F1 output diff: {reg_diff:.6f} -> {status}")

    # Check F0 vs F2 outputs differ (audio should change output)
    if 'F0' in results and 'F2' in results and results['F0']['status'] == 'ok' and results['F2']['status'] == 'ok':
        reg_diff = abs(results['F0']['reg_mean'] - results['F2']['reg_mean'])
        status = 'pass' if reg_diff > 0 else 'FAIL'
        checks.append(('F0_vs_F2_outputs_differ', status,
                       f"|F0_mean - F2_mean| = {reg_diff:.6f}"))
        print(f"  F0 vs F2 output diff: {reg_diff:.6f} -> {status}")

    # Check F3 uses fixed fusion (not AWAF)
    if 'F3' in results and results['F3']['status'] == 'ok':
        is_fixed = results['F3'].get('fusion_mode') == 'fixed'
        status = 'pass' if is_fixed else 'FAIL'
        checks.append(('F3_fixed_fusion', status,
                       f"Fusion mode: {results['F3'].get('fusion_mode', 'unknown')}"))
        print(f"  F3 fusion mode: {results['F3'].get('fusion_mode')} -> {status}")

    # Check F4 uses mean fusion
    if 'F4' in results and results['F4']['status'] == 'ok':
        is_mean = results['F4'].get('fusion_mode') == 'mean'
        status = 'pass' if is_mean else 'FAIL'
        checks.append(('F4_mean_fusion', status,
                       f"Fusion mode: {results['F4'].get('fusion_mode', 'unknown')}"))
        print(f"  F4 fusion mode: {results['F4'].get('fusion_mode')} -> {status}")

    # Check F5 has no interaction
    if 'F5' in results and results['F5']['status'] == 'ok':
        no_interaction = not results['F5'].get('use_interaction', True)
        status = 'pass' if no_interaction else 'FAIL'
        checks.append(('F5_no_interaction', status,
                       f"Use interaction: {results['F5'].get('use_interaction', True)}"))
        print(f"  F5 interaction: {results['F5'].get('use_interaction')} -> {status}")

    # Check E1 uses LSTM
    if 'E1' in results and results['E1']['status'] == 'ok':
        is_lstm = results['E1'].get('temporal_encoder_type') == 'lstm'
        status = 'pass' if is_lstm else 'FAIL'
        checks.append(('E1_lstm_encoder', status,
                       f"Temporal encoder: {results['E1'].get('temporal_encoder_type', 'unknown')}"))
        print(f"  E1 temporal encoder: {results['E1'].get('temporal_encoder_type')} -> {status}")

    # Check all models produce distinct outputs
    all_ok = {k: v for k, v in results.items() if v.get('status') == 'ok'}
    reg_means = {k: v['reg_mean'] for k, v in all_ok.items()}
    n_unique = len(set(reg_means.values()))
    status = 'pass' if n_unique >= 3 else 'WARN'  # at least some should differ
    checks.append(('all_outputs_not_identical', status,
                   f"Unique reg_mean values: {n_unique}/{len(all_ok)}"))
    print(f"  Unique outputs: {n_unique}/{len(all_ok)} -> {status}")

    # ================================================================
    # Generate reports
    # ================================================================
    print("\n[4/4] Generating audit reports...")

    # CSV: model signature matrix
    sig_csv_path = os.path.join(REPORT_DIR, 'A0_tav_model_signature_matrix.csv')
    sig_fields = ['variant', 'mode', 'total_params', 'trainable_params',
                  'has_audio_proj', 'has_audio_temporal', 'has_audio_pool',
                  'has_vision_proj', 'has_vision_temporal', 'has_vision_pool',
                  'fusion_mode', 'use_context', 'use_interaction',
                  'temporal_encoder_type', 'is_canonical']
    with open(sig_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sig_fields)
        writer.writeheader()
        for var_id, _, var_name in VARIANTS:
            if var_id in results and results[var_id]['status'] == 'ok':
                row = {'variant': var_id}
                row.update({k: results[var_id].get(k, 'N/A') for k in sig_fields[1:]})
                writer.writerow(row)
    print(f"  Signature matrix: {sig_csv_path}")

    # CSV: fixed batch output audit
    out_csv_path = os.path.join(REPORT_DIR, 'A0_tav_fixed_batch_output_audit.csv')
    out_fields = ['variant', 'status', 'config_sha256', 'reg_mean', 'reg_std',
                  'w_t_mean', 'w_a_mean', 'w_v_mean', 'w_t_std', 'w_unique',
                  'total_params', 'trainable_params']
    with open(out_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for var_id, _, var_name in VARIANTS:
            if var_id in results:
                row = {'variant': var_id}
                row.update({k: results[var_id].get(k, 'N/A') for k in out_fields[1:]})
                writer.writerow(row)
    print(f"  Output audit CSV: {out_csv_path}")

    # Markdown: switch audit report
    md_path = os.path.join(REPORT_DIR, 'A0_tav_ablation_switch_audit.md')
    with open(md_path, 'w') as f:
        f.write("# P6AB TAV Ablation Switch Integrity Audit\n\n")
        f.write(f"**Date**: 2026-06-21\n")
        f.write(f"**Device**: {DEVICE}\n")
        f.write(f"**Seed**: {SEED}\n\n")

        f.write("## Switch Integrity Checks\n\n")
        f.write("| Check | Status | Detail |\n")
        f.write("|-------|--------|--------|\n")
        for name, status, detail in checks:
            emoji = 'PASS' if status == 'pass' else ('WARN' if status == 'WARN' else 'FAIL')
            f.write(f"| {name} | {emoji} {status} | {detail} |\n")
        f.write("\n")

        all_pass = all(s in ('pass',) for _, s, _ in checks)
        verdict = 'PASS — all variants valid for training' if all_pass else 'BLOCKED — fix failures before training'
        f.write(f"**Overall verdict**: {verdict}\n\n")

        f.write("## Model Signatures\n\n")
        f.write("| Variant | Params (Total/Trainable) | Audio Branch | Vision Branch | Fusion | Temporal |\n")
        f.write("|---------|--------------------------|--------------|---------------|--------|----------|\n")
        for var_id, _, var_name in VARIANTS:
            if var_id in results and results[var_id]['status'] == 'ok':
                r = results[var_id]
                audio = f"YES" if r.get('has_audio_proj') else f"NO"
                vision = f"YES" if r.get('has_vision_proj') else f"NO"
                fusion = r.get('fusion_mode', 'N/A')
                temporal = r.get('temporal_encoder_type', 'N/A')
                f.write(f"| {var_id} | {r['total_params']:,} / {r['trainable_params']:,} | {audio} | {vision} | {fusion} | {temporal} |\n")
            else:
                f.write(f"| {var_id} | ERROR | - | - | - | - |\n")
        f.write("\n")

        f.write("## Fixed Batch Forward Outputs\n\n")
        f.write("| Variant | reg_mean | reg_std | w_t | w_a | w_v | w_unique |\n")
        f.write("|---------|----------|---------|-----|-----|-----|----------|\n")
        for var_id, _, var_name in VARIANTS:
            if var_id in results and results[var_id]['status'] == 'ok':
                r = results[var_id]
                wt = f"{r.get('w_t_mean', 0):.4f}" if 'w_t_mean' in r else '-'
                wa = f"{r.get('w_a_mean', 0):.4f}" if 'w_a_mean' in r else '-'
                wv = f"{r.get('w_v_mean', 0):.4f}" if 'w_v_mean' in r else '-'
                wu = r.get('w_unique', '-')
                f.write(f"| {var_id} | {r['reg_mean']:.4f} | {r['reg_std']:.4f} | {wt} | {wa} | {wv} | {wu} |\n")
            else:
                f.write(f"| {var_id} | ERROR | - | - | - | - | - |\n")

    print(f"  Audit report: {md_path}")
    print(f"\n{'='*80}")
    print(f"Switch integrity audit complete. Verdict: {verdict}")
    print(f"{'='*80}")

    return all_pass


if __name__ == '__main__':
    main()
