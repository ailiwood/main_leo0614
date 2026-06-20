#!/usr/bin/env python
"""P6W: Generate fair ablation configs with equal training budget."""
import os, yaml, hashlib, json

BASE = 'configs/experiments/p6w_fair_ablation'
COMMIT = '6e40eb8'

def hash_dict(d):
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]

def write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

# ===== MOSI PROTOCOL =====
MOSI_DATA = {'csv_path': 'data/mosi/label.csv', 'dataset': 'mosi',
    'feature_root': 'data/features_strong_sequence_mosi_v3_T40', 'formal_mode': True}
MOSI_MODEL_BASE = {'mode': 'text_audio_residual', 'audio_dim': 768, 'hidden_dim': 256,
    'slstm_num_layers': 1, 'slstm_dropout': 0.2, 'text_model_name': 'roberta-large',
    'text_mlp_hidden': 512, 'text_dropout': 0.1, 'lora_r': 16, 'lora_alpha': 32,
    'lora_dropout': 0.05, 'lora_targets': ['query','value'],
    'tau_init': 3.0, 'awaf_dropout': 0.1, 'use_modality_dropout': True,
    'modality_dropout_prob': 0.1, 'use_modal_layernorm': True, 'freeze_text_base': False}
MOSI_TRAIN = {'batch_size': 4, 'grad_accum_steps': 16, 'epochs': 20,
    'early_stopping_patience': 6, 'min_epochs': 5, 'lr': 3e-5, 'lr_text_lora': 3e-6,
    'weight_decay': 0.03, 'seed': 42, 'test_final_once': True}

mosi_ablations = [
    ('control_awaf_slstm', 'awaf', 'slstm'),
    ('fusion_mean', 'mean', 'slstm'),
    ('fusion_concat', 'concat', 'slstm'),
    ('fusion_gated', 'gated', 'slstm'),
    ('awaf_no_interaction', 'awaf', 'slstm'),
    ('encoder_gru', 'awaf', 'gru'),
    ('encoder_no_temporal', 'awaf', 'none'),
]

for name, fusion, enc in mosi_ablations:
    model = dict(MOSI_MODEL_BASE)
    model['fusion_type'] = fusion
    model['temporal_encoder'] = enc
    cfg = {
        'ablation_protocol': 'P6W_fair', 'source_commit': COMMIT,
        'phase': 'P6W-FAIR', 'dataset': 'mosi', 'ablation_name': name,
        'fusion_type': fusion, 'temporal_encoder': enc, 'seed': 42,
        'epochs': 20, 'min_epochs': 5, 'early_stopping_patience': 6,
        'notes': f'MOSI fair ablation: fusion={fusion}, encoder={enc}, epochs=20',
        'data': MOSI_DATA, 'model': model, 'training': MOSI_TRAIN,
        'output': {'exp_name': f'{name}_s42', 'root': 'outputs/P6W_fair_ablation/mosi'},
    }
    write_yaml(f'{BASE}/mosi/{name}_s42.yaml', cfg)
print(f'MOSI: {len(mosi_ablations)} configs')

# ===== MOSEI PROTOCOL =====
MOSEI_DATA = {'csv_path': 'data/processed/mosei_full/label.csv', 'dataset': 'mosei',
    'feature_root': 'data/processed/mosei_full', 'formal_mode': True}
MOSEI_MODEL_BASE = {'mode': 'text_audio_residual', 'audio_dim': 74, 'hidden_dim': 256,
    'slstm_num_layers': 1, 'slstm_dropout': 0.2, 'text_model_name': 'roberta-large',
    'text_mlp_hidden': 512, 'text_dropout': 0.1, 'lora_r': 16, 'lora_alpha': 32,
    'lora_dropout': 0.05, 'lora_targets': ['query','value'],
    'tau_init': 3.0, 'awaf_dropout': 0.1, 'use_modality_dropout': True,
    'modality_dropout_prob': 0.1, 'use_modal_layernorm': True, 'freeze_text_base': False}
MOSEI_TRAIN = {'batch_size': 8, 'grad_accum_steps': 8, 'epochs': 12,
    'early_stopping_patience': 4, 'min_epochs': 3, 'lr': 3e-5, 'lr_text_lora': 3e-6,
    'weight_decay': 0.03, 'seed': 42, 'test_final_once': True}

mosei_ablations = [
    ('control_awaf_slstm', 'awaf', 'slstm'),
    ('fusion_mean', 'mean', 'slstm'),
    ('fusion_gated', 'gated', 'slstm'),
    ('awaf_no_interaction', 'awaf', 'slstm'),
    ('encoder_gru', 'awaf', 'gru'),
    ('encoder_no_temporal', 'awaf', 'none'),
]

for name, fusion, enc in mosei_ablations:
    model = dict(MOSEI_MODEL_BASE)
    model['fusion_type'] = fusion
    model['temporal_encoder'] = enc
    cfg = {
        'ablation_protocol': 'P6W_fair', 'source_commit': COMMIT,
        'phase': 'P6W-FAIR', 'dataset': 'mosei', 'ablation_name': name,
        'fusion_type': fusion, 'temporal_encoder': enc, 'seed': 42,
        'epochs': 12, 'min_epochs': 3, 'early_stopping_patience': 4,
        'notes': f'MOSEI fair ablation: fusion={fusion}, encoder={enc}, epochs=12',
        'data': MOSEI_DATA, 'model': model, 'training': MOSEI_TRAIN,
        'output': {'exp_name': f'{name}_s42', 'root': 'outputs/P6W_fair_ablation/mosei'},
    }
    write_yaml(f'{BASE}/mosei/{name}_s42.yaml', cfg)
print(f'MOSEI: {len(mosei_ablations)} configs')
print(f'Total: {len(mosi_ablations) + len(mosei_ablations)} fair ablation configs')
