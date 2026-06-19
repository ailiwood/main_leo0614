#!/usr/bin/env python
"""P6V: Generate all ablation configs for MOSI and MOSEI."""
import os, yaml

BASE = 'configs/experiments/p6v_ablation'

def write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

# === MOSI configs ===
MOSI_DATA = {'csv_path': 'data/mosi/label.csv', 'dataset': 'mosi',
             'feature_root': 'data/features_strong_sequence_mosi_v3_T40', 'formal_mode': True}
MOSI_MODEL_BASE = {'mode': 'text_audio_residual', 'audio_dim': 768, 'hidden_dim': 256,
    'slstm_num_layers': 1, 'slstm_dropout': 0.2, 'text_model_name': 'roberta-large',
    'text_mlp_hidden': 512, 'text_dropout': 0.1, 'lora_r': 16, 'lora_alpha': 32,
    'lora_dropout': 0.05, 'lora_targets': ['query','value'],
    'tau_init': 3.0, 'awaf_dropout': 0.1, 'use_modality_dropout': True,
    'modality_dropout_prob': 0.1, 'use_modal_layernorm': True, 'freeze_text_base': False}
MOSI_TRAIN = {'batch_size': 4, 'grad_accum_steps': 16, 'epochs': 8,
    'early_stopping_patience': 4, 'min_epochs': 3, 'lr': 3e-5, 'lr_text_lora': 3e-6,
    'weight_decay': 0.03, 'seed': 42, 'test_final_once': True}

mosi_ablations = [
    ('main_awaf_slstm', 'awaf', True, True, 'slstm'),
    ('fusion_mean', 'mean', True, True, 'slstm'),
    ('fusion_concat', 'concat', True, True, 'slstm'),
    ('fusion_gated', 'gated', True, True, 'slstm'),
    ('fusion_fixed', 'fixed', True, True, 'slstm'),
    ('awaf_no_interaction', 'awaf', True, False, 'slstm'),
    ('awaf_no_context', 'awaf', False, True, 'slstm'),
    ('encoder_gru', 'awaf', True, True, 'gru'),
    ('encoder_lstm', 'awaf', True, True, 'lstm'),
    ('encoder_no_temporal', 'awaf', True, True, 'none'),
]
for name, fusion, ctx, inter, enc in mosi_ablations:
    model = dict(MOSI_MODEL_BASE)
    model['fusion_type'] = fusion
    model['awaf_context'] = ctx
    model['awaf_interaction'] = inter
    model['temporal_encoder'] = enc
    if enc == 'gru':
        model['slstm_num_layers'] = 1
    write_yaml(f'{BASE}/mosi/{name}_s42.yaml', {
        'phase': 'P6V-ABLATION', 'dataset': 'mosi', 'ablation_name': name,
        'fusion_type': fusion, 'temporal_encoder': enc, 'seed': 42,
        'notes': f'MOSI ablation: fusion={fusion}, encoder={enc}',
        'data': MOSI_DATA, 'model': model, 'training': MOSI_TRAIN,
        'output': {'exp_name': f'{name}_s42', 'root': 'outputs/P6V_ablation/mosi'},
    })

# === MOSEI configs ===
MOSEI_DATA = {'csv_path': 'data/processed/mosei_full/label.csv', 'dataset': 'mosei',
              'feature_root': 'data/processed/mosei_full', 'formal_mode': True}
MOSEI_MODEL_BASE = {'mode': 'text_audio_residual', 'audio_dim': 74, 'hidden_dim': 256,
    'slstm_num_layers': 1, 'slstm_dropout': 0.2, 'text_model_name': 'roberta-large',
    'text_mlp_hidden': 512, 'text_dropout': 0.1, 'lora_r': 16, 'lora_alpha': 32,
    'lora_dropout': 0.05, 'lora_targets': ['query','value'],
    'tau_init': 3.0, 'awaf_dropout': 0.1, 'use_modality_dropout': True,
    'modality_dropout_prob': 0.1, 'use_modal_layernorm': True, 'freeze_text_base': False}
MOSEI_TRAIN = {'batch_size': 8, 'grad_accum_steps': 8, 'epochs': 4,
    'early_stopping_patience': 3, 'min_epochs': 3, 'lr': 3e-5, 'lr_text_lora': 3e-6,
    'weight_decay': 0.03, 'seed': 42, 'test_final_once': True}

mosei_ablations = [
    ('main_awaf_slstm', 'awaf', True, True, 'slstm'),
    ('fusion_mean', 'mean', True, True, 'slstm'),
    ('fusion_gated', 'gated', True, True, 'slstm'),
    ('awaf_no_interaction', 'awaf', True, False, 'slstm'),
    ('encoder_gru', 'awaf', True, True, 'gru'),
    ('encoder_no_temporal', 'awaf', True, True, 'none'),
]
for name, fusion, ctx, inter, enc in mosei_ablations:
    model = dict(MOSEI_MODEL_BASE)
    model['fusion_type'] = fusion
    model['awaf_context'] = ctx
    model['awaf_interaction'] = inter
    model['temporal_encoder'] = enc
    write_yaml(f'{BASE}/mosei/{name}_s42.yaml', {
        'phase': 'P6V-ABLATION', 'dataset': 'mosei', 'ablation_name': name,
        'fusion_type': fusion, 'temporal_encoder': enc, 'seed': 42,
        'notes': f'MOSEI ablation: fusion={fusion}, encoder={enc}',
        'data': MOSEI_DATA, 'model': model, 'training': MOSEI_TRAIN,
        'output': {'exp_name': f'{name}_s42', 'root': 'outputs/P6V_ablation/mosei'},
    })

print(f'Generated {len(mosi_ablations)} MOSI + {len(mosei_ablations)} MOSEI ablation configs')
print(f'Total: {len(mosi_ablations) + len(mosei_ablations)} configs')
