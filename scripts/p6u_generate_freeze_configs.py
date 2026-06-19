#!/usr/bin/env python
"""P6U: Generate all freeze configs."""
import os, yaml

base = 'configs/experiments/p6u_freeze'

def write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

# MOSI Main
write_yaml(f'{base}/mosi/main_text_audio_s42.yaml', {
    'phase': 'P6U-FREEZE', 'dataset': 'mosi',
    'model_type': 'main', 'modality_mode': 'text_audio',
    'seed': 42, 'epochs': 20, 'notes': 'BEST MOSI main model (P6K). Frozen for thesis.',
    'data': {'csv_path': 'data/mosi/label.csv', 'dataset': 'mosi',
             'feature_root': 'data/features_strong_sequence_mosi_v3_T40', 'formal_mode': True},
    'model': {'mode': 'text_audio_residual', 'audio_dim': 768, 'hidden_dim': 256,
              'slstm_num_layers': 1, 'slstm_dropout': 0.2, 'text_model_name': 'roberta-large',
              'text_mlp_hidden': 512, 'text_dropout': 0.1, 'lora_r': 16, 'lora_alpha': 32,
              'lora_dropout': 0.05, 'lora_targets': ['query','value'],
              'awaf_fusion_mode': 'awaf', 'tau_init': 3.0, 'awaf_dropout': 0.1,
              'use_modality_dropout': True, 'modality_dropout_prob': 0.1,
              'use_modal_layernorm': True, 'freeze_text_base': False},
    'training': {'batch_size': 4, 'grad_accum_steps': 16, 'epochs': 20,
                 'early_stopping_patience': 6, 'min_epochs': 5,
                 'lr': 3e-5, 'lr_text_lora': 3e-6, 'weight_decay': 0.03,
                 'seed': 42, 'test_final_once': True},
})
print('MOSI main done')

# MOSEI Main
write_yaml(f'{base}/mosei/main_text_audio_s42.yaml', {
    'phase': 'P6U-FREEZE', 'dataset': 'mosei',
    'model_type': 'main', 'modality_mode': 'text_audio',
    'seed': 42, 'epochs': 12, 'notes': 'MOSEI main (P6T). audio branch unused.',
    'data': {'csv_path': 'data/processed/mosei_full/label.csv', 'dataset': 'mosei',
             'feature_root': 'data/processed/mosei_full', 'formal_mode': True},
    'model': {'mode': 'text_audio_residual', 'audio_dim': 74, 'hidden_dim': 256,
              'slstm_num_layers': 1, 'slstm_dropout': 0.2, 'text_model_name': 'roberta-large',
              'text_mlp_hidden': 512, 'text_dropout': 0.1, 'lora_r': 16, 'lora_alpha': 32,
              'lora_dropout': 0.05, 'lora_targets': ['query','value'],
              'awaf_fusion_mode': 'awaf', 'tau_init': 3.0, 'awaf_dropout': 0.1,
              'use_modality_dropout': True, 'modality_dropout_prob': 0.1,
              'use_modal_layernorm': True, 'freeze_text_base': False},
    'training': {'batch_size': 8, 'grad_accum_steps': 8, 'epochs': 12,
                 'early_stopping_patience': 4, 'min_epochs': 3,
                 'lr': 3e-5, 'lr_text_lora': 3e-6, 'weight_decay': 0.03,
                 'seed': 42, 'test_final_once': True},
})
print('MOSEI main done')

# MOSEI Text-Only Diagnostic
write_yaml(f'{base}/mosei/text_only_diagnostic_s42.yaml', {
    'phase': 'P6U-FREEZE', 'dataset': 'mosei', 'model_type': 'diagnostic',
    'modality_mode': 'text_only',
    'notes': 'Diagnostic only. Not the final main model.',
    'data': {'csv_path': 'data/processed/mosei_full/label.csv', 'dataset': 'mosei',
             'feature_root': 'data/processed/mosei_full', 'formal_mode': True},
    'model': {'mode': 'text_only', 'hidden_dim': 256, 'slstm_num_layers': 1,
              'slstm_dropout': 0.2, 'text_model_name': 'roberta-large',
              'text_mlp_hidden': 512, 'text_dropout': 0.1,
              'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05,
              'lora_targets': ['query','value'], 'freeze_text_base': False},
    'training': {'batch_size': 8, 'epochs': 20, 'seed': 42,
                 'early_stopping_patience': 6, 'min_epochs': 5,
                 'lr': 3e-5, 'lr_text_lora': 3e-6, 'weight_decay': 0.03,
                 'test_final_once': True},
})
print('MOSEI diag done')

# MOSEI Baselines
for model in ['misa_lite','selfmm_lite','mult_lite','lmf_lite','tfn_lite','mlcl_lite','dlf_lite']:
    write_yaml(f'{base}/mosei/baselines/{model}_s42.yaml', {
        'phase': 'P6U-FREEZE', 'dataset': 'mosei', 'model_name': model,
        'model_type': 'baseline', 'modality_mode': 'text_audio', 'seed': 42,
        'epochs': 12, 'min_epochs': 3, 'early_stopping_patience': 4,
        'monitor_metric': 'ACC2_Non0', 'test_final_once': True,
        'data': {'csv_path': 'data/processed/mosei_full/label.csv', 'dataset': 'mosei',
                 'feature_root': 'data/processed/mosei_full', 'formal_mode': True},
        'model': {'model_name': model, 'modality_mode': 'text_audio',
                  'audio_input_dim': 74, 'text_input_dim': 1024, 'hidden_dim': 128,
                  'dropout': 0.2, 'use_pretrained_text': True,
                  'roberta_cache_dir': 'data/processed/mosei_full/roberta_cache'},
        'training': {'batch_size': 64, 'epochs': 12, 'early_stopping_patience': 4,
                     'min_epochs': 3, 'lr': 3e-4, 'weight_decay': 0.01, 'seed': 42,
                     'test_final_once': True},
    })
print('7 MOSEI baselines done')

# MOSI Baselines
for model in ['tfn_lite','mult_lite','selfmm_lite','lmf_lite']:
    write_yaml(f'{base}/mosi/baselines/{model}_s42.yaml', {
        'phase': 'P6U-FREEZE', 'dataset': 'mosi', 'model_name': model,
        'model_type': 'baseline', 'modality_mode': 'text_audio', 'seed': 42,
        'epochs': 12, 'min_epochs': 3, 'early_stopping_patience': 4,
        'monitor_metric': 'ACC2_Non0', 'test_final_once': True,
        'data': {'csv_path': 'data/mosi/label.csv', 'dataset': 'mosi',
                 'feature_root': 'data/features_strong_sequence_mosi_v3_T40', 'formal_mode': True},
        'model': {'model_name': model, 'modality_mode': 'text_audio',
                  'audio_input_dim': 768, 'text_input_dim': 1024, 'hidden_dim': 128,
                  'dropout': 0.2, 'use_pretrained_text': False},
        'training': {'batch_size': 64, 'epochs': 12, 'early_stopping_patience': 4,
                     'min_epochs': 3, 'lr': 3e-4, 'weight_decay': 0.01, 'seed': 42,
                     'test_final_once': True},
    })
print('4 MOSI baselines done')
print('\nAll 14 P6U freeze configs generated!')
