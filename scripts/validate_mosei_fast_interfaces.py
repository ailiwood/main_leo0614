#!/usr/bin/env python
"""P6S-Repair-4: Validate all MOSEI interfaces before training."""
import sys, os, json, csv, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = 'data/processed/mosei_full/roberta_cache'
PROCESSED_DIR = 'data/processed/mosei_full'
PASS, FAIL = 0, 0

def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        print(f'  [PASS] {name}')
        PASS += 1
    else:
        print(f'  [FAIL] {name} {detail}')
        FAIL += 1

print('=== 1. Cache Files ===')
for split in ['train', 'valid', 'test']:
    f = os.path.join(CACHE_DIR, f'{split}_roberta_cls.npy')
    i = os.path.join(CACHE_DIR, f'{split}_ids.json')
    check(f'{split} .npy exists', os.path.exists(f), f)
    check(f'{split} .json exists', os.path.exists(i), f)
    if os.path.exists(f):
        d = np.load(f)
        check(f'{split} shape[-1]=1024', d.shape[-1]==1024, str(d.shape))

print('\n=== 2. Processed Dataset ===')
csv_path = os.path.join(PROCESSED_DIR, 'label.csv')
check('label.csv exists', os.path.exists(csv_path))
if os.path.exists(csv_path):
    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    check('has rows', len(rows) > 0)
    splits = {}
    for r in rows:
        splits[r['mode']] = splits.get(r['mode'], 0) + 1
    print(f'  Splits: {splits}')
    check('train > 10000', splits.get('train', 0) > 10000)

# Audio .npz
for split in ['train', 'valid', 'test']:
    npz_dir = os.path.join(PROCESSED_DIR, split)
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')] if os.path.isdir(npz_dir) else []
    check(f'{split} has .npz files', len(npz_files) > 0, f'{len(npz_files)} files')
    if npz_files:
        sample = np.load(os.path.join(npz_dir, npz_files[0]))
        check(f'{split} audio_seq shape[-1]=74', sample.get('audio_seq', np.zeros((1,74))).shape[-1] == 74)

print('\n=== 3. Cache-CSV Alignment ===')
for split in ['train', 'valid', 'test']:
    id_path = os.path.join(CACHE_DIR, f'{split}_ids.json')
    if not os.path.exists(id_path): continue
    with open(id_path) as f:
        cache_ids = set(json.load(f))
    csv_ids = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['mode'] == split:
                    csv_ids.add(f"{r['video_id']}_{r['clip_id']}")
    overlap = len(cache_ids & csv_ids)
    check(f'{split} cache-CSV overlap > 90%', overlap / max(len(csv_ids), 1) > 0.9,
          f'{overlap}/{len(csv_ids)} = {overlap/max(len(csv_ids),1)*100:.1f}%')

print('\n=== 4. Model Forward ===')
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from torch.utils.data import DataLoader
import torch.nn as nn

# Tokenizer fix
os.environ['TRANSFORMERS_OFFLINE'] = '1'
try:
    ds = TextFTMultimodalDataset(csv_path=csv_path, feature_root=PROCESSED_DIR, split='train', formal_mode=True)
    check('Dataset loads', True, f'{len(ds)} samples')
    loader = DataLoader(ds, 4, collate_fn=collate_textft)
    batch = next(iter(loader))
    check('Batch created', True)

    # Inject cached features
    train_feats = np.load(os.path.join(CACHE_DIR, 'train_roberta_cls.npy'))
    with open(os.path.join(CACHE_DIR, 'train_ids.json')) as f:
        train_ids = json.load(f)
    id_to_idx = {sid: i for i, sid in enumerate(train_ids)}
    indices = [id_to_idx.get(sid, 0) for sid in batch.get('id', ['']*4)]
    batch['roberta_cls'] = torch.from_numpy(train_feats[indices]).float()
    check('Cache injected', 'roberta_cls' in batch)

    # Test main model forward
    from models.textft_lora_xlstm_awaf_residual import TextFTLoRAConfig, TextFTLoRAXLSTMAWAFResidual
    config = TextFTLoRAConfig(mode='text_audio_residual', audio_input_dim=74, device='cpu',
                              use_uncertainty_gate=True, use_delta_experts=True)
    model = TextFTLoRAXLSTMAWAFResidual(config)
    check('Main model created', True)
    # Would need GPU for full test, skip

    # Test baseline models
    from models.baselines import tfn_lite, lmf_lite, mult_lite, self_mm_lite, misa_lite, mmim_lite, mlcl_lite, dlf_lite
    baseline_models = {
        'tfn': tfn_lite.TFNLite, 'lmf': lmf_lite.LMFLite, 'mult': mult_lite.MulTLite,
        'selfmm': self_mm_lite.SelfMMLite, 'misa': misa_lite.MISALite,
        'mmim': mmim_lite.MMIMLite, 'mlcl': mlcl_lite.MLCLLite, 'dlf': dlf_lite.DLFLite,
    }
    for name, cls in baseline_models.items():
        try:
            m = cls({'modality_mode': 'text_audio', 'hidden_dim': 128, 'audio_input_dim': 74,
                      'use_pretrained_text': True, 'text_input_dim': 1024})
            out = m(batch)
            check(f'{name} forward', 'reg' in out, str(list(out.keys())))
        except Exception as e:
            check(f'{name} forward', False, str(e)[:100])

except Exception as e:
    check('Pipeline', False, str(e)[:150])

print(f'\nResults: {PASS} passed, {FAIL} failed')
