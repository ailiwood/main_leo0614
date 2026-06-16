"""
data/sequence_dataset.py — Standard MMSA/MLCL sequence feature dataset

Loads MLCL-format features: (features_tuple, label, id) per sample
Text: list of word strings (needs GloVe/BERT embedding)
Audio: (T_a, 47) COVAREP
Vision: (T_v, 74) FACET

Features are ALIGNED to word-level: T_text = T_audio = T_vision ~ 12
"""
import os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter


class MLCLMOSIDataset(Dataset):
    """MLCL-format MOSI sequence dataset."""

    def __init__(self, split='train', pkl_path=None, max_seq_len=50,
                 text_pad_dim=300, audio_dim=47, vision_dim=74):
        if pkl_path is None:
            pkl_path = 'external/features_mlcl/mosi_mcl.pkl'
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

        split_map = {'train':'train','val':'dev','dev':'dev','test':'test'}
        self.split = split_map.get(split, split)
        self.samples = self.data[self.split]
        self.max_seq_len = max_seq_len
        self.text_pad_dim = text_pad_dim
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        feats_tuple, label_arr, sid = self.samples[idx]
        txt_words = feats_tuple[0]  # list of word strings
        audio = np.array(feats_tuple[1]).astype(np.float32)  # (T,47)
        vision = np.array(feats_tuple[2]).astype(np.float32)  # (T,74)
        label = float(label_arr)

        # Truncate to max_seq_len
        T = min(len(txt_words), self.max_seq_len)
        audio = audio[:T]; vision = vision[:T]
        txt_words = txt_words[:T]

        # Build masks (1 for real, 0 for padding)
        t_mask = torch.ones(T, dtype=torch.long)
        a_mask = torch.ones(T, dtype=torch.long)
        v_mask = torch.ones(T, dtype=torch.long)

        return {
            'text_raw': txt_words,    # list of str
            'text': txt_words,        # placeholder — GloVe embedding done in collate or pre-processing
            'audio': torch.from_numpy(audio),   # (T, 47)
            'vision': torch.from_numpy(vision),  # (T, 74)
            'text_mask': t_mask, 'audio_mask': a_mask, 'vision_mask': v_mask,
            'label': torch.tensor([label], dtype=torch.float32),
            'id': sid, 'T': T,
        }


def collate_sequence(batch, max_seq_len=50):
    """Pad sequences in batch to max_seq_len."""
    T_max = max_seq_len  # Use fixed max for simplicity
    B = len(batch)

    # Device-independent: stay on CPU during collation
    audio_dim = batch[0]['audio'].shape[-1]
    vision_dim = batch[0]['vision'].shape[-1]

    text_batch = []; audio_batch = []; vision_batch = []
    text_mask_b = []; audio_mask_b = []; vision_mask_b = []
    labels = []; ids = []

    for item in batch:
        T = item['T']
        # Text: just store raw words for now (GloVe embedding TBD)
        text_batch.append(item['text_raw'])

        # Audio: pad to T_max
        a = item['audio']
        a_pad = torch.zeros(T_max, audio_dim)
        a_pad[:T] = a[:T_max]
        audio_batch.append(a_pad)

        # Vision: pad to T_max
        v = item['vision']
        v_pad = torch.zeros(T_max, vision_dim)
        v_pad[:T] = v[:T_max]
        vision_batch.append(v_pad)

        # Masks
        a_mask = torch.zeros(T_max, dtype=torch.long)
        a_mask[:T] = 1
        audio_mask_b.append(a_mask)
        v_mask = torch.zeros(T_max, dtype=torch.long)
        v_mask[:T] = 1
        vision_mask_b.append(v_mask)
        t_mask = torch.zeros(T_max, dtype=torch.long)
        t_mask[:T] = 1
        text_mask_b.append(t_mask)

        labels.append(item['label'])
        ids.append(item['id'])

    return {
        'text': text_batch,  # list of lists of strings
        'audio': torch.stack(audio_batch),
        'vision': torch.stack(vision_batch),
        'text_mask': torch.stack(text_mask_b),
        'audio_mask': torch.stack(audio_mask_b),
        'vision_mask': torch.stack(vision_mask_b),
        'label': torch.stack(labels),
        'id': ids,
    }


if __name__ == '__main__':
    print("=== MLCL MOSI Sequence Dataset Test ===\n")
    for sp in ['train','dev','test']:
        ds = MLCLMOSIDataset(sp)
        print('{}: N={}'.format(sp, len(ds)))
        s = ds[0]; tw = len(s['text_raw'])
        au_s = list(s['audio'].shape); vi_s = list(s['vision'].shape)
        print('  text_words={}, audio={}, vision={}, T={}, label={:.1f}'.format(
            tw, au_s, vi_s, s['T'], s['label'].item()))

    from torch.utils.data import DataLoader
    ds = MLCLMOSIDataset('train')
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_sequence)
    batch = next(iter(loader))
    au_s = list(batch['audio'].shape); vi_s = list(batch['vision'].shape)
    print('\nBatch: audio={}, vision={}'.format(au_s, vi_s))
    print('  masks: audio={}, vision={}'.format(
        batch['audio_mask'].sum().item(), batch['vision_mask'].sum().item()))

    import sys; sys.path.insert(0, '..')
    from models.ours_xlstm_fusion import OursXLSTMFusion
    cfg = {'text_dim':300,'audio_dim':47,'vision_dim':74,'hidden_dim':256,
           'proj_dropout':0.1,'slstm_num_layers':1,'slstm_dropout':0.3,
           'slstm_pooling':'masked_mean','awaf_fusion_mode':'awaf',
           'awaf_modality_dropout':True,'head_dropout':0.3,
           'use_aux_heads':False,'use_deconv':False,'use_cme':False,'use_data2vec_audio':False}
    print('\n  Random text embedding forward test:')
    txt_rand = torch.randn(4, 50, 300)
    m = OursXLSTMFusion(cfg)
    o = m(txt_rand, batch['audio'], batch['vision'],
          text_mask=batch['text_mask'], audio_mask=batch['audio_mask'],
          vision_mask=batch['vision_mask'])
    print('  reg={}, weights={}'.format(list(o['reg'].shape), list(o['awaf_weights'].shape)))
    dev = (o['awaf_weights'].sum(-1)-1).abs().max().item()
    print('  AWAF sum(w) max_dev={:.2e}'.format(dev))
    print('  Sequence model forward PASSED!')
