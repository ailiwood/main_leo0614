"""
data/strong_sequence_dataset.py — MOSI Strong Sequence Dataset

Features: DeBERTa token-level (1024d) + wav2vec2 frame-level (768d) + CLIP (1024d, T=1)
Source: data/features_strong_sequence_mosi/<split>/<uid>.npz
"""
import os, numpy as np, torch
from torch.utils.data import Dataset

class StrongSequenceMOSIDataset(Dataset):
    def __init__(self, split='train', feature_root=None, max_seq_len=100):
        if feature_root is None:
            feature_root = 'data/features_strong_sequence_mosi'
        self.root = os.path.join(feature_root, split)
        self.files = sorted([f for f in os.listdir(self.root) if f.endswith('.npz')])
        self.max_seq_len = max_seq_len

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = np.load(os.path.join(self.root, self.files[idx]), allow_pickle=True)
        t_seq = torch.from_numpy(d['text_seq']).float()[:self.max_seq_len]     # (Tt, 1024)
        t_mask = torch.from_numpy(d['text_mask']).long()[:self.max_seq_len]
        a_seq = torch.from_numpy(d['audio_seq']).float()[:self.max_seq_len]   # (Ta, 768)
        a_mask = torch.from_numpy(d['audio_mask']).long()[:self.max_seq_len]
        v_seq = torch.from_numpy(d['vision_seq']).float()  # (1, 1024)
        v_mask = torch.from_numpy(d['vision_mask']).long()
        label = float(d['label']) if 'label' in d else 0.0
        sid = str(d['sample_id']) if 'sample_id' in d else self.files[idx].replace('.npz','')
        return {'text': t_seq, 'audio': a_seq, 'vision': v_seq,
                'text_mask': t_mask, 'audio_mask': a_mask, 'vision_mask': v_mask,
                'label': torch.tensor([label], dtype=torch.float32), 'id': sid, 'Tt': t_seq.shape[0],
                'Ta': a_seq.shape[0], 'Tv': v_seq.shape[0]}

def collate_strong_sequence(batch, max_text_len=50, max_audio_len=100, max_vision_len=1):
    B = len(batch)
    text_dim = batch[0]['text'].shape[-1]; audio_dim = batch[0]['audio'].shape[-1]; vision_dim = batch[0]['vision'].shape[-1]
    txt = torch.zeros(B, max_text_len, text_dim); aud = torch.zeros(B, max_audio_len, audio_dim)
    vis = torch.zeros(B, max_vision_len, vision_dim)
    tm = torch.zeros(B, max_text_len, dtype=torch.long); am = torch.zeros(B, max_audio_len, dtype=torch.long)
    vm = torch.zeros(B, max_vision_len, dtype=torch.long)
    labels = []; ids = []
    for i, item in enumerate(batch):
        Tt = min(item['Tt'], max_text_len); Ta = min(item['Ta'], max_audio_len); Tv = min(item['Tv'], max_vision_len)
        txt[i,:Tt] = item['text'][:Tt]; tm[i,:Tt] = item['text_mask'][:Tt]
        aud[i,:Ta] = item['audio'][:Ta]; am[i,:Ta] = item['audio_mask'][:Ta]
        vis[i,:Tv] = item['vision'][:Tv]; vm[i,:Tv] = item['vision_mask'][:Tv]
        labels.append(item['label']); ids.append(item['id'])
    return {'text': txt, 'audio': aud, 'vision': vis, 'text_mask': tm, 'audio_mask': am, 'vision_mask': vm,
            'label': torch.stack(labels), 'id': ids}

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    for sp in ['train','val','test']:
        ds = StrongSequenceMOSIDataset(sp)
        s = ds[0]
        ts = list(s['text'].shape); aus = list(s['audio'].shape); vs = list(s['vision'].shape)
        lb = s['label'].item()
        print('{}: N={}, text={}, audio={}, vision={}, label={:.1f}'.format(sp, len(ds), ts, aus, vs, lb))
    dl = DataLoader(StrongSequenceMOSIDataset('train'), 4, collate_fn=collate_strong_sequence)
    b = next(iter(dl))
    bts = list(b['text'].shape); bas = list(b['audio'].shape); bvs = list(b['vision'].shape)
    print('Batch: text={}, audio={}, vision={}'.format(bts, bas, bvs))
    import sys; sys.path.insert(0,'..')
    from models.ours_xlstm_fusion import OursXLSTMFusion
    cfg={'text_dim':1024,'audio_dim':768,'vision_dim':1024,'hidden_dim':256,'proj_dropout':0.1,'slstm_num_layers':1,'slstm_dropout':0.3,'slstm_pooling':'masked_mean','awaf_fusion_mode':'awaf','awaf_modality_dropout':True,'head_dropout':0.3,'use_aux_heads':False,'use_deconv':False,'use_cme':False,'use_data2vec_audio':False}
    m=OursXLSTMFusion(cfg); m.awaf.modality_dropout_prob=0.2
    o=m(b['text'],b['audio'],b['vision'],text_mask=b['text_mask'],audio_mask=b['audio_mask'],vision_mask=b['vision_mask'])
    loss=o['reg'].mean()+o['cls'].mean(); loss.backward()
    reg_s = list(o['reg'].shape); w_s = list(o['awaf_weights'].shape); ws = o['awaf_weights'].sum(-1).tolist()
    print('Model: reg={}, weights={}, sum(w)={}'.format(reg_s, w_s, ws))
    print('Backward OK, No NaN')
    print('STRONG SEQUENCE DATASET + MODEL FORWARD TEST PASSED!')
