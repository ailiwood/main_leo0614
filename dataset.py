"""
mosei_roberta_dataset.py
========================
新数据集类: 读 RoBERTa-large 1024d text + CASP audio/vision

与 mosei_bert_dataset.py 并行, 不覆盖 (因为 text dim 不同)

依赖文件: data/CMU-MOSEI/{train,val,test}_roberta.pkl
调用方:   train_main_v6.py (RoBERTa 版本)

每个样本输出 dict:
- text:              (50, 1024) RoBERTa-large last_hidden_state
- text_mask:         (50,) RoBERTa attention mask
- audio:             (64, 80) CASP audio
- vision:            (64, 176) CASP vision
- label:             (1,) sentiment in [-3, 3]
- raw_label:         scalar sentiment
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class MOSEIRobertaDataset(Dataset):
    DATA_DIR = r'D:\business\pycharm\project\Tri_modal_ER\data\CMU-MOSEI'

    def __init__(self, split='train'):
        pkl_path = os.path.join(self.DATA_DIR, f'{split}_roberta.pkl')
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        self.text = np.nan_to_num(d['text'])
        self.text_mask = d['text_mask']
        self.audio = np.nan_to_num(d['audio'])
        self.vision = np.nan_to_num(d['vision'])
        self.labels = d['regression_labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': torch.from_numpy(self.text[idx]).float(),
            'text_mask': torch.from_numpy(self.text_mask[idx]).long(),
            'audio': torch.from_numpy(self.audio[idx]).float(),
            'vision': torch.from_numpy(self.vision[idx]).float(),
            'label': torch.tensor([self.labels[idx]], dtype=torch.float32),
            'raw_label': float(self.labels[idx]),
        }


if __name__ == '__main__':
    for sp in ['train', 'val', 'test']:
        ds = MOSEIRobertaDataset(sp)
        sample = ds[0]
        print(f'{sp}: N={len(ds)}, text={sample["text"].shape}, '
              f'audio={sample["audio"].shape}, vision={sample["vision"].shape}, '
              f'label={sample["raw_label"]:.3f}')
