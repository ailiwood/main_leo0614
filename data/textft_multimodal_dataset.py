"""
P6D: TextFT Multimodal Dataset — combines raw text with frozen audio/vision features.
"""
import csv, os, numpy as np, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextFTMultimodalDataset(Dataset):
    def __init__(self, csv_path='data/mosi/label.csv', feature_root='data/features_strong_sequence_mosi_v3_T40',
                 split='train', tokenizer_name='roberta-large', max_text_len=128, max_audio_len=100, max_vision_len=50):
        self.feature_root = feature_root
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.max_vision_len = max_vision_len

        # Read labels and raw text
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        split_map = {'train': 'train', 'val': 'valid', 'test': 'test'}
        self.data = [r for r in rows if r.get('mode', 'train') == split_map.get(split, split)]

        # Tokenizer for raw text
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Feature directory
        self.feat_dir = os.path.join(feature_root, split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        sample_id = f"{r['video_id']}_{r['clip_id']}"

        # Raw text → tokens
        text = r['text']
        tokens = self.tokenizer(text, padding='max_length', truncation=True,
                                max_length=self.max_text_len, return_tensors='pt')

        # Label
        label = float(r['label'])

        # Load frozen audio/vision features
        feat_path = os.path.join(self.feat_dir, f'{sample_id}.npz')
        if os.path.exists(feat_path):
            feat = np.load(feat_path, allow_pickle=True)
            audio_seq = torch.from_numpy(feat['audio_seq']).float()[:self.max_audio_len]
            audio_mask = torch.from_numpy(feat['audio_mask']).long()[:self.max_audio_len]
            vision_seq = torch.from_numpy(feat['vision_seq']).float()[:self.max_vision_len]
            vision_mask = torch.from_numpy(feat['vision_mask']).long()[:self.max_vision_len]
        else:
            # Fallback: zero features
            audio_seq = torch.zeros(self.max_audio_len, 768)
            audio_mask = torch.zeros(self.max_audio_len, dtype=torch.long)
            vision_seq = torch.zeros(self.max_vision_len, 768)
            vision_mask = torch.zeros(self.max_vision_len, dtype=torch.long)

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'audio': audio_seq, 'audio_mask': audio_mask,
            'vision': vision_seq, 'vision_mask': vision_mask,
            'label': torch.tensor([label], dtype=torch.float32),
            'id': sample_id,
        }


def collate_textft(batch):
    """Collate for TextFT multimodal data."""
    B = len(batch)
    max_al = batch[0]['audio'].size(0); max_vl = batch[0]['vision'].size(0)
    audio_dim = batch[0]['audio'].size(-1); vision_dim = batch[0]['vision'].size(-1)
    max_tl = batch[0]['input_ids'].size(0)

    input_ids = torch.zeros(B, max_tl, dtype=torch.long)
    attention_mask = torch.zeros(B, max_tl, dtype=torch.long)
    audio = torch.zeros(B, max_al, audio_dim)
    audio_mask = torch.zeros(B, max_al, dtype=torch.long)
    vision = torch.zeros(B, max_vl, vision_dim)
    vision_mask = torch.zeros(B, max_vl, dtype=torch.long)
    labels = torch.zeros(B, 1)
    ids = []

    for i, item in enumerate(batch):
        tl = min(item['input_ids'].size(0), max_tl)
        input_ids[i, :tl] = item['input_ids'][:tl]
        attention_mask[i, :tl] = item['attention_mask'][:tl]
        al = min(item['audio'].size(0), max_al)
        audio[i, :al] = item['audio'][:al]
        audio_mask[i, :al] = item['audio_mask'][:al]
        vl = min(item['vision'].size(0), max_vl)
        vision[i, :vl] = item['vision'][:vl]
        vision_mask[i, :vl] = item['vision_mask'][:vl]
        labels[i] = item['label']
        ids.append(item['id'])

    return {'input_ids': input_ids, 'attention_mask': attention_mask,
            'audio': audio, 'audio_mask': audio_mask,
            'vision': vision, 'vision_mask': vision_mask,
            'label': labels, 'id': ids}
