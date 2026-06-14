"""
model_main_v6.py — v9: RoBERTa-large + v6 架构 (冲 87%)
========================================================

与 model_main_v3.py 完全相同, 唯一区别: TEXT_DIM 从 768 改 1024
其余 ModalEncoder / GatedFusion / Head / 训练逻辑全部复用 v3 写法

下游: train_main_v6.py 训这个, 3-seed 集成 + val 阈值校准

数据流: data/CMU-MOSEI/{train,val,test}_roberta.pkl -> MOSEIRobertaDataset -> MainModelV9
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, n_layers=2, n_heads=8,
                 seq_len=64, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pe = nn.Parameter(torch.zeros(1, seq_len + 1, hidden_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pe, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation='gelu'
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask=None):
        h = self.norm_in(self.proj(x))
        B = h.size(0)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pe[:, :h.size(1), :]
        if key_padding_mask is not None:
            kp = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device),
                            key_padding_mask], dim=1)
        else:
            kp = None
        h = self.enc(h, src_key_padding_mask=kp)
        return self.norm_out(h[:, 0, :])


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim, n_mod=3, dropout=0.2):
        super().__init__()
        self.n_mod = n_mod
        self.gate_proj = nn.Linear(hidden_dim * n_mod, n_mod)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * n_mod, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, embs):
        concat = torch.cat(embs, dim=-1)
        gates = torch.softmax(self.gate_proj(concat), dim=-1)
        weighted = torch.cat([e * gates[:, i:i+1] for i, e in enumerate(embs)], dim=-1)
        return self.fuse(weighted), gates


class Head(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class MainModelV9(nn.Module):
    """v6 架构 + RoBERTa-large 1024d text"""
    TEXT_DIM = 1024   # <-- 唯一区别
    AUDIO_DIM = 80
    VISION_DIM = 176

    def __init__(self, hidden_dim=256, head_dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_enc = ModalEncoder(self.TEXT_DIM, hidden_dim, n_layers=2,
                                      n_heads=8, seq_len=50, dropout=head_dropout)
        self.audio_enc = ModalEncoder(self.AUDIO_DIM, hidden_dim, n_layers=2,
                                       n_heads=8, seq_len=64, dropout=head_dropout)
        self.vision_enc = ModalEncoder(self.VISION_DIM, hidden_dim, n_layers=2,
                                        n_heads=8, seq_len=64, dropout=head_dropout)
        self.fusion = GatedFusion(hidden_dim, n_mod=3, dropout=head_dropout)
        self.head_M = Head(hidden_dim, 128, dropout=head_dropout)
        self.head_T = Head(hidden_dim, 64, dropout=head_dropout)
        self.head_A = Head(hidden_dim, 64, dropout=head_dropout)
        self.head_V = Head(hidden_dim, 64, dropout=head_dropout)
        self.head_pol = Head(hidden_dim, 128, dropout=head_dropout)

    def forward(self, text, audio, vision, text_mask=None):
        text_pad = (text_mask == 0) if text_mask is not None else None
        t_emb = self.text_enc(text, key_padding_mask=text_pad)
        a_emb = self.audio_enc(audio)
        v_emb = self.vision_enc(vision)
        fused, gates = self.fusion([t_emb, a_emb, v_emb])
        return {
            'M': self.head_M(fused),
            'T': self.head_T(t_emb),
            'A': self.head_A(a_emb),
            'V': self.head_V(v_emb),
            'polarity': self.head_pol(fused),
            'gates': gates,
            'fused': fused,
        }


if __name__ == '__main__':
    m = MainModelV9(hidden_dim=256)
    text = torch.randn(2, 50, 1024)  # RoBERTa-large
    audio = torch.randn(2, 64, 80)
    vision = torch.randn(2, 64, 176)
    mask = torch.ones(2, 50, dtype=torch.long)
    out = m(text, audio, vision, mask)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.shape}')
    print(f'Total params: {sum(p.numel() for p in m.parameters()):,}')
