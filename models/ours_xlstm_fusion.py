"""
models/ours_xlstm_fusion.py — 主模型 (支持候选模块挂载)

P3B 更新：支持 DEConv / CME / Data2Vec-Audio 候选模块的挂载与消融切换。
"""
import torch
import torch.nn as nn
import yaml
from typing import Dict, Optional

from models.encoders.slstm import SLSTMEncoder
from models.fusion.awaf import AdaptiveWeightedAttentionFusion
from models.heads import RegressionHead, ClassificationHead, UnimodalHeads


class OursXLSTMFusion(nn.Module):
    """主模型: sLSTM + AWAF + 可选增强模块。"""

    def __init__(self, config: dict):
        super().__init__()

        self.text_dim = config.get('text_dim', 1024)
        self.audio_dim = config.get('audio_dim', 1024)
        self.vision_dim = config.get('vision_dim', 1024)
        self.hidden_dim = config.get('hidden_dim', 256)

        # 候选模块开关
        self.use_deconv = config.get('use_deconv', False)
        self.use_cme = config.get('use_cme', False)
        self.use_data2vec_audio = config.get('use_data2vec_audio', False)

        # --- 投影层 ---
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.get('proj_dropout', 0.1)),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.get('proj_dropout', 0.1)),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.get('proj_dropout', 0.1)),
        )

        # --- DEConv (vision enhancement) ---
        if self.use_deconv:
            from models.enhancements.deconv import DynamicFeatureEnhancer
            self.deconv = DynamicFeatureEnhancer(
                dim=self.hidden_dim,
                kernel_size=config.get('deconv_kernel', 3),
                dropout=config.get('deconv_dropout', 0.1),
            )
        else:
            self.deconv = None

        # --- sLSTM 编码器 ---
        slstm_kwargs = dict(
            input_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
            num_layers=config.get('slstm_num_layers', 1),
            dropout=config.get('slstm_dropout', 0.0),
            bidirectional=config.get('slstm_bidirectional', False),
            pooling=config.get('slstm_pooling', 'masked_mean'),
        )
        self.text_slstm = SLSTMEncoder(**slstm_kwargs)
        self.audio_slstm = SLSTMEncoder(**slstm_kwargs)
        self.vision_slstm = SLSTMEncoder(**slstm_kwargs)

        self.text_align = nn.Identity()
        self.audio_align = nn.Identity()
        self.vision_align = nn.Identity()

        # --- CME (cross-modal enhancement) ---
        if self.use_cme:
            cme_type = config.get('cme_type', 'lightweight')  # 'lightweight' | 'gated'
            if cme_type == 'gated':
                from models.enhancements.cme_residual import GatedCME
                self.cme = GatedCME(
                    dim=self.hidden_dim,
                    num_heads=config.get('cme_num_heads', 4),
                    alpha_init=config.get('cme_alpha_init', 0.1),
                    dropout=config.get('cme_dropout', 0.1),
                )
            else:
                from models.enhancements.cme import LightweightCME
                self.cme = LightweightCME(
                    dim=self.hidden_dim,
                    num_heads=config.get('cme_num_heads', 4),
                    dropout=config.get('cme_dropout', 0.1),
                )
        else:
            self.cme = None

        # --- AWAF ---
        self.awaf = AdaptiveWeightedAttentionFusion(
            hidden_dim=self.hidden_dim,
            fusion_mode=config.get('awaf_fusion_mode', 'awaf'),
            tau_init=config.get('awaf_tau_init', 1.0),
            dropout=config.get('awaf_dropout', 0.1),
            use_modality_dropout=config.get('awaf_modality_dropout', True),
        )

        # --- Heads ---
        head_dropout = config.get('head_dropout', 0.3)
        self.head_reg = RegressionHead(self.hidden_dim, config.get('head_reg_hidden', 128), head_dropout)
        self.head_cls = ClassificationHead(self.hidden_dim, config.get('head_cls_hidden', 128), head_dropout)
        self.use_aux = config.get('use_aux_heads', False)
        if self.use_aux:
            self.aux_heads = UnimodalHeads(self.hidden_dim, modalities=['T', 'A', 'V'])
        else:
            self.aux_heads = None

    def _encode_modality(self, x, proj, slstm, align, mask=None, deconv=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if mask is None:
                mask = torch.ones(x.size(0), 1, device=x.device)
        h = proj(x)
        if deconv is not None:
            h = deconv(h)
        out = slstm(h, mask=mask, return_all=True)
        return align(out['pooled'])

    def forward(self, text, audio, vision,
                text_mask=None, audio_mask=None, vision_mask=None):
        # Vision: optional DEConv
        h_v = self._encode_modality(vision, self.vision_proj, self.vision_slstm,
                                     self.vision_align, vision_mask, self.deconv)
        h_t = self._encode_modality(text, self.text_proj, self.text_slstm,
                                     self.text_align, text_mask)
        h_a = self._encode_modality(audio, self.audio_proj, self.audio_slstm,
                                     self.audio_align, audio_mask)

        # CME
        if self.cme is not None:
            h_t, h_a, h_v = self.cme(h_t, h_a, h_v)

        # AWAF
        awaf_out = self.awaf(h_t, h_a, h_v, return_weights=True)

        reg_pred = self.head_reg(awaf_out['Z'])
        cls_pred = self.head_cls(awaf_out['Z'])

        aux_out = {}
        if self.aux_heads is not None:
            aux_out = self.aux_heads({'T': h_t, 'A': h_a, 'V': h_v})

        return {
            'reg': reg_pred, 'cls': cls_pred, 'aux': aux_out,
            'awaf_weights': awaf_out['weights'],
            'h_t': h_t, 'h_a': h_a, 'h_v': h_v,
        }
