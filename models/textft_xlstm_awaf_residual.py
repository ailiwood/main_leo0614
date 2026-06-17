"""
P6D: TextFT-xLSTM-AWAF Residual — RoBERTa-large fine-tuned text backbone
     + frozen audio/vision features with sLSTM + AWAF residual fusion.

Architecture:
  Text:   RoBERTa-large → CLS/mean pool → DeepMLP → reg_text_base, cls_text_base
  Audio:  frozen 768d → sLSTM → attention pool → h_a
  Vision: frozen 768d → sLSTM → attention pool → h_v
  AWAF:   [h_text_residual, h_a, h_v] → delta_reg/cls, weights, gate
  Final:  reg = reg_text_base + gate_reg * δ_reg * bounded_delta_reg
"""
from __future__ import annotations
import torch, torch.nn as nn
from typing import Dict, Optional
from transformers import AutoModel, AutoConfig

from models.encoders.slstm import SLSTMEncoder
from models.fusion.awaf import AdaptiveWeightedAttentionFusion
from models.pooling.attention_pooling import MaskedAttentionPooling
from models.modules.uncertainty_residual_gate import UncertaintyGuidedResidualGate


class TextFTXLSTMAWAFResidual(nn.Module):
    def __init__(
        self,
        text_model_name: str = 'roberta-large',
        audio_dim: int = 768,
        vision_dim: int = 768,
        hidden_dim: int = 256,
        text_dropout: float = 0.1,
        text_pooling: str = 'cls',
        slstm_num_layers: int = 1,
        slstm_dropout: float = 0.2,
        awaf_fusion_mode: str = 'awaf',
        awaf_modality_dropout: bool = True,
        use_uncertainty_gate: bool = True,
        use_delta_experts: bool = True,
        use_bounded_delta: bool = True,
        max_delta: float = 1.0,
        delta_scale_init: float = 0.05,
        freeze_bottom_k_layers: int = 12,
        ablation: str = 'none',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_pooling = text_pooling
        self.ablation = ablation
        self.use_uncertainty_gate = use_uncertainty_gate
        self.use_delta_experts = use_delta_experts
        self.use_bounded_delta = use_bounded_delta
        self.max_delta = max_delta

        # --- Text branch: RoBERTa-large ---
        config = AutoConfig.from_pretrained(text_model_name)
        self.text_dim = config.hidden_size  # 1024 for roberta-large
        self.roberta = AutoModel.from_pretrained(text_model_name)
        self.text_dropout_layer = nn.Dropout(text_dropout)

        # Optionally freeze bottom layers
        if freeze_bottom_k_layers > 0:
            for i, layer in enumerate(self.roberta.encoder.layer):
                if i < freeze_bottom_k_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        # Text MLP head
        self.text_mlp = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), nn.GELU(), nn.Dropout(text_dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(text_dropout),
        )
        self.reg_text_head = nn.Linear(hidden_dim, 1)
        self.cls_text_head = nn.Linear(hidden_dim, 1)

        # --- Audio branch: frozen 768d + sLSTM ---
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(slstm_dropout)
        )
        self.audio_slstm = SLSTMEncoder(hidden_dim, hidden_dim, num_layers=slstm_num_layers,
                                         dropout=slstm_dropout, bidirectional=False, pooling='masked_mean')
        self.audio_pool = MaskedAttentionPooling(hidden_dim, dropout=slstm_dropout)

        # --- Vision branch: frozen 768d + sLSTM ---
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(slstm_dropout)
        )
        self.vision_slstm = SLSTMEncoder(hidden_dim, hidden_dim, num_layers=slstm_num_layers,
                                          dropout=slstm_dropout, bidirectional=False, pooling='masked_mean')
        self.vision_pool = MaskedAttentionPooling(hidden_dim, dropout=slstm_dropout)

        # --- AWAF ---
        awaf_mode = 'mean' if ablation == 'no_awaf_mean_residual' else awaf_fusion_mode
        self.awaf = AdaptiveWeightedAttentionFusion(
            hidden_dim=hidden_dim, fusion_mode=awaf_mode, tau_init=1.0,
            dropout=slstm_dropout, use_modality_dropout=awaf_modality_dropout, modality_dropout_prob=0.1)

        # Delta experts
        def _make_delta_head():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
                nn.GELU(), nn.Dropout(slstm_dropout), nn.Linear(hidden_dim // 2, 1))

        self.delta_reg_t = _make_delta_head(); self.delta_reg_a = _make_delta_head(); self.delta_reg_v = _make_delta_head()
        self.delta_cls_t = _make_delta_head(); self.delta_cls_a = _make_delta_head(); self.delta_cls_v = _make_delta_head()

        self.delta_scale_reg = nn.Parameter(torch.tensor(delta_scale_init))
        self.delta_scale_cls = nn.Parameter(torch.tensor(delta_scale_init))

        if use_uncertainty_gate:
            self.residual_gate = UncertaintyGuidedResidualGate(hidden_dim, gate_hidden_dim=128, dropout=0.1)

    def _text_forward(self, input_ids, attention_mask):
        """RoBERTa forward → h_text_base, reg_text_base, cls_text_base."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if self.text_pooling == 'cls':
            h = outputs.last_hidden_state[:, 0, :]  # [B, 1024]
        else:
            # mean pooling over non-padding tokens
            mask = attention_mask.float().unsqueeze(-1)
            h = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        h = self.text_dropout_layer(h)
        h_text = self.text_mlp(h)
        return h_text, self.reg_text_head(h_text), self.cls_text_head(h_text)

    def _slstm_branch(self, x, mask, proj, slstm, pool):
        h = proj(x)
        out = slstm(h, mask)
        pooled, _ = pool(out['H'], mask)
        return pooled

    def _bound_delta(self, delta):
        return self.max_delta * torch.tanh(delta) if self.use_bounded_delta else delta

    def _expert_delta(self, h_t, h_a, h_v, weights, kind):
        if kind == 'reg':
            d_t, d_a, d_v = self.delta_reg_t(h_t), self.delta_reg_a(h_a), self.delta_reg_v(h_v)
        else:
            d_t, d_a, d_v = self.delta_cls_t(h_t), self.delta_cls_a(h_a), self.delta_cls_v(h_v)
        return weights[:, 0:1]*d_t + weights[:, 1:2]*d_a + weights[:, 2:3]*d_v, d_t, d_a, d_v

    def forward(self, input_ids=None, attention_mask=None,
                audio=None, audio_mask=None, vision=None, vision_mask=None,
                labels=None, return_all=False):
        B = input_ids.size(0)
        device = input_ids.device

        # --- Text ---
        h_text_base, reg_text_base, cls_text_base = self._text_forward(input_ids, attention_mask)

        # --- Audio ---
        if self.ablation == 'no_audio':
            h_a = torch.zeros(B, self.hidden_dim, device=device)
        else:
            h_a = self._slstm_branch(audio, audio_mask, self.audio_proj, self.audio_slstm, self.audio_pool)

        # --- Vision ---
        if self.ablation == 'no_vision':
            h_v = torch.zeros(B, self.hidden_dim, device=device)
        else:
            h_v = self._slstm_branch(vision, vision_mask, self.vision_proj, self.vision_slstm, self.vision_pool)

        # --- AWAF residual ---
        if self.ablation == 'no_residual':
            reg, cls = reg_text_base, cls_text_base
            awaf_weights = torch.zeros(B, 3, device=device); awaf_weights[:, 0] = 1.0
            delta_reg = delta_cls = bounded_delta_reg = bounded_delta_cls = torch.zeros_like(reg_text_base)
            effective_delta_reg = torch.zeros_like(reg_text_base)
            gate_reg = gate_cls = torch.ones_like(reg_text_base)
        else:
            # Use text_base representation projected to hidden_dim for AWAF input
            h_t_residual = h_text_base  # already [B, H]
            awaf_out = self.awaf(h_t_residual, h_a, h_v)
            z_residual = awaf_out['Z']
            awaf_weights = awaf_out['weights']

            if self.use_delta_experts:
                delta_reg, dreg_t, dreg_a, dreg_v = self._expert_delta(h_t_residual, h_a, h_v, awaf_weights, 'reg')
                delta_cls, dcls_t, dcls_a, dcls_v = self._expert_delta(h_t_residual, h_a, h_v, awaf_weights, 'cls')
            else:
                delta_reg = self.delta_reg_z(z_residual); delta_cls = self.delta_cls_z(z_residual)

            bounded_delta_reg = self._bound_delta(delta_reg)
            bounded_delta_cls = self._bound_delta(delta_cls)

            if self.use_uncertainty_gate:
                gate_out = self.residual_gate(h_text_base, z_residual, reg_text_base, cls_text_base,
                                               awaf_weights, bounded_delta_reg)
                gate_reg = gate_out['gate_reg']; gate_cls = gate_out['gate_cls']
            else:
                gate_reg = torch.ones_like(reg_text_base); gate_cls = torch.ones_like(cls_text_base)

            effective_delta_reg = gate_reg * self.delta_scale_reg * bounded_delta_reg
            effective_delta_cls = gate_cls * self.delta_scale_cls * bounded_delta_cls
            reg = reg_text_base + effective_delta_reg
            cls = cls_text_base + effective_delta_cls

        result = {
            'reg': reg, 'cls': cls,
            'reg_text_base': reg_text_base, 'cls_text_base': cls_text_base,
            'delta_reg': delta_reg, 'delta_cls': delta_cls,
            'bounded_delta_reg': bounded_delta_reg, 'bounded_delta_cls': bounded_delta_cls,
            'effective_delta_reg': effective_delta_reg,
            'awaf_weights': awaf_weights,
            'residual_gate_reg': gate_reg, 'residual_gate_cls': gate_cls,
            'delta_scale_reg': self.delta_scale_reg, 'delta_scale_cls': self.delta_scale_cls,
        }
        if return_all:
            result.update({'h_text_base': h_text_base, 'h_a': h_a, 'h_v': h_v})
        return result
