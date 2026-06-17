"""
models/deeptext_xlstm_awaf_residual_v2.py

P5E: DeepText-xLSTM-UGR-AWAF Residual V2

相比 P5D：
1. 使用 uncertainty-guided residual gate；
2. 使用 bounded delta: max_delta * tanh(delta_raw)；
3. 使用 modality-specific delta experts，并由 AWAF weights 直接加权；
4. 输出 effective_delta_reg，供 delta target loss 显式监督。

该文件是关键替换/新增候选。建议 CC 在 P5E 中作为新模型文件接入，
不要直接覆盖 P5D 主模型，直到实验证明有效。
"""
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn

from models.encoders.slstm import SLSTMEncoder
from models.fusion.awaf import AdaptiveWeightedAttentionFusion
from models.pooling.attention_pooling import MaskedAttentionPooling
from models.heads import RegressionHead, ClassificationHead, UnimodalHeads
from models.modules.uncertainty_residual_gate import UncertaintyGuidedResidualGate


class DeepTextXLSTMAWAFResidualV2(nn.Module):
    def __init__(
        self,
        text_dim: int = 1024,
        audio_dim: int = 768,
        vision_dim: int = 768,
        hidden_dim: int = 256,
        text_mlp_hidden: int = 512,
        text_mlp_layers: int = 3,
        text_mlp_dropout: float = 0.3,
        slstm_num_layers: int = 1,
        slstm_dropout: float = 0.3,
        slstm_pooling: str = "masked_mean",
        awaf_fusion_mode: str = "awaf",
        awaf_tau_init: float = 1.0,
        awaf_dropout: float = 0.1,
        awaf_modality_dropout: bool = True,
        awaf_modality_dropout_prob: float = 0.1,
        delta_scale_init: float = 0.1,
        max_delta: float = 1.5,
        use_bounded_delta: bool = True,
        use_uncertainty_gate: bool = True,
        use_delta_experts: bool = True,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.1,
        head_dropout: float = 0.3,
        ablation: str = "none",
        use_aux_heads: bool = False,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.ablation = ablation
        self.use_bounded_delta = use_bounded_delta
        self.use_uncertainty_gate = use_uncertainty_gate
        self.use_delta_experts = use_delta_experts
        self.max_delta = float(max_delta)

        # Text branch: NO sLSTM
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(slstm_dropout * 0.5)
        )
        self.text_pool = MaskedAttentionPooling(hidden_dim, dropout=head_dropout * 0.5)
        text_layers = []
        in_dim = hidden_dim
        for i in range(text_mlp_layers):
            out_dim = text_mlp_hidden if i < text_mlp_layers - 1 else hidden_dim
            text_layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(text_mlp_dropout)])
            in_dim = out_dim
        self.text_mlp = nn.Sequential(*text_layers)
        self.reg_text_head = RegressionHead(hidden_dim, hidden_dim // 2, head_dropout)
        self.cls_text_head = ClassificationHead(hidden_dim, hidden_dim // 2, head_dropout)

        # Audio / Vision xLSTM residual branches
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(slstm_dropout * 0.5))
        self.vision_proj = nn.Sequential(nn.Linear(vision_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(slstm_dropout * 0.5))
        self.audio_slstm = SLSTMEncoder(hidden_dim, hidden_dim, num_layers=slstm_num_layers, dropout=slstm_dropout, bidirectional=False, pooling=slstm_pooling)
        self.vision_slstm = SLSTMEncoder(hidden_dim, hidden_dim, num_layers=slstm_num_layers, dropout=slstm_dropout, bidirectional=False, pooling=slstm_pooling)
        self.audio_pool = MaskedAttentionPooling(hidden_dim, dropout=head_dropout * 0.5)
        self.vision_pool = MaskedAttentionPooling(hidden_dim, dropout=head_dropout * 0.5)

        # Text residual branch: projection + pool only
        self.text_residual_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(slstm_dropout * 0.5))
        self.text_residual_pool = MaskedAttentionPooling(hidden_dim, dropout=head_dropout * 0.5)

        # AWAF over residual representations
        awaf_mode = "mean" if ablation == "no_awaf_mean_residual" else awaf_fusion_mode
        self.awaf = AdaptiveWeightedAttentionFusion(
            hidden_dim=hidden_dim,
            fusion_mode=awaf_mode,
            tau_init=awaf_tau_init,
            dropout=awaf_dropout,
            use_modality_dropout=awaf_modality_dropout,
            modality_dropout_prob=awaf_modality_dropout_prob,
        )

        # Delta experts: each modality can propose correction. AWAF weights directly aggregate experts.
        def make_delta_head():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(head_dropout), nn.Linear(hidden_dim // 2, 1)
            )

        self.delta_reg_t = make_delta_head()
        self.delta_reg_a = make_delta_head()
        self.delta_reg_v = make_delta_head()
        self.delta_cls_t = make_delta_head()
        self.delta_cls_a = make_delta_head()
        self.delta_cls_v = make_delta_head()

        # Fallback single z heads, useful for ablation
        self.delta_reg_z = make_delta_head()
        self.delta_cls_z = make_delta_head()

        self.delta_scale_reg = nn.Parameter(torch.tensor(float(delta_scale_init)))
        self.delta_scale_cls = nn.Parameter(torch.tensor(float(delta_scale_init)))

        if use_uncertainty_gate:
            self.residual_gate = UncertaintyGuidedResidualGate(hidden_dim, gate_hidden_dim=gate_hidden_dim, dropout=gate_dropout)

        self.use_aux_heads = use_aux_heads
        if use_aux_heads:
            self.aux_heads = UnimodalHeads(hidden_dim, hidden_dim // 4, hidden_dim // 4, dropout=head_dropout, modalities=["A", "V"])

        self._init_info = {
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "model_name": "DeepTextXLSTMAWAFResidualV2",
        }

    def _pool_text(self, text: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.text_proj(text)
        pooled, _ = self.text_pool(h, mask)
        h_base = self.text_mlp(pooled)
        return {"h_text_base": h_base, "reg_text_base": self.reg_text_head(h_base), "cls_text_base": self.cls_text_head(h_base)}

    def _slstm_branch(self, x: torch.Tensor, mask: Optional[torch.Tensor], proj: nn.Module, slstm: nn.Module, pool: nn.Module) -> torch.Tensor:
        h = proj(x)
        out = slstm(h, mask)
        h_seq = out["H"]
        pooled, _ = pool(h_seq, mask)
        return pooled

    def _text_residual(self, text: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.text_residual_proj(text)
        pooled, _ = self.text_residual_pool(h, mask)
        return pooled

    def _bound_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if self.use_bounded_delta:
            return self.max_delta * torch.tanh(delta)
        return delta

    def _weighted_expert_delta(self, h_t: torch.Tensor, h_a: torch.Tensor, h_v: torch.Tensor, weights: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "reg":
            d_t, d_a, d_v = self.delta_reg_t(h_t), self.delta_reg_a(h_a), self.delta_reg_v(h_v)
        else:
            d_t, d_a, d_v = self.delta_cls_t(h_t), self.delta_cls_a(h_a), self.delta_cls_v(h_v)
        delta = weights[:, 0:1] * d_t + weights[:, 1:2] * d_a + weights[:, 2:3] * d_v
        return delta, d_t, d_a, d_v

    def forward(self, text, audio, vision, text_mask=None, audio_mask=None, vision_mask=None, return_all: bool = False) -> Dict[str, torch.Tensor]:
        B = text.size(0)
        text_out = self._pool_text(text, text_mask)
        h_text_base = text_out["h_text_base"]
        reg_text_base = text_out["reg_text_base"]
        cls_text_base = text_out["cls_text_base"]

        if self.ablation == "no_audio":
            h_a = torch.zeros(B, self.hidden_dim, device=text.device, dtype=text.dtype)
        else:
            h_a = self._slstm_branch(audio, audio_mask, self.audio_proj, self.audio_slstm, self.audio_pool)

        if self.ablation == "no_vision":
            h_v = torch.zeros(B, self.hidden_dim, device=text.device, dtype=text.dtype)
        else:
            h_v = self._slstm_branch(vision, vision_mask, self.vision_proj, self.vision_slstm, self.vision_pool)

        h_t_residual = self._text_residual(text, text_mask)

        if self.ablation == "no_residual":
            awaf_weights = torch.zeros(B, 3, device=text.device, dtype=text.dtype)
            awaf_weights[:, 0] = 1.0
            zero = torch.zeros_like(reg_text_base)
            return {
                "reg": reg_text_base, "cls": cls_text_base, "aux": {}, "awaf_weights": awaf_weights,
                "reg_text_base": reg_text_base, "cls_text_base": cls_text_base,
                "delta_reg": zero, "delta_cls": zero, "effective_delta_reg": zero,
                "delta_scale_reg": self.delta_scale_reg, "delta_scale_cls": self.delta_scale_cls,
            }

        awaf_out = self.awaf(h_t_residual, h_a, h_v)
        z_residual = awaf_out["Z"]
        awaf_weights = awaf_out["weights"]

        if self.use_delta_experts:
            delta_reg_raw, dreg_t, dreg_a, dreg_v = self._weighted_expert_delta(h_t_residual, h_a, h_v, awaf_weights, "reg")
            delta_cls_raw, dcls_t, dcls_a, dcls_v = self._weighted_expert_delta(h_t_residual, h_a, h_v, awaf_weights, "cls")
        else:
            delta_reg_raw = self.delta_reg_z(z_residual)
            delta_cls_raw = self.delta_cls_z(z_residual)
            dreg_t = dreg_a = dreg_v = delta_reg_raw
            dcls_t = dcls_a = dcls_v = delta_cls_raw

        bounded_delta_reg = self._bound_delta(delta_reg_raw)
        bounded_delta_cls = self._bound_delta(delta_cls_raw)

        if self.use_uncertainty_gate:
            gate_out = self.residual_gate(h_text_base, z_residual, reg_text_base, cls_text_base, awaf_weights, bounded_delta_reg)
            gate_reg = gate_out["gate_reg"]
            gate_cls = gate_out["gate_cls"]
        else:
            gate_out = {}
            gate_reg = torch.ones_like(reg_text_base)
            gate_cls = torch.ones_like(cls_text_base)

        effective_delta_reg = gate_reg * self.delta_scale_reg * bounded_delta_reg
        effective_delta_cls = gate_cls * self.delta_scale_cls * bounded_delta_cls
        reg = reg_text_base + effective_delta_reg
        cls = cls_text_base + effective_delta_cls

        aux = {}
        if self.use_aux_heads:
            aux = self.aux_heads({"A": h_a, "V": h_v})

        result = {
            "reg": reg,
            "cls": cls,
            "aux": aux,
            "awaf_weights": awaf_weights,
            "reg_text_base": reg_text_base,
            "cls_text_base": cls_text_base,
            "delta_reg": delta_reg_raw,
            "delta_cls": delta_cls_raw,
            "bounded_delta_reg": bounded_delta_reg,
            "bounded_delta_cls": bounded_delta_cls,
            "effective_delta_reg": effective_delta_reg,
            "effective_delta_cls": effective_delta_cls,
            "delta_scale_reg": self.delta_scale_reg,
            "delta_scale_cls": self.delta_scale_cls,
            "residual_gate_reg": gate_reg,
            "residual_gate_cls": gate_cls,
            "delta_reg_t": dreg_t,
            "delta_reg_a": dreg_a,
            "delta_reg_v": dreg_v,
            "delta_cls_t": dcls_t,
            "delta_cls_a": dcls_a,
            "delta_cls_v": dcls_v,
        }
        result.update(gate_out)
        if return_all:
            result.update({"h_text_base": h_text_base, "h_t_residual": h_t_residual, "h_a": h_a, "h_v": h_v, "z_residual": z_residual})
        return result
