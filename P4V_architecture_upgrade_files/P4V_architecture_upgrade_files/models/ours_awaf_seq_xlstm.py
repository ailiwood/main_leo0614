"""
models/ours_awaf_seq_xlstm.py

P4V 新增主模型候选：OursAWAFSeqXLSTM

结构：
    strong sequence features
      → modality-specific projection
      → three independent sLSTM encoders
      → CrossModalTransformerEncoder
      → MaskedAttentionPooling per modality
      → AdaptiveWeightedAttentionFusion (AWAF)
      → regression/classification heads

注意：
- 不覆盖 C0 (`models/ours_xlstm_fusion.py`)；
- Cross-modal Transformer 负责序列级交互；
- AWAF 保留为样本级动态权重融合和解释模块。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.encoders.slstm import SLSTMEncoder
from models.fusion.awaf import AdaptiveWeightedAttentionFusion
from models.heads import RegressionHead, ClassificationHead, UnimodalHeads
from models.interaction.cross_modal_transformer import CrossModalTransformerEncoder
from models.pooling.attention_pooling import MaskedAttentionPooling


class OursAWAFSeqXLSTM(nn.Module):
    """AWAF-Seq + Cross-modal Transformer 主模型候选。"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.text_dim = config.get("text_dim", 1024)
        self.audio_dim = config.get("audio_dim", 768)
        self.vision_dim = config.get("vision_dim", 1024)
        self.hidden_dim = config.get("hidden_dim", 256)

        proj_dropout = config.get("proj_dropout", 0.1)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(proj_dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(proj_dropout),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(proj_dropout),
        )

        slstm_kwargs = dict(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=config.get("slstm_num_layers", 1),
            dropout=config.get("slstm_dropout", 0.2),
            bidirectional=config.get("slstm_bidirectional", False),
            pooling=config.get("slstm_pooling", "masked_mean"),
        )
        self.text_slstm = SLSTMEncoder(**slstm_kwargs)
        self.audio_slstm = SLSTMEncoder(**slstm_kwargs)
        self.vision_slstm = SLSTMEncoder(**slstm_kwargs)

        slstm_out_dim = self.text_slstm.output_dim
        if slstm_out_dim != self.hidden_dim:
            # 若未来启用 bidirectional，可在此投影回 hidden_dim。
            self.text_post = nn.Linear(slstm_out_dim, self.hidden_dim)
            self.audio_post = nn.Linear(slstm_out_dim, self.hidden_dim)
            self.vision_post = nn.Linear(slstm_out_dim, self.hidden_dim)
        else:
            self.text_post = nn.Identity()
            self.audio_post = nn.Identity()
            self.vision_post = nn.Identity()

        self.cross_modal = CrossModalTransformerEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=config.get("cross_num_layers", 2),
            num_heads=config.get("cross_num_heads", 4),
            ffn_dim=config.get("cross_ffn_dim", 512),
            dropout=config.get("cross_dropout", 0.2),
            use_modality_embedding=config.get("cross_use_modality_embedding", True),
        )

        pooling_dropout = config.get("pooling_dropout", 0.1)
        self.text_pool = MaskedAttentionPooling(self.hidden_dim, dropout=pooling_dropout)
        self.audio_pool = MaskedAttentionPooling(self.hidden_dim, dropout=pooling_dropout)
        self.vision_pool = MaskedAttentionPooling(self.hidden_dim, dropout=pooling_dropout)

        self.awaf = AdaptiveWeightedAttentionFusion(
            hidden_dim=self.hidden_dim,
            fusion_mode=config.get("awaf_fusion_mode", "awaf"),
            tau_init=config.get("awaf_tau_init", 1.0),
            dropout=config.get("awaf_dropout", 0.2),
            use_modality_dropout=config.get("awaf_modality_dropout", True),
        )
        # 兼容当前 AWAF 实现：若模块内部有 modality_dropout_prob 属性，则显式设置。
        if hasattr(self.awaf, "modality_dropout_prob"):
            self.awaf.modality_dropout_prob = config.get("modality_dropout_prob", 0.2)

        head_dropout = config.get("head_dropout", 0.3)
        self.head_reg = RegressionHead(
            self.hidden_dim,
            config.get("head_reg_hidden", 128),
            head_dropout,
        )
        self.head_cls = ClassificationHead(
            self.hidden_dim,
            config.get("head_cls_hidden", 128),
            head_dropout,
        )
        self.use_aux = config.get("use_aux_heads", False)
        self.aux_heads = (
            UnimodalHeads(self.hidden_dim, modalities=["T", "A", "V"])
            if self.use_aux else None
        )

    @staticmethod
    def _ensure_seq(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected [B,D] or [B,T,D], got {tuple(x.shape)}")
        return x

    @staticmethod
    def _ensure_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.long)
        return mask.to(device=x.device).long()

    def _encode_modality(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        proj: nn.Module,
        slstm: SLSTMEncoder,
        post: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        x = self._ensure_seq(x)
        mask = self._ensure_mask(x, mask)
        z = proj(x)
        out = slstm(z, mask=mask, return_all=True)
        H = post(out["H"])
        pooled = post(out["pooled"])
        return {"H": H, "pooled": pooled, "mask": mask}

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        t = self._encode_modality(text, text_mask, self.text_proj, self.text_slstm, self.text_post)
        a = self._encode_modality(audio, audio_mask, self.audio_proj, self.audio_slstm, self.audio_post)
        v = self._encode_modality(vision, vision_mask, self.vision_proj, self.vision_slstm, self.vision_post)

        cross = self.cross_modal(
            t["H"], a["H"], v["H"],
            text_mask=t["mask"], audio_mask=a["mask"], vision_mask=v["mask"],
        )

        h_t, attn_t = self.text_pool(cross["H_t"], cross["text_mask"], return_weights=True)
        h_a, attn_a = self.audio_pool(cross["H_a"], cross["audio_mask"], return_weights=True)
        h_v, attn_v = self.vision_pool(cross["H_v"], cross["vision_mask"], return_weights=True)

        awaf_out = self.awaf(h_t, h_a, h_v, return_weights=True)
        z = awaf_out["Z"]
        reg_pred = self.head_reg(z)
        cls_pred = self.head_cls(z)

        aux_out = {}
        if self.aux_heads is not None:
            aux_out = self.aux_heads({"T": h_t, "A": h_a, "V": h_v})

        return {
            "reg": reg_pred,
            "cls": cls_pred,
            "aux": aux_out,
            "awaf_weights": awaf_out["weights"],
            "h_t": h_t,
            "h_a": h_a,
            "h_v": h_v,
            "pool_attn_t": attn_t,
            "pool_attn_a": attn_a,
            "pool_attn_v": attn_v,
            "cross_hidden_t": cross["H_t"],
            "cross_hidden_a": cross["H_a"],
            "cross_hidden_v": cross["H_v"],
        }
