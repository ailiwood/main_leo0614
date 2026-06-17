"""
models/ours_text_guided_awaf_xlstm.py

P5A 主模型候选：Text-Guided AWAF-xLSTM

结构：
text/audio/vision → projection → sLSTM → CrossModalTransformer → attention pooling
→ ModalityReliabilityGate → Text-Anchor Residual Correction → AWAF → heads

Forward 接口兼容 StrictTrainer。
"""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn

from models.encoders.slstm import SLSTMEncoder
from models.fusion.awaf import AdaptiveWeightedAttentionFusion
from models.heads import ClassificationHead, RegressionHead, UnimodalHeads
from models.interaction.cross_modal_transformer import CrossModalTransformerEncoder
from models.interaction.modality_reliability import ModalityReliabilityGate
from models.pooling.attention_pooling import MaskedAttentionPooling


class TextGuidedAWAFXlstm(nn.Module):
    """Text-Guided AWAF-xLSTM 主模型候选。"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.text_dim = int(config.get("text_dim", 1024))
        self.audio_dim = int(config.get("audio_dim", 768))
        self.vision_dim = int(config.get("vision_dim", 768))
        self.hidden_dim = int(config.get("hidden_dim", 256))

        proj_dropout = float(config.get("proj_dropout", 0.1))
        self.text_proj = self._make_projection(self.text_dim, self.hidden_dim, proj_dropout)
        self.audio_proj = self._make_projection(self.audio_dim, self.hidden_dim, proj_dropout)
        self.vision_proj = self._make_projection(self.vision_dim, self.hidden_dim, proj_dropout)

        slstm_kwargs = dict(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=int(config.get("slstm_num_layers", 1)),
            dropout=float(config.get("slstm_dropout", 0.2)),
            bidirectional=bool(config.get("slstm_bidirectional", False)),
            pooling=str(config.get("slstm_pooling", "masked_mean")),
        )
        self.text_slstm = SLSTMEncoder(**slstm_kwargs)
        self.audio_slstm = SLSTMEncoder(**slstm_kwargs)
        self.vision_slstm = SLSTMEncoder(**slstm_kwargs)

        slstm_out_dim = self.text_slstm.output_dim
        if slstm_out_dim != self.hidden_dim:
            self.text_post = nn.Linear(slstm_out_dim, self.hidden_dim)
            self.audio_post = nn.Linear(slstm_out_dim, self.hidden_dim)
            self.vision_post = nn.Linear(slstm_out_dim, self.hidden_dim)
        else:
            self.text_post = nn.Identity()
            self.audio_post = nn.Identity()
            self.vision_post = nn.Identity()

        self.cross_modal = CrossModalTransformerEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=int(config.get("cross_num_layers", 2)),
            num_heads=int(config.get("cross_num_heads", 4)),
            ffn_dim=int(config.get("cross_ffn_dim", 512)),
            dropout=float(config.get("cross_dropout", 0.2)),
            use_modality_embedding=bool(config.get("cross_use_modality_embedding", True)),
        )

        pooling_dropout = float(config.get("pooling_dropout", 0.1))
        self.text_pool = MaskedAttentionPooling(self.hidden_dim, dropout=pooling_dropout)
        self.audio_pool = MaskedAttentionPooling(self.hidden_dim, dropout=pooling_dropout)
        self.vision_pool = MaskedAttentionPooling(self.hidden_dim, dropout=pooling_dropout)

        self.use_reliability_gate = bool(config.get("use_reliability_gate", True))
        self.reliability_gate = ModalityReliabilityGate(
            hidden_dim=self.hidden_dim,
            dropout=float(config.get("reliability_dropout", 0.1)),
            hidden_ratio=float(config.get("reliability_hidden_ratio", 1.0)),
        )

        self.audio_to_text = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.get("anchor_dropout", 0.1))),
        )
        self.vision_to_text = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.get("anchor_dropout", 0.1))),
        )
        self.anchor_norm = nn.LayerNorm(self.hidden_dim)

        self.awaf = AdaptiveWeightedAttentionFusion(
            hidden_dim=self.hidden_dim,
            fusion_mode=str(config.get("awaf_fusion_mode", "awaf")),
            tau_init=float(config.get("awaf_tau_init", 1.0)),
            dropout=float(config.get("awaf_dropout", 0.2)),
            use_modality_dropout=bool(config.get("awaf_modality_dropout", True)),
        )
        if hasattr(self.awaf, "modality_dropout_prob"):
            self.awaf.modality_dropout_prob = float(config.get("modality_dropout_prob", 0.2))

        init_lambda = float(config.get("anchor_lambda_init", 0.5))
        init_lambda = min(max(init_lambda, 1e-4), 1 - 1e-4)
        self.final_lambda_raw = nn.Parameter(torch.logit(torch.tensor(init_lambda, dtype=torch.float32)))
        self.final_norm = nn.LayerNorm(self.hidden_dim)

        head_dropout = float(config.get("head_dropout", 0.3))
        self.head_reg = RegressionHead(self.hidden_dim, int(config.get("head_reg_hidden", 128)), head_dropout)
        self.head_cls = ClassificationHead(self.hidden_dim, int(config.get("head_cls_hidden", 128)), head_dropout)

        self.use_aux = bool(config.get("use_aux_heads", True))
        self.aux_heads = UnimodalHeads(self.hidden_dim, modalities=["T", "A", "V"]) if self.use_aux else None

    @staticmethod
    def _make_projection(in_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

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

    def _encode_modality(self, x, mask, proj, slstm, post) -> Dict[str, torch.Tensor]:
        x = self._ensure_seq(x)
        mask = self._ensure_mask(x, mask)
        z = proj(x)
        out = slstm(z, mask=mask, return_all=True)
        return {"H": post(out["H"]), "pooled": post(out["pooled"]), "mask": mask}

    def forward(self, text, audio, vision, text_mask=None, audio_mask=None, vision_mask=None) -> Dict[str, torch.Tensor]:
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

        if self.use_reliability_gate:
            rel = self.reliability_gate(h_t, h_a, h_v)
        else:
            rel = {
                "reliability": torch.ones(h_t.size(0), 3, device=h_t.device, dtype=h_t.dtype),
                "r_t": torch.ones(h_t.size(0), 1, device=h_t.device, dtype=h_t.dtype),
                "r_a": torch.ones(h_t.size(0), 1, device=h_t.device, dtype=h_t.dtype),
                "r_v": torch.ones(h_t.size(0), 1, device=h_t.device, dtype=h_t.dtype),
            }
        r_t, r_a, r_v = rel["r_t"], rel["r_a"], rel["r_v"]

        h_anchor = self.anchor_norm(h_t + r_a * self.audio_to_text(h_a) + r_v * self.vision_to_text(h_v))
        h_t_cal = self.anchor_norm(h_t + r_t * h_t)
        h_a_cal = self.anchor_norm(h_a + r_a * h_a)
        h_v_cal = self.anchor_norm(h_v + r_v * h_v)

        awaf_out = self.awaf(h_t_cal, h_a_cal, h_v_cal, return_weights=True)
        z_awaf = awaf_out["Z"]
        anchor_lambda = torch.sigmoid(self.final_lambda_raw)
        z = self.final_norm(z_awaf + anchor_lambda * h_anchor)

        aux_out = self.aux_heads({"T": h_t_cal, "A": h_a_cal, "V": h_v_cal}) if self.aux_heads is not None else {}
        return {
            "reg": self.head_reg(z),
            "cls": self.head_cls(z),
            "aux": aux_out,
            "awaf_weights": awaf_out["weights"],
            "h_t": h_t_cal,
            "h_a": h_a_cal,
            "h_v": h_v_cal,
            "h_anchor": h_anchor,
            "z_awaf": z_awaf,
            "z_final": z,
            "reliability": rel["reliability"],
            "anchor_lambda": anchor_lambda.detach().view(1),
            "pool_attn_t": attn_t,
            "pool_attn_a": attn_a,
            "pool_attn_v": attn_v,
            "cross_hidden_t": cross["H_t"],
            "cross_hidden_a": cross["H_a"],
            "cross_hidden_v": cross["H_v"],
        }


OursTextGuidedAWAFXLSTM = TextGuidedAWAFXlstm
