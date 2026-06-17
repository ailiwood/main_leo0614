"""
models/interaction/modality_reliability.py

P5A 新增模块：ModalityReliabilityGate

用途：估计样本级三模态可靠性，让 audio/vision 以 residual correction 的方式修正文本文本锚点，同时保留 AWAF 作为最终样本级权重解释模块。
"""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn


class ModalityReliabilityGate(nn.Module):
    """样本级三模态可靠性门控。

    输入 h_t/h_a/h_v: [B,D]
    输出 reliability: [B,3]，顺序为 text/audio/vision，范围 [0,1]。
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, hidden_ratio: float = 1.0) -> None:
        super().__init__()
        in_dim = hidden_dim * 7
        mid_dim = max(hidden_dim, int(hidden_dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 3),
        )

    def forward(self, h_t: torch.Tensor, h_a: torch.Tensor, h_v: torch.Tensor) -> Dict[str, torch.Tensor]:
        if h_t.dim() != 2 or h_a.dim() != 2 or h_v.dim() != 2:
            raise ValueError("h_t/h_a/h_v must be [B,D].")
        if not (h_t.shape == h_a.shape == h_v.shape):
            raise ValueError(f"Shape mismatch: {h_t.shape}, {h_a.shape}, {h_v.shape}")

        feat = torch.cat([
            h_t,
            h_a,
            h_v,
            torch.abs(h_t - h_a),
            torch.abs(h_t - h_v),
            h_t * h_a,
            h_t * h_v,
        ], dim=-1)
        reliability = torch.sigmoid(self.mlp(feat))
        return {
            "reliability": reliability,
            "r_t": reliability[:, 0:1],
            "r_a": reliability[:, 1:2],
            "r_v": reliability[:, 2:3],
        }
