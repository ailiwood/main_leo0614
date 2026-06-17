"""
engine/losses_polarity.py

P5A 可选损失：PolarityMarginLoss。
用于弱极性样本的 reg_sign 约束，不替代 MAE/L1。
"""

from __future__ import annotations
import torch
import torch.nn as nn


class PolarityMarginLoss(nn.Module):
    """回归输出极性 margin loss。

    对 y != 0 的样本：loss=max(0, margin - sign(y)*reg_pred)。
    """

    def __init__(self, margin: float = 0.2, ignore_zero_eps: float = 1e-6) -> None:
        super().__init__()
        self.margin = margin
        self.ignore_zero_eps = ignore_zero_eps

    def forward(self, reg_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reg_pred = reg_pred.view(-1)
        target = target.view(-1).to(reg_pred.device).float()
        valid = target.abs() > self.ignore_zero_eps
        if valid.sum() == 0:
            return reg_pred.new_tensor(0.0)
        sign = torch.where(target[valid] > 0, 1.0, -1.0).to(reg_pred.device)
        return torch.relu(self.margin - sign * reg_pred[valid]).mean()
