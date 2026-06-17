"""
engine/residual_losses_v2.py

P5E loss for DeepText-xLSTM-UGR-AWAF Residual.

关键修复：
1. sample reweight 使用 reduction='none'，真正逐样本加权；
2. 新增 delta target loss，显式监督 residual 学习 label - text_base；
3. 新增 margin sign loss，增强非零样本符号边界；
4. 返回 group-level diagnostics，便于 weak_neg/near_zero 分析。
"""
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn.functional as F


class ResidualLossV2:
    def __init__(
        self,
        reg_loss_weight: float = 1.0,
        cls_loss_weight: float = 0.5,
        sign_consistency_weight: float = 0.1,
        delta_reg_weight: float = 0.02,
        delta_target_loss_weight: float = 0.2,
        margin_sign_loss_weight: float = 0.05,
        awaf_entropy_reg_weight: float = 0.0,
        sample_reweight_enabled: bool = False,
        weak_neg_weight: float = 1.5,
        weak_pos_weight: float = 1.1,
        near_zero_weight: float = 1.0,
        strong_sample_weight: float = 1.0,
        sign_focal_enabled: bool = False,
        sign_focal_gamma: float = 2.0,
        sign_focal_alpha_neg: float = 1.2,
        margin: float = 0.35,
        eps: float = 1e-8,
    ) -> None:
        self.reg_loss_weight = reg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.sign_consistency_weight = sign_consistency_weight
        self.delta_reg_weight = delta_reg_weight
        self.delta_target_loss_weight = delta_target_loss_weight
        self.margin_sign_loss_weight = margin_sign_loss_weight
        self.awaf_entropy_reg_weight = awaf_entropy_reg_weight
        self.sample_reweight_enabled = sample_reweight_enabled
        self.weak_neg_weight = weak_neg_weight
        self.weak_pos_weight = weak_pos_weight
        self.near_zero_weight = near_zero_weight
        self.strong_sample_weight = strong_sample_weight
        self.sign_focal_enabled = sign_focal_enabled
        self.sign_focal_gamma = sign_focal_gamma
        self.sign_focal_alpha_neg = sign_focal_alpha_neg
        self.margin = margin
        self.eps = eps

    @staticmethod
    def _groups(labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = labels.view(-1)
        return {
            "weak_neg": (y > -1.0) & (y < 0.0),
            "weak_pos": (y > 0.0) & (y < 1.0),
            "near_zero": y.abs() <= 0.5,
            "strong_neg": y <= -1.0,
            "strong_pos": y >= 1.0,
            "nonzero": y != 0.0,
        }

    def _weights(self, labels: torch.Tensor) -> torch.Tensor:
        y = labels.view(-1)
        weights = torch.ones_like(y) * self.strong_sample_weight
        if not self.sample_reweight_enabled:
            return weights.view(-1, 1)
        weak_neg = (y > -1.0) & (y < 0.0)
        weak_pos = (y > 0.0) & (y < 1.0)
        near_zero = y.abs() <= 0.5
        weights[weak_neg] = torch.maximum(weights[weak_neg], torch.tensor(self.weak_neg_weight, device=y.device, dtype=y.dtype))
        weights[weak_pos] = torch.maximum(weights[weak_pos], torch.tensor(self.weak_pos_weight, device=y.device, dtype=y.dtype))
        weights[near_zero] = torch.maximum(weights[near_zero], torch.tensor(self.near_zero_weight, device=y.device, dtype=y.dtype))
        return weights.view(-1, 1)

    @staticmethod
    def _weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (loss * weights).sum() / weights.sum().clamp(min=1e-8)

    def _bce_loss(self, logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.sign_focal_enabled:
            prob = torch.sigmoid(logits)
            p_t = torch.where(target > 0.5, prob, 1.0 - prob)
            focal = (1.0 - p_t).pow(self.sign_focal_gamma)
            alpha = torch.where(target > 0.5, torch.ones_like(target), torch.ones_like(target) * self.sign_focal_alpha_neg)
            bce = bce * focal * alpha
        return self._weighted_mean(bce, weights)

    def compute(self, output: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = labels.view(-1, 1).float()
        weights = self._weights(y)
        polarity = (y >= 0).float()

        # 1. Final regression loss
        per_reg = F.l1_loss(output["reg"], y, reduction="none")
        reg_loss = self._weighted_mean(per_reg, weights)

        # 2. Final classification loss
        cls_loss = torch.tensor(0.0, device=y.device)
        if self.cls_loss_weight > 0 and output.get("cls") is not None:
            cls_loss = self._bce_loss(output["cls"], polarity, weights)

        # 3. Sign consistency from regression prediction
        sign_loss = torch.tensor(0.0, device=y.device)
        if self.sign_consistency_weight > 0:
            sign_logit = 2.0 * output["reg"]
            sign_loss = self._bce_loss(sign_logit, polarity, weights)

        # 4. Delta regularization
        delta_abs_loss = torch.tensor(0.0, device=y.device)
        raw_delta = output.get("delta_reg")
        effective_delta = output.get("effective_delta_reg")
        if raw_delta is not None and self.delta_reg_weight > 0:
            delta_abs_loss = raw_delta.abs().mean()

        # 5. Explicit residual target loss
        delta_target_loss = torch.tensor(0.0, device=y.device)
        if self.delta_target_loss_weight > 0 and output.get("reg_text_base") is not None:
            target_delta = y - output["reg_text_base"].detach()
            pred_delta = effective_delta if effective_delta is not None else (output["reg"] - output["reg_text_base"])
            per_delta = F.smooth_l1_loss(pred_delta, target_delta, reduction="none")
            delta_target_loss = self._weighted_mean(per_delta, weights)

        # 6. Margin sign loss: encourage nonzero samples to stay away from 0 in correct direction
        margin_sign_loss = torch.tensor(0.0, device=y.device)
        if self.margin_sign_loss_weight > 0:
            nonzero = (y != 0).float()
            sign = torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))
            margin_violation = F.relu(self.margin - sign * output["reg"])
            denom = (weights * nonzero).sum().clamp(min=1e-8)
            margin_sign_loss = (margin_violation * weights * nonzero).sum() / denom

        # 7. AWAF entropy regularization
        entropy_reg = torch.tensor(0.0, device=y.device)
        w = output.get("awaf_weights")
        if w is not None and self.awaf_entropy_reg_weight != 0:
            wc = w.clamp(min=self.eps)
            entropy = -(wc * torch.log(wc)).sum(-1).mean()
            entropy_reg = -self.awaf_entropy_reg_weight * entropy

        total = (
            self.reg_loss_weight * reg_loss
            + self.cls_loss_weight * cls_loss
            + self.sign_consistency_weight * sign_loss
            + self.delta_reg_weight * delta_abs_loss
            + self.delta_target_loss_weight * delta_target_loss
            + self.margin_sign_loss_weight * margin_sign_loss
            + entropy_reg
        )

        result = {
            "total": total,
            "reg": reg_loss,
            "cls": cls_loss,
            "sign_consistency": sign_loss,
            "delta_reg": delta_abs_loss,
            "delta_target": delta_target_loss,
            "margin_sign": margin_sign_loss,
            "entropy_reg": entropy_reg,
            "sample_weight_mean": weights.mean().detach(),
        }

        # Lightweight diagnostics
        with torch.no_grad():
            groups = self._groups(y)
            for name, mask in groups.items():
                if mask.any():
                    result[f"mae_{name}"] = F.l1_loss(output["reg"][mask], y[mask], reduction="mean").detach()
        return result
