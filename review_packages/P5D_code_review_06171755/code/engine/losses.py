"""
engine/losses.py — P5C Loss Functions for DeepText-xLSTM-AWAF Residual

支持的损失函数:
  - reg_loss:       L1 / SmoothL1 回归损失
  - cls_loss:       BCEWithLogits 分类损失
  - sign_consistency: 回归符号与标签符号一致性损失
  - aux_loss:       单模态辅助头损失
  - delta_reg:      delta 正则化 (鼓励小 delta，避免残差过度修正)
  - awaf_entropy_reg: AWAF 权重熵正则化 (鼓励多样性)

Usage:
    from engine.losses import compute_losses
    losses = compute_losses(output, labels, loss_weights)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ResidualLossComputer:
    """
    DeepText-xLSTM-AWAF Residual 专用损失计算器。

    Loss = w_reg * L_reg
         + w_cls * L_cls
         + w_sign * L_sign_consistency
         + w_aux * L_aux
         + w_delta * L_delta_reg
         + w_entropy * L_awaf_entropy
    """

    def __init__(
        self,
        # Loss weights
        reg_loss_weight: float = 1.0,
        cls_loss_weight: float = 0.5,
        sign_consistency_weight: float = 0.1,
        aux_loss_weight: float = 0.0,
        delta_reg_weight: float = 0.05,
        awaf_entropy_reg_weight: float = 0.0,
        # Loss types
        reg_loss_type: str = 'l1',  # 'l1' | 'smooth_l1'
        # P5D: Sample reweight
        sample_reweight_enabled: bool = False,
        weak_neg_weight: float = 1.5,
        weak_pos_weight: float = 1.2,
        near_zero_weight: float = 1.2,
        strong_sample_weight: float = 1.0,
        # P5D: Focal sign loss
        sign_focal_enabled: bool = False,
        sign_focal_gamma: float = 2.0,
        sign_focal_alpha_neg: float = 1.2,
        eps: float = 1e-8,
    ):
        self.reg_loss_weight = reg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.sign_consistency_weight = sign_consistency_weight
        self.aux_loss_weight = aux_loss_weight
        self.delta_reg_weight = delta_reg_weight
        self.awaf_entropy_reg_weight = awaf_entropy_reg_weight
        self.sample_reweight_enabled = sample_reweight_enabled
        self.weak_neg_weight = weak_neg_weight
        self.weak_pos_weight = weak_pos_weight
        self.near_zero_weight = near_zero_weight
        self.strong_sample_weight = strong_sample_weight
        self.sign_focal_enabled = sign_focal_enabled
        self.sign_focal_gamma = sign_focal_gamma
        self.sign_focal_alpha_neg = sign_focal_alpha_neg
        self.eps = eps

        if reg_loss_type == 'l1':
            self.reg_loss_fn = nn.L1Loss()
        elif reg_loss_type == 'smooth_l1':
            self.reg_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown reg_loss_type: {reg_loss_type}")

        self.cls_loss_fn = nn.BCEWithLogitsLoss()

    def to_device(self, device: torch.device):
        """Placeholder — losses are stateless, but kept for API compatibility."""
        return self

    def _get_sample_groups(self, labels: torch.Tensor):
        """P5D: Classify samples into groups for reweight analysis.

        weak_neg: -1.0 < label < 0
        weak_pos: 0 < label < 1.0
        near_zero: abs(label) <= 0.5 (overlaps with weak_neg/weak_pos)
        strong_neg: label <= -1.0
        strong_pos: label >= 1.0
        """
        lbl = labels.view(-1)
        return {
            'weak_neg': (lbl > -1.0) & (lbl < 0.0),
            'weak_pos': (lbl > 0.0) & (lbl < 1.0),
            'near_zero': lbl.abs() <= 0.5,
            'strong_neg': lbl <= -1.0,
            'strong_pos': lbl >= 1.0,
            'all': torch.ones_like(lbl, dtype=torch.bool),
        }

    def _compute_sample_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """P5D: Compute per-sample weights based on label groups."""
        if not self.sample_reweight_enabled:
            return torch.ones_like(labels.view(-1))
        lbl = labels.view(-1)
        weights = torch.ones_like(lbl) * self.strong_sample_weight
        # weak_neg: -1.0 < label < 0
        wneg = (lbl > -1.0) & (lbl < 0.0)
        weights[wneg] = self.weak_neg_weight
        # weak_pos: 0 < label < 1.0
        wpos = (lbl > 0.0) & (lbl < 1.0)
        weights[wpos] = self.weak_pos_weight
        # near_zero: abs(label) <= 0.5
        nz = lbl.abs() <= 0.5
        weights[nz] = max(weights[nz].max().item(), self.near_zero_weight)  # don't reduce weak weight
        return weights

    def compute(
        self,
        output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            output: model forward output dict
            labels: [B, 1] or [B] 回归标签 [-3, 3]

        Returns:
            dict with 'total', 'reg', 'cls', 'sign_consistency', 'aux', 'delta_reg', 'entropy_reg'
        """
        labels = labels.view(-1, 1).float()
        polarity = (labels >= 0).float()  # [B, 1]

        # P5D: Sample weights
        sample_w = self._compute_sample_weights(labels).view(-1, 1)  # [B, 1]
        sample_groups = self._get_sample_groups(labels)

        # --- 1. Regression Loss ---
        reg_loss = self.reg_loss_fn(output['reg'], labels)
        if self.sample_reweight_enabled:
            reg_loss = (reg_loss * sample_w).mean()  # element-wise L1 * weight

        # --- 2. Classification Loss ---
        if self.cls_loss_weight > 0:
            cls_loss = self.cls_loss_fn(output['cls'], polarity)
            if self.sample_reweight_enabled:
                cls_loss = (cls_loss * sample_w).mean()
        else:
            cls_loss = torch.tensor(0.0, device=labels.device)

        # --- 3. Sign Consistency Loss (P5D: +focal) ---
        if self.sign_consistency_weight > 0:
            sign_logit = 2.0 * output['reg'].view(-1)
            sign_target = polarity.view(-1)
            if self.sign_focal_enabled:
                # Focal BCE: -α * (1-p)^γ * y*log(p) - (1-α) * p^γ * (1-y)*log(1-p)
                bce = F.binary_cross_entropy_with_logits(sign_logit, sign_target, reduction='none')
                probs = torch.sigmoid(sign_logit)
                p_t = torch.where(sign_target == 1, probs, 1 - probs)
                focal_weight = (1 - p_t) ** self.sign_focal_gamma
                # Alpha: weight negative samples more
                alpha = torch.where(sign_target == 1, 1.0, self.sign_focal_alpha_neg)
                sign_loss = (alpha * focal_weight * bce).mean()
            else:
                sign_loss = self.cls_loss_fn(sign_logit, sign_target)
            if self.sample_reweight_enabled:
                sign_loss = sign_loss * sample_w.mean()  # global scaling
        else:
            sign_loss = torch.tensor(0.0, device=labels.device)

        # --- 4. Auxiliary Loss ---
        aux_loss = torch.tensor(0.0, device=labels.device)
        if output.get('aux') and self.aux_loss_weight > 0:
            aux_count = 0
            for k, v in output['aux'].items():
                if k.endswith('_reg'):
                    aux_loss += self.reg_loss_fn(v, labels)
                    aux_count += 1
                elif k.endswith('_cls'):
                    aux_loss += self.cls_loss_fn(v, polarity)
                    aux_count += 1
            if aux_count > 0:
                aux_loss = aux_loss / aux_count

        # --- 5. Delta Regularization ---
        # 鼓励 delta 不要过大 — 文本主判别应占主导
        if self.delta_reg_weight > 0:
            delta_reg = output.get('delta_reg')
            if delta_reg is not None:
                delta_reg_loss = delta_reg.abs().mean()
            else:
                delta_reg_loss = torch.tensor(0.0, device=labels.device)
        else:
            delta_reg_loss = torch.tensor(0.0, device=labels.device)

        # --- 6. AWAF Entropy Regularization ---
        # 负熵: 鼓励 AWAF 权重不要过于集中 (collapse prevention)
        entropy_reg = torch.tensor(0.0, device=labels.device)
        w = output.get('awaf_weights')
        if w is not None and self.awaf_entropy_reg_weight > 0:
            # Entropy = -sum(w * log(w)), we want to maximize (negative loss)
            w_clamped = w.clamp(self.eps)
            entropy = -(w_clamped * torch.log(w_clamped)).sum(-1).mean()
            # Negative sign: maximizing entropy means minimizing -entropy
            entropy_reg = -self.awaf_entropy_reg_weight * entropy

        # --- Total ---
        total = (
            self.reg_loss_weight * reg_loss
            + self.cls_loss_weight * cls_loss
            + self.sign_consistency_weight * sign_loss
            + self.aux_loss_weight * aux_loss
            + self.delta_reg_weight * delta_reg_loss
            + entropy_reg
        )

        return {
            'total': total,
            'reg': reg_loss,
            'cls': cls_loss,
            'sign_consistency': sign_loss,
            'aux': aux_loss,
            'delta_reg': delta_reg_loss,
            'entropy_reg': entropy_reg,
        }
        # P5D: Group-level regression loss for diagnostics
        if self.sample_reweight_enabled:
            group_reg_losses = {}
            with torch.no_grad():
                for grp_name, grp_mask in sample_groups.items():
                    if grp_name == 'all':
                        continue
                    if grp_mask.any():
                        grp_rl = self.reg_loss_fn(
                            output['reg'][grp_mask], labels[grp_mask]
                        ).mean()
                        group_reg_losses[f'reg_loss_{grp_name}'] = grp_rl
            result.update(group_reg_losses)
        return result


# Backward-compatible function interface
def compute_losses(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_weights: Optional[Dict[str, float]] = None,
    reg_loss_type: str = 'l1',
) -> Dict[str, torch.Tensor]:
    """
    便捷函数：计算所有损失。

    Args:
        output: model output dict
        labels: [B, 1] or [B]
        loss_weights: optional override of default weights
        reg_loss_type: 'l1' or 'smooth_l1'

    Returns:
        loss dict
    """
    weights = {
        'reg_loss_weight': 1.0,
        'cls_loss_weight': 0.5,
        'sign_consistency_weight': 0.1,
        'aux_loss_weight': 0.0,
        'delta_reg_weight': 0.05,
        'awaf_entropy_reg_weight': 0.0,
    }
    if loss_weights:
        weights.update(loss_weights)

    computer = ResidualLossComputer(
        reg_loss_weight=weights['reg_loss_weight'],
        cls_loss_weight=weights['cls_loss_weight'],
        sign_consistency_weight=weights['sign_consistency_weight'],
        aux_loss_weight=weights['aux_loss_weight'],
        delta_reg_weight=weights['delta_reg_weight'],
        awaf_entropy_reg_weight=weights['awaf_entropy_reg_weight'],
        reg_loss_type=reg_loss_type,
    )
    return computer.compute(output, labels)


# ============================================================
# 单元测试
# ============================================================
if __name__ == '__main__':
    print("=== Losses 单元测试 ===\n")

    B, H = 4, 256
    torch.manual_seed(42)

    # Mock output
    output = {
        'reg': torch.randn(B, 1) * 1.5,
        'cls': torch.randn(B, 1),
        'awaf_weights': torch.softmax(torch.randn(B, 3), dim=-1),
        'delta_reg': torch.randn(B, 1) * 0.2,
        'aux': {
            'A_reg': torch.randn(B, 1),
            'A_cls': torch.randn(B, 1),
            'V_reg': torch.randn(B, 1),
            'V_cls': torch.randn(B, 1),
        },
    }
    labels = torch.randn(B, 1) * 2.0  # range ~[-3, 3]

    # Test with default weights
    losses = compute_losses(output, labels, loss_weights={'aux_loss_weight': 0.1, 'delta_reg_weight': 0.05})
    for k, v in losses.items():
        print(f"  {k:20s}: {v.item():.6f}")
    assert not torch.isnan(losses['total']), "NaN in loss!"
    print("  ✅ Default weights OK")

    # Test with all weights = 0 except reg
    losses_reg = compute_losses(output, labels, loss_weights={
        'reg_loss_weight': 1.0, 'cls_loss_weight': 0.0,
        'sign_consistency_weight': 0.0, 'aux_loss_weight': 0.0,
        'delta_reg_weight': 0.0,
    })
    print(f"  reg_only total: {losses_reg['total'].item():.6f}")
    print("  ✅ Reg-only OK")

    # Test backward compatibility
    total = losses['total']
    total.backward()
    print("  ✅ Backward OK")

    # Test smooth_l1
    losses_sl1 = compute_losses(output, labels, reg_loss_type='smooth_l1')
    print(f"  smooth_l1 total: {losses_sl1['total'].item():.6f}")
    print("  ✅ SmoothL1 OK")

    # Test no delta_reg in output
    out_no_delta = {k: v for k, v in output.items() if k != 'delta_reg'}
    losses_nd = compute_losses(out_no_delta, labels, loss_weights={'delta_reg_weight': 0.05})
    print(f"  no_delta total: {losses_nd['total'].item():.6f} (delta_reg should be 0)")
    print("  ✅ No-delta OK")

    print("\n=== Losses 单元测试全部通过 ===")
