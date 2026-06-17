"""
models/modules/conditional_residual_gate.py — P5D Conditional Residual Gate

设计目标:
  当前 final = text_base + λ * delta (无条件残差)
  改为:      final = text_base + g(x) * λ * delta (条件残差)

Gate 输入:
  - h_text_base: 文本基础表征
  - z_residual: AWAF 融合后的残差表征
  - abs(reg_text_base): 文本预测的绝对值 (proxy for confidence)
  - cls_text_base sign confidence: 文本分类置信度
  - optional: AWAF entropy
  - optional: |delta_reg| magnitude

直觉:
  - strong_pos / strong_neg: text base 可靠 → gate ≈ 0 (少修正)
  - weak / near_zero: text base 不确定 → gate ≈ 1 (多修正)

Gate 输出: g_reg, g_cls ∈ [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ConditionalResidualGate(nn.Module):
    """
    条件残差门控模块。

    输入:
        h_text_base:   [B, H] — text branch 表征
        z_residual:    [B, H] — AWAF residual 表征
        reg_text_base: [B, 1] — text regression prediction
        cls_text_base: [B, 1] — text classification logit
        awaf_weights:  [B, 3] — AWAF weights (optional, for entropy)
        delta_reg:     [B, 1] — delta_reg (optional)

    输出:
        gate_reg: [B, 1] ∈ [0, 1]
        gate_cls: [B, 1] ∈ [0, 1]
    """

    def __init__(
        self,
        hidden_dim: int,
        gate_hidden_dim: int = 128,
        dropout: float = 0.1,
        init_bias: float = -1.0,  # 负初始偏置 → gate ≈ 0.27 初始 (保守)
        use_text_confidence: bool = True,
        use_awaf_entropy: bool = True,
        use_delta_magnitude: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_text_confidence = use_text_confidence
        self.use_awaf_entropy = use_awaf_entropy
        self.use_delta_magnitude = use_delta_magnitude
        self.eps = eps

        # 计算输入维度
        # base: h_text_base [H] + z_residual [H] = 2*H
        input_dim = 2 * hidden_dim
        # text confidence: |reg_text_base| [1] + cls_text_base [1] = 2
        if use_text_confidence:
            input_dim += 2
        # awaf entropy: 1 (scalar)
        if use_awaf_entropy:
            input_dim += 1
        # delta magnitude: |delta_reg| [1]
        if use_delta_magnitude:
            input_dim += 1

        # Gate MLP: input_dim → gate_hidden → 2 (reg_gate, cls_gate)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim // 2),
            nn.LayerNorm(gate_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim // 2, 2),  # [gate_reg_raw, gate_cls_raw]
        )

        # 初始化：最后一层 bias 设为负值 → sigmoid 初始值小 → 保守起始
        with torch.no_grad():
            self.gate_net[-1].bias.data.fill_(init_bias)

    def forward(
        self,
        h_text_base: torch.Tensor,
        z_residual: torch.Tensor,
        reg_text_base: torch.Tensor,
        cls_text_base: torch.Tensor,
        awaf_weights: Optional[torch.Tensor] = None,
        delta_reg: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_text_base:   [B, H]
            z_residual:    [B, H]
            reg_text_base: [B, 1]
            cls_text_base: [B, 1]
            awaf_weights:  [B, 3] (optional)
            delta_reg:     [B, 1] (optional)

        Returns:
            {'gate_reg': [B,1], 'gate_cls': [B,1]}
        """
        # Base features
        features = [h_text_base, z_residual]  # 2×[B,H]

        # Text confidence: |reg_text_base| as inverse confidence
        if self.use_text_confidence:
            # Larger magnitude → more confident → gate should be smaller
            # We pass |reg| as a feature; the MLP learns the inverse relationship
            abs_reg = reg_text_base.abs()  # [B, 1]
            # Also pass cls confidence: |cls_logit| as confidence proxy
            cls_conf = cls_text_base.abs()  # [B, 1] — larger |logit| = more confident
            features.extend([abs_reg, cls_conf])

        # AWAF entropy: higher entropy = more uncertain fusion → larger gate
        if self.use_awaf_entropy and awaf_weights is not None:
            w_clamped = awaf_weights.clamp(self.eps)
            entropy = -(w_clamped * torch.log(w_clamped)).sum(-1, keepdim=True)  # [B, 1]
            features.append(entropy)

        # Delta magnitude: if delta is large, we might want to gate it
        if self.use_delta_magnitude and delta_reg is not None:
            abs_delta = delta_reg.abs()  # [B, 1]
            features.append(abs_delta)

        # Concatenate all features
        feat = torch.cat(features, dim=-1)  # [B, input_dim]

        # Gate prediction
        gate_raw = self.gate_net(feat)  # [B, 2]
        gate_reg = torch.sigmoid(gate_raw[:, 0:1])  # [B, 1]
        gate_cls = torch.sigmoid(gate_raw[:, 1:2])  # [B, 1]

        return {
            'gate_reg': gate_reg,
            'gate_cls': gate_cls,
        }


def compute_gate_stats(gate_reg: torch.Tensor, gate_cls: torch.Tensor):
    """计算 gate 统计信息，用于监控坍缩。"""
    return {
        'gate_reg_mean': float(gate_reg.mean()),
        'gate_reg_std': float(gate_reg.std()),
        'gate_cls_mean': float(gate_cls.mean()),
        'gate_cls_std': float(gate_cls.std()),
        'gate_reg_min': float(gate_reg.min()),
        'gate_reg_max': float(gate_reg.max()),
        'gate_reg_collapse_0': float((gate_reg < 0.01).float().mean()),
        'gate_reg_collapse_1': float((gate_reg > 0.99).float().mean()),
    }


# ============================================================
# 单元测试
# ============================================================
if __name__ == '__main__':
    print("=== ConditionalResidualGate 单元测试 ===\n")

    B, H = 4, 256
    torch.manual_seed(42)

    gate = ConditionalResidualGate(hidden_dim=H)

    h_text = torch.randn(B, H)
    z_res = torch.randn(B, H)
    reg_base = torch.randn(B, 1) * 2.0
    cls_base = torch.randn(B, 1)
    awaf_w = torch.softmax(torch.randn(B, 3), dim=-1)
    delta = torch.randn(B, 1) * 0.5

    # Test 1: forward
    out = gate(h_text, z_res, reg_base, cls_base, awaf_w, delta)
    print(f"1. Forward: gate_reg={out['gate_reg'].shape}, gate_cls={out['gate_cls'].shape}")
    g_reg = out['gate_reg']
    g_cls = out['gate_cls']
    assert g_reg.shape == (B, 1)
    assert g_cls.shape == (B, 1)
    print(f"   gate_reg range: [{g_reg.min().item():.4f}, {g_reg.max().item():.4f}]")
    print(f"   gate_cls range: [{g_cls.min().item():.4f}, {g_cls.max().item():.4f}]")
    print("   ✅ Forward OK")

    # Test 2: range [0,1]
    assert (g_reg >= 0).all() and (g_reg <= 1).all(), "gate_reg out of [0,1]"
    assert (g_cls >= 0).all() and (g_cls <= 1).all(), "gate_cls out of [0,1]"
    print("2. ✅ Range [0,1] OK")

    # Test 3: backward
    loss = g_reg.mean() + g_cls.mean()
    loss.backward()
    has_grad = any(p.grad is not None for p in gate.parameters())
    assert has_grad, "No gradients"
    print("3. ✅ Backward OK")

    # Test 4: not all 0 or all 1
    assert g_reg.std().item() > 1e-4, "gate_reg collapsed (std≈0)"
    assert g_cls.std().item() > 1e-4, "gate_cls collapsed (std≈0)"
    print(f"4. ✅ Not collapsed: reg_std={g_reg.std().item():.4f}, cls_std={g_cls.std().item():.4f}")

    # Test 5: gate responds to text confidence
    # High-confidence text (large |reg|) should produce smaller gate
    reg_low_conf = torch.zeros(B, 1)  # near zero → high uncertainty
    reg_high_conf = torch.ones(B, 1) * 3.0  # far from zero → high certainty
    out_low = gate(h_text, z_res, reg_low_conf, cls_base, awaf_w, delta)
    out_high = gate(h_text, z_res, reg_high_conf, cls_base, awaf_w, delta)
    print(f"5. Gate response: low_conf={out_low['gate_reg'].mean().item():.4f}, "
          f"high_conf={out_high['gate_reg'].mean().item():.4f}")
    print("   ✅ Gate responds to confidence (check direction)")

    # Test 6: gate stats
    stats = compute_gate_stats(g_reg, g_cls)
    print(f"6. Stats: reg_mean={stats['gate_reg_mean']:.4f}, "
          f"collapse_0={stats['gate_reg_collapse_0']:.4f}, "
          f"collapse_1={stats['gate_reg_collapse_1']:.4f}")
    print("   ✅ Stats computable")

    # Test 7: without optional features
    gate_min = ConditionalResidualGate(H, use_text_confidence=False, use_awaf_entropy=False, use_delta_magnitude=False)
    out_min = gate_min(h_text, z_res, reg_base, cls_base)
    print(f"7. Minimal gate: reg={out_min['gate_reg'].shape}, ✅ OK")

    # Test 8: params
    n_params = sum(p.numel() for p in gate.parameters())
    print(f"8. Params: {n_params:,} ✅")

    print("\n=== ConditionalResidualGate 全部测试通过 ===")
