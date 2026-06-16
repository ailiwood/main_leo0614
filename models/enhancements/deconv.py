"""
models/enhancements/deconv.py — Dynamic Feature Enhancer (通用视觉增强模块)

基于 DEConv 思路重写，去除硬编码 1024→32×32。
改为通用动态特征增强：1D temporal conv + channel mixing + residual gate。

挂载位置:
  vision feature → projection → DynamicFeatureEnhancer → vision sLSTM

T=1 边界说明 (P3B):
  当前 vision 为 T=1 clip-level 单向量，不是 image frame 序列。
  因此"空间卷积"没有意义，实际效果为"clip-level embedding 动态增强"。
  这必须在报告中明确标注，不伪装成真实时空卷积。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFeatureEnhancer(nn.Module):
    """
    通用动态特征增强模块。

    对 vision feature [B, T, D] 做：
      1. 1D depthwise temporal conv (捕获局部时序变化)
      2. Channel mixing (跨 channel 交互)
      3. Learnable residual gate (控制增强强度)

    T=1 时退化为纯 FC 增强。

    Args:
        dim: 特征维度 D
        kernel_size: 1D conv kernel size (T≥kernel_size 时有效)
        expand_ratio: channel mixing 扩展比例
        dropout: dropout rate
    """

    def __init__(self, dim: int, kernel_size: int = 3, expand_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        inner_dim = int(dim * expand_ratio)

        # 1D depthwise temporal conv (处理时序)
        self.temporal_conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim, bias=False
        )
        self.temporal_norm = nn.LayerNorm(dim)

        # Channel mixing (类似 DEConv 多分支融合的思想，但更通用)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )
        self.channel_norm = nn.LayerNorm(dim)

        # Learnable residual gate
        self.residual_gate = nn.Parameter(torch.zeros(1))  # 初始接近 0，即接近 identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] 或 [B, D]

        Returns:
            enhanced: same shape as input
        """
        input_is_2d = (x.dim() == 2)
        if input_is_2d:
            x = x.unsqueeze(1)  # [B, 1, D]

        B, T, D_in = x.shape

        # ---- Temporal conv (1D over time) ----
        # [B, T, D] → [B, D, T] → conv → [B, D, T] → [B, T, D]
        if T >= self.kernel_size:
            x_t = x.transpose(1, 2)  # [B, D, T]
            x_t = self.temporal_conv(x_t)  # [B, D, T]
            x_t = x_t.transpose(1, 2)  # [B, T, D]
            residual = x + self.temporal_norm(x_t)
        else:
            # T=1: skip temporal conv (kernel > sequence)
            residual = x

        # ---- Channel mixing ----
        enhanced = self.channel_mix(residual)  # [B, T, D]
        enhanced = self.channel_norm(enhanced)

        # ---- Residual gate ----
        gate = torch.sigmoid(self.residual_gate)
        output = residual + gate * enhanced

        if input_is_2d:
            output = output.squeeze(1)
        return output


# ============================================================
# 随机张量测试
# ============================================================
if __name__ == '__main__':
    print("=== DynamicFeatureEnhancer test ===\n")

    enhancer = DynamicFeatureEnhancer(dim=256)

    # Test [B, D]
    x_2d = torch.randn(4, 256)
    out_2d = enhancer(x_2d)
    print(f"  [B,D] input {list(x_2d.shape)} → output {list(out_2d.shape)}")
    assert out_2d.shape == x_2d.shape
    assert not torch.isnan(out_2d).any()

    # Test [B, T, D]
    x_3d = torch.randn(4, 5, 256)
    out_3d = enhancer(x_3d)
    print(f"  [B,T,D] input {list(x_3d.shape)} → output {list(out_3d.shape)}")
    assert out_3d.shape == x_3d.shape
    assert not torch.isnan(out_3d).any()

    # Backward test
    loss = out_3d.mean()
    loss.backward()
    print("  backward OK ✅")

    # Parameter count
    n = sum(p.numel() for p in enhancer.parameters())
    print(f"  params: {n:,}")

    print("\n  All tests passed ✅")
