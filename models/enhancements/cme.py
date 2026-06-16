"""
models/enhancements/cme.py — LightweightCME: Cross-Modal Enhancement

轻量跨模态交互模块。纯 PyTorch 自实现，不依赖 AGPL 代码。

核心: 每个模态 attend 其他两个模态，增强自身表示。

挂载位置:
  sLSTM 输出 h_t/h_a/h_v → LightweightCME → enhanced h_t/h_a/h_v → AWAF

CME 只做跨模态交互增强，AWAF 仍然负责最终样本级三模态权重融合。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCME(nn.Module):
    """
    轻量跨模态增强模块。

    对每个模态 m:
      - 以 h_m 为 query
      - 以其他两个模态的拼接为 key/value
      - Multi-head cross-attention → context c_m
      - enhanced_h_m = LayerNorm(h_m + c_m)

    不与 AWAF context enhancement 重复:
      - CME: raw h_t/h_a/h_v → cross-attend → enhanced h
      - AWAF context: enhanced h → attend other → 生成样本级权重

    Args:
        dim: 隐藏维度
        num_heads: 注意力头数
        dropout: dropout rate
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.head_dim = dim // num_heads

        # 共享投影（所有模态→query/key/value）
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(dim, dim)

        # 层归一化
        self.norm_t = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def _cross_attend(self, h_query: torch.Tensor,
                      h_other1: torch.Tensor, h_other2: torch.Tensor) -> torch.Tensor:
        """
        单模态跨模态 attention。

        Args:
            h_query:  [B, D] 当前模态 (作为 query)
            h_other1: [B, D] 另一个模态
            h_other2: [B, D] 第三个模态

        Returns:
            context: [B, D] 跨模态上下文
        """
        B, D = h_query.shape

        # Key/Value: 拼接其他两个模态 [B, 2, D]
        kv = torch.stack([h_other1, h_other2], dim=1)  # [B, 2, D]

        # 投影
        q = self.q_proj(h_query).view(B, self.num_heads, self.head_dim)  # [B, NH, HD]
        k = self.k_proj(kv).view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)  # [B, NH, 2, HD]
        v = self.v_proj(kv).view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)  # [B, NH, 2, HD]

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / scale  # [B, NH, 1, 2]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v).squeeze(2)  # [B, NH, HD]
        context = context.reshape(B, D)  # [B, D]
        context = self.out_proj(context)

        return context

    def forward(self, h_t: torch.Tensor, h_a: torch.Tensor, h_v: torch.Tensor):
        """
        Args:
            h_t, h_a, h_v: 各 [B, D] — sLSTM 池化后摘要

        Returns:
            enhanced_h_t, enhanced_h_a, enhanced_h_v: 各 [B, D]
        """
        # Text: attend to Audio + Vision
        c_t = self._cross_attend(h_t, h_a, h_v)
        eh_t = self.norm_t(h_t + c_t)

        # Audio: attend to Text + Vision
        c_a = self._cross_attend(h_a, h_t, h_v)
        eh_a = self.norm_a(h_a + c_a)

        # Vision: attend to Text + Audio
        c_v = self._cross_attend(h_v, h_t, h_a)
        eh_v = self.norm_v(h_v + c_v)

        return eh_t, eh_a, eh_v


# ============================================================
# 随机张量测试
# ============================================================
if __name__ == '__main__':
    print("=== LightweightCME test ===\n")

    B, D = 4, 256
    cme = LightweightCME(dim=D, num_heads=4)

    h_t = torch.randn(B, D)
    h_a = torch.randn(B, D)
    h_v = torch.randn(B, D)

    eh_t, eh_a, eh_v = cme(h_t, h_a, h_v)

    for name, tensor in [('eh_t', eh_t), ('eh_a', eh_a), ('eh_v', eh_v)]:
        print(f"  {name}: {list(tensor.shape)}")
        assert tensor.shape == (B, D)
        assert not torch.isnan(tensor).any(), f"NaN in {name}"

    # Backward test
    loss = eh_t.mean() + eh_a.mean() + eh_v.mean()
    loss.backward()
    print("  backward OK ✅")

    # Verify that different inputs produce different outputs
    eh_t2, _, _ = cme(h_t + 1.0, h_a, h_v)
    diff = (eh_t - eh_t2).abs().max().item()
    print(f"  sensitivity check: max_diff={diff:.4f} (should be > 0)")
    assert diff > 1e-3, "CME not sensitive to input change!"

    n = sum(p.numel() for p in cme.parameters())
    print(f"  params: {n:,}")
    print("\n  All tests passed ✅")
