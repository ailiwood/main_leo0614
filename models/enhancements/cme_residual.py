"""
models/enhancements/cme_residual.py — Gated/Residual CME

保守版 CME：加入可学习 alpha gate 控制 CME 的影响强度。
alpha 初始接近 0 → CME 初始接近 identity → 避免破坏 C0 表示。

纯 PyTorch 自实现，不依赖 AGPL 代码。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCME(nn.Module):
    """
    Residual/Gated Cross-Modal Enhancement.

    每个模态 m:
      c_m = CrossAttend(h_m, other_modals)  ← [B, D]
      enhanced = LayerNorm(h_m + alpha_m * c_m)

    alpha_m: per-modality learnable scalar, init 0.1

    Args:
        dim: 隐藏维度
        num_heads: 注意力头数
        alpha_init: alpha 初始值 (默认 0.1，接近 identity)
        dropout: dropout rate
    """
    def __init__(self, dim: int, num_heads: int = 4, alpha_init: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        # 共享投影
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # 可学习 alpha gate (per-modality)
        self.alpha_t = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_a = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_v = nn.Parameter(torch.tensor(alpha_init))

        # LayerNorm
        self.norm_t = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _cross_attend(self, h_query, h_other1, h_other2):
        B, D = h_query.shape
        kv = torch.stack([h_other1, h_other2], dim=1)  # [B, 2, D]

        q = self.q_proj(h_query).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(kv).view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5
        attn = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, v).squeeze(2).reshape(B, D)
        return self.out_proj(ctx)

    def forward(self, h_t, h_a, h_v):
        """
        Returns enhanced_h_t, enhanced_h_a, enhanced_h_v
        """
        c_t = self._cross_attend(h_t, h_a, h_v)
        c_a = self._cross_attend(h_a, h_t, h_v)
        c_v = self._cross_attend(h_v, h_t, h_a)

        eh_t = self.norm_t(h_t + self.alpha_t * self.dropout(c_t))
        eh_a = self.norm_a(h_a + self.alpha_a * self.dropout(c_a))
        eh_v = self.norm_v(h_v + self.alpha_v * self.dropout(c_v))

        return eh_t, eh_a, eh_v


if __name__ == '__main__':
    print("=== GatedCME test ===")
    B, D = 4, 256
    cme = GatedCME(D, num_heads=4)
    h_t, h_a, h_v = torch.randn(B, D), torch.randn(B, D), torch.randn(B, D)
    eh_t, eh_a, eh_v = cme(h_t, h_a, h_v)
    for n, t in [('eh_t', eh_t), ('eh_a', eh_a), ('eh_v', eh_v)]:
        assert t.shape == (B, D); assert not torch.isnan(t).any()
    loss = eh_t.mean() + eh_a.mean() + eh_v.mean(); loss.backward()
    print(f"  Shapes OK, backward OK. alpha=[{cme.alpha_t.item():.2f}, {cme.alpha_a.item():.2f}, {cme.alpha_v.item():.2f}]")
    print(f"  Params: {sum(p.numel() for p in cme.parameters()):,}")
    print("  All tests passed")
