"""
models/heads.py — 多任务预测头

包含:
  - RegressionHead: 连续情感强度回归 [B] → pred ∈ R
  - ClassificationHead: 二分类 [B] → logit
  - 单模态辅助头: 每个模态独立的 reg/cls 头
"""
import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """情感强度回归头: 输出连续值 ([-3, 3] 范围)"""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d]
        Returns:
            pred: [B, 1] 回归预测值
        """
        return self.net(x)


class ClassificationHead(nn.Module):
    """二分类头: 输出 logit (positive if ≥ 0)"""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d]
        Returns:
            logit: [B, 1] 二分类 logit (≥0 → positive)
        """
        return self.net(x)


class UnimodalHeads(nn.Module):
    """
    单模态辅助头。

    对文本、音频、视觉分别输出回归和分类预测。
    用于辅助训练，不参与推理。
    """

    def __init__(
        self,
        in_dim: int,
        reg_hidden: int = 64,
        cls_hidden: int = 64,
        dropout: float = 0.3,
        modalities: list = None,
    ):
        """
        Args:
            in_dim: 单模态特征维度
            reg_hidden: 回归头 hidden
            cls_hidden: 分类头 hidden
            dropout: dropout rate
            modalities: 哪些模态有辅助头，默认 ['T', 'A', 'V']
        """
        super().__init__()
        self.modalities = modalities or ['T', 'A', 'V']

        for mod in self.modalities:
            self.add_module(f'reg_{mod}', RegressionHead(in_dim, reg_hidden, dropout))
            self.add_module(f'cls_{mod}', ClassificationHead(in_dim, cls_hidden, dropout))

    def forward(self, h_dict: dict) -> dict:
        """
        Args:
            h_dict: {'T': [B,d], 'A': [B,d], 'V': [B,d]} 或类似

        Returns:
            {'T_reg': [B,1], 'T_cls': [B,1], 'A_reg': [B,1], ...}
        """
        results = {}
        for mod in self.modalities:
            h = h_dict.get(mod, h_dict.get(mod.lower()))
            if h is not None:
                results[f'{mod}_reg'] = getattr(self, f'reg_{mod}')(h)
                results[f'{mod}_cls'] = getattr(self, f'cls_{mod}')(h)
        return results
