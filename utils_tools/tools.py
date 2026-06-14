"""
工具函数模块
"""

import torch

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dict_to_str(d):
    """将字典转换为字符串"""
    result = []
    for k, v in d.items():
        if isinstance(v, float):
            result.append(f"{k}: {v:.4f}")
        else:
            result.append(f"{k}: {v}")
    return ', '.join(result)


# ===== Compatibility for legacy BERT-style modules =====
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from transformers.activations import ACT2FN
except Exception:
    def gelu(x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(x):
        return x * torch.sigmoid(x)

    ACT2FN = {
        "gelu": gelu,
        "relu": F.relu,
        "swish": swish,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }

BertLayerNorm = nn.LayerNorm