"""
models/baselines/base_baseline.py — Baseline-Lite 基类

统一接口:
  输入: text, audio, vision, text_mask, audio_mask, vision_mask, label, sample_id
  输出: reg, cls_logits(可选), loss_terms, debug(可选)

必须支持: text_only, text_audio, text_audio_vision
"""
import torch.nn as nn
from typing import Dict, Optional


class BaseBaseline(nn.Module):
    """Baseline-lite 基类。所有 baseline 模型继承此类。"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.mode = config.get('mode', 'text_audio_vision')

    def forward(self, batch: Dict) -> Dict:
        raise NotImplementedError

    def compute_loss(self, output: Dict, batch: Dict) -> Dict:
        raise NotImplementedError

    @staticmethod
    def get_modalities(mode: str):
        """Return modality config given mode."""
        return {
            'use_text': mode in ('text_only', 'text_audio', 'text_audio_vision'),
            'use_audio': mode in ('audio_only', 'text_audio', 'text_audio_vision', 'av_only'),
            'use_vision': mode in ('vision_only', 'text_audio_vision', 'av_only'),
        }
