"""
models/deeptext_xlstm_awaf_residual.py — DeepText-xLSTM-AWAF Residual 主模型

P5C 新架构 (D031):
  Text branch:     DeBERTa → AttnPool → DeepMLP → reg_text_base, cls_text_base
  Audio branch:    wav2vec2 → sLSTM → AttnPool → h_a
  Vision branch:   CLIP frames → sLSTM → AttnPool → h_v
  Text residual:   DeBERTa → AttnPool (NO sLSTM) → h_t_residual
  AWAF residual:   AWAF(h_t_residual, h_a, h_v) → z_residual, awaf_weights
                    → delta_reg, delta_cls
  Final:           reg = reg_text_base + δ_reg * delta_reg
                   cls = cls_text_base + δ_cls * delta_cls

核心设计原则:
  1. Text branch 是主判别分支 — 使用 DeepMLP，不使用 sLSTM
  2. xLSTM 仅用于 audio / vision 时序残差增强
  3. AWAF 负责样本级 residual correction 权重
  4. Final = text_base + λ * residual_correction
  P5D 新增:
  - ConditionalResidualGate: g(x) * λ * delta (条件残差修正)

Ablation 支持:
  - no_residual:           仅 text branch (baseline)
  - no_audio:              残差分支去掉 audio
  - no_vision:             残差分支去掉 vision
  - no_awaf_mean_residual: AWAF → mean pooling 在 residual branch
  - text_slstm_on:         在 text residual branch 启用 sLSTM (消融验证)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.encoders.slstm import SLSTMEncoder
from models.fusion.awaf import AdaptiveWeightedAttentionFusion
from models.pooling.attention_pooling import MaskedAttentionPooling
from models.heads import RegressionHead, ClassificationHead, UnimodalHeads
from models.modules.conditional_residual_gate import ConditionalResidualGate


class DeepTextXLSTMAWAFResidual(nn.Module):
    """
    DeepText-xLSTM-AWAF Residual 主模型。

    输入:
        text [B, Tt, Dt], text_mask [B, Tt]
        audio [B, Ta, Da], audio_mask [B, Ta]
        vision [B, Tv, Dv], vision_mask [B, Tv]

    输出 dict:
        reg, cls, aux, awaf_weights,
        reg_text_base, cls_text_base, delta_reg, delta_cls,
        delta_scale_reg, delta_scale_cls,
        h_text_base, h_t_residual, h_a, h_v
    """

    def __init__(
        self,
        # --- 维度配置 ---
        text_dim: int = 1024,
        audio_dim: int = 768,
        vision_dim: int = 768,
        hidden_dim: int = 256,
        # --- Text branch ---
        text_mlp_hidden: int = 512,
        text_mlp_layers: int = 3,
        text_mlp_dropout: float = 0.3,
        # --- sLSTM (audio/vision only) ---
        slstm_num_layers: int = 1,
        slstm_dropout: float = 0.3,
        slstm_pooling: str = 'masked_mean',
        # --- AWAF ---
        awaf_fusion_mode: str = 'awaf',
        awaf_tau_init: float = 1.0,
        awaf_dropout: float = 0.1,
        awaf_modality_dropout: bool = True,
        awaf_modality_dropout_prob: float = 0.1,
        # --- Residual ---
        delta_scale_init: float = 0.1,
        # --- Heads ---
        head_dropout: float = 0.3,
        # --- Ablation ---
        ablation: str = 'none',  # 'none'|'no_residual'|'no_audio'|'no_vision'|'no_awaf_mean_residual'|'text_slstm_on'
        # --- Aux ---
        use_aux_heads: bool = False,
        # --- P5D: Conditional Residual Gate ---
        use_residual_gate: bool = False,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.1,
        gate_init_bias: float = -1.0,
        gate_use_text_confidence: bool = True,
        gate_use_awaf_entropy: bool = True,
        gate_use_delta_magnitude: bool = True,
        # --- Misc ---
        eps: float = 1e-8,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.ablation = ablation
        self.eps = eps

        # ============================================================
        # 1. Text Branch (主判别分支, NO sLSTM)
        # ============================================================
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(slstm_dropout * 0.5),
        )
        self.text_pool = MaskedAttentionPooling(hidden_dim, dropout=head_dropout * 0.5)

        # DeepMLP: 多层 MLP + LayerNorm + GELU
        text_mlp_layers_list = []
        in_dim = hidden_dim
        for i in range(text_mlp_layers):
            out_dim = text_mlp_hidden if i < text_mlp_layers - 1 else hidden_dim
            text_mlp_layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(text_mlp_dropout),
            ])
            in_dim = out_dim
        self.text_mlp = nn.Sequential(*text_mlp_layers_list)

        # Text base heads
        self.reg_text_head = RegressionHead(hidden_dim, hidden_dim // 2, head_dropout)
        self.cls_text_head = ClassificationHead(hidden_dim, hidden_dim // 2, head_dropout)

        # ============================================================
        # 2. Audio Branch (残差增强分支, WITH sLSTM)
        # ============================================================
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(slstm_dropout * 0.5),
        )
        self.audio_slstm = SLSTMEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=slstm_num_layers,
            dropout=slstm_dropout,
            bidirectional=False,
            pooling=slstm_pooling,
        )
        self.audio_pool = MaskedAttentionPooling(hidden_dim, dropout=head_dropout * 0.5)

        # ============================================================
        # 3. Vision Branch (残差增强分支, WITH sLSTM)
        # ============================================================
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(slstm_dropout * 0.5),
        )
        self.vision_slstm = SLSTMEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=slstm_num_layers,
            dropout=slstm_dropout,
            bidirectional=False,
            pooling=slstm_pooling,
        )
        self.vision_pool = MaskedAttentionPooling(hidden_dim, dropout=head_dropout * 0.5)

        # ============================================================
        # 4. Text Residual Branch (轻量, 默认 NO sLSTM)
        # ============================================================
        self.text_residual_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(slstm_dropout * 0.5),
        )
        self.text_residual_pool = MaskedAttentionPooling(hidden_dim, dropout=head_dropout * 0.5)

        # 可选 text sLSTM (仅 text_slstm_on 消融时启用)
        self._text_slstm_on = (ablation == 'text_slstm_on')
        if self._text_slstm_on:
            self.text_residual_slstm = SLSTMEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=slstm_num_layers,
                dropout=slstm_dropout,
                bidirectional=False,
                pooling=slstm_pooling,
            )

        # ============================================================
        # 5. AWAF Residual Fusion
        # ============================================================
        # 根据消融模式选择 awaf 模式
        if ablation == 'no_awaf_mean_residual':
            awaf_mode = 'mean'
        else:
            awaf_mode = awaf_fusion_mode

        self.awaf = AdaptiveWeightedAttentionFusion(
            hidden_dim=hidden_dim,
            fusion_mode=awaf_mode,
            tau_init=awaf_tau_init,
            dropout=awaf_dropout,
            use_modality_dropout=awaf_modality_dropout,
            modality_dropout_prob=awaf_modality_dropout_prob,
        )

        # delta heads: z_residual → delta_reg, delta_cls
        self.delta_reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.delta_cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 可学习 residual scale
        self.delta_scale_reg = nn.Parameter(torch.tensor(delta_scale_init))
        self.delta_scale_cls = nn.Parameter(torch.tensor(delta_scale_init))

        # ============================================================
        # 6. Auxiliary Heads (可选)
        # ============================================================
        self.use_residual_gate = use_residual_gate
        self.use_aux_heads = use_aux_heads
        if use_residual_gate:
            self.residual_gate = ConditionalResidualGate(
                hidden_dim=hidden_dim,
                gate_hidden_dim=gate_hidden_dim,
                dropout=gate_dropout,
                init_bias=gate_init_bias,
                use_text_confidence=gate_use_text_confidence,
                use_awaf_entropy=gate_use_awaf_entropy,
                use_delta_magnitude=gate_use_delta_magnitude,
            )
        if use_aux_heads:
            self.aux_heads = UnimodalHeads(
                in_dim=hidden_dim,
                reg_hidden=hidden_dim // 4,
                cls_hidden=hidden_dim // 4,
                dropout=head_dropout,
                modalities=['A', 'V'],  # only audio/vision aux heads
            )

        self._log_init()

    def _log_init(self):
        """记录初始化信息（供外部 logger 使用）。"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._init_info = {
            'total_params': n_params,
            'trainable_params': n_trainable,
            'text_dim': self.text_dim,
            'audio_dim': self.audio_dim,
            'vision_dim': self.vision_dim,
            'hidden_dim': self.hidden_dim,
            'ablation': self.ablation,
            'text_slstm_on': self._text_slstm_on,
            'delta_scale_reg_init': self.delta_scale_reg.item(),
            'delta_scale_cls_init': self.delta_scale_cls.item(),
        }

    def _process_text_branch(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Text branch: projection → attention pool → DeepMLP → reg/cls_base."""
        h = self.text_proj(text)                # [B, Tt, H]
        h_pooled, _ = self.text_pool(h, text_mask)  # [B, H]
        h_base = self.text_mlp(h_pooled)         # [B, H]
        reg_base = self.reg_text_head(h_base)    # [B, 1]
        cls_base = self.cls_text_head(h_base)    # [B, 1]
        return {
            'h_text_base': h_base,
            'reg_text_base': reg_base,
            'cls_text_base': cls_base,
        }

    def _process_branch_with_slstm(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        proj: nn.Module,
        slstm: nn.Module,
        pool: nn.Module,
        branch_name: str,
    ) -> torch.Tensor:
        """通用 xLSTM 分支: projection → sLSTM → attention pool → h."""
        h = proj(x)                              # [B, T, H]
        slstm_out = slstm(h, mask)               # dict with 'H', 'pooled'
        h_seq = slstm_out['H']                   # [B, T, H]
        h_pooled, _ = pool(h_seq, mask)          # [B, H]
        return h_pooled

    def _process_audio_branch(
        self,
        audio: torch.Tensor,
        audio_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._process_branch_with_slstm(
            audio, audio_mask,
            self.audio_proj, self.audio_slstm, self.audio_pool,
            'audio',
        )

    def _process_vision_branch(
        self,
        vision: torch.Tensor,
        vision_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._process_branch_with_slstm(
            vision, vision_mask,
            self.vision_proj, self.vision_slstm, self.vision_pool,
            'vision',
        )

    def _process_text_residual_branch(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Text residual branch: projection → [(optional sLSTM)] → attention pool → h_t_residual."""
        h = self.text_residual_proj(text)            # [B, Tt, H]

        if self._text_slstm_on:
            # 消融模式：启用 text sLSTM
            slstm_out = self.text_residual_slstm(h, text_mask)
            h_seq = slstm_out['H']                   # [B, Tt, H]
        else:
            h_seq = h                                # [B, Tt, H] (直接使用投影后特征)

        h_pooled, _ = self.text_residual_pool(h_seq, text_mask)  # [B, H]
        return h_pooled

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            text:  [B, Tt, Dt]
            audio: [B, Ta, Da]
            vision: [B, Tv, Dv]
            text_mask:  [B, Tt] (1=有效, 0=padding)
            audio_mask: [B, Ta]
            vision_mask: [B, Tv]
            return_all: 是否返回全部中间表示

        Returns:
            dict 至少包含: reg, cls, awaf_weights
        """
        B = text.size(0)

        # --- 1. Text Branch (主判别) ---
        text_out = self._process_text_branch(text, text_mask)
        reg_text_base = text_out['reg_text_base']  # [B, 1]
        cls_text_base = text_out['cls_text_base']  # [B, 1]
        h_text_base = text_out['h_text_base']      # [B, H]

        # --- 2. Audio Branch (残差增强) ---
        if self.ablation == 'no_audio':
            h_a = torch.zeros(B, self.hidden_dim, device=text.device, dtype=text.dtype)
        else:
            h_a = self._process_audio_branch(audio, audio_mask)  # [B, H]

        # --- 3. Vision Branch (残差增强) ---
        if self.ablation == 'no_vision':
            h_v = torch.zeros(B, self.hidden_dim, device=text.device, dtype=text.dtype)
        else:
            h_v = self._process_vision_branch(vision, vision_mask)  # [B, H]

        # --- 4. Text Residual Branch ---
        h_t_residual = self._process_text_residual_branch(text, text_mask)  # [B, H]

        # --- 5. AWAF Residual Fusion ---
        if self.ablation == 'no_residual':
            # 不使用残差：直接返回 text base
            reg = reg_text_base
            cls = cls_text_base
            awaf_weights = torch.zeros(B, 3, device=text.device, dtype=text.dtype)
            awaf_weights[:, 0] = 1.0  # text weight = 1
            delta_reg = torch.zeros_like(reg_text_base)
            delta_cls = torch.zeros_like(cls_text_base)
            z_residual = h_text_base
        else:
            awaf_out = self.awaf(h_t_residual, h_a, h_v)  # {Z, weights}
            z_residual = awaf_out['Z']                      # [B, H]
            awaf_weights = awaf_out['weights']              # [B, 3]

            delta_reg = self.delta_reg_head(z_residual)     # [B, 1]
            delta_cls = self.delta_cls_head(z_residual)     # [B, 1]

            # P5D: Conditional Residual Gate
            residual_gate_reg = None
            residual_gate_cls = None
            if self.use_residual_gate and self.ablation != 'no_residual':
                gate_out = self.residual_gate(
                    h_text_base=h_text_base,
                    z_residual=z_residual,
                    reg_text_base=reg_text_base,
                    cls_text_base=cls_text_base,
                    awaf_weights=awaf_weights,
                    delta_reg=delta_reg,
                )
                residual_gate_reg = gate_out['gate_reg']  # [B, 1]
                residual_gate_cls = gate_out['gate_cls']  # [B, 1]
                # Conditional residual correction
                reg = reg_text_base + residual_gate_reg * self.delta_scale_reg * delta_reg
                cls = cls_text_base + residual_gate_cls * self.delta_scale_cls * delta_cls
            else:
                # Original residual correction (unconditional)
                reg = reg_text_base + self.delta_scale_reg * delta_reg
                cls = cls_text_base + self.delta_scale_cls * delta_cls

        # --- 6. Auxiliary Heads ---
        aux = {}
        if self.use_aux_heads and self.ablation != 'no_residual':
            aux_input = {
                'A': h_a if self.ablation != 'no_audio' else h_text_base,
                'V': h_v if self.ablation != 'no_vision' else h_text_base,
            }
            aux = self.aux_heads(aux_input)

        # --- 7. NaN 防护 ---
        if torch.isnan(reg).any():
            reg = torch.where(torch.isnan(reg), reg_text_base, reg)
        if torch.isnan(cls).any():
            cls = torch.where(torch.isnan(cls), cls_text_base, cls)

        result = {
            'reg': reg,
            'cls': cls,
            'aux': aux,
            'awaf_weights': awaf_weights,
            'reg_text_base': reg_text_base,
            'cls_text_base': cls_text_base,
            'delta_reg': delta_reg if self.ablation != 'no_residual' else delta_reg,
            'delta_cls': delta_cls if self.ablation != 'no_residual' else delta_cls,
            'delta_scale_reg': self.delta_scale_reg,
            'delta_scale_cls': self.delta_scale_cls,
        }
        # P5D: Conditional Residual Gate outputs
        if residual_gate_reg is not None:
            result['residual_gate_reg'] = residual_gate_reg
        if residual_gate_cls is not None:
            result['residual_gate_cls'] = residual_gate_cls

        if return_all:
            result.update({
                'h_text_base': h_text_base,
                'h_t_residual': h_t_residual,
                'h_a': h_a,
                'h_v': h_v,
                'z_residual': z_residual if self.ablation != 'no_residual' else h_text_base,
            })

        return result


# ============================================================
# 随机张量 forward test
# ============================================================
if __name__ == '__main__':
    print("=== DeepText-xLSTM-AWAF Residual 随机张量 forward test ===\n")

    B, Tt, Ta, Tv = 2, 8, 10, 5
    Dt, Da, Dv = 1024, 768, 768
    H = 256
    device = 'cpu'

    text = torch.randn(B, Tt, Dt)
    audio = torch.randn(B, Ta, Da)
    vision = torch.randn(B, Tv, Dv)
    text_mask = torch.ones(B, Tt)
    audio_mask = torch.ones(B, Ta)
    vision_mask = torch.ones(B, Tv)

    # Padding test
    text_mask[0, -2:] = 0
    audio_mask[0, -3:] = 0

    ablations = ['none', 'no_residual', 'no_audio', 'no_vision', 'no_awaf_mean_residual', 'text_slstm_on']

    for ab in ablations:
        print(f"\n--- Ablation: {ab} ---")
        model = DeepTextXLSTMAWAFResidual(
            text_dim=Dt, audio_dim=Da, vision_dim=Dv, hidden_dim=H,
            ablation=ab,
        )
        model.eval()
        out = model(text, audio, vision, text_mask, audio_mask, vision_mask)

        print(f"  reg:           {out['reg'].shape}  range [{out['reg'].min().item():.4f}, {out['reg'].max().item():.4f}]")
        print(f"  cls:           {out['cls'].shape}")
        print(f"  awaf_weights:  {out['awaf_weights'].shape}  sum(w) max_dev={((out['awaf_weights'].sum(-1) - 1).abs().max().item()):.2e}")
        print(f"  reg_text_base: {out['reg_text_base'].shape}")
        print(f"  delta_reg:     {out['delta_reg'].shape}")
        print(f"  delta_scale:   reg={out['delta_scale_reg'].item():.4f} cls={out['delta_scale_cls'].item():.4f}")
        print(f"  params:        {model._init_info['total_params']:,}")

        # NaN check
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                print(f"  ❌ NaN in {k}!")
                break
        else:
            print(f"  ✅ No NaN")

        # Backward test
        loss = out['reg'].mean() + out['cls'].mean()
        loss.backward()
        print(f"  ✅ Backward OK (loss={loss.item():.4f})")

    # --- MOSEI SDK dims test ---
    print(f"\n--- MOSEI SDK dims (300/74/35) ---")
    model_mosei = DeepTextXLSTMAWAFResidual(
        text_dim=300, audio_dim=74, vision_dim=35, hidden_dim=H,
    )
    model_mosei.eval()
    text_m = torch.randn(B, 5, 300)
    audio_m = torch.randn(B, 8, 74)
    vision_m = torch.randn(B, 3, 35)
    out_m = model_mosei(text_m, audio_m, vision_m)
    print(f"  reg: {out_m['reg'].shape}, cls: {out_m['cls'].shape}")
    print(f"  ✅ MOSEI SDK dims OK")
    print(f"  params: {model_mosei._init_info['total_params']:,}")

    # --- no_residual: verify reg == reg_text_base ---
    print(f"\n--- Verify no_residual: reg == reg_text_base ---")
    model_nr = DeepTextXLSTMAWAFResidual(text_dim=Dt, audio_dim=Da, vision_dim=Dv, hidden_dim=H, ablation='no_residual')
    model_nr.eval()
    out_nr = model_nr(text, audio, vision, text_mask, audio_mask, vision_mask)
    reg_diff = (out_nr['reg'] - out_nr['reg_text_base']).abs().max().item()
    print(f"  |reg - reg_text_base|_max = {reg_diff:.2e} (期望 0.0)")
    assert reg_diff < 1e-6, f"no_residual: reg should equal reg_text_base, got diff={reg_diff}"
    print(f"  ✅ no_residual verified")

    # --- AWAF sum=1 ---
    print(f"\n--- AWAF sum(w)=1 ---")
    model_awaf = DeepTextXLSTMAWAFResidual(text_dim=Dt, audio_dim=Da, vision_dim=Dv, hidden_dim=H, ablation='none')
    model_awaf.eval()
    out_awaf = model_awaf(text, audio, vision, text_mask, audio_mask, vision_mask)
    w_sum_dev = (out_awaf['awaf_weights'].sum(-1) - 1).abs().max().item()
    print(f"  max|sum(w)-1| = {w_sum_dev:.2e}")
    assert w_sum_dev < 1e-4, f"AWAF sum(w) should be 1.0"
    print(f"  ✅ AWAF sum(w)=1")

    # --- delta_scale learnable ---
    print(f"\n--- delta_scale learnable ---")
    model_ds = DeepTextXLSTMAWAFResidual(text_dim=Dt, audio_dim=Da, vision_dim=Dv, hidden_dim=H)
    ds_before = model_ds.delta_scale_reg.item()
    opt = torch.optim.SGD(model_ds.parameters(), lr=1.0)
    out_ds = model_ds(text, audio, vision, text_mask, audio_mask, vision_mask)
    loss_ds = out_ds['reg'].mean()
    loss_ds.backward()
    opt.step()
    ds_after = model_ds.delta_scale_reg.item()
    print(f"  delta_scale_reg: {ds_before:.4f} → {ds_after:.4f} (changed={ds_before != ds_after})")
    assert ds_before != ds_after, "delta_scale should be learnable"

    print(f"\n=== All tests passed ===")
