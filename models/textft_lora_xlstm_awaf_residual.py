"""
models/textft_lora_xlstm_awaf_residual.py — P6H-R TextFT LoRA xLSTM AWAF Residual 主模型

完整三模态主模型: RoBERTa-large + Minimal LoRA + sLSTM + AWAF + UGR + Delta Residual

P6H-R 修复开关 (均通过 config 控制):
  use_modal_layernorm   : AWAF 前各模态独立 LayerNorm (防权重崩溃)
  tau_init              : AWAF 温度 (默认 3.0, 原 1.0)
  awaf_uniform_mix      : 权重均匀混合 ε (默认 0.0)
  lambda_awaf_entropy   : AWAF entropy regularization (默认 0.0)
  delta_scale_init      : Delta 修正幅度 (默认 0.2, 原 0.02)
  max_delta             : Delta 上限 (默认 0.5)
  gate_init_bias        : Gate 最后一层 bias (默认 +2.0, 原 -1.5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from .modules.minimal_lora import apply_lora_to_roberta, mark_only_lora_as_trainable
from .encoders.slstm import SLSTMEncoder
from .pooling.attention_pooling import MaskedAttentionPooling
from .fusion.awaf import AdaptiveWeightedAttentionFusion
from .modules.uncertainty_residual_gate import UncertaintyGuidedResidualGate


@dataclass
class TextFTLoRAConfig:
    """P6H-R 主模型配置。"""
    # --- Text ---
    text_model_name: str = "roberta-large"
    text_hidden_dim: int = 1024          # RoBERTa-large hidden
    text_mlp_hidden: int = 512
    text_dropout: float = 0.1

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = ('query', 'value')

    # --- Common ---
    hidden_dim: int = 256                # 统一投影维度 H
    audio_input_dim: int = 768
    vision_input_dim: int = 768

    # --- sLSTM ---
    slstm_num_layers: int = 1
    slstm_dropout: float = 0.2
    slstm_bidirectional: bool = False

    # --- AWAF ---
    awaf_fusion_mode: str = "awaf"
    tau_init: float = 3.0                # [P6H-R] 原 1.0
    awaf_dropout: float = 0.1
    use_modality_dropout: bool = True
    modality_dropout_prob: float = 0.1
    use_modal_layernorm: bool = True     # [P6H-R]
    awaf_uniform_mix: float = 0.0        # [P6H-R]
    lambda_awaf_entropy: float = 0.0     # [P6H-R]

    # --- Gate ---
    use_uncertainty_gate: bool = True
    gate_hidden_dim: int = 128
    gate_dropout: float = 0.1
    gate_init_bias: float = 2.0          # [P6H-R] 原 -1.5
    gate_margin_init: float = 0.75
    gate_temperature_init: float = 0.35

    # --- Delta ---
    use_delta_experts: bool = True
    use_bounded_delta: bool = True
    max_delta: float = 0.5
    delta_scale_init: float = 0.2        # [P6H-R] 原 0.02

    # --- Training hints ---
    device: str = "cuda"


class TextFTLoRAXLSTMAWAFResidual(nn.Module):
    """
    P6H-R 完整三模态主模型。

    数据流:
      Text:   RoBERTa-large + LoRA → CLS → TextMLP → h_t
      Audio:  Frozen 768d → Linear(768,H) → sLSTM → MaskedAttnPool → h_a
      Vision: Frozen 768d → Linear(768,H) → sLSTM → MaskedAttnPool → h_v
      Fusion: AWAF(h_t,h_a,h_v) → Z + weights
      Delta:  per-modality delta experts → bounded_delta
      Gate:   UGR(h_t, Z, ...) → gate_reg, gate_cls
      Output: reg = text_base_reg + gate_reg * dsr * delta_reg
              cls = text_base_cls + gate_cls * dsc * delta_cls
    """

    def __init__(self, config: TextFTLoRAConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim
        D = config.text_hidden_dim
        DEVICE = config.device

        # ============================================================
        # 1. RoBERTa + Minimal LoRA
        # ============================================================
        from transformers import AutoModel
        self.roberta = AutoModel.from_pretrained(config.text_model_name)
        self.roberta = apply_lora_to_roberta(
            self.roberta,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_patterns=list(config.lora_targets),
        )
        mark_only_lora_as_trainable(self.roberta)

        # ============================================================
        # 2. Text MLP: RoBERTa CLS (1024d) → 512 → 256
        # ============================================================
        self.text_mlp = nn.Sequential(
            nn.Linear(D, config.text_mlp_hidden),
            nn.LayerNorm(config.text_mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.text_dropout),
            nn.Linear(config.text_mlp_hidden, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(config.text_dropout),
        )
        self.reg_head_text = nn.Linear(H, 1)  # text_base regression
        self.cls_head_text = nn.Linear(H, 1)  # text_base classification proxy

        # ============================================================
        # 3. Audio branch (frozen 768d features → sLSTM)
        # ============================================================
        self.audio_proj = nn.Sequential(
            nn.Linear(config.audio_input_dim, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(config.slstm_dropout),
        )
        self.audio_slstm = SLSTMEncoder(
            H, H, config.slstm_num_layers,
            dropout=config.slstm_dropout,
            bidirectional=config.slstm_bidirectional,
            pooling='masked_mean',
        )
        self.audio_pool = MaskedAttentionPooling(H, dropout=config.slstm_dropout)

        # ============================================================
        # 4. Vision branch (frozen 768d features → sLSTM)
        # ============================================================
        self.vision_proj = nn.Sequential(
            nn.Linear(config.vision_input_dim, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(config.slstm_dropout),
        )
        self.vision_slstm = SLSTMEncoder(
            H, H, config.slstm_num_layers,
            dropout=config.slstm_dropout,
            bidirectional=config.slstm_bidirectional,
            pooling='masked_mean',
        )
        self.vision_pool = MaskedAttentionPooling(H, dropout=config.slstm_dropout)

        # ============================================================
        # 5. AWAF (with P6H-R repair switches)
        # ============================================================
        self.awaf = AdaptiveWeightedAttentionFusion(
            hidden_dim=H,
            fusion_mode=config.awaf_fusion_mode,
            tau_init=config.tau_init,
            dropout=config.awaf_dropout,
            use_modality_dropout=config.use_modality_dropout,
            modality_dropout_prob=config.modality_dropout_prob,
            use_modal_layernorm=config.use_modal_layernorm,
            awaf_uniform_mix=config.awaf_uniform_mix,
            return_diagnostics=True,
        )

        # ============================================================
        # 6. Uncertainty-Guided Residual Gate
        # ============================================================
        self.use_gate = config.use_uncertainty_gate
        if self.use_gate:
            self.gate = UncertaintyGuidedResidualGate(
                hidden_dim=H,
                gate_hidden_dim=config.gate_hidden_dim,
                dropout=config.gate_dropout,
                init_bias=config.gate_init_bias,
                margin_init=config.gate_margin_init,
                prior_temperature_init=config.gate_temperature_init,
            )

        # ============================================================
        # 7. Delta experts (per-modality, configurable scale)
        # ============================================================
        self.use_delta = config.use_delta_experts
        if self.use_delta:
            def _make_delta_expert():
                return nn.Sequential(
                    nn.Linear(H, H // 2),
                    nn.LayerNorm(H // 2),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(H // 2, 1),
                )
            self.delta_reg_t = _make_delta_expert()
            self.delta_reg_a = _make_delta_expert()
            self.delta_reg_v = _make_delta_expert()
            self.delta_cls_t = _make_delta_expert()
            self.delta_cls_a = _make_delta_expert()
            self.delta_cls_v = _make_delta_expert()

            # [P6H-R] Delta scale: configurable, learnable
            self.delta_scale_reg = nn.Parameter(torch.tensor(config.delta_scale_init))
            self.delta_scale_cls = nn.Parameter(torch.tensor(config.delta_scale_init))

            self.max_delta = config.max_delta
            self.use_bounded_delta = config.use_bounded_delta

    # ================================================================
    # Forward
    # ================================================================
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys:
                input_ids, attention_mask: text tokens [B, T_text]
                audio, audio_mask: audio features [B, T_audio, 768]
                vision, vision_mask: vision features [B, T_vision, 768]
                label: ground truth [B, 1] (optional)

        Returns:
            dict with:
                reg: final regression prediction [B, 1]
                cls: final classification proxy [B, 1]
                reg_text_base: text-only regression [B, 1]
                cls_text_base: text-only classification [B, 1]
                awaf_weights: [B, 3] (w_t, w_a, w_v)
                awaf_Z: AWAF fusion vector [B, H]
                awaf_diagnostics: dict with entropy, tau, l2 norms
                gate_reg, gate_cls: gate values [B, 1]
                delta_reg: raw delta [B, 1]
                effective_delta_reg: gated delta [B, 1]
        """
        DEVICE = self.config.device
        ids = batch['input_ids'].to(DEVICE)
        am = batch['attention_mask'].to(DEVICE)
        a = batch['audio'].to(DEVICE)
        am_a = batch['audio_mask'].to(DEVICE)
        v = batch['vision'].to(DEVICE)
        vm_v = batch['vision_mask'].to(DEVICE)

        # --- Text: RoBERTa + MLP ---
        ro = self.roberta(input_ids=ids, attention_mask=am)
        ht = ro.last_hidden_state[:, 0, :]  # CLS token [B, 1024]
        ht = self.text_mlp(ht)              # [B, H]
        rtb = self.reg_head_text(ht)        # [B, 1] text_base regression
        ctb = self.cls_head_text(ht)        # [B, 1] text_base classification

        # --- Audio: projection → sLSTM → pool ---
        ha = self.audio_proj(a)                       # [B, T_a, H]
        hao = self.audio_slstm(ha, am_a)              # dict with 'H': [B, T_a, H]
        hap, _ = self.audio_pool(hao['H'], am_a)      # [B, H]

        # --- Vision: projection → sLSTM → pool ---
        hv = self.vision_proj(v)                       # [B, T_v, H]
        hvo = self.vision_slstm(hv, vm_v)              # dict with 'H': [B, T_v, H]
        hvp, _ = self.vision_pool(hvo['H'], vm_v)      # [B, H]

        # --- AWAF fusion ---
        aw = self.awaf(ht, hap, hvp)                   # dict: 'Z', 'weights', 'diagnostics'
        z = aw['Z']                                    # [B, H]
        w = aw['weights']                              # [B, 3]
        awaf_diag = aw.get('diagnostics', {})          # entropy, tau, l2 norms

        # --- Delta experts ---
        if self.use_delta:
            # Regression delta
            dr_t = self.delta_reg_t(ht)   # [B, 1]
            dr_a = self.delta_reg_a(hap)  # [B, 1]
            dr_v = self.delta_reg_v(hvp)  # [B, 1]
            dr = w[:, 0:1] * dr_t + w[:, 1:2] * dr_a + w[:, 2:3] * dr_v  # weighted sum [B, 1]

            # Classification delta
            dc_t = self.delta_cls_t(ht)
            dc_a = self.delta_cls_a(hap)
            dc_v = self.delta_cls_v(hvp)
            dc = w[:, 0:1] * dc_t + w[:, 1:2] * dc_a + w[:, 2:3] * dc_v

            # Bounded delta
            if self.use_bounded_delta:
                bdr = self.max_delta * torch.tanh(dr)
                bdc = self.max_delta * torch.tanh(dc)
            else:
                bdr = dr
                bdc = dc

            dsr = self.delta_scale_reg
            dsc = self.delta_scale_cls
        else:
            bdr = torch.zeros_like(rtb)
            bdc = torch.zeros_like(ctb)
            dsr = torch.tensor(0.0, device=DEVICE)
            dsc = torch.tensor(0.0, device=DEVICE)
            dr = bdr

        # --- Uncertainty gate ---
        if self.use_gate:
            go = self.gate(ht, z, rtb, ctb, w, dr)
            gr = go['gate_reg']     # [B, 1]
            gc = go['gate_cls']     # [B, 1]
        else:
            gr = torch.ones_like(rtb)
            gc = torch.ones_like(ctb)

        # --- Final prediction ---
        edr = gr * dsr * bdr       # effective delta regression
        edc = gc * dsc * bdc       # effective delta classification
        reg = rtb + edr
        cls = ctb + edc

        return {
            'reg': reg,
            'cls': cls,
            'reg_text_base': rtb,
            'cls_text_base': ctb,
            'awaf_weights': w,
            'awaf_Z': z,
            'awaf_diagnostics': awaf_diag,
            'gate_reg': gr if self.use_gate else torch.ones_like(rtb),
            'gate_cls': gc if self.use_gate else torch.ones_like(ctb),
            'delta_reg': dr,
            'bounded_delta_reg': bdr,
            'effective_delta_reg': edr,
            'delta_scale_reg': dsr,
        }

    # ================================================================
    # Helpers
    # ================================================================
    def collect_trainable_params(self):
        """收集所有可训练参数 (用于 optimizer)。"""
        params = []
        # RoBERTa + LoRA
        params += list(self.roberta.parameters())
        # Text
        params += list(self.text_mlp.parameters())
        params += list(self.reg_head_text.parameters())
        params += list(self.cls_head_text.parameters())
        # Audio/Vision branches
        for m in [self.audio_proj, self.audio_slstm, self.audio_pool,
                  self.vision_proj, self.vision_slstm, self.vision_pool,
                  self.awaf]:
            params += list(m.parameters())
        # Gate
        if self.use_gate:
            params += list(self.gate.parameters())
        # Delta
        if self.use_delta:
            for m in [self.delta_reg_t, self.delta_reg_a, self.delta_reg_v,
                      self.delta_cls_t, self.delta_cls_a, self.delta_cls_v]:
                params += list(m.parameters())
            params += [self.delta_scale_reg, self.delta_scale_cls]
        return params

    def count_trainable(self) -> Dict[str, float]:
        """统计可训练 / 总参数量 (M)。"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_M': total / 1e6, 'trainable_M': trainable / 1e6}
