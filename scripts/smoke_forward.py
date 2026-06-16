"""
scripts/smoke_forward.py — 随机张量 forward test

验证所有核心模块:
  1. SLSTMCell + SLSTMEncoder
  2. AdaptiveWeightedAttentionFusion (全部 7 种融合模式)
  3. OursXLSTMFusion 主模型 (T=1 + T=5)
  4. AWAF sum(w) = 1
  5. 全部 fusion mode 可切换
  6. metrics.py 边界情况
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from utils.metrics import compute_all_metrics
from utils.seed import set_seed
from models.encoders.slstm import SLSTMCell, SLSTMEncoder
from models.fusion.awaf import AdaptiveWeightedAttentionFusion
from models.ours_xlstm_fusion import OursXLSTMFusion


def test_metrics():
    """测试统一指标模块。"""
    print("=" * 60)
    print("1. Metrics 测试")
    print("=" * 60)

    np.random.seed(42)
    N = 100
    targets = np.random.uniform(-3, 3, N)
    targets[:10] = 0.0  # zero labels
    preds = targets + np.random.normal(0, 0.5, N)
    logits = np.where(targets >= 0, 1.0, -1.0) + np.random.normal(0, 0.3, N)

    m = compute_all_metrics(preds, logits, targets)
    for k, v in m.items():
        print(f"  {k:12s}: {v:.4f}")
    assert not np.isnan(m['MAE']), "MAE is NaN"
    assert m['Corr'] > 0, f"Corr should be positive, got {m['Corr']}"
    print("  ✅ 通过\n")


def test_slstm_cell():
    """测试 sLSTM 单元。"""
    print("=" * 60)
    print("2. SLSTMCell 测试")
    print("=" * 60)

    B, D, H = 2, 128, 256
    cell = SLSTMCell(D, H)

    # 单步
    x_t = torch.randn(B, D)
    h_t, state = cell(x_t)
    print(f"  Input {x_t.shape} → Hidden {h_t.shape}")
    assert not torch.isnan(h_t).any()
    assert not torch.isinf(h_t).any()

    # 多步展开
    T = 10
    x_seq = torch.randn(B, T, D)
    h_prev = torch.zeros(B, H)
    c_prev = torch.zeros(B, H)
    n_prev = torch.zeros(B, H)
    m_prev = torch.full((B, H), -float('inf'))
    state = (h_prev, c_prev, n_prev, m_prev)

    outputs = []
    for t in range(T):
        h_t, state = cell(x_seq[:, t, :], state)
        outputs.append(h_t)
    H_all = torch.stack(outputs, dim=1)
    print(f"  Sequence {x_seq.shape} → H {H_all.shape}")
    assert not torch.isnan(H_all).any()
    print("  ✅ 通过\n")


def test_slstm_encoder():
    """测试 sLSTM 编码器。"""
    print("=" * 60)
    print("3. SLSTMEncoder 测试")
    print("=" * 60)

    B, T, D, H = 4, 5, 128, 256

    for pooling in ['masked_mean', 'last_valid']:
        enc = SLSTMEncoder(D, H, num_layers=2, dropout=0.1, pooling=pooling)
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T)
        mask[0, -2:] = 0  # last 2 padded

        out = enc(x, mask)
        print(f"  [{pooling:15s}] pooled: {out['pooled'].shape}, H: {out['H'].shape}")
        assert out['pooled'].shape == (B, H)
        assert not torch.isnan(out['pooled']).any()

    # T=1
    x_t1 = torch.randn(B, 1, D)
    out_t1 = enc(x_t1, torch.ones(B, 1))
    print(f"  [T=1         ] pooled: {out_t1['pooled'].shape}")
    assert out_t1['pooled'].shape == (B, H)

    # Bidirectional
    enc_bi = SLSTMEncoder(D, H, num_layers=1, bidirectional=True)
    x = torch.randn(B, T, D)
    out_bi = enc_bi(x)
    print(f"  [bidirectional] pooled: {out_bi['pooled'].shape} (expected {(B, H*2)})")
    assert out_bi['pooled'].shape == (B, H * 2)
    print("  ✅ 通过\n")


def test_awaf():
    """测试 AWAF 全部融合模式。"""
    print("=" * 60)
    print("4. AWAF 测试")
    print("=" * 60)

    B, d = 4, 256
    h_t = torch.randn(B, d)
    h_a = torch.randn(B, d)
    h_v = torch.randn(B, d)

    modes = ['awaf', 'awaf_no_context', 'awaf_no_interaction',
             'mean', 'concat', 'gated', 'fixed']

    for mode in modes:
        awaf = AdaptiveWeightedAttentionFusion(d, fusion_mode=mode)
        out = awaf(h_t, h_a, h_v)

        if mode not in ('concat',):
            w_sum = out['weights'].sum(dim=-1)
            max_dev = (w_sum - 1.0).abs().max().item()
            assert max_dev < 1e-4, f"[{mode}] sum(w) max dev = {max_dev:.2e}"
            print(f"  [{mode:25s}] sum(w) max_dev: {max_dev:.2e} ✅")
        else:
            print(f"  [{mode:25s}] (no weight check) ✅")

    # Modality dropout test
    awaf_train = AdaptiveWeightedAttentionFusion(d, use_modality_dropout=True)
    awaf_train.train()
    out_md = awaf_train(h_t, h_a, h_v)
    print(f"  [modality_dropout   ] weights: {out_md['weights'][0].tolist()} ✅")

    print()


def test_main_model():
    """测试主模型。"""
    print("=" * 60)
    print("5. OursXLSTMFusion 主模型测试")
    print("=" * 60)

    config = {
        'text_dim': 1024, 'audio_dim': 1024, 'vision_dim': 1024,
        'hidden_dim': 256, 'slstm_num_layers': 1, 'slstm_dropout': 0.0,
        'slstm_bidirectional': False, 'slstm_pooling': 'masked_mean',
        'awaf_fusion_mode': 'awaf', 'awaf_tau_init': 1.0, 'awaf_dropout': 0.1,
        'awaf_modality_dropout': True,
        'head_reg_hidden': 128, 'head_cls_hidden': 128, 'head_dropout': 0.3,
        'use_aux_heads': True, 'proj_dropout': 0.1,
    }
    B = 4
    model = OursXLSTMFusion(config)
    model.train()

    # T=1 (clip-level)
    text = torch.randn(B, 1024)
    audio = torch.randn(B, 1024)
    vision = torch.randn(B, 1024)
    out = model(text, audio, vision)
    print(f"  [T=1] reg={out['reg'].shape}, cls={out['cls'].shape}, weights={out['awaf_weights'].shape}")
    ws = out['awaf_weights'].sum(dim=-1)
    assert (ws - 1.0).abs().max() < 1e-4
    assert out['reg'].shape == (B, 1)

    # T=5 (序列)
    T = 5
    text_seq = torch.randn(B, T, 1024)
    audio_seq = torch.randn(B, T, 1024)
    vision_seq = torch.randn(B, T, 1024)
    mask = torch.ones(B, T)
    out_seq = model(text_seq, audio_seq, vision_seq, mask, mask, mask)
    print(f"  [T=5] reg={out_seq['reg'].shape}, weights={out_seq['awaf_weights'].shape}")
    assert out_seq['reg'].shape == (B, 1)

    # 测试 backward
    loss = out['reg'].mean() + out['cls'].mean()
    loss.backward()
    print(f"  backward ✅")

    # 参数量
    n = sum(p.numel() for p in model.parameters())
    print(f"  params: {n:,}")

    # 测试所有 fusion mode
    for mode in ['awaf', 'awaf_no_context', 'awaf_no_interaction', 'mean', 'concat', 'gated', 'fixed']:
        cfg = config.copy()
        cfg['awaf_fusion_mode'] = mode
        m = OursXLSTMFusion(cfg)
        m.eval()
        o = m(text, audio, vision)
        assert o['reg'].shape == (B, 1)
    print(f"  all 7 fusion modes ✅")
    print()


def main():
    set_seed(42)
    print("\n" + "=" * 60)
    print("  SMOKE FORWARD TEST — 随机张量验证")
    print("=" * 60 + "\n")

    test_metrics()
    test_slstm_cell()
    test_slstm_encoder()
    test_awaf()
    test_main_model()

    print("=" * 60)
    print("  全部随机张量 forward test 通过 ✅")
    print("=" * 60)


if __name__ == '__main__':
    main()
