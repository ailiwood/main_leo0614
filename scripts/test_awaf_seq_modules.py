"""
scripts/test_awaf_seq_modules.py

P4V AWAF-Seq 模块单元测试。
运行：
    python scripts/test_awaf_seq_modules.py
"""
import os
import sys

import torch

# 允许从项目根目录或 scripts/ 目录运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.interaction.cross_modal_transformer import CrossModalTransformerEncoder
from models.pooling.attention_pooling import MaskedAttentionPooling
from models.ours_awaf_seq_xlstm import OursAWAFSeqXLSTM


def assert_close(name, x, y, tol=1e-5):
    diff = (x - y).abs().max().item()
    print(f"  {name}: max_abs_diff={diff:.2e}")
    assert diff < tol, f"{name} diff {diff} >= {tol}"


def test_cross_modal_transformer(device):
    print("[1] CrossModalTransformerEncoder")
    B, D = 3, 256
    Tt, Ta, Tv = 7, 11, 1
    Ht = torch.randn(B, Tt, D, device=device, requires_grad=True)
    Ha = torch.randn(B, Ta, D, device=device, requires_grad=True)
    Hv = torch.randn(B, Tv, D, device=device, requires_grad=True)
    mt = torch.ones(B, Tt, device=device, dtype=torch.long); mt[0, -2:] = 0
    ma = torch.ones(B, Ta, device=device, dtype=torch.long); ma[1, -3:] = 0
    mv = torch.ones(B, Tv, device=device, dtype=torch.long)

    enc = CrossModalTransformerEncoder(D, num_layers=1, num_heads=4, ffn_dim=512).to(device)
    out = enc(Ht, Ha, Hv, mt, ma, mv)
    assert out["H_t"].shape == (B, Tt, D)
    assert out["H_a"].shape == (B, Ta, D)
    assert out["H_v"].shape == (B, Tv, D)
    loss = out["joint"].mean()
    loss.backward()
    print("  OK")


def test_attention_pooling(device):
    print("[2] MaskedAttentionPooling")
    B, T, D = 4, 9, 256
    H = torch.randn(B, T, D, device=device, requires_grad=True)
    mask = torch.ones(B, T, device=device, dtype=torch.long)
    mask[0, -4:] = 0
    pool = MaskedAttentionPooling(D).to(device)
    pooled, attn = pool(H, mask)
    assert pooled.shape == (B, D)
    assert attn.shape == (B, T)
    assert_close("attn_sum", attn.sum(dim=1), torch.ones(B, device=device), tol=1e-5)
    assert attn[0, -4:].abs().max().item() < 1e-6
    pooled.mean().backward()
    print("  OK")


def test_full_model(device):
    print("[3] OursAWAFSeqXLSTM forward/backward + mask invariance")
    cfg = {
        "text_dim": 1024,
        "audio_dim": 768,
        "vision_dim": 1024,
        "hidden_dim": 256,
        "proj_dropout": 0.1,
        "slstm_num_layers": 1,
        "slstm_dropout": 0.2,
        "slstm_pooling": "masked_mean",
        "cross_num_layers": 1,
        "cross_num_heads": 4,
        "cross_ffn_dim": 512,
        "cross_dropout": 0.1,
        "pooling_dropout": 0.1,
        "awaf_fusion_mode": "awaf",
        "awaf_modality_dropout": False,  # invariance 测试中关闭随机 dropout
        "head_dropout": 0.1,
        "use_aux_heads": False,
    }
    torch.manual_seed(1234)
    model = OursAWAFSeqXLSTM(cfg).to(device)
    model.eval()

    B = 2
    text = torch.randn(B, 8, 1024, device=device)
    audio = torch.randn(B, 12, 768, device=device)
    vision = torch.randn(B, 1, 1024, device=device)
    tm = torch.ones(B, 8, device=device, dtype=torch.long); tm[0, -3:] = 0
    am = torch.ones(B, 12, device=device, dtype=torch.long); am[1, -5:] = 0
    vm = torch.ones(B, 1, device=device, dtype=torch.long)

    with torch.no_grad():
        out_clean = model(text, audio, vision, tm, am, vm)
        text_noisy = text.clone(); text_noisy[0, -3:] = 999.0
        audio_noisy = audio.clone(); audio_noisy[1, -5:] = -999.0
        out_noisy = model(text_noisy, audio_noisy, vision, tm, am, vm)

    assert_close("reg_padding_invariance", out_clean["reg"], out_noisy["reg"], tol=1e-4)
    assert_close("awaf_padding_invariance", out_clean["awaf_weights"], out_noisy["awaf_weights"], tol=1e-4)
    assert_close(
        "awaf_sum",
        out_clean["awaf_weights"].sum(dim=-1),
        torch.ones(B, device=device),
        tol=1e-5,
    )

    model.train()
    out = model(text, audio, vision, tm, am, vm)
    loss = out["reg"].mean() + out["cls"].mean()
    loss.backward()
    assert not torch.isnan(out["reg"]).any()
    print("  OK")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    test_cross_modal_transformer(device)
    test_attention_pooling(device)
    test_full_model(device)
    print("\nALL AWAF-SEQ MODULE TESTS PASSED")


if __name__ == "__main__":
    main()
