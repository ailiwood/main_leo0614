"""Random tensor smoke test for Text-Guided AWAF-xLSTM."""
from __future__ import annotations
import argparse
from typing import Dict
import torch
from models.ours_text_guided_awaf_xlstm import OursTextGuidedAWAFXLSTM


def make_config(dataset: str) -> Dict:
    if dataset == "mosei":
        return {"text_dim": 300, "audio_dim": 74, "vision_dim": 35, "hidden_dim": 256, "use_aux_heads": True}
    return {"text_dim": 1024, "audio_dim": 768, "vision_dim": 768, "hidden_dim": 256, "use_aux_heads": True}


def random_batch(cfg: Dict, device: torch.device):
    B, Tt, Ta, Tv = 3, 11, 37, 20
    text = torch.randn(B, Tt, cfg["text_dim"], device=device)
    audio = torch.randn(B, Ta, cfg["audio_dim"], device=device)
    vision = torch.randn(B, Tv, cfg["vision_dim"], device=device)
    tm = torch.ones(B, Tt, dtype=torch.long, device=device)
    am = torch.ones(B, Ta, dtype=torch.long, device=device)
    vm = torch.ones(B, Tv, dtype=torch.long, device=device)
    tm[1, -3:] = 0; am[1, -5:] = 0; vm[1, -4:] = 0
    return text, audio, vision, tm, am, vm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    cfg = make_config(args.dataset)
    model = OursTextGuidedAWAFXLSTM(cfg).to(device)
    model.train()
    text, audio, vision, tm, am, vm = random_batch(cfg, device)
    out = model(text, audio, vision, tm, am, vm)
    assert out["reg"].shape == (text.size(0), 1)
    assert out["cls"].shape == (text.size(0), 1)
    assert out["awaf_weights"].shape == (text.size(0), 3)
    assert out["reliability"].shape == (text.size(0), 3)
    sum_dev = (out["awaf_weights"].sum(-1) - 1).abs().max().item()
    assert sum_dev < 1e-5, sum_dev
    assert 0.0 <= out["reliability"].min().item() <= out["reliability"].max().item() <= 1.0
    loss = out["reg"].abs().mean() + out["cls"].abs().mean()
    loss.backward()
    print("[OK] Text-Guided AWAF-xLSTM smoke passed")
    print(f"dataset={args.dataset}, device={device}, sum_w_max_dev={sum_dev:.3e}")
    print(f"reliability_mean={out['reliability'].mean().item():.4f}")


if __name__ == "__main__":
    main()
