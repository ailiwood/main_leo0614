"""
scripts/train_awaf_seq.py

P4V AWAF-Seq 训练入口。

示例：
python scripts/train_awaf_seq.py \
  --config configs/models/ours_awaf_seq_xlstm.yaml \
  --data_config configs/data/mosi_strong_sequence.yaml \
  --seed 42 \
  --output_dir outputs/P4V_awafseq_upgrade/awaf_seq/seed42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.strong_sequence_dataset import StrongSequenceMOSIDataset, collate_strong_sequence
from engine.strict_trainer import StrictTrainer
from models.ours_awaf_seq_xlstm import OursAWAFSeqXLSTM
from models.ours_xlstm_fusion import OursXLSTMFusion


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(config: Dict):
    name = str(config.get("model_name", "ours_awaf_seq_xlstm")).lower()
    if name in {"ours_awaf_seq_xlstm", "awaf_seq", "awafseq"}:
        return OursAWAFSeqXLSTM(config)
    if name in {"ours_c0_strong_sequence", "ours_xlstm_fusion", "c0"}:
        return OursXLSTMFusion(config)
    raise ValueError(f"Unknown model_name: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    train_cfg = model_cfg.get("train", {}) or {}

    epochs = args.epochs or int(train_cfg.get("epochs", 40))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 16))
    feature_root = data_cfg.get("feature_root", "data/features_strong_sequence_mosi")
    max_text_len = int(data_cfg.get("max_text_len", 50))
    max_audio_len = int(data_cfg.get("max_audio_len", 100))
    max_vision_len = int(data_cfg.get("max_vision_len", 1))

    collate_fn = partial(
        collate_strong_sequence,
        max_text_len=max_text_len,
        max_audio_len=max_audio_len,
        max_vision_len=max_vision_len,
    )

    train_ds = StrongSequenceMOSIDataset("train", feature_root=feature_root, max_seq_len=max(max_text_len, max_audio_len))
    val_ds = StrongSequenceMOSIDataset("val", feature_root=feature_root, max_seq_len=max(max_text_len, max_audio_len))
    test_ds = StrongSequenceMOSIDataset("test", feature_root=feature_root, max_seq_len=max(max_text_len, max_audio_len))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_model(model_cfg)
    trainer = StrictTrainer(
        model,
        device=device,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        reg_loss_weight=float(train_cfg.get("reg_loss_weight", 1.0)),
        cls_loss_weight=float(train_cfg.get("cls_loss_weight", 0.3)),
        aux_loss_weight=float(train_cfg.get("aux_loss_weight", 0.0)),
        awaf_entropy_reg_weight=float(train_cfg.get("awaf_entropy_reg_weight", 0.0)),
    )

    patience = int(train_cfg.get("early_stopping_patience", 8))
    no_improve = 0
    best_seen = float("inf")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)

    with open(out_dir / "config_merged.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_cfg, "data": data_cfg, "seed": args.seed}, f, indent=2, ensure_ascii=False)
    with open(out_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write(command)

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch=epoch)
        _, record = trainer.check_val(epoch, val_loader)
        val_mae = float(record["val_MAE"])
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_MAE={val_mae:.4f}, best_epoch={trainer.best_epoch}")

        if val_mae < best_seen - 1e-8:
            best_seen = val_mae
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience}).")
            break

    test_result = trainer.final_test(test_loader)
    trainer.save_run(str(out_dir), {"model": model_cfg, "data": data_cfg, "seed": args.seed}, command, epoch, test_result)

    print("Final test metrics:")
    print(json.dumps(test_result["metrics_regsign"], indent=2))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
