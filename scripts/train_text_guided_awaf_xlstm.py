"""P5A training entry template for Text-Guided AWAF-xLSTM.

CC should adapt build_dataloaders() to the current repository dataset APIs.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Any, Dict
import torch
import yaml
from models.ours_text_guided_awaf_xlstm import OursTextGuidedAWAFXLSTM
from utils.seed import set_seed


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_dataloaders(dataset: str, data_config: Dict[str, Any], batch_size: int):
    if dataset == "mosi":
        from data.strong_sequence_dataset import build_strong_sequence_loaders
        return build_strong_sequence_loaders(data_config, batch_size=batch_size)
    if dataset == "mosei_sdk":
        from data.mosei_sdk_dataset import build_mosei_sdk_loaders
        return build_mosei_sdk_loaders(data_config, batch_size=batch_size)
    raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mosi", "mosei_sdk"], required=True)
    p.add_argument("--data_config", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = model_cfg.get("train", {})
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    epochs = args.epochs or int(train_cfg.get("epochs", 60))
    lr = args.lr or float(train_cfg.get("lr", 1e-4))
    actual_config = {"args": vars(args), "data_config": data_cfg, "model_config": model_cfg,
                     "resolved": {"batch_size": batch_size, "epochs": epochs, "lr": lr}}
    with open(Path(args.output_dir) / "actual_config.json", "w", encoding="utf-8") as f:
        json.dump(actual_config, f, indent=2, ensure_ascii=False)
    loaders = build_dataloaders(args.dataset, data_cfg, batch_size=batch_size)
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
    model = OursTextGuidedAWAFXLSTM(model_cfg)
    from engine.strict_trainer import StrictTrainer
    trainer = StrictTrainer(
        model=model, device=args.device, lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        reg_loss_weight=float(model_cfg.get("reg_loss_weight", 1.0)),
        cls_loss_weight=float(model_cfg.get("cls_loss_weight", 0.3)),
        aux_loss_weight=float(model_cfg.get("aux_loss_weight", 0.1)),
    )
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch=epoch)
        _, record = trainer.check_val(epoch, val_loader)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_MAE={record['val_MAE']:.4f} "
              f"val_ACC2_NZ={record['val_ACC2_Non0_regsign']:.2f}")
    test_result = trainer.final_test(test_loader)
    trainer.save_run(args.output_dir, actual_config, " ".join(os.sys.argv), epochs, test_result)
    print(f"[DONE] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
