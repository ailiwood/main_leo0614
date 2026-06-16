"""
scripts/train_main.py — 统一训练入口

支持:
  --config:      YAML 配置文件路径
  --dataset:     mosi (当前) | mosei (未来)
  --phase:       P2_smoke (default)
  --seed:        随机种子
  --epochs:      训练轮数
  --device:      cuda | cpu
  --output_dir:  输出目录 (默认自动生成)

P2 smoke test 用法:
  python scripts/train_main.py --epochs 1 --seed 42
"""
import sys
import os
import argparse
import json
import time
from datetime import datetime

# 将项目根加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.metrics import compute_all_metrics
from data.dataset import TMDCMOSIDataset, collate_fn
from models.ours_xlstm_fusion import OursXLSTMFusion
from engine.trainer import Trainer


def load_config(config_path: str) -> dict:
    """加载 YAML 配置并展开。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 展开嵌套: 如果顶层有 model/training/dataset 等 key，展开
    flattened = {}
    for section in ['model', 'training', 'dataset', 'output']:
        if section in cfg:
            flattened.update(cfg[section])

    # 保留未展开的
    for k, v in cfg.items():
        if k not in ['model', 'training', 'dataset', 'output']:
            flattened[k] = v

    return flattened


def build_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def main():
    parser = argparse.ArgumentParser(description='Train main model')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--dataset', type=str, default='mosi')
    parser.add_argument('--phase', type=str, default='P2_smoke')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--fusion_mode', type=str, default='awaf')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true', help='仅验证数据加载，不训练')
    args = parser.parse_args()

    # --- 加载配置 ---
    config = load_config(args.config) if os.path.exists(args.config) else {}

    # 命令行参数覆盖
    config['seed'] = args.seed
    config['epochs'] = args.epochs
    config['lr'] = args.lr
    config['hidden_dim'] = args.hidden_dim
    config['awaf_fusion_mode'] = args.fusion_mode
    config.setdefault('text_dim', 1024)
    config.setdefault('audio_dim', 1024)
    config.setdefault('vision_dim', 1024)
    config.setdefault('slstm_num_layers', 1)
    config.setdefault('slstm_dropout', 0.0)
    config.setdefault('slstm_bidirectional', False)
    config.setdefault('slstm_pooling', 'masked_mean')
    config.setdefault('awaf_tau_init', 1.0)
    config.setdefault('awaf_dropout', 0.1)
    config.setdefault('awaf_modality_dropout', True)
    config.setdefault('head_reg_hidden', 128)
    config.setdefault('head_cls_hidden', 128)
    config.setdefault('head_dropout', 0.3)
    config.setdefault('use_aux_heads', True)
    config.setdefault('proj_dropout', 0.1)
    config.setdefault('device', args.device)

    device = config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        config['device'] = 'cpu'

    # --- 随机种子 ---
    set_seed(args.seed)

    # --- 输出目录 ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = build_timestamp()
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', args.phase, args.dataset, 'ours_xlstm_fusion',
            f'{timestamp}_seed{args.seed}'
        )

    print("=" * 60)
    print(f"  Train Main Model — {args.phase}")
    print("=" * 60)
    print(f"  Config:     {args.config}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Seed:       {args.seed}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")
    print()

    # --- 数据加载 ---
    print("加载数据...")
    if args.dataset == 'mosi':
        train_ds = TMDCMOSIDataset(split='train')
        val_ds = TMDCMOSIDataset(split='val')
        test_ds = TMDCMOSIDataset(split='test')
    else:
        raise ValueError(f"Dataset {args.dataset} not yet supported in P2")

    print(f"  train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    if args.dry_run:
        print("\n[Dry run] 验证第一个 batch...")
        batch = next(iter(train_loader))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__}")
        print("Data loading OK ✅")
        return

    # --- 模型 ---
    print("\n构建模型...")
    model = OursXLSTMFusion(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # --- 训练器 ---
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=config.get('weight_decay', 0.01),
        reg_loss_weight=config.get('reg_loss_weight', 1.0),
        cls_loss_weight=config.get('cls_loss_weight', 0.5),
        aux_loss_weight=config.get('aux_loss_weight', 0.1),
        use_cls_loss=config.get('use_cls_loss', True),
    )

    # --- 训练 ---
    print(f"\n开始训练 ({args.epochs} epoch(s))...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch=epoch)

        # 每个 epoch 后评估
        eval_result = trainer.evaluate(test_loader)

        print(f"\n  Epoch {epoch}/{args.epochs}")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Test Metrics:")
        for k, v in eval_result['metrics'].items():
            print(f"      {k:12s}: {v:.3f}")

        # 检查 AWAF 权重
        w = eval_result['awaf_weights']
        print(f"    AWAF weights mean: [{w[:,0].mean():.3f}, {w[:,1].mean():.3f}, {w[:,2].mean():.3f}]")
        print(f"    AWAF weights std:  [{w[:,0].std():.3f}, {w[:,1].std():.3f}, {w[:,2].std():.3f}]")
        w_sum = w.sum(dim=-1)
        print(f"    AWAF sum(w) max_dev: {(w_sum - 1.0).abs().max().item():.2e}")

    elapsed = time.time() - t_start
    print(f"\n训练完成, 耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # --- 保存产物 ---
    print("\n保存产物...")
    command = ' '.join(sys.argv)
    trainer.save_run(
        output_dir=output_dir,
        config=config,
        command=command,
        train_loss=train_loss,
        eval_result=eval_result,
        epoch=args.epochs,
    )

    print(f"\n{'='*60}")
    print(f"  P2 Smoke Test 完成")
    print(f"  结果: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
