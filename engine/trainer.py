"""
engine/trainer.py — 训练器

支持:
  - 多 epoch 训练 + 逐 epoch 评估
  - 多任务 loss (reg + cls + aux)
  - 统一 metrics (utils/metrics.py)
  - 产物保存: config.yaml, log, metrics, predictions, awaf_weights, model checkpoints
  - loss/metric curves (PNG)
"""
import os
import json
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # 无头模式，不弹窗
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Optional, List

from utils.metrics import compute_all_metrics
from models.fusion.awaf import save_awaf_weights_csv


class Trainer:
    """多 epoch 训练器，支持逐 epoch 指标追踪和完整产物保存。"""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        reg_loss_weight: float = 1.0,
        cls_loss_weight: float = 0.5,
        aux_loss_weight: float = 0.1,
        use_cls_loss: bool = True,
        awaf_entropy_reg_weight: float = 0.0,
        eps: float = 1e-8,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.reg_loss_weight = reg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.use_cls_loss = use_cls_loss
        self.awaf_entropy_reg_weight = awaf_entropy_reg_weight
        self.eps = eps

        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # 逐 epoch 指标历史
        self.history: List[Dict] = []

    def _compute_loss(self, output: Dict, labels: torch.Tensor) -> Dict:
        labels = labels.view(-1, 1).float().to(self.device)
        polar_tgt = (labels >= 0).float()

        reg_loss = self.l1_loss(output['reg'], labels)

        if self.use_cls_loss:
            cls_loss = self.bce_loss(output['cls'], polar_tgt)
        else:
            cls_loss = torch.tensor(0.0, device=self.device)

        aux_loss = torch.tensor(0.0, device=self.device)
        if output.get('aux') and self.aux_loss_weight > 0:
            n_aux = 0
            for k, v in output['aux'].items():
                if k.endswith('_reg'):
                    aux_loss += self.l1_loss(v, labels)
                    n_aux += 1
                elif k.endswith('_cls'):
                    aux_loss += self.bce_loss(v, polar_tgt)
                    n_aux += 1
            if n_aux > 0:
                aux_loss = aux_loss / n_aux

        # AWAF entropy regularization
        entropy_reg = torch.tensor(0.0, device=self.device)
        awaf_w = output.get('awaf_weights', None)
        if awaf_w is not None and self.awaf_entropy_reg_weight > 0:
            # H(w) = -sum(w_i * log(w_i + eps))
            w_clamped = awaf_w.clamp(min=self.eps)
            entropy = -(w_clamped * torch.log(w_clamped)).sum(dim=-1)  # [B]
            entropy_reg = entropy.mean()  # scalar
            # loss = original_loss - lambda * entropy  (higher entropy = more balanced)
            entropy_reg = -self.awaf_entropy_reg_weight * entropy_reg

        total = (self.reg_loss_weight * reg_loss +
                 self.cls_loss_weight * cls_loss +
                 self.aux_loss_weight * aux_loss +
                 entropy_reg)

        return {'total': total, 'reg': reg_loss, 'cls': cls_loss,
                'aux': aux_loss, 'entropy_reg': entropy_reg}

    def train_epoch(self, loader: DataLoader, epoch: int = 1) -> float:
        """训练一个 epoch，返回平均 loss。"""
        self.model.train()
        total_loss = 0.0
        cnt = 0

        pbar = tqdm(loader, desc=f'Train E{epoch}', leave=False)
        for batch in pbar:
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            vision = batch['vision'].to(self.device)
            labels = batch['label'].to(self.device)

            text_mask = batch.get('text_mask', None)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(
                text, audio, vision,
                text_mask=text_mask, audio_mask=None, vision_mask=None,
            )

            if torch.isnan(output['reg']).any():
                continue

            losses = self._compute_loss(output, labels)
            loss = losses['total']

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * text.size(0)
            cnt += text.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / cnt if cnt > 0 else float('nan')

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """评估，返回全部预测、权重、标签。"""
        self.model.eval()
        all_reg, all_cls, all_labels = [], [], []
        all_ids, all_weights = [], []
        total_loss = 0.0
        cnt = 0

        for batch in tqdm(loader, desc='Eval', leave=False):
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            vision = batch['vision'].to(self.device)
            labels = batch['label'].to(self.device)
            text_mask = batch.get('text_mask', None)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)

            output = self.model(
                text, audio, vision,
                text_mask=text_mask, audio_mask=None, vision_mask=None,
            )

            losses = self._compute_loss(output, labels)
            total_loss += losses['total'].item() * text.size(0)
            cnt += text.size(0)

            all_reg.append(output['reg'].cpu())
            all_cls.append(output['cls'].cpu())
            all_labels.append(labels.cpu())
            all_weights.append(output['awaf_weights'].cpu())

            ids = batch.get('id', [str(i) for i in range(len(labels))])
            all_ids.extend(ids)

        reg_preds = torch.cat(all_reg, dim=0)
        cls_preds = torch.cat(all_cls, dim=0)
        targets = torch.cat(all_labels, dim=0)
        awaf_w = torch.cat(all_weights, dim=0)
        avg_loss = total_loss / cnt if cnt > 0 else float('nan')

        metrics = compute_all_metrics(reg_preds, cls_preds, targets)

        # AWAF 权重统计
        awaf_stats = {
            'w_t_mean': float(awaf_w[:, 0].mean()),
            'w_a_mean': float(awaf_w[:, 1].mean()),
            'w_v_mean': float(awaf_w[:, 2].mean()),
            'w_t_std': float(awaf_w[:, 0].std()),
            'w_a_std': float(awaf_w[:, 1].std()),
            'w_v_std': float(awaf_w[:, 2].std()),
            'sum_w_max_dev': float((awaf_w.sum(dim=-1) - 1.0).abs().max()),
            # AWAF entropy
            'awaf_entropy_mean': float((-awaf_w.clamp(min=1e-8) * torch.log(awaf_w.clamp(min=1e-8))).sum(dim=-1).mean()),
            'awaf_entropy_std': float((-awaf_w.clamp(min=1e-8) * torch.log(awaf_w.clamp(min=1e-8))).sum(dim=-1).std()),
            # Collapse ratio
            'collapse_ratio_0.8': float((awaf_w[:, 0] > 0.8).float().mean()),
            'collapse_ratio_0.9': float((awaf_w[:, 0] > 0.9).float().mean()),
            'max_weight_mean': float(awaf_w.max(dim=-1)[0].mean()),
        }

        return {
            'reg_preds': reg_preds,
            'cls_preds': cls_preds,
            'targets': targets,
            'metrics': metrics,
            'awaf_weights': awaf_w,
            'awaf_stats': awaf_stats,
            'ids': all_ids,
            'loss': avg_loss,
        }

    def record_epoch(self, epoch: int, train_loss: float, eval_result: Dict, elapsed: float):
        """记录一个 epoch 的所有指标到 history。"""
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_result['loss'],
            'time_sec': elapsed,
        }
        record.update(eval_result['metrics'])
        record.update(eval_result['awaf_stats'])
        self.history.append(record)

    def save_history_csv(self, output_dir: str):
        """保存逐 epoch 指标 CSV。"""
        if not self.history:
            return
        path = os.path.join(output_dir, 'metrics_epoch.csv')
        keys = list(self.history[0].keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    def plot_curves(self, output_dir: str):
        """绘制 loss 和 metrics 曲线。"""
        if not self.history:
            return
        epochs = [r['epoch'] for r in self.history]

        # Loss curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs, [r['train_loss'] for r in self.history], 'b-', label='Train Loss')
        ax1.plot(epochs, [r['eval_loss'] for r in self.history], 'r-', label='Eval Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curve'); ax1.legend(); ax1.grid(True, alpha=0.3)

        # Metrics curve
        metric_keys = ['ACC2_Non0', 'MAE', 'Corr']
        colors = ['g-', 'r-', 'b-']
        for mk, c in zip(metric_keys, colors):
            if mk in self.history[0]:
                ax2.plot(epochs, [r[mk] for r in self.history], c, label=mk)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Value')
        ax2.set_title('Metrics Curve'); ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
        plt.close(fig)

        # AWAF weight evolution
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, mod in enumerate(['w_t_mean', 'w_a_mean', 'w_v_mean']):
            if mod in self.history[0]:
                ax.plot(epochs, [r[mod] for r in self.history],
                        ['b-', 'r-', 'g-'][i], label=mod)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Weight')
        ax.set_title('AWAF Weight Evolution'); ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, 'awaf_weights_curve.png'), dpi=150)
        plt.close(fig)

    def save_run(
        self,
        output_dir: str,
        config: dict,
        command: str,
        epoch: int = 1,
    ):
        """保存完整 run 产物。"""
        os.makedirs(output_dir, exist_ok=True)

        # config.yaml (保存为 json 简化)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # command.txt
        with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
            f.write(command)

        # metrics_epoch.csv
        self.save_history_csv(output_dir)

        # metrics_best.json (最后一个 epoch 的指标)
        if self.history:
            best = max(self.history, key=lambda r: r.get('ACC2_Non0', 0))
            with open(os.path.join(output_dir, 'metrics_best.json'), 'w') as f:
                json.dump(best, f, indent=2)

        # predictions_test.csv (使用最近一次 eval)
        # 这里需要重新 evaluate 一次获取 predictions
        # 实际在 train loop 中保存最后一个 epoch 的 predictions

        # best_model.pth
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, os.path.join(output_dir, 'best_model.pth'))

        # last_model.pth
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
        }, os.path.join(output_dir, 'last_model.pth'))

        # Curves
        self.plot_curves(output_dir)

    def save_predictions_and_weights(self, output_dir: str, eval_result: Dict):
        """保存预测和权重 CSV。"""
        # predictions_test.csv
        with open(os.path.join(output_dir, 'predictions_test.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'reg_pred', 'cls_logit', 'target'])
            reg = eval_result['reg_preds'].numpy().flatten()
            cls_l = eval_result['cls_preds'].numpy().flatten()
            tgt = eval_result['targets'].numpy().flatten()
            ids = eval_result.get('ids', range(len(reg)))
            for i in range(len(reg)):
                writer.writerow([ids[i] if i < len(ids) else i, reg[i], cls_l[i], tgt[i]])

        # awaf_weights_test.csv
        save_awaf_weights_csv(
            eval_result['awaf_weights'],
            eval_result.get('ids', []),
            os.path.join(output_dir, 'awaf_weights_test.csv'),
        )
