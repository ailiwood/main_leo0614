"""
评估指标计算模块
"""

import numpy as np
import torch


class MetricsTop:
    def __init__(self, train_mode='regression'):
        self.train_mode = train_mode

    def getMetics(self, dataset_name):
        if self.train_mode == 'regression':
            return RegressionMetrics(dataset_name)


class RegressionMetrics:
    """回归任务评估指标"""
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __call__(self, preds, targets):
        """
        计算评估指标
        Args:
            preds: tensor, 预测值
            targets: tensor, 真实值
        """
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # MAE
        mae = np.mean(np.abs(preds - targets))

        # Correlation
        if len(preds) > 1:
            corr = np.corrcoef(preds, targets)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        # ACC2 (Non-zero)
        pred_binary = (preds >= 0).astype(int)
        target_binary = (targets >= 0).astype(int)
        acc2 = np.mean(pred_binary == target_binary)

        # F1 (Non-zero)
        tp = np.sum((pred_binary == 1) & (target_binary == 1))
        fp = np.sum((pred_binary == 1) & (target_binary == 0))
        fn = np.sum((pred_binary == 0) & (target_binary == 1))
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = 0.0

        return {
            'MAE': mae,
            'Corr': corr,
            'Has0_acc_2': acc2 * 100,
            'Has0_f1': f1 * 100
        }
