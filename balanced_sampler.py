"""
BalancedSampler — 让每个 batch 内正负样本 50/50 (P1 修复)
针对 MOSEI 71% 正样本不平衡问题

v2 修复 (2026-06-13): 改成 BatchSampler 模式 — yield 整个 batch list
DataLoader 用 batch_sampler= 参数 (不能再带 batch_size / shuffle / sampler)
"""
import random
import numpy as np
from torch.utils.data import Sampler


class BalancedSentimentSampler(Sampler):
    """
    BatchSampler: 每个 batch 内 50% 正样本 + 50% 负样本
    - 正样本: sentiment >= 0
    - 负样本: sentiment < 0
    - 大的类别下采样, 小的类别上采样 (with replacement)
    用法: DataLoader(dataset, batch_sampler=this, num_workers=0)
    """

    def __init__(self, labels, batch_size=16, num_batches_per_epoch=None, seed=42):
        labels = np.asarray(labels).flatten()
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.pos_idx = np.where(labels >= 0)[0]
        self.neg_idx = np.where(labels < 0)[0]
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError(f"need both pos and neg: pos={len(self.pos_idx)}, neg={len(self.neg_idx)}")
        self.n_per_class = max(batch_size // 2, 1)
        if num_batches_per_epoch is None:
            num_batches_per_epoch = max(len(self.pos_idx), len(self.neg_idx)) // max(self.n_per_class, 1)
        self.num_batches = max(num_batches_per_epoch, 1)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            pos_batch = self.rng.choices(self.pos_idx.tolist(), k=self.n_per_class)
            neg_batch = self.rng.choices(self.neg_idx.tolist(), k=self.n_per_class)
            batch = pos_batch + neg_batch
            self.rng.shuffle(batch)
            # v2 修复: yield 整个 list, 让 DataLoader 走 batch_sampler 路径
            yield batch


# 兼容名
BalancedSentimentBatchSampler = BalancedSentimentSampler


def get_balanced_sampler(dataset, batch_size=16, seed=42):
    """从 dataset 提取 labels 并构造 BatchSampler"""
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        lbl = item.get('label', item.get('raw_label', None))
        if lbl is None:
            continue
        if hasattr(lbl, 'item'):
            lbl = lbl.item() if lbl.dim() == 0 else lbl.flatten()[0].item()
        labels.append(float(lbl))
    return BalancedSentimentSampler(labels, batch_size=batch_size, seed=seed)