# -*- coding: utf-8 -*-
"""
损失函数集合
============
- SupervisedContrastiveLoss: 监督对比损失
  同类样本互为正样本（pull together），异类为负样本（push apart）。
  要求输入 embeddings 已做 L2 归一化。
"""
import torch
import torch.nn as nn


class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比损失（SupCon Loss, Khosla et al. 2020）

    公式（对每个样本 i）：
        L_i = -1 / (|P(i)|) * Σ_{p∈P(i)} log( exp(z_i·z_p / τ) / Σ_{a≠i} exp(z_i·z_a / τ) )
    其中 P(i) = 与 i 同类的所有其他样本，τ 为温度系数。

    参数:
        temperature: 温度系数，越小对比越强（默认 0.07）
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: [batch_size, dim]  已 L2 归一化
        labels:     [batch_size]        类别标签
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # 相似度矩阵 [batch, batch]
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 对角线（自身）置为 -inf，不参与分母
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        similarity = similarity.masked_fill(mask_self, float("-inf"))

        # 构建正样本掩码：同类别且非自身
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T) & ~mask_self  # [batch, batch]

        # 数值稳定：每行减去最大值（log-sum-exp trick）
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # exp(logits)，对角线为 0
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(mask_self, 0.0)

        # log 分母 = log(Σ_{a≠i} exp(z_i·z_a / τ))
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 只对正样本求平均
        # 注意：不能用 mask_pos.float() * log_prob，因为对角线 log_prob = -inf，
        # 0.0 * (-inf) = nan。用 torch.where 安全选取正样本位置。
        pos_count = mask_pos.float().sum(dim=1).clamp(min=1.0)  # 避免无正样本
        pos_log_prob = torch.where(mask_pos, log_prob, torch.zeros_like(log_prob))
        mean_log_prob_pos = pos_log_prob.sum(dim=1) / pos_count

        loss = -mean_log_prob_pos.mean()
        return loss
