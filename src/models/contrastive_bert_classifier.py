# -*- coding: utf-8 -*-
"""
对比学习 BERT 分类模型
======================
在 BERTClassifier 基础上增加投影头（projection head），用于监督对比学习。

架构：
1. BERT 预训练模型 -> <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 表示 [batch, hidden_size]
2. 分类头：Dropout + Linear(hidden_size -> num_classes)
3. 投影头：Linear(hidden_size -> hidden_size) + ReLU + Linear(hidden_size -> projection_dim)
   输出经 L2 归一化，供 Supervised Contrastive Loss 使用

forward 返回 (logits, embeddings)，embeddings 已 L2 归一化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class ContrastiveBERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes=6, dropout=0.15,
                 projection_dim=128):
        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # 对比学习投影头（2 层 MLP）
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, input_ids, attention_mask):
        """
        返回:
            logits:     [batch, num_classes]  分类 logits
            embeddings: [batch, projection_dim]  L2 归一化后的对比特征
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # 分类分支
        cls_dropped = self.dropout(cls_output)
        logits = self.classifier(cls_dropped)  # [batch, num_classes]

        # 对比学习分支：L2 归一化（稳定 SCL 数值尺度）
        embeddings = self.projection(cls_output)  # [batch, projection_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return logits, embeddings
