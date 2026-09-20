"""
BERT 文本分类模型

架构：
1. BERT 预训练模型：提取文本的上下文语义表示
   - 输入: input_ids, attention_mask
   - 输出: last_hidden_state [batch_size, seq_len, hidden_size]
   - 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 位置的输出作为句子表示 [batch_size, hidden_size]

2. Dropout + 全连接层：分类
   - hidden_size (BERT-base = 768) -> num_classes

关键概念：
- <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 标记：BERT 用第一个 token 的输出作为整个句子的表示
- 微调（Fine-tuning）：在预训练权重基础上，用下游任务数据继续训练
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes=2, dropout=0.1):
        super().__init__()

        # 加载预训练 BERT 模型
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        # BERT-base 的隐藏层维度是 768
        hidden_size = self.bert.config.hidden_size

        # 分类头：Dropout + Linear
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len] 输入token的ID
        attention_mask: [batch_size, seq_len] 注意力掩码（1=真实token, 0=padding）
        """
        # BERT 前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        # 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> 位置（第0个token）的输出作为句子表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Dropout + 分类
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [batch_size, num_classes]

        return logits
