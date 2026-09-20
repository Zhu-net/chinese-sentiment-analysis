"""
TextCNN 模型（2014 年 Yoon Kim 提出）

架构原理：
1. Embedding 层：把每个词的 ID 转为稠密向量（词向量）
   - 输入: [batch_size, seq_len] (整数ID)
   - 输出: [batch_size, seq_len, embed_dim] (浮点向量)

2. 多个并行的一维卷积层（不同 kernel_size）：
   - kernel_size=2: 捕捉二元词组信息（如 "不好"）
   - kernel_size=3: 捕捉三元词组信息（如 "非常好"）
   - kernel_size=4: 捕捉四元词组信息
   每个卷积核会在序列上滑动，提取 n-gram 特征

3. 最大池化（Max Pooling）：
   - 对每个卷积核的输出取最大值
   - 作用：捕捉"最重要"的特征，同时把变长序列转为定长向量

4. 全连接层 + Dropout + Softmax：
   - 拼接所有卷积核的池化结果
   - 通过线性层映射到类别数

为什么用多个不同大小的卷积核？
→ 因为情感词可能是 2 字词（难吃）、3 字词（非常好）、4 字词（太不划算了）等
  不同大小的卷积核能捕捉不同粒度的短语特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=128,
                 kernel_sizes=(2, 3, 4), num_classes=2, dropout=0.5):
        super().__init__()

        # ===== 1. 词嵌入层 =====
        # vocab_size: 词表大小
        # embed_dim: 每个词的向量维度（128维）
        # padding_idx=0: ID=0 (<PAD>) 的向量固定为全0，不参与训练
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # ===== 2. 多个并行的一维卷积层 =====
        # nn.ModuleList 可以存放多个子模块
        # 每个卷积层: in_channels=embed_dim, out_channels=num_filters
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        # ===== 3. Dropout 防止过拟合 =====
        self.dropout = nn.Dropout(dropout)

        # ===== 4. 全连接分类层 =====
        # 输入维度 = 卷积核数量 * 卷积核种类数
        # 输出维度 = 类别数（2：正面/负面）
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        """
        前向传播：定义数据如何流过网络
        x: [batch_size, seq_len] 输入的整数ID序列
        """
        # Step 1: 词嵌入
        # x: [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        x = self.embedding(x)

        # Step 2: 调整维度以适配 Conv1d
        # Conv1d 要求输入: [batch_size, in_channels, seq_len]
        # 所以需要把 embed_dim 换到中间维度
        # permute(0, 2, 1): 交换第1维和第2维
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]

        # Step 3: 对每个卷积核做卷积 + 激活 + 最大池化
        pooled_outputs = []
        for conv in self.convs:
            # 卷积: [batch_size, num_filters, seq_len - kernel_size + 1]
            conv_out = conv(x)
            # ReLU 激活函数
            conv_out = F.relu(conv_out)
            # 最大池化: 在最后一维（序列维）上取最大值
            # -> [batch_size, num_filters]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            pooled_outputs.append(pooled)

        # Step 4: 拼接所有卷积核的输出
        # -> [batch_size, num_filters * len(kernel_sizes)]
        x = torch.cat(pooled_outputs, dim=1)

        # Step 5: Dropout + 全连接层
        x = self.dropout(x)
        logits = self.fc(x)  # [batch_size, num_classes]

        return logits
