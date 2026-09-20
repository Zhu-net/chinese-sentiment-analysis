"""
BiLSTM 模型（双向长短期记忆网络）

LSTM 原理简介：
- LSTM 是一种循环神经网络（RNN）的变体，解决了普通 RNN 的梯度消失问题
- 通过"门"机制（输入门、遗忘门、输出门）控制信息的流动
- 适合处理序列数据（如文本），能捕捉长距离依赖

双向 LSTM (BiLSTM)：
- 同时从左到右（正向）和从右到左（反向）处理文本
- 正向捕捉"上文"信息，反向捕捉"下文"信息
- 拼接两个方向的输出，获得更丰富的上下文表示

架构：
1. Embedding 层：词ID -> 词向量
2. BiLSTM 层：捕捉上下文语义
3. 池化/取最后时刻：将变长序列转为定长向量
4. 全连接层 + Dropout：分类
"""
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 双向 LSTM
        # input_size: 输入特征维度（= embed_dim）
        # hidden_size: 隐藏状态维度
        # num_layers: LSTM 层数（堆叠多层 LSTM）
        # bidirectional=True: 双向
        # batch_first=True: 输入格式为 [batch_size, seq_len, features]
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 全连接分类层
        # 输入维度 = hidden_dim * 2（双向拼接）
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch_size, seq_len] 输入的词ID序列
        """
        # Step 1: 词嵌入
        # -> [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)

        # Step 2: BiLSTM
        # output: [batch_size, seq_len, hidden_dim * 2] 每个时刻的输出
        # (h_n, c_n): 最后时刻的隐藏状态和细胞状态
        output, (h_n, c_n) = self.lstm(embedded)

        # Step 3: 取最后时刻的输出作为句子表示
        # 或者用平均池化/最大池化，这里用最后时刻输出
        # 对于双向 LSTM，h_n 的形状是 [num_layers * 2, batch_size, hidden_dim]
        # 取最后一层的正向和反向隐藏状态拼接
        # h_n[-2]: 最后一层正向, h_n[-1]: 最后一层反向
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch_size, hidden_dim * 2]

        # Step 4: Dropout + 全连接
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)  # [batch_size, num_classes]

        return logits
