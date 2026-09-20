"""
PyTorch 文本数据集类
负责将文本数据转化为模型可以处理的张量（Tensor）

核心概念：
1. Dataset: PyTorch 的数据集抽象，需要实现 __len__ 和 __getitem__
2. DataLoader: 批量加载数据，自动 shuffle、并行加载
3. 词表 (Vocab): 把词语映射为整数 ID，再通过 Embedding 层转为向量
"""
import jieba
from collections import Counter
import torch
from torch.utils.data import Dataset


class Vocab:
    """词表类：词语 <-> 整数ID 的映射"""

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}  # PAD 填充, UNK 未知词
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        """从文本列表构建词表"""
        counter = Counter()
        for text in texts:
            words = jieba.lcut(text)
            counter.update(words)

        # 按词频排序，只保留出现次数 >= min_freq 的词
        idx = len(self.word2idx)
        for word, freq in counter.most_common():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text, max_len):
        """将文本编码为整数序列，并做 padding/truncation"""
        words = jieba.lcut(text)
        ids = [self.word2idx.get(w, 1) for w in words]  # 1 = <UNK>
        # 截断或填充到 max_len
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))  # 0 = <PAD>
        return ids

    def __len__(self):
        return len(self.word2idx)


class TextDataset(Dataset):
    """文本数据集，继承 PyTorch 的 Dataset 类"""

    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """返回第 idx 个样本（文本张量 + 标签）"""
        text = self.texts[idx]
        label = self.labels[idx]
        # 文本 -> 整数序列
        ids = self.vocab.encode(text, self.max_len)
        # 转为 PyTorch 张量
        input_ids = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return input_ids, label
