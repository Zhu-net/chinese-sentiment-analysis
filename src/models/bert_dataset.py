"""
BERT 数据集类
使用 HuggingFace 的 BERT Tokenizer（子词分词），而不是 jieba

BERT Tokenizer 的特点：
- 使用 WordPiece 子词分词，能处理未登录词（OOV）
- 自动添加 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>、[SEP] 等特殊标记
- 生成 input_ids、attention_mask、token_type_ids
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # BERT tokenizer 编码
        # truncation=True: 超过 max_len 截断
        # padding='max_length': 填充到 max_len
        # return_tensors='pt': 返回 PyTorch 张量
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # encoding 中的张量形状是 [1, max_len]，需要 squeeze 掉 batch 维
        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # [max_len]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [max_len]
            "labels": torch.tensor(label, dtype=torch.long),
        }
