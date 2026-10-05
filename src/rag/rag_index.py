# -*- coding: utf-8 -*-
"""
RAG 向量索引：本地 bert-base-chinese + mean-pooling，无需外网下载新模型
"""
import os
import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def mean_pooling(last_hidden, attention_mask):
    """对 token embedding 做 mean-pooling（排除 padding）"""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_emb = torch.sum(last_hidden * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return sum_emb / sum_mask


class RAGIndex:
    def __init__(self, index_dir, pretrained_model="saved_models/bert-base-chinese",
                 max_len=128, device=None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_model = pretrained_model
        self.max_len = max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载本地 bert-base-chinese 做编码器
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.encoder = AutoModel.from_pretrained(pretrained_model).to(self.device)
        self.encoder.eval()

        self.texts = []
        self.labels = []
        self.vectors = None

    def _encode(self, texts, batch_size=128):
        all_vecs = []
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start:start + batch_size]]
            enc = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=self.max_len, return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.encoder(input_ids, attention_mask=attention_mask)
                vecs = mean_pooling(outputs.last_hidden_state, attention_mask)
            all_vecs.append(vecs.cpu().numpy())
        return np.vstack(all_vecs)

    def build(self, csv_path):
        """从 CSV 构建索引并缓存"""
        cache_path = self.index_dir / "index.pkl"
        meta_path = self.index_dir / "meta.json"
        df = pd.read_csv(csv_path)
        texts = df["text"].tolist()
        labels = df[["label", "label_name", "label_cn"]].values.tolist()
        print(f"RAG 索引：编码 {len(texts)} 条语料 ...")
        vectors = self._encode(texts)
        # L2 归一化（方便余弦相似度 = 点积）
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        with open(cache_path, "wb") as f:
            pickle.dump({"texts": texts, "labels": labels, "vectors": vectors}, f)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"size": len(texts), "model": self.pretrained_model}, f)
        print(f"索引已缓存: {cache_path}")
        self.texts = texts
        self.labels = labels
        self.vectors = vectors

    def load(self):
        """加载缓存索引"""
        cache_path = self.index_dir / "index.pkl"
        if not cache_path.exists():
            return False
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.labels = data["labels"]
        self.vectors = data["vectors"]
        return True

    def search(self, query_text, top_k=5):
        """检索最相似的 top_k 条已标注样本"""
        if self.vectors is None:
            raise RuntimeError("索引未加载，请先 build() 或 load()")
        q_vec = self._encode([query_text])[0]
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-12)
        scores = np.dot(self.vectors, q_vec)  # 余弦相似度
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results = []
        for idx in top_idx:
            results.append({
                "text": self.texts[idx],
                "label": self.labels[idx][2],  # label_cn
                "similarity": round(float(scores[idx]), 4),
            })
        return results


if __name__ == "__main__":
    from src.utils import load_config
    cfg = load_config()
    rag_cfg = cfg.get("rag", {})
    index_dir = rag_cfg.get("index_dir", "data/rag_index")
    corpus_file = rag_cfg.get("corpus_file", "data/processed_emotion/train_augmented_v2_cleaned_fixed2.csv")
    pretrained = cfg["model"]["pretrained_model"]

    index = RAGIndex(index_dir=index_dir, pretrained_model=pretrained)
    if index.load():
        print(f"索引已加载，共 {len(index.texts)} 条")
    else:
        index.build(corpus_file)

    # 测试检索
    print("\n测试检索：")
    q = "好焦虑，事情太多做不完"
    print(f"查询: {q}")
    for r in index.search(q, top_k=3):
        print(f"  [{r['similarity']:.3f}] {r['label']} | {r['text'][:40]}")
