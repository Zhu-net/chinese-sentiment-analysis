# -*- coding: utf-8 -*-
"""
情绪 6 分类推理模块
==================
加载微调好的 BERT 情绪模型，输出：
- 主情绪（中/英文 + emoji）
- 情绪对应的极性（正面/负面，兼容二分类语义）
- 6 类概率分布 + Top-3 情绪
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config
from src.models.bert_classifier import BERTClassifier


class EmotionPredictor:
    def __init__(self, device=None):
        cfg = load_config()
        self.ecfg = cfg["emotion"]
        save_dir = Path(cfg["train"]["save_dir"])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            save_dir / self.ecfg["model_file"], map_location=self.device, weights_only=False
        )
        model_cfg = checkpoint["config"]

        # tokenizer：优先微调时保存的，其次骨架模型目录
        tok_dir = save_dir / self.ecfg["tokenizer_dir"]
        tok_path = str(tok_dir) if tok_dir.exists() else checkpoint["pretrained_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

        self.model = BERTClassifier(
            pretrained_model_name=checkpoint["pretrained_model"],
            num_classes=model_cfg["num_classes"],
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.max_len = model_cfg["max_len"]
        self.labels_en = self.ecfg["labels"]
        self.labels_cn = self.ecfg["labels_cn"]
        self.emojis = self.ecfg["emojis"]
        self.polarity_map = self.ecfg["polarity"]
        print(f"情绪模型加载成功，设备: {self.device}")

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        encoding = self.tokenizer(
            text, truncation=True, padding=True,  # 动态 padding
            max_length=self.max_len, return_tensors="pt",
        )
        logits = self.model(
            encoding["input_ids"].to(self.device),
            encoding["attention_mask"].to(self.device),
        )
        probs = F.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())

        # Top-3
        top3_vals, top3_ids = torch.topk(probs, k=3)
        top3 = [
            {
                "emotion": self.labels_en[int(i)],
                "emotion_cn": self.labels_cn[int(i)],
                "emoji": self.emojis[int(i)],
                "probability": round(float(v), 4),
            }
            for v, i in zip(top3_vals, top3_ids)
        ]

        emotion_en = self.labels_en[pred_id]
        return {
            "text": text,
            "emotion": emotion_en,
            "emotion_cn": self.labels_cn[pred_id],
            "emoji": self.emojis[pred_id],
            "polarity": self.polarity_map[emotion_en],
            "confidence": round(float(probs[pred_id]), 4),
            "probabilities": {
                self.labels_cn[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
            "top3": top3,
        }

    @torch.no_grad()
    def predict_batch(self, texts, batch_size=32, with_probabilities=True) -> list:
        results = []
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start:start + batch_size]]
            enc = self.tokenizer(
                batch, truncation=True, padding=True,  # 动态 padding
                max_length=self.max_len, return_tensors="pt",
            )
            logits = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
            probs = F.softmax(logits, dim=1)
            pred_ids = torch.argmax(probs, dim=1)
            for j, text in enumerate(batch):
                pid = int(pred_ids[j])
                en = self.labels_en[pid]
                item = {
                    "text": text,
                    "emotion": en,
                    "emotion_cn": self.labels_cn[pid],
                    "emoji": self.emojis[pid],
                    "polarity": self.polarity_map[en],
                    "confidence": round(float(probs[j][pid]), 4),
                }
                if with_probabilities:
                    # 与单条 predict 接口对称：返回 6 类全概率
                    item["probabilities"] = {
                        self.labels_cn[i]: round(float(p), 4)
                        for i, p in enumerate(probs[j])
                    }
                results.append(item)
        return results


# 全局单例
_emotion_predictor = None


def get_emotion_predictor():
    global _emotion_predictor
    if _emotion_predictor is None:
        _emotion_predictor = EmotionPredictor()
    return _emotion_predictor
