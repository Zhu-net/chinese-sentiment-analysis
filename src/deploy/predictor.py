"""
模型推理模块
加载训练好的 BERT 模型，提供统一的预测接口
"""
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

# HuggingFace 国内镜像（防止模型加载时访问外网）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.utils import load_config
from src.models.bert_classifier import BERTClassifier


class SentimentPredictor:
    """情感分析预测器"""

    def __init__(self, model_path=None, device=None):
        cfg = load_config()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载配置
        if model_path is None:
            model_path = Path(cfg["train"]["save_dir"]) / "bert_best.pt"

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint["config"]
        pretrained_model = checkpoint["pretrained_model"]

        # 加载 tokenizer（优先用保存的，其次用原始预训练模型的）
        tokenizer_path = Path(cfg["train"]["save_dir"]) / "bert_tokenizer"
        if tokenizer_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        # 加载模型
        self.model = BERTClassifier(
            pretrained_model_name=pretrained_model,
            num_classes=model_config["num_classes"],
            dropout=0.0,  # 推理时不需要 dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()  # 设置为评估模式

        self.max_len = model_config["max_len"]
        self.label_map = {0: "负面", 1: "正面"}

        print(f"模型加载成功，设备: {self.device}")

    @torch.no_grad()
    def predict(self, text):
        """
        单条文本预测
        返回: {label, confidence, probabilities}
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,  # 动态 padding：短文本按实际长度计算，节省算力
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits = self.model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1)  # 转为概率

        pred_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_label].item()

        return {
            "text": text,
            "label": self.label_map[pred_label],
            "label_id": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "负面": round(probabilities[0][0].item(), 4),
                "正面": round(probabilities[0][1].item(), 4),
            },
        }

    @torch.no_grad()
    def predict_batch(self, texts, batch_size=32):
        """批量预测"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,  # 动态 padding：按 batch 内最长文本对齐
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1)
            pred_labels = torch.argmax(probabilities, dim=1)

            for j, text in enumerate(batch_texts):
                pred_label = pred_labels[j].item()
                results.append({
                    "text": text,
                    "label": self.label_map[pred_label],
                    "label_id": pred_label,
                    "confidence": round(probabilities[j][pred_label].item(), 4),
                })
        return results


# 全局单例（避免重复加载模型）
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor
