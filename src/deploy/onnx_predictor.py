# -*- coding: utf-8 -*-
"""
ONNX 推理器
===========
用 onnxruntime 替代 PyTorch 推理。
优势：不需要 torch 依赖，镜像体积从 ~6GB 降至 ~1GB，冷启动快。

支持自动降级：ONNX 文件不存在时自动回退到 PyTorch 推理器。
"""
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config


class OnnxEmotionPredictor:
    """ONNX 情绪模型推理器（接口与 EmotionPredictor 完全兼容）"""

    def __init__(self, device=None):
        cfg = load_config()
        self.ecfg = cfg["emotion"]
        save_dir = Path(cfg["train"]["save_dir"])
        onnx_path = save_dir / "bert_emotion.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}，请先运行 export_onnx.py")

        # 加载 tokenizer
        tok_dir = save_dir / self.ecfg["tokenizer_dir"]
        tok_path = str(tok_dir) if tok_dir.exists() else cfg["model"]["pretrained_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

        # ONNX Runtime session（CPU）
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name_ids = self.session.get_inputs()[0].name
        self.input_name_mask = self.session.get_inputs()[1].name

        self.max_len = cfg["data"]["max_length"]
        self.labels_en = self.ecfg["labels"]
        self.labels_cn = self.ecfg["labels_cn"]
        self.emojis = self.ecfg["emojis"]
        self.polarity_map = self.ecfg["polarity"]
        print(f"ONNX 情绪模型加载成功（onnxruntime CPU）")

    def _softmax(self, logits):
        """数值稳定的 softmax"""
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def predict(self, text: str) -> dict:
        enc = self.tokenizer(
            str(text), truncation=True, padding=True,  # 动态 padding
            max_length=self.max_len, return_tensors="np",
        )
        logits = self.session.run(None, {
            self.input_name_ids: enc["input_ids"].astype(np.int64),
            self.input_name_mask: enc["attention_mask"].astype(np.int64),
        })[0]
        probs = self._softmax(logits)[0]
        pred_id = int(np.argmax(probs))

        # Top-3
        top3_ids = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "emotion": self.labels_en[int(i)],
                "emotion_cn": self.labels_cn[int(i)],
                "emoji": self.emojis[int(i)],
                "probability": round(float(probs[i]), 4),
            }
            for i in top3_ids
        ]

        en = self.labels_en[pred_id]
        return {
            "text": text,
            "emotion": en,
            "emotion_cn": self.labels_cn[pred_id],
            "emoji": self.emojis[pred_id],
            "polarity": self.polarity_map[en],
            "confidence": round(float(probs[pred_id]), 4),
            "probabilities": {
                self.labels_cn[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
            "top3": top3,
        }

    def predict_batch(self, texts, batch_size=32, with_probabilities=True) -> list:
        results = []
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start:start + batch_size]]
            enc = self.tokenizer(
                batch, truncation=True, padding=True,  # 动态 padding
                max_length=self.max_len, return_tensors="np",
            )
            logits = self.session.run(None, {
                self.input_name_ids: enc["input_ids"].astype(np.int64),
                self.input_name_mask: enc["attention_mask"].astype(np.int64),
            })[0]
            probs = self._softmax(logits)
            pred_ids = np.argmax(probs, axis=1)
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
                    item["probabilities"] = {
                        self.labels_cn[i]: round(float(p), 4)
                        for i, p in enumerate(probs[j])
                    }
                results.append(item)
        return results


class OnnxBinaryPredictor:
    """ONNX 二分类推理器（接口与 SentimentPredictor 兼容）"""

    def __init__(self, device=None):
        cfg = load_config()
        save_dir = Path(cfg["train"]["save_dir"])
        onnx_path = save_dir / "bert_binary.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")

        tok_dir = save_dir / "bert_tokenizer"
        tok_path = str(tok_dir) if tok_dir.exists() else cfg["model"]["pretrained_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name_ids = self.session.get_inputs()[0].name
        self.input_name_mask = self.session.get_inputs()[1].name

        self.max_len = cfg["data"]["max_length"]
        self.label_map = {0: "负面", 1: "正面"}
        print(f"ONNX 二分类模型加载成功（onnxruntime CPU）")

    def _softmax(self, logits):
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def predict(self, text: str) -> dict:
        enc = self.tokenizer(
            str(text), truncation=True, padding=True,  # 动态 padding
            max_length=self.max_len, return_tensors="np",
        )
        logits = self.session.run(None, {
            self.input_name_ids: enc["input_ids"].astype(np.int64),
            self.input_name_mask: enc["attention_mask"].astype(np.int64),
        })[0]
        probs = self._softmax(logits)[0]
        pred_id = int(np.argmax(probs))
        return {
            "text": text,
            "label": self.label_map[pred_id],
            "label_id": pred_id,
            "confidence": round(float(probs[pred_id]), 4),
            "probabilities": {
                "负面": round(float(probs[0]), 4),
                "正面": round(float(probs[1]), 4),
            },
        }

    def predict_batch(self, texts, batch_size=32) -> list:
        results = []
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start:start + batch_size]]
            enc = self.tokenizer(
                batch, truncation=True, padding=True,  # 动态 padding
                max_length=self.max_len, return_tensors="np",
            )
            logits = self.session.run(None, {
                self.input_name_ids: enc["input_ids"].astype(np.int64),
                self.input_name_mask: enc["attention_mask"].astype(np.int64),
            })[0]
            probs = self._softmax(logits)
            pred_ids = np.argmax(probs, axis=1)
            for j, text in enumerate(batch):
                pid = int(pred_ids[j])
                results.append({
                    "text": text,
                    "label": self.label_map[pid],
                    "label_id": pid,
                    "confidence": round(float(probs[j][pid]), 4),
                })
        return results


# 智能工厂：优先 ONNX，回退 PyTorch
_onnx_emotion = None
_onnx_binary = None

def get_onnx_emotion_predictor():
    """优先返回 ONNX 推理器，不存在则回退到 PyTorch"""
    global _onnx_emotion
    if _onnx_emotion is None:
        try:
            _onnx_emotion = OnnxEmotionPredictor()
        except FileNotFoundError:
            from src.deploy.emotion_predictor import get_emotion_predictor
            _onnx_emotion = get_emotion_predictor()
    return _onnx_emotion

def get_onnx_binary_predictor():
    global _onnx_binary
    if _onnx_binary is None:
        try:
            _onnx_binary = OnnxBinaryPredictor()
        except FileNotFoundError:
            from src.deploy.predictor import get_predictor
            _onnx_binary = get_predictor()
    return _onnx_binary
