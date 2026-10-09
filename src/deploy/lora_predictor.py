# -*- coding: utf-8 -*-
"""
LoRA 合并模型的情绪预测后端（P1 第三路线）
=========================================
加载 saved_models/qwen15b_emotion_merged 的 bf16 完整权重（不依赖 peft/bnb），
输出结构与 EmotionPredictor 对齐并额外含：
  model_type="lora"、reason、valid_json、latency_ms
confidence 本期取 one-hot（合法 JSON=1.0，局限见评测报告 §5.3）。
"""
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config
from src.sft.prompts import build_messages
from src.sft.parse_output import parse_emotion_json


class LoRAEmotionPredictor:
    def __init__(self, device=None):
        cfg = load_config()
        self.ecfg = cfg["emotion"]
        fcfg = cfg.get("llm_finetune", {})
        self.model_path = fcfg.get("model_path",
                                   "saved_models/qwen15b_emotion_merged")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"LoRA 合并模型不存在: {self.model_path}（先运行 train_qlora + export_merged）")
        self.max_new_tokens = int(fcfg.get("max_new_tokens", 64))
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.labels_en = self.ecfg["labels"]
        self.labels_cn = self.ecfg["labels_cn"]
        self.emojis = self.ecfg["emojis"]
        self.polarity_map = self.ecfg["polarity"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(151643)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=torch.bfloat16,
            device_map=str(self.device)).eval()
        print(f"LoRA 情绪模型加载成功，设备: {self.device}，路径: {self.model_path}")

    def _format_result(self, text, pred_id, pred_cn, reason, valid, latency_ms):
        if valid:
            en = self.labels_en[pred_id]
            return {
                "text": text,
                "emotion": en,
                "emotion_cn": pred_cn,
                "emoji": self.emojis[pred_id],
                "polarity": self.polarity_map[en],
                "confidence": 1.0,  # one-hot 置信（生成式无概率，局限见报告）
                "reason": reason,
                "valid_json": True,
                "latency_ms": round(latency_ms, 1),
                "model_type": "lora",
            }
        return {
            "text": text, "emotion": None, "emotion_cn": None,
            "emoji": None, "polarity": None, "confidence": 0.0,
            "reason": reason, "valid_json": False,
            "latency_ms": round(latency_ms, 1), "model_type": "lora",
        }

    @torch.no_grad()
    def _generate(self, texts):
        self.tokenizer.padding_side = "left"
        prompts = [self.tokenizer.apply_chat_template(
            build_messages(str(t)), tokenize=False, add_generation_prompt=True)
            for t in texts]
        inp = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=256).to(self.device)
        out = self.model.generate(
            **inp, max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id)
        gen = out[:, inp["input_ids"].shape[1]:]
        self.tokenizer.padding_side = "right"
        return [self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in gen]

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()
        raw = self._generate([text])[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        lid, cn, reason, valid = parse_emotion_json(raw)
        return self._format_result(text, lid, cn, reason, valid, latency_ms)

    @torch.no_grad()
    def predict_batch(self, texts, batch_size=16) -> list:
        results, t0_all = [], time.perf_counter()
        for start in range(0, len(texts), batch_size):
            chunk = [str(t) for t in texts[start:start + batch_size]]
            raws = self._generate(chunk)
            for t, raw in zip(chunk, raws):
                lid, cn, reason, valid = parse_emotion_json(raw)
                results.append(self._format_result(
                    t, lid, cn, reason, valid, 0.0))
        total_ms = (time.perf_counter() - t0_all) * 1000
        for r in results:
            r["batch_total_ms"] = round(total_ms, 1)
        return results


_lora_predictor = None


def get_lora_predictor():
    global _lora_predictor
    if _lora_predictor is None:
        _lora_predictor = LoRAEmotionPredictor()
    return _lora_predictor
