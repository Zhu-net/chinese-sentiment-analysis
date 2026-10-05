# -*- coding: utf-8 -*-
"""
RAG 检索增强精判：BERT 低置信度 → 检索相似样本 → DeepSeek 精判
"""
import os
import sys
import json
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.utils import load_config
from src.llm.llm_client import DeepSeekClient
from src.rag.rag_index import RAGIndex

LABELS_CN = ["开心", "感激", "悲伤", "愤怒", "恐惧", "焦虑"]
LABELS_EN = ["happy", "grateful", "sad", "angry", "fear", "anxious"]
CN_TO_IDX = {cn: i for i, cn in enumerate(LABELS_CN)}

PROMPT_TMPL = """你是中文微博情绪分类专家。请根据下面的参考样本，判断目标文本的主导情绪。

六类情绪定义：
- 开心：高兴、喜悦、满足、期待
- 感激：感谢、感恩
- 悲伤：难过、失落、无奈、沮丧、哀叹、委屈
- 愤怒：生气、发怒、指责、强烈不满
- 恐惧：害怕、惊恐、恐慌（对已发生危险的本能害怕）
- 焦虑：紧张、担忧、不安、纠结、烦躁（对未发生事件/持续压力的焦灼，如考试、工作、等结果）

参考样本（相似度高的已标注数据）：
{examples}

目标文本：
"{text}"

请输出 JSON（不要多余文字）：
{{"emotion": "六类之一", "confidence": "高/中/低", "reason": "20字内中文理由"}}
"""


class RAGRefiner:
    def __init__(self):
        cfg = load_config()
        self.rag_cfg = cfg.get("rag", {})
        self.enabled = self.rag_cfg.get("enabled", False)
        if not self.enabled:
            return

        self.threshold = float(self.rag_cfg.get("confidence_threshold", 0.75))
        self.anxious_gap = float(self.rag_cfg.get("anxious_gap_threshold", 0.3))
        self.top_k = int(self.rag_cfg.get("top_k", 5))

        # 加载向量索引
        index_dir = self.rag_cfg.get("index_dir", "data/rag_index")
        corpus_file = self.rag_cfg.get(
            "corpus_file",
            "data/processed_emotion/train_augmented_v2_cleaned_fixed2.csv",
        )
        pretrained = cfg["model"]["pretrained_model"]
        self.index = RAGIndex(index_dir=index_dir, pretrained_model=pretrained)
        if not self.index.load():
            self.index.build(corpus_file)

        # LLM
        api_key = self.rag_cfg.get("deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY")
        self.llm = DeepSeekClient(api_key=api_key)

    def _build_prompt(self, text, similar):
        lines = []
        for i, r in enumerate(similar, 1):
            lines.append(f"{i}. [{r['label']}] {r['text']}")
        examples = "\n".join(lines)
        return PROMPT_TMPL.format(examples=examples, text=text)

    def _parse_llm_json(self, raw):
        """解析 LLM 返回的 JSON，容错处理 markdown 代码块"""
        s = raw.strip()
        if "```" in s:
            s = s.split("```")[1]
            if s.startswith("json"):
                s = s[4:]
        s = s.strip()
        # 找到第一个 { 和最后一个 }
        l, r = s.find("{"), s.rfind("}")
        if l == -1 or r == -1:
            raise ValueError(f"无法解析 JSON: {raw[:100]}")
        data = json.loads(s[l:r + 1])
        emotion = data.get("emotion", "").strip()
        if emotion not in LABELS_CN:
            raise ValueError(f"LLM 返回无效情绪: {emotion}")
        return {
            "emotion_cn": emotion,
            "confidence_text": data.get("confidence", ""),
            "reason": str(data.get("reason", "")).strip()[:60],
        }

    def refine(self, bert_result):
        """
        对 BERT 低置信度结果做 RAG 精判
        :param bert_result: EmotionPredictor.predict() 的返回值
        :return: 精判后的 dict（含 refined / refine_source / reason / similar_samples）
        """
        result = dict(bert_result)
        result["refined"] = False
        result["refine_source"] = "bert"
        result["reason"] = ""
        result["similar_samples"] = []

        if not self.enabled:
            return result

        # 触发条件 1：低置信度
        low_conf = bert_result["confidence"] < self.threshold
        # 触发条件 2：预测为焦虑但 top1-top2 概率差过小（焦虑系统性过判的双保险）
        anxious_uncertain = False
        if bert_result["emotion"] == "anxious":
            top3 = bert_result.get("top3", [])
            if len(top3) >= 2:
                gap = top3[0]["probability"] - top3[1]["probability"]
                # 用 <=：阈值设为 1.0 时覆盖"top1=1.0/top2=0.0"的满格置信情况
                anxious_uncertain = gap <= self.anxious_gap

        if not low_conf and not anxious_uncertain:
            return result

        # 低置信度：检索 + LLM 精判
        text = bert_result["text"]
        try:
            similar = self.index.search(text, top_k=self.top_k)
            prompt = self._build_prompt(text, similar)
            raw = self.llm.generate(prompt, temperature=0.0, max_tokens=300)
            parsed = self._parse_llm_json(raw)

            pred_cn = parsed["emotion_cn"]
            pred_idx = CN_TO_IDX[pred_cn]
            result.update({
                "emotion": LABELS_EN[pred_idx],
                "emotion_cn": pred_cn,
                "emoji": bert_result["emoji"],  # 保持原 emoji 结构（若需可重映射）
                "confidence": 0.95,  # LLM 精判视为高置信
                "refined": True,
                "refine_source": "rag_llm",
                "reason": parsed["reason"],
                "similar_samples": similar,
            })
            # 修正 emoji / polarity（与 6 类一致）
            cfg = load_config()
            ecfg = cfg["emotion"]
            result["emoji"] = ecfg["emojis"][pred_idx]
            result["polarity"] = ecfg["polarity"][LABELS_EN[pred_idx]]
        except Exception as e:
            result["refine_source"] = "rag_failed"
            result["reason"] = f"RAG 失败，回退 BERT: {str(e)[:80]}"

        return result


# 全局单例
_refiner = None


def get_rag_refiner():
    global _refiner
    if _refiner is None:
        _refiner = RAGRefiner()
    return _refiner


if __name__ == "__main__":
    from src.deploy.emotion_predictor import get_emotion_predictor

    predictor = get_emotion_predictor()
    refiner = get_rag_refiner()

    tests = [
        "今天心情真好，阳光明媚",           # 高置信，不触发 RAG
        "好焦虑，事情太多做不完",           # 低置信，触发 RAG
        "被客户经理放了鸽子…气呼呼的回到办公室",  # 边界模糊
    ]
    for t in tests:
        r = predictor.predict(t)
        print(f"\n输入: {t}")
        print(f"BERT: {r['emotion_cn']} ({r['confidence']:.3f})")
        r2 = refiner.refine(r)
        print(f"RAG:  {r2['emotion_cn']} (refined={r2['refined']}, source={r2['refine_source']})")
        if r2["reason"]:
            print(f"     理由: {r2['reason']}")
