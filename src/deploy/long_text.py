# -*- coding: utf-8 -*-
"""
长文本情绪分析
==============
- 智能分句：句号/问号/叹号/分号/换行切分；超长句再按逗号切；过滤碎片
- 逐句复用 EmotionPredictor.predict_batch（批量推理，100 句约 1 秒级）
- 聚合：情绪分布、主导情绪、正/负面句占比、情绪转换点
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_config

_CFG = load_config()
_MAX_SENTENCES = int(_CFG["app"]["long_text"]["max_sentences"])
_MIN_LEN = int(_CFG["app"]["long_text"]["min_sentence_len"])

# 中文标签 -> 极性映射（pos_score 计算用）
_CN2POLARITY = {
    _cn: _CFG["emotion"]["polarity"][_en]
    for _en, _cn in zip(_CFG["emotion"]["labels"], _CFG["emotion"]["labels_cn"])
}

# 句末标点（保留可读的切分，不保留标点本身）
_SENT_SPLIT = re.compile(r"[。！？!?；;\n]+")
# 过长句子的二级切分
_COMMA_SPLIT = re.compile(r"[，,、]+")
# 单句软上限：超过则触发二级切分
_HARD_SENT_LEN = 60


def split_sentences(text: str) -> list:
    """中文长文本 -> 干净的句子列表"""
    sentences = []
    for chunk in _SENT_SPLIT.split(str(text)):
        chunk = chunk.strip()
        if len(chunk) < _MIN_LEN:
            continue
        if len(chunk) <= _HARD_SENT_LEN:
            sentences.append(chunk)
        else:
            # 长句按逗号二次切分，相邻碎片合并到至少 4 字
            buf = ""
            for piece in _COMMA_SPLIT.split(chunk):
                piece = piece.strip()
                if not piece:
                    continue
                if len(buf) < 4:
                    buf += piece
                else:
                    sentences.append(buf)
                    buf = piece
            if buf:
                sentences.append(buf)
    return sentences[:_MAX_SENTENCES]


def analyze_long_text(text: str, predictor) -> dict:
    """长文本分析主流程。

    predictor: EmotionPredictor 实例（复用已加载模型，不重复占显存）
    返回：
    {
      "sentences": [{idx, text, emotion, emotion_cn, emoji, polarity,
                     confidence, pos_score}],
      "summary": {total, pos_count, neg_count, neutral_count,
                  pos_ratio, neg_ratio, dominant_emotion, dominant_emotion_en,
                  dominant_ratio, distribution, switches,
                  overall_polarity, overall_pos_score}
    }
    pos_score: Σ正面类概率 - Σ负面类概率，范围 [-1, 1]，给走势图做纵轴
    overall_polarity: 整段极性（正面/负面/中性），由全句 pos_score 均值 + 阈值(±0.2)判定
    """
    sentences = split_sentences(text)
    if not sentences:
        return {"sentences": [], "summary": {}}

    raw_results = predictor.predict_batch(sentences, batch_size=32)

    sent_items = []
    dist = Counter()
    pos_n = neg_n = neu_n = 0
    prev_polarity = None
    switches = 0
    pos_scores = []

    for idx, (sent, r) in enumerate(zip(sentences, raw_results)):
        probs = r.get("probabilities")
        if probs:
            # 严格口径：pos_score = Σ正面类概率 - Σ负面类概率，范围 [-1, 1]
            pos_score = (
                sum(float(p) for cn, p in probs.items() if _CN2POLARITY.get(cn) == "正面")
                - sum(float(p) for cn, p in probs.items() if _CN2POLARITY.get(cn) == "负面")
            )
        else:
            # 兜底：推理器未返回概率时退化为带符号置信度
            pos_score = float(r["confidence"]) if r["polarity"] == "正面" else -float(r["confidence"])
        if r["polarity"] == "正面":
            pos_n += 1
        else:
            neg_n += 1
        pos_scores.append(pos_score)

        # 极性转换点（正面↔负面）
        if prev_polarity is not None and r["polarity"] != prev_polarity:
            switches += 1
        prev_polarity = r["polarity"]

        dist[r["emotion_cn"]] += 1
        sent_items.append({
            "idx": idx,
            "text": sent,
            "emotion": r["emotion"],
            "emotion_cn": r["emotion_cn"],
            "emoji": r["emoji"],
            "polarity": r["polarity"],
            "confidence": r["confidence"],
            "pos_score": round(pos_score, 4),
        })

    total = len(sent_items)
    dom_cn, dom_cnt = dist.most_common(1)[0]
    # 中文 -> 英文键（给存储/导出用）
    cn2en = dict(zip(_CFG["emotion"]["labels_cn"], _CFG["emotion"]["labels"]))

    # 整段极性：全句 pos_score 均值 + 阈值判定
    mean_pos = sum(pos_scores) / total if total else 0.0
    if mean_pos >= 0.2:
        overall_polarity = "正面"
    elif mean_pos <= -0.2:
        overall_polarity = "负面"
    else:
        overall_polarity = "中性"

    return {
        "sentences": sent_items,
        "summary": {
            "total": total,
            "pos_count": pos_n,
            "neg_count": neg_n,
            "neutral_count": neu_n,
            "pos_ratio": round(pos_n / total, 4),
            "neg_ratio": round(neg_n / total, 4),
            "dominant_emotion": dom_cn,
            "dominant_emotion_en": cn2en.get(dom_cn),
            "dominant_ratio": round(dom_cnt / total, 4),
            "distribution": dict(dist),
            "switches": switches,
            "overall_polarity": overall_polarity,
            "overall_pos_score": round(mean_pos, 4),
        },
    }
