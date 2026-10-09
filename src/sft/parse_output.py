# -*- coding: utf-8 -*-
"""LoRA 模型输出的容错 JSON 解析（训练选模 / FastAPI 后端 / 评测共用）"""
import json
import re

from src.sft.prompts import CN2LABEL


def parse_emotion_json(text: str):
    """
    解析模型生成文本。
    返回: (label_id: int|None, emotion_cn: str|None, reason: str, valid: bool)
    解析失败或枚举非法时 valid=False，调用方按失败处理（如回退 BERT）。
    """
    if not text:
        return None, None, "", False
    s = text.strip()

    # 去除 markdown 代码块包裹
    if "```" in s:
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = s.rstrip("`").strip()

    # 截取首个 {...}（容忍前后多余字符）
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        frag = m.group(0)
        try:
            obj = json.loads(frag)
        except Exception:
            # 容错：去除尾部逗号等常见小瑕疵后再试一次
            try:
                obj = json.loads(re.sub(r",\s*}", "}", frag))
            except Exception:
                return None, None, "", False
        emo = str(obj.get("emotion", "")).strip()
        if emo in CN2LABEL:
            reason = str(obj.get("reason", "")).strip()
            return CN2LABEL[emo], emo, reason, True
        return None, None, "", False

    # 无 JSON 时的兜底：直接命中枚举词（标记为非法 JSON，由上层统计）
    return None, None, "", False
